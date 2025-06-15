import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import mean

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from loguru import logger as log
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class Args:
    env_id: str = "dm_control/cartpole-balance-v0"
    exp_name: str = "ppo"
    seed: int = 0
    num_envs: int = 1
    num_steps: int = 2048
    clip_coef: float = 0.2
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 10
    total_timesteps: int = 1000000
    anneal_lr: bool = True


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        #! below are important
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        obs_dim = np.prod(envs.single_observation_space.shape)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        # XXX: why log?
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(dim=1), probs.entropy().sum(dim=1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)


def main():
    args = tyro.cli(Args)
    batch_size = args.num_envs * args.num_steps
    mini_batch_size = batch_size // 32
    num_iterations = args.total_timesteps // batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
    seed_everything(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__lr={args.lr:.0e}__seed={args.seed}__{timestamp}"
    writer = SummaryWriter(f"logdir/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, False, run_name) for i in range(args.num_envs)])

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.from_numpy(obs).float().to(device)
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obss = torch.zeros((args.num_steps + 1, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps + 1, args.num_envs) + envs.single_action_space.shape, device=device)
    log_probs = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    values = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    advantages = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps + 1, args.num_envs), device=device)

    obss[0] = obs
    dones[0] = torch.zeros(args.num_envs).to(device)

    start_time = time.time()
    global_step = 0

    for iteration in tqdm(range(num_iterations)):
        ## anneal lr
        if args.anneal_lr:
            lr = args.lr * (1 - iteration / num_iterations)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        ## collect rollout
        for t in range(args.num_steps):
            global_step += args.num_envs

            with torch.no_grad():
                action, log_prob, entropy = agent.get_action(obs)
                value = agent.get_value(obs)

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)

            actions[t] = action
            log_probs[t] = log_prob
            values[t] = value.view(-1)
            rewards[t] = torch.from_numpy(reward).float().to(device).view(-1)
            obss[t + 1] = torch.from_numpy(next_obs).float().to(device)
            dones[t + 1] = torch.from_numpy(next_done).float().to(device)

            obs = obss[t + 1]

            if "final_info" in infos:
                episode_returns = []
                episode_lengths = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        print(f"global_step={global_step}, episode_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return_mean", np.mean(episode_returns), global_step)
                writer.add_scalar("charts/episodic_return_min", np.min(episode_returns), global_step)
                writer.add_scalar("charts/episodic_return_max", np.max(episode_returns), global_step)
                writer.add_scalar("charts/episodic_length_mean", np.mean(episode_lengths), global_step)

        values[args.num_steps] = agent.get_value(obss[args.num_steps]).view(-1)

        ## bootstrap (XXX: what is this?)
        with torch.no_grad():
            advantages[args.num_steps] = 0
            for t in reversed(range(args.num_steps)):
                # Generalized Advantage Estimation (core-5, see original paper)
                delta = rewards[t] + args.gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
                advantages[t] = delta + args.gamma * args.gae_lambda * advantages[t + 1] * (1 - dones[t + 1])
            returns = advantages + values

        ## flatten and random shuffle
        random_indices = torch.randperm(args.num_envs * args.num_steps)
        b_obs = obss[:-1].view(-1, *envs.single_observation_space.shape)[random_indices]
        b_log_probs = log_probs[:-1].view(-1)[random_indices]
        b_actions = actions[:-1].view(-1, *envs.single_action_space.shape)[random_indices]
        b_returns = returns[:-1].view(-1)[random_indices]
        b_advantages = advantages[:-1].view(-1)[random_indices]
        b_values = values[:-1].view(-1)[random_indices]

        ## optimize
        for epoch in range(args.epochs):
            # Mini-batch updates (core-6)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size

                #! important to use old actions
                _, new_log_probs, entropy = agent.get_action(b_obs[start:end], b_actions[start:end])
                new_value = agent.get_value(b_obs[start:end]).view(-1)
                logratio = new_log_probs - b_log_probs[start:end]
                ratio = logratio.exp()

                mb_advantages = b_advantages[start:end]

                # Normalize the advantages (core-7)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # actor loss
                a_loss1 = mb_advantages * ratio
                a_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                a_loss = -torch.min(a_loss1, a_loss2).mean()

                # critic loss
                c_loss = 0.5 * (new_value - b_returns[start:end]).pow(2).mean()

                # total loss
                loss = a_loss + 0.5 * c_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"SPS: {global_step / (time.time() - start_time):.2f}")

    writer.close()


if __name__ == "__main__":
    main()
