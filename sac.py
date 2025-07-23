from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from dataclasses import dataclass
from datetime import datetime
from itertools import chain

os.environ["MUJOCO_GL"] = "egl"  # significantly faster rendering compared to glfw and osmesa

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from loguru import logger as log
from torch import Tensor
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rvrl.envs.env_factory import create_vector_env
from rvrl.utils.reproducibility import enable_deterministic_run, seed_everything


########################################################
## Standalone utils
########################################################
class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int],
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        state_type = np.uint8 if len(observation_shape) < 3 else np.float32

        self.observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=state_type)
        self.next_observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=state_type)
        self.action = np.empty((self.capacity, self.num_envs, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: Tensor,
        next_observation: Tensor,
        done: Tensor,
        terminated: Tensor,
    ):
        self.observation[self.buffer_index] = observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().cpu().numpy()
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        assert self.full or (self.buffer_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else self.buffer_index, batch_size)
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = torch.as_tensor(flatten(self.observation)[flattened_index], device=self.device).float()
        next_observation = torch.as_tensor(flatten(self.next_observation)[flattened_index], device=self.device).float()
        action = torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
        reward = torch.as_tensor(flatten(self.reward)[flattened_index], device=self.device)
        done = torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
        terminated = torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "terminated": terminated,
        }
        return sample


########################################################
## Args
########################################################
@dataclass
class Args:
    exp_name: str = "sac"
    seed: int = 0
    device: str = "cuda"
    deterministic: bool = True
    env_id: str = "dm_control/walker-walk-v0"
    num_envs: int = 1
    buffer_size: int = 1_000_000
    total_timesteps: int = 1000_000
    prefill: int = 5000

    ## train
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99  # discount factor
    policy_frequency: int = 2
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    alpha: float = 0.2
    tau: float = 0.005


args = tyro.cli(Args)


########################################################
## Networks
########################################################
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(np.prod(env.single_observation_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * np.prod(env.single_action_space.shape)),
        )

    def forward(self, obs):
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std_min, log_std_max = -5, 2
        log_std = log_std_min + (log_std_max - log_std_min) * (F.tanh(log_std) + 1) / 2
        return mean, log_std

    def get_action(self, obs):
        # assume action is bounded in [-1, 1], which is the value range of tanh. Otherwise we need to add an affine transform

        ## Option 1
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        action_dist = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )  # ! use cache_size=1 to avoid atanh which could cause nan
        action_dist = Independent(action_dist, 1)
        action = action_dist.rsample()
        return action, action_dist.log_prob(action).unsqueeze(-1)

        ## Option 2
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action**2) + 1e-6)  # ! 1e-6 to avoid nan
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.qnet = nn.Sequential(
            nn.Linear(np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.qnet(x)


########################################################
## Main
########################################################

## setup
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
seed_everything(args.seed)
if args.deterministic:
    enable_deterministic_run()


## logger
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__seed={args.seed}__{_timestamp}"
logdir = f"logdir/{run_name}"
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

## env and replay buffer
envs = create_vector_env(args.env_id, "proprio", args.num_envs, args.seed)
buffer = ReplayBuffer(
    envs.single_observation_space.shape, envs.single_action_space.shape[0], device, args.num_envs, args.buffer_size
)

## networks
actor = Actor(envs).to(device)
qf1 = Critic(envs).to(device)
qf2 = Critic(envs).to(device)
qf1_target = Critic(envs).to(device)
qf2_target = Critic(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_optimizer = torch.optim.Adam(chain(qf1.parameters(), qf2.parameters()), lr=args.critic_lr)


def main():
    global_step = 0
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    episodic_return = torch.zeros(args.num_envs, device=device)

    obs, _ = envs.reset(seed=args.seed)
    alpha = args.alpha
    while global_step < args.total_timesteps:
        ## Step the environment and add to buffer
        with torch.inference_mode():
            if global_step < args.prefill:
                action = torch.as_tensor(envs.action_space.sample(), device=device)
            else:
                action, _ = actor.get_action(obs)
            next_obs, reward, terminated, truncated, info = envs.step(action)
            done = torch.logical_or(terminated, truncated)
            buffer.add(obs, action, reward, next_obs, done, terminated)
            obs = next_obs
            episodic_return += reward
            if done.any():
                writer.add_scalar("reward/episodic_return", episodic_return[done].mean().item(), global_step)
                print(f"global_step={global_step}, episodic_return={episodic_return[done].mean().item()}")
                episodic_return[done] = 0

        ## Update the model
        if global_step >= args.prefill:
            data = buffer.sample(args.batch_size)
            with torch.no_grad():
                next_action, next_log_prob = actor.get_action(data["next_observation"])
                qf1_next_target = qf1_target(data["next_observation"], next_action)
                qf2_next_target = qf2_target(data["next_observation"], next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_prob
                next_q = data["reward"] + (1 - data["done"]) * args.gamma * min_qf_next_target

            q1 = qf1(data["observation"], data["action"])
            q2 = qf2(data["observation"], data["action"])
            q1_loss = F.mse_loss(q1, next_q)
            q2_loss = F.mse_loss(q2, next_q)
            critic_loss = q1_loss + q2_loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    action, log_prob = actor.get_action(data["observation"])
                    q1 = qf1(data["observation"], action)
                    q2 = qf2(data["observation"], action)
                    min_q = torch.min(q1, q2)
                    actor_loss = (alpha * log_prob - min_q).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

            ## Target network update
            if global_step % args.target_network_frequency == 0:
                for target_param, param in zip(qf1_target.parameters(), qf1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for target_param, param in zip(qf2_target.parameters(), qf2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        global_step += args.num_envs
        pbar.update(args.num_envs)


if __name__ == "__main__":
    main()
