from __future__ import annotations

try:
    import isaacgym
except ImportError:
    pass

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Sequence

os.environ["MUJOCO_GL"] = "egl"  # significantly faster rendering compared to glfw and osmesa
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic run

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from loguru import logger as log
from rich.logging import RichHandler
from torch import Tensor
from torch.distributions import (
    AffineTransform,
    ComposeTransform,
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
    kl_divergence,
)
from torch.distributions.utils import probs_to_logits
from torch.utils.tensorboard.writer import SummaryWriter

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Experimental
########################################################
from dm_control import suite
from gymnasium import spaces

from src.wrapper.numpy_to_torch_wrapper import NumpyToTorch


class DMCWrapper:
    def __init__(self, env_id: str, seed: int, width: int = 64, height: int = 64, decimation: int = 1):
        domain_name = env_id.split("-")[0]
        task_name = env_id.split("-")[1]
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
        self.width = width
        self.height = height
        self.decimation = decimation
        action_spec = self.env.action_spec()
        self._action_space = spaces.Box(
            low=-1, high=1, shape=action_spec.shape, dtype=np.float32
        )  # XXX: may only work for walker

    def reset(self) -> tuple[np.ndarray, dict]:
        self.env.reset()
        obs = self.env.physics.render(width=self.width, height=self.height, camera_id=0)
        obs = np.transpose(obs, (2, 0, 1)).copy() / 255.0 - 0.5  # (H, W, 3) -> (3, H, W)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0
        for _ in range(self.decimation):
            data = self.env.step(action)
            reward += data.reward
            done = data.last()
            if done:
                break
        obs = self.env.physics.render(width=self.width, height=self.height, camera_id=0)
        obs = np.transpose(obs, (2, 0, 1)).copy() / 255.0 - 0.5  # (H, W, 3) -> (3, H, W)
        return obs, reward, done, done, {}

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def observation_space(self):
        return spaces.Box(low=-0.5, high=0.5, shape=(3, self.width, self.height), dtype=np.float32)

    @property
    def action_space(self):
        return self._action_space

    @property
    def metadata(self):
        return {}

    @property
    def episode_length(self) -> int:
        return int(self.env._step_limit / self.decimation)


########################################################
## Standalone utils
########################################################
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class ReplayBuffer(object):
    def __init__(
        self, observation_shape: Sequence[int], action_size: int, device: str | torch.device, capacity: int = 5000000
    ):
        self.device = device
        self.capacity = capacity
        self.observation = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.next_observation = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.action = np.empty((self.capacity, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation: Tensor, action: Tensor, reward: Tensor, next_observation: Tensor, done: Tensor) -> None:
        self.observation[self.buffer_index] = observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
        self.done[self.buffer_index] = done.detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> dict[str, Tensor]:
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (last_filled_index > batch_size), "too short dataset or too long chunk_size"
        sample_idx = np.random.randint(0, self.capacity if self.full else last_filled_index, batch_size).reshape(-1, 1)
        chunk_length = np.arange(chunk_size).reshape(1, -1)
        sample_idx = (sample_idx + chunk_length) % self.capacity

        observation = torch.as_tensor(self.observation[sample_idx], device=self.device).float()
        next_observation = torch.as_tensor(self.next_observation[sample_idx], device=self.device).float()
        action = torch.as_tensor(self.action[sample_idx], device=self.device)
        reward = torch.as_tensor(self.reward[sample_idx], device=self.device)
        done = torch.as_tensor(self.done[sample_idx], device=self.device)

        return {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        }


def compute_lambda_values(
    rewards: Tensor, values: Tensor, continues: Tensor, horizon_length: int, device: torch.device, gae_lambda: float
) -> Tensor:
    """
    Compute lambda returns (λ-returns) for Generalized Advantage Estimation (GAE).

    The lambda return is computed recursively as:
    R_t^λ = r_t + γ * [(1 - λ) * V(s_{t+1}) + λ * R_{t+1}^λ]

    Args:
        rewards: (batch_size, time_step) - rewards at each timestep (r_t)
        values: (batch_size, time_step) - value estimates at each timestep (V(s_t))
        horizon_length: int - length of the planning horizon
        device: torch.device - device to compute on
        gae_lambda: float - lambda parameter for GAE (λ, typically 0.95)

    Returns:
        Tensor: (batch_size, horizon_length-1) - lambda returns (R_t^λ)
    """
    # Remove last timestep since we need t+1 values
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]

    # Initialize with the last value estimate
    last = next_values[:, -1]

    # Compute the base term: r_t + γ * (1 - λ) * V(s_{t+1})
    inputs = rewards + continues * next_values * (1 - gae_lambda)

    # Compute lambda returns backward in time
    outputs = []
    for index in reversed(range(horizon_length - 1)):
        # R_t^λ = [r_t + γ * (1 - λ) * V(s_{t+1})] + γ * λ * R_{t+1}^λ
        last = inputs[:, index] + continues[:, index] * gae_lambda * last
        outputs.append(last)

    # Reverse to get chronological order and move to device
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns


########################################################
## Args
########################################################
@dataclass
class Args:
    env_id: str = "dm_control/walker-walk-v0"
    exp_name: str = "dreamerv3"
    num_envs: int = 1
    seed: int = 0
    device: str = "cuda"
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    num_iterations: int = 1000
    batch_size: int = 50
    batch_length: int = 50
    deterministic_size: int = 512
    stochastic_length: int = 16
    stochastic_classes: int = 16
    embedded_obs_size: int = 1024
    horizon: int = 15
    gae_lambda: float = 0.95

    @property
    def stochastic_size(self):
        return self.stochastic_length * self.stochastic_classes


args = tyro.cli(Args)


########################################################
## Networks
########################################################
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.Tanh(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        embedded_obs = self.encoder(obs)
        return embedded_obs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 512),
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 64, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2),
            nn.Tanh(),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        x = torch.cat([posterior, deterministic], dim=-1)
        input_shape = x.shape
        x = x.flatten(0, 1)
        mean = self.decoder(x)
        mean = mean.unflatten(0, input_shape[:2])
        std = 1  # XXX: why std is 1?
        dist = Independent(Normal(mean, std), 3)  # 3 is number of dimensions for observation space, shape is (3, H, W)
        return dist


class RecurrentModel(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.linear = nn.Linear(args.stochastic_size + envs.single_action_space.shape[0], 200)
        self.act = nn.Tanh()
        self.recurrent = nn.GRUCell(200, args.deterministic_size)

    def forward(self, state: Tensor, action: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.act(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x


class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.deterministic_size, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, args.stochastic_size),
        )

    def forward(self, deterministic: Tensor) -> tuple[Tensor, Tensor]:
        raw_logits = self.net(deterministic).view(-1, args.stochastic_length, args.stochastic_classes)
        prob = F.softmax(raw_logits, dim=-1)
        uniform_prob = torch.ones_like(prob) / args.stochastic_classes
        mixed_prob = 0.99 * prob + 0.01 * uniform_prob
        logits = probs_to_logits(mixed_prob)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        sample = dist.rsample()
        return sample.view(-1, args.stochastic_size), logits


class RepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.embedded_obs_size + args.deterministic_size, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, args.stochastic_size),
        )

    def forward(self, embedded_obs: Tensor, deterministic: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([embedded_obs, deterministic], dim=-1)
        raw_logits = self.net(x).view(-1, args.stochastic_length, args.stochastic_classes)
        prob = F.softmax(raw_logits, dim=-1)
        uniform_prob = torch.ones_like(prob) / args.stochastic_classes
        mixed_prob = 0.99 * prob + 0.01 * uniform_prob
        logits = probs_to_logits(mixed_prob)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        sample = dist.rsample()
        return sample.view(-1, args.stochastic_size), logits


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.stochastic_size + args.deterministic_size, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 2),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        x = torch.cat([posterior, deterministic], dim=-1)
        mean, log_std = self.net(x).chunk(2, dim=-1)
        dist = Independent(Normal(mean, torch.exp(log_std)), 1)
        return dist


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(args.stochastic_size + args.deterministic_size, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, envs.single_action_space.shape[0] * 2),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        log_std_min, log_std_max = -5, 2
        x = torch.cat([posterior, deterministic], dim=-1)
        mean, log_std = self.actor(x).chunk(2, dim=-1)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (F.tanh(log_std) + 1)
        std = torch.exp(log_std)

        action_dist = TransformedDistribution(Normal(mean, std), TanhTransform())  # XXX: assume action is in [-1, 1]
        action = action_dist.rsample()
        return action_dist, action


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(args.stochastic_size + args.deterministic_size, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 2),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        x = torch.cat([posterior, deterministic], dim=-1)
        mean, log_std = self.critic(x).chunk(2, dim=-1)
        dist = Independent(Normal(mean, torch.exp(log_std)), 1)
        return dist


########################################################
## Main
########################################################

## setup
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
seed_everything(args.seed)
enable_deterministic_run()
envs = gym.vector.SyncVectorEnv([
    lambda: DMCWrapper(args.env_id.removeprefix("dm_control/"), args.seed, decimation=2) for _ in range(args.num_envs)
])
envs = NumpyToTorch(envs, device=device)
buffer = ReplayBuffer(envs.single_observation_space.shape, envs.single_action_space.shape[0], device)

## writer
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__seed={args.seed}__{_timestamp}"
writer = SummaryWriter(f"logdir/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)


## networks
encoder = Encoder().to(device)
decoder = Decoder().to(device)
recurrent_model = RecurrentModel(envs).to(device)
transition_model = TransitionModel().to(device)
representation_model = RepresentationModel().to(device)
reward_predictor = RewardPredictor().to(device)
actor = Actor(envs).to(device)
critic = Critic().to(device)
model_params = chain(
    encoder.parameters(),
    decoder.parameters(),
    recurrent_model.parameters(),
    transition_model.parameters(),
    representation_model.parameters(),
    reward_predictor.parameters(),
)
model_optimizer = torch.optim.Adam(model_params, lr=args.model_lr)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

cnt_episode = 0


## core functions
@torch.inference_mode()
def rollout(num_episodes: int):
    global cnt_episode
    for epi in range(num_episodes):
        posterior = torch.zeros(1, args.stochastic_size, device=device)
        deterministic = torch.zeros(1, args.deterministic_size, device=device)
        action = torch.zeros(1, envs.single_action_space.shape[0], device=device)
        obs, _ = envs.reset()
        reward_sum = torch.zeros(envs.num_envs, device=device)
        while True:
            embeded_obs = encoder(obs)
            deterministic = recurrent_model(posterior, action, deterministic)
            posterior, _ = representation_model(embeded_obs, deterministic)
            _, action = actor(posterior, deterministic)
            next_obs, reward, terminated, truncated, _ = envs.step(action)
            reward_sum += reward
            done = torch.logical_or(terminated, truncated)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if done.all():
                break
        cnt_episode += 1
        print(f"Episode {cnt_episode}, Return: {reward_sum.mean().item()}")
        writer.add_scalar("charts/episodic_return", reward_sum.mean().item(), cnt_episode)


def world_model_learning(data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    posterior = torch.zeros(args.batch_size, args.stochastic_size, device=device)
    deterministic = torch.zeros(args.batch_size, args.deterministic_size, device=device)
    embeded_obs = encoder(data["observation"].flatten(0, 1)).unflatten(0, (args.batch_size, args.batch_length))

    deterministics, priors_logits, posteriors, posteriors_logits = [], [], [], []
    for t in range(1, args.batch_length):
        deterministic = recurrent_model(posterior, data["action"][:, t - 1], deterministic)
        _, prior_logits = transition_model(deterministic)
        posterior, posterior_logits = representation_model(embeded_obs[:, t], deterministic)

        deterministics.append(deterministic)
        priors_logits.append(prior_logits)
        posteriors.append(posterior)
        posteriors_logits.append(posterior_logits)

    deterministics = torch.stack(deterministics, dim=1).to(device)
    priors_logits = torch.stack(priors_logits, dim=1).to(device)
    posteriors = torch.stack(posteriors, dim=1).to(device)
    posteriors_logits = torch.stack(posteriors_logits, dim=1).to(device)

    reconstructed_obs_dist = decoder(posteriors, deterministics)
    reconstructed_obs_loss = -reconstructed_obs_dist.log_prob(data["observation"][:, 1:]).mean()
    reward_dist = reward_predictor(posteriors, deterministics)
    reward_loss = -reward_dist.log_prob(data["reward"][:, 1:]).mean()

    prior_dist = Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1)
    prior_dist_sg = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1)
    posterior_dist = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1)
    posterior_dist_sg = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1)
    prior_kl = kl_divergence(posterior_dist_sg, prior_dist)
    posterior_kl = kl_divergence(prior_dist_sg, posterior_dist)
    prior_kl = torch.max(prior_kl, torch.tensor(1.0, device=device))
    posterior_kl = torch.max(posterior_kl, torch.tensor(1.0, device=device))
    kl_loss = (1.0 * prior_kl + 0.1 * posterior_kl).mean()

    model_loss = reconstructed_obs_loss + reward_loss + kl_loss

    model_optimizer.zero_grad()
    model_loss.backward()
    nn.utils.clip_grad_norm_(model_params, 100)
    model_optimizer.step()

    return posteriors, deterministics


def behavior_learning(posteriers_: Tensor, deterministics_: Tensor):
    ## reuse the `posteriors` and `deterministics` from model learning, important to detach them!
    state = posteriers_.detach().view(-1, args.stochastic_size)
    deterministic = deterministics_.detach().view(-1, args.deterministic_size)

    states = []
    deterministics = []
    logprobs = []
    # entropies = []
    for t in range(args.horizon):
        action_dist, action = actor(state, deterministic)
        deterministic = recurrent_model(state, action, deterministic)
        state, _ = transition_model(deterministic)

        states.append(state)
        deterministics.append(deterministic)
        logprobs.append(action_dist.log_prob(action))
        # entropies.append(action_dist.entropy())

    states = torch.stack(states, dim=1)
    deterministics = torch.stack(deterministics, dim=1)
    logprobs = torch.stack(logprobs, dim=1)
    # entropies = torch.stack(entropies, dim=1)

    predicted_rewards = reward_predictor(states, deterministics).mean
    values = critic(states, deterministics).mean
    continues = torch.ones_like(values) * 0.997
    lambda_values = compute_lambda_values(predicted_rewards, values, continues, args.horizon, device, args.gae_lambda)
    # TODO: do advantage normalization
    advantages = lambda_values - values[:, :-1]

    # actor-critic policy gradient
    actor_loss = -(advantages.detach() * logprobs[:, 1:]).mean()  # + 0.0003 * entropies.mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 100)
    actor_optimizer.step()

    value_dist = critic(states[:, :-1].detach(), deterministics[:, :-1].detach())
    value_loss = -value_dist.log_prob(lambda_values.detach()).mean()
    critic_optimizer.zero_grad()
    value_loss.backward()


def main():
    rollout(5)
    for i in range(args.num_iterations):
        data = buffer.sample(args.batch_size, args.batch_length)
        posteriors, deterministics = world_model_learning(data)
        behavior_learning(posteriors, deterministics)
        rollout(1)


if __name__ == "__main__":
    main()
