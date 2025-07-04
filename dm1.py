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

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from loguru import logger as log
from rich.logging import RichHandler
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, TanhTransform, TransformedDistribution, kl_divergence
from torch.utils.tensorboard import SummaryWriter

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
        self.env = suite.load(domain_name, task_name)
        self.seed = seed
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
        obs = np.transpose(obs, (2, 0, 1)).copy()  # (H, W, 3) -> (3, H, W)
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
        obs = np.transpose(obs, (2, 0, 1)).copy()  # (H, W, 3) -> (3, H, W)
        return obs, reward, done, done, {}

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=(3, self.width, self.height), dtype=np.uint8)

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


class ReplayBuffer(object):
    def __init__(
        self, observation_shape: Sequence[int], action_size: int, device: str | torch.device, capacity: int = 5000000
    ):
        self.device = device
        self.capacity = capacity

        state_type = np.uint8 if len(observation_shape) < 3 else np.float32

        self.observation = np.empty((self.capacity, *observation_shape), dtype=state_type)
        self.next_observation = np.empty((self.capacity, *observation_shape), dtype=state_type)
        self.action = np.empty((self.capacity, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)

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
    ):
        self.observation[self.buffer_index] = observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
        self.done[self.buffer_index] = done.detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> dict[str, Tensor]:
        """
        (batch_size, chunk_size, input_size)
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (last_filled_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else last_filled_index, batch_size).reshape(
            -1, 1
        )
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity

        observation = torch.as_tensor(self.observation[sample_index], device=self.device).float()
        next_observation = torch.as_tensor(self.next_observation[sample_index], device=self.device).float()

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        }
        return sample


def compute_lambda_values(
    rewards: Tensor, values: Tensor, continues: Tensor, horizon_length: int, device: torch.device, gae_lambda: float
) -> Tensor:
    """
    rewards : (batch_size, time_step)
    values : (batch_size, time_step)
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - gae_lambda)

    outputs = []
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * gae_lambda * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns


########################################################
## Args
########################################################
@dataclass
class Args:
    env_id: str = "dm_control/walker-walk-v0"
    exp_name: str = "dreamerv1"
    num_envs: int = 1
    seed: int = 0
    device: str = "cuda"
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    num_iterations: int = 1000
    batch_size: int = 50
    batch_length: int = 50
    stochastic_size: int = 30  # XXX: what is this?
    deterministic_size: int = 200  # XXX: what is this?
    embedded_obs_size: int = 1024
    horizon: int = 15
    gae_lambda: float = 0.95


args = tyro.cli(Args)


########################################################
## Networks
########################################################
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        B = obs.shape[0]
        embedded_obs = self.encoder(obs)
        return embedded_obs.view(B, -1)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 1024),
            nn.Unflatten(1, (1024, 1, 1)),  # XXX: why 1024?
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        # posterior: (batch_size, batch_length-1, stochastic_size)
        # deterministic: (batch_size, batch_length-1, deterministic_size)
        input_shape = posterior.shape
        posterior = posterior.flatten(0, 1)
        deterministic = deterministic.flatten(0, 1)
        x = torch.cat([posterior, deterministic], dim=1)
        mean = self.decoder(x)
        mean = mean.unflatten(0, input_shape[:2])
        std = 1  # XXX: why std is 1?
        dist = Independent(Normal(mean, std), 3)  # 3 is number of dimensions for observation space, shape is (3, H, W)
        return dist


class RecurrentModel(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.linear = nn.Linear(args.stochastic_size + envs.single_action_space.shape[0], 200)
        self.act = nn.ELU()
        self.recurrent = nn.GRUCell(200, args.deterministic_size)

    def forward(self, embedded_obs: Tensor, action: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([embedded_obs, action], dim=1)
        x = self.act(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x


class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.deterministic_size, 200),
            nn.ELU(),
            nn.Linear(200, args.stochastic_size * 2),
        )

    def forward(self, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        mean, std = self.net(deterministic).chunk(2, dim=1)
        std = F.softplus(std) + 0.1  # XXX: why add 0.1?
        prior_dist = Normal(mean, std)
        prior = prior_dist.rsample()
        return prior_dist, prior


class RepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.embedded_obs_size + args.deterministic_size, 200),
            nn.ELU(),
            nn.Linear(200, args.stochastic_size * 2),
        )

    def forward(self, embedded_obs: Tensor, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        x = torch.cat([embedded_obs, deterministic], dim=1)
        mean, std = self.net(x).chunk(2, dim=1)
        std = F.softplus(std) + 0.1  # XXX: why add 0.1?
        posterior_dist = Normal(mean, std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 200),
            nn.ELU(),
            nn.Linear(200, 1),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        # posterior: (batch_size, batch_length-1, stochastic_size)
        # deterministic: (batch_size, batch_length-1, deterministic_size)
        input_shape = posterior.shape
        posterior = posterior.flatten(0, 1)
        deterministic = deterministic.flatten(0, 1)
        x = torch.cat([posterior, deterministic], dim=1)
        mean = self.net(x)
        mean = mean.unflatten(0, input_shape[:2])
        std = 1  # XXX: why std is 1?
        reward_dist = Independent(Normal(mean, std), 1)
        return reward_dist


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 400),
            nn.ELU(),
            nn.Linear(400, envs.single_action_space.shape[0] * 2),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([posterior, deterministic], dim=-1)
        mean, std = self.actor(x).chunk(2, dim=-1)
        mean = 5 * F.tanh(mean / 5)  # XXX: what is this?
        std = F.softplus(std + 5) + 1e-4  # XXX: why add 5? why add 1e-4?
        action_dist = TransformedDistribution(Normal(mean, std), TanhTransform())  # XXX: why use TanhTransform?
        action_dist = Independent(action_dist, 1)
        action = action_dist.rsample()  #! important to use rsample()
        return action


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(args.stochastic_size + args.deterministic_size, 400),
            nn.ELU(),
            nn.Linear(400, 400),
            nn.ELU(),
            nn.Linear(400, 1),
        )

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        x = torch.cat([posterior, deterministic], dim=-1)
        mean = self.critic(x)
        std = 1  # XXX: why std is 1?
        dist = Independent(Normal(mean, std), 1)
        return dist


########################################################
## Main
########################################################

## setup
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
seed_everything(args.seed)
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__seed={args.seed}__{_timestamp}"

envs = gym.vector.SyncVectorEnv([
    lambda: DMCWrapper(args.env_id.removeprefix("dm_control/"), args.seed, decimation=2) for _ in range(args.num_envs)
])
envs = NumpyToTorch(envs, device=device)
writer = SummaryWriter(f"logdir/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
buffer = ReplayBuffer(envs.single_observation_space.shape, envs.single_action_space.shape[0], device)

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


def rollout(envs, num_episodes: int):
    for epi in range(num_episodes):
        posterior = torch.zeros(1, args.stochastic_size, device=device)
        deterministic = torch.zeros(1, args.deterministic_size, device=device)
        action = torch.zeros(1, envs.single_action_space.shape[0], device=device)
        obs, _ = envs.reset()
        while True:
            embeded_obs = encoder(obs)
            deterministic = recurrent_model(posterior, action, deterministic)
            _, prior = representation_model(embeded_obs.view(1, -1), deterministic)
            action = actor(prior, deterministic)
            next_obs, reward, terminated, truncated, info = envs.step(action)
            done = torch.logical_or(terminated, truncated)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if done.all():
                break


def dynamic_learning(data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    posterior = torch.zeros(args.batch_size, args.stochastic_size, device=device)
    deterministic = torch.zeros(args.batch_size, args.deterministic_size, device=device)
    embeded_obs = encoder(data["observation"].flatten(0, 1)).unflatten(0, (args.batch_size, args.batch_length))

    priors = []
    prior_means = []
    prior_stds = []
    posteriors = []
    posterior_means = []
    posterior_stds = []
    deterministics = []
    for t in range(1, args.batch_length):
        deterministic = recurrent_model(posterior, data["action"][:, t - 1], deterministic)
        prior_dist, prior = transition_model(deterministic)
        posterior_dist, posterior = representation_model(embeded_obs[:, t], deterministic)

        priors.append(prior)
        prior_means.append(prior_dist.mean)
        prior_stds.append(prior_dist.scale)
        posteriors.append(posterior)
        posterior_means.append(posterior_dist.mean)
        posterior_stds.append(posterior_dist.scale)
        deterministics.append(deterministic)

    priors = torch.stack(priors, dim=1).to(device)
    prior_means = torch.stack(prior_means, dim=1).to(device)
    prior_stds = torch.stack(prior_stds, dim=1).to(device)
    posteriors = torch.stack(posteriors, dim=1).to(device)
    posterior_means = torch.stack(posterior_means, dim=1).to(device)
    posterior_stds = torch.stack(posterior_stds, dim=1).to(device)
    deterministics = torch.stack(deterministics, dim=1).to(device)

    reconstructed_obs_dist = decoder(posteriors, deterministics)
    reconstructed_obs_loss = -reconstructed_obs_dist.log_prob(data["observation"][:, 1:]).mean()
    reward_dist = reward_predictor(posteriors, deterministics)
    reward_loss = -reward_dist.log_prob(data["reward"][:, 1:]).mean()

    ## XXX: why recreate the distribution? what does Independent do?
    prior_dist = Independent(Normal(prior_means, prior_stds), 1)
    posterior_dist = Independent(Normal(posterior_means, posterior_stds), 1)
    kl_loss = kl_divergence(posterior_dist, prior_dist).mean()

    ## TODO: add coefficients for loss terms
    model_loss = reconstructed_obs_loss + reward_loss + kl_loss

    model_optimizer.zero_grad()
    model_loss.backward()
    nn.utils.clip_grad_norm_(model_params, 100)
    model_optimizer.step()

    return posteriors, deterministics


def behavior_learning(posteriors_: Tensor, deterministics_: Tensor):
    ## reuse the `posteriors` and `deterministics` from model learning, important to detach them!
    state = posteriors_.detach().view(-1, args.stochastic_size)
    deterministic = deterministics_.detach().view(-1, args.deterministic_size)

    states = []
    deterministics = []
    for t in range(args.horizon):
        action = actor(state, deterministic)  # XXX: why don't use prior? previously used prior
        deterministic = recurrent_model(state, action, deterministic)
        _, state = transition_model(deterministic)
        states.append(state)
        deterministics.append(deterministic)

    states = torch.stack(states, dim=1)
    deterministics = torch.stack(deterministics, dim=1)
    predicted_rewards = reward_predictor(states, deterministics).mean
    values = critic(states, deterministics).mean
    continues = torch.ones_like(values) * 0.99
    lambda_values = compute_lambda_values(predicted_rewards, values, continues, args.horizon, device, args.gae_lambda)

    actor_loss = -lambda_values.mean()  # direct policy gradient
    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 100)
    actor_optimizer.step()

    value_dist = critic(states[:, :-1].detach(), deterministics[:, :-1].detach())
    value_loss = -value_dist.log_prob(lambda_values.detach()).mean()
    critic_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 100)
    critic_optimizer.step()


def main():
    ## rollout before training
    tic = time.time()
    rollout(envs, 5)
    toc = time.time()
    log.info(f"Time taken for pre-rollout: {toc - tic:.2f} seconds")
    log.info(f"Pre-collected buffer size: {len(buffer)}")

    ## training loop
    for iteration in range(args.num_iterations):
        tic = time.time()
        for sample_index in range(100):
            data = buffer.sample(args.batch_size, 50)
            posteriors, deterministics = dynamic_learning(data)
            behavior_learning(posteriors, deterministics)
        toc = time.time()
        log.info(f"Time taken for training iteration {iteration}: {toc - tic:.2f} seconds")

        tic = time.time()
        rollout(envs, 1)
        toc = time.time()
        log.info(f"Time taken for rollout: {toc - tic:.2f} seconds")


if __name__ == "__main__":
    main()
