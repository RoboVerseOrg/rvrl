"""Wrapper that tracks the cumulative rewards and episode lengths."""
## refer to https://gymnasium.farama.org/_modules/gymnasium/wrappers/vector/common/#RecordEpisodeStatistics

from __future__ import annotations

import time

import gymnasium as gym
import torch
from gymnasium.vector.vector_env import VectorEnv


class RecordEpisodeStatisticsTensor(gym.Wrapper):
    def __init__(
        self,
        env: VectorEnv,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            stats_key: The info key to save the data
        """
        super().__init__(env)
        self._stats_key = stats_key

        self.episode_count = 0

        self.episode_start_times: torch.Tensor = torch.zeros((self.num_envs,))
        self.episode_returns: torch.Tensor = torch.zeros((self.num_envs,))
        self.episode_lengths: torch.Tensor = torch.zeros((self.num_envs,), dtype=int)
        self.prev_dones: torch.Tensor = torch.zeros((self.num_envs,), dtype=bool)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            assert isinstance(reset_mask, torch.ndarray), (
                f"`options['reset_mask': mask]` must be a numpy array, got {type(reset_mask)}"
            )
            assert reset_mask.shape == (self.num_envs,), (
                f"`options['reset_mask': mask]` must have shape `({self.num_envs},)`, got {reset_mask.shape}"
            )
            assert reset_mask.dtype == torch.bool_, (
                f"`options['reset_mask': mask]` must have `dtype=torch.bool_`, got {reset_mask.dtype}"
            )
            assert torch.any(reset_mask), (
                f"`options['reset_mask': mask]` must contain a boolean array, got reset_mask={reset_mask}"
            )

            self.episode_start_times[reset_mask] = time.perf_counter()
            self.episode_returns[reset_mask] = 0
            self.episode_lengths[reset_mask] = 0
            self.prev_dones[reset_mask] = False
        else:
            self.episode_start_times = torch.full((self.num_envs,), time.perf_counter())
            self.episode_returns = torch.zeros(self.num_envs)
            self.episode_lengths = torch.zeros(self.num_envs, dtype=int)
            self.prev_dones = torch.zeros(self.num_envs, dtype=bool)

        return obs, info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Steps through the environment, recording the episode statistics."""
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        assert isinstance(infos, dict)

        self.episode_returns[self.prev_dones] = 0
        self.episode_returns[~self.prev_dones] += rewards.cpu()[~self.prev_dones]

        self.episode_lengths[self.prev_dones] = 0
        self.episode_lengths[~self.prev_dones] += 1

        self.episode_start_times[self.prev_dones] = time.perf_counter()

        self.prev_dones = dones = torch.logical_or(terminations, truncations).cpu()  # XXX
        num_dones = torch.sum(dones)

        if num_dones:
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(
                    f"Attempted to add episode stats with key '{self._stats_key}' but this key already exists in info: {list(infos.keys())}"
                )
            else:
                episode_time_length = torch.round(time.perf_counter() - self.episode_start_times, decimals=6)
                infos[self._stats_key] = {
                    "r": torch.where(dones, self.episode_returns, 0.0),
                    "l": torch.where(dones, self.episode_lengths, 0),
                    "t": torch.where(dones, episode_time_length, 0.0),
                }
                infos[f"_{self._stats_key}"] = dones

            self.episode_count += num_dones

        return (
            obs,
            rewards,
            terminations,
            truncations,
            infos,
        )
