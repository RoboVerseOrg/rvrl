from __future__ import annotations

from typing import Any, Literal

import metasim  # noqa: F401
import torch
from metasim.task.registry import get_task_class

from rvrl.envs import BaseVecEnv


class RoboverseEnv(BaseVecEnv):
    def __init__(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        device: str | torch.device,
        obs_mode: Literal["rgb", "state"],
        image_size: tuple[int, int],
    ):
        assert obs_mode == "state", "Only state mode is supported"  # TODO: support rgb
        task_cls = get_task_class(task_name)
        scenario = task_cls.scenario.update(
            robots=["franka"],  # TODO: make this configurable
            simulator="mjx",  # TODO: make this configurable
            num_envs=num_envs,
            headless=True,
            cameras=[],
        )
        self.envs = task_cls(scenario, device=device)

    def reset(self, seed: int | None = None, options: Any | None = None):
        obs, extra = self.envs.reset()
        return obs, extra

    def step(self, action):
        obs, reward, terminations, truncations, info = self.envs.step(action)
        return obs, reward, terminations, truncations, info

    def render(self):
        raise NotImplementedError

    @property
    def single_observation_space(self):
        return self.envs.observation_space

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def single_action_space(self):
        return self.envs.action_space

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        return self.envs.num_envs
