from __future__ import annotations

from typing import Any

import hydra
import torch
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from rvrl.envs import BaseVecEnv


class IsaacGymEnv(BaseVecEnv):
    def __init__(self, task_name: str, num_envs: int = 1, seed: int = 0, device: str = "cuda"):
        with hydra.initialize(config_path="../../refs/minimal-stable-PPO/configs"):
            cfg = hydra.compose(config_name="config", overrides=[f"task={task_name.replace('isaacgymenv/', '')}"])
        cfg.task.env.numEnvs = num_envs
        cfg.sim_device = device
        cfg.rl_device = device
        cfg.seed = seed

        self.envs: VecTask = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task),
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=False,
            force_render=True,
        )

    def reset(self, seed: int | None = None, options: Any | None = None):
        obs = self.envs.reset()
        return obs["obs"], {}

    def step(self, action: torch.Tensor):
        obs, reward, done, info = self.envs.step(action)
        return obs["obs"], reward, done, done, info

    def render(self):
        raise NotImplementedError

    @property
    def single_observation_space(self):
        return self.envs.observation_space

    @property
    def single_action_space(self):
        return self.envs.action_space

    @property
    def num_envs(self) -> int:
        return self.envs.num_envs
