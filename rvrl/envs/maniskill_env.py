# Refer to https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo_rgb.py and https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo.py

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import Tensor

from rvrl.envs import BaseVecEnv


class ManiskillVecEnv(BaseVecEnv):
    def __init__(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        device: str | torch.device,
        obs_mode: Literal["rgb", "proprio"],
        image_size: tuple[int, int] | None = None,
    ):
        self.env = gym.make(
            task_name,
            num_envs=num_envs,
            obs_mode="rgb" if obs_mode == "rgb" else "state",
            render_mode="all" if obs_mode == "rgb" else "rgb_array",
            sim_backend="physx_cuda",
            control_mode="pd_ee_delta_pose",  # XXX: make this configurable, currently same as tdmpc2
            robot_uids="panda",  # XXX: make this configurable; this will affect agent_sensor
            sensor_configs={"width": image_size[0], "height": image_size[1]} if obs_mode == "rgb" else {},
        )
        if obs_mode == "rgb":
            self.env = FlattenRGBDObservationWrapper(self.env, rgb=True, depth=False, state=False)
        if isinstance(self.env.action_space, gym.spaces.Dict):
            self.env = FlattenActionSpaceWrapper(self.env)
        self.env = ManiSkillVectorEnv(self.env, num_envs)
        self._obs_mode = obs_mode

        self._action_space = self.env.action_space
        self._single_action_space = self.env.single_action_space
        if obs_mode == "rgb":
            self._obs_space = gym.spaces.Box(0, 255, shape=(num_envs, 3, image_size[0], image_size[1]), dtype=np.uint8)
            self._single_obs_space = gym.spaces.Box(0, 255, shape=(3, image_size[0], image_size[1]), dtype=np.uint8)
        else:
            self._obs_space = self.env.observation_space
            self._single_obs_space = self.env.single_observation_space
        self._obs_space.seed(seed)
        self._single_obs_space.seed(seed)
        self._action_space.seed(seed)
        self._single_action_space.seed(seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options={} if options is None else options)
        return obs["rgb"].permute(0, 3, 1, 2) if self._obs_mode == "rgb" else obs, info

    def step(self, action: Tensor):
        obs, reward, terminations, truncations, info = self.env.step(action)
        return (
            obs["rgb"].permute(0, 3, 1, 2) if self._obs_mode == "rgb" else obs,
            reward,
            terminations,
            truncations,
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def single_observation_space(self):
        return self._single_obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def num_envs(self) -> int:
        return self.env.num_envs
