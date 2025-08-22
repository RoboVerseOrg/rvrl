from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_env.specs import Array, BoundedArray, DiscreteArray
from gymnasium import spaces


def dm_spec2gym_space(spec) -> spaces.Space[Any]:
    """Converts a dm_env spec to a gymnasium space.
    Reference: https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/utils/dm_env.py
    """
    if isinstance(spec, (OrderedDict, dict)):
        return spaces.Dict({key: dm_spec2gym_space(value) for key, value in copy.copy(spec).items()})
    elif type(spec) is BoundedArray:
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(
            low=low,
            high=high,
            shape=spec.shape,
            dtype=spec.dtype,
        )
    elif type(spec) is Array:
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        elif spec.dtype == "bool":
            low = 0
            high = 1
        else:
            raise TypeError(f"Unknown dtype {spec.dtype} for spec {spec}.")

        return spaces.Box(
            low=low,
            high=high,
            shape=spec.shape,
            dtype=spec.dtype,
        )
    elif type(spec) is DiscreteArray:
        return spaces.Discrete(spec.num_values)
    else:
        raise NotImplementedError(f"Cannot convert dm_spec to gymnasium space, unknown spec: {spec}, please report.")


_DEFAULT_CAMERA_ID = {
    "quadruped": 2,  # same as dreamerv3
}


class DMControlProprioEnv(gym.Env):
    def __init__(self, env_id: str, seed: int, action_repeat: int = 1):
        domain_name = env_id.split("-")[0]
        task_name = env_id.split("-")[1]
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
        self.action_repeat = action_repeat
        self.observation_space = dm_spec2gym_space(self.env.observation_spec())
        self.action_space = dm_spec2gym_space(self.env.action_spec())
        print(f"{self.observation_space=}")
        print(f"{self.action_space=}")

    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[np.ndarray, dict]:
        timestep = self.env.reset()
        obs = timestep.observation
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0
        for _ in range(self.action_repeat):
            timestep = self.env.step(action)
            reward += timestep.reward
            done = timestep.last()
            if done:
                break
        truncated = timestep.last() and timestep.discount == 1.0
        terminated = timestep.last() and timestep.discount == 0.0
        obs = timestep.observation
        return obs, reward, terminated, truncated, {}

    def render(self):
        raise NotImplementedError

    def close(self):
        pass


class DMControlRgbEnv(gym.Env):
    def __init__(self, env_id: str, seed: int, width: int = 64, height: int = 64, action_repeat: int = 1):
        domain_name = env_id.split("-")[0]
        task_name = env_id.split("-")[1]
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
        self.width = width
        self.height = height
        self.action_repeat = action_repeat
        self._obs_space = spaces.Box(low=0, high=1, shape=(3, self.width, self.height), dtype=np.float32)
        self._true_action_space = dm_spec2gym_space(self.env.action_spec())
        self._norm_action_space = spaces.Box(low=-1, high=1, shape=self._true_action_space.shape, dtype=np.float32)

        self._obs_space.seed(seed)
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._camera_id = _DEFAULT_CAMERA_ID.get(domain_name, 0)

    def _convert_action(self, action) -> np.ndarray:
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[np.ndarray, dict]:
        self.env.reset()
        obs = self.env.physics.render(width=self.width, height=self.height, camera_id=self._camera_id)
        obs = np.transpose(obs, (2, 0, 1)).copy() / 255.0  # (H, W, 3) -> (3, H, W)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = self._convert_action(action)
        reward = 0
        for _ in range(self.action_repeat):
            timestep = self.env.step(action)
            reward += timestep.reward
            done = timestep.last()
            if done:
                break
        truncated = timestep.last() and timestep.discount == 1.0
        terminated = timestep.last() and timestep.discount == 0.0
        obs = self.env.physics.render(width=self.width, height=self.height, camera_id=self._camera_id)
        obs = np.transpose(obs, (2, 0, 1)).copy() / 255.0  # (H, W, 3) -> (3, H, W)
        return obs, reward, terminated, truncated, {}

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def metadata(self):
        return {}

    @property
    def episode_length(self) -> int:
        return int(self.env._step_limit / self.action_repeat)
