from __future__ import annotations

import gymnasium as gym
import numpy as np
from dm_control import suite
from gymnasium import spaces


class DMControlRgbEnv(gym.Env):
    def __init__(self, env_id: str, seed: int, width: int = 64, height: int = 64, action_repeat: int = 1):
        domain_name = env_id.split("-")[0]
        task_name = env_id.split("-")[1]
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
        self.width = width
        self.height = height
        self.action_repeat = action_repeat
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
        for _ in range(self.action_repeat):
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
        return int(self.env._step_limit / self.action_repeat)
