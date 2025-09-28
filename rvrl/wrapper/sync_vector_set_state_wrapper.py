from __future__ import annotations

from typing import Union

import numpy as np
import torch

from gymnasium import Env
from gymnasium.vector.sync_vector_env import SyncVectorEnv

Device = Union[str, torch.device]


class SyncVectorSetStateWrapper(SyncVectorEnv):
    def __init__(self, env: Env, device: Device | None = None):
        super().__init__(env, device)
        self._raw_state = None

    def set_state_async(self, state: np.ndarray):
        # self._raw_state = iterate(self.observation_space, state)
        self._raw_state = state

    def set_state_wait(self, index: np.ndarray | None = None):
        for i, (env, raw_state) in enumerate(zip(self.envs, self._raw_state)):
            if index is None:
                env.set_state(raw_state)
            elif index[i]:
                env.set_state(raw_state)

    def set_state(self, state: np.ndarray, index: np.ndarray | None = None):
        self.set_state_async(state)
        self.set_state_wait(index=index)
