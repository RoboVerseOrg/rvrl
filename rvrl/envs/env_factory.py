"""Unified environment factory for creating different types of environments."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from rvrl.envs import BaseVecEnv
from rvrl.wrapper.numpy_to_torch_wrapper import NumpyToTorch

SEED_SPACING = 1_000_000


def create_vector_env(
    env_id: str,
    obs_mode: Literal["rgb", "state", "both"],
    num_envs: int,
    seed: int,
    action_repeat: int = 1,
    image_size: tuple[int, int] = (64, 64),
    device: str = "cuda",
) -> BaseVecEnv:
    """
    Create a vectorized environment.

    Args:
        env_id: Environment identifier
        obs_mode: Observation type
        num_envs: Number of parallel environments
        seed: Seed for the environment
        action_repeat: Action repeat for the environment
        image_size: Image size for RGB observation. Only used when :param:`obs_mode` is "rgb". Type is (width, height) tuple.
        device: Device to run on (for IsaacLab)

    Returns:
        Vectorized environment
    """
    if env_id.startswith("dm_control/"):
        from .dmc_env import DMControlEnv

        env_fns = [
            lambda: DMControlEnv(
                env_id.replace("dm_control/", ""),
                seed + i * SEED_SPACING,
                image_size,
                obs_mode,
                action_repeat=action_repeat,
            )
            for i in range(num_envs)
        ]
        if obs_mode == "state":
            env_fns = [lambda: gym.wrappers.FlattenObservation(func()) for func in env_fns]
        envs = gym.vector.SyncVectorEnv(env_fns)
        envs = NumpyToTorch(envs, device)
        return envs
    elif env_id.startswith("humanoid_bench/"):
        from .humanoid_bench_env import HumanoidBenchEnv

        env_fns = [
            lambda: HumanoidBenchEnv(
                env_id.replace("humanoid_bench/", ""), seed + i * SEED_SPACING, image_size, obs_mode
            )
            for i in range(num_envs)
        ]
        envs = gym.vector.SyncVectorEnv(env_fns)
        envs = NumpyToTorch(envs, device)
        return envs
    elif env_id.startswith("isaaclab/"):
        from .isaaclab_env import IsaacLabEnv

        envs = IsaacLabEnv(env_id.replace("isaaclab/", ""), num_envs, seed=seed)
        return envs
    elif env_id.startswith("isaacgymenv/"):
        from .isaacgym_env import IsaacGymEnv

        assert obs_mode == "state", "isaacgymenv only supports state observation"
        envs = IsaacGymEnv(env_id.replace("isaacgymenv/", ""), num_envs, seed, device)
        return envs
    elif env_id.startswith("gym/"):  # gymnasium native envs

        def make_env(env_id, seed):
            def thunk():
                env = gym.make(env_id)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        env_fns = [make_env(env_id.replace("gym/", ""), seed + i * SEED_SPACING) for i in range(num_envs)]
        envs = gym.vector.SyncVectorEnv(env_fns)
        envs = NumpyToTorch(envs, device)
        return envs
    elif env_id.startswith("maniskill/"):
        from .maniskill_env import ManiskillVecEnv

        envs = ManiskillVecEnv(env_id.replace("maniskill/", ""), num_envs, seed, device, obs_mode, image_size)
        return envs
    else:
        raise ValueError(f"Unknown environment: {env_id}")
