"""Unified environment factory for creating different types of environments."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from rvrl.envs import BaseVecEnv
from rvrl.wrapper.numpy_to_torch_wrapper import NumpyToTorch

SEED_SPACING = 1_000_000


## Used for DMControl
def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)

        # TODO: is below necessary?
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


## Main function to create vectorized environment
def create_vector_env(
    env_id: str,
    obs_type: Literal["rgb", "proprio"],
    num_envs: int,
    seed: int,
    action_repeat: int = 1,
    image_size: tuple[int, int] = (64, 64),
    capture_video: bool = False,
    run_name: str = "",
    device: str = "cuda",
    **kwargs,
) -> BaseVecEnv:
    """
    Create a vectorized environment.

    Args:
        env_id: Environment identifier
        obs_type: Observation type
        num_envs: Number of parallel environments
        seed: Seed for the environment
        action_repeat: Action repeat for the environment
        image_size: Image size for RGB observation. Only used when :param:`obs_type` is "rgb". Type is (width, height) tuple.
        capture_video: Whether to record video
        run_name: Name for video recording
        device: Device to run on (for IsaacLab)
        **kwargs: Additional environment arguments

    Returns:
        Vectorized environment
    """
    if env_id.startswith("dm_control/"):
        if obs_type == "proprio":
            from .dmc_env import DMControlProprioEnv

            env_fns = [
                lambda: gym.wrappers.FlattenObservation(
                    DMControlProprioEnv(
                        env_id.replace("dm_control/", ""),
                        seed + i * SEED_SPACING,
                        action_repeat=action_repeat,
                    )
                )
                for i in range(num_envs)
            ]
            envs = gym.vector.SyncVectorEnv(env_fns)
            envs = NumpyToTorch(envs, device)
            return envs
        elif obs_type == "rgb":
            from .dmc_env import DMControlRgbEnv

            env_fns = [
                lambda: DMControlRgbEnv(
                    env_id.replace("dm_control/", ""),
                    seed + i * SEED_SPACING,
                    width=image_size[0],
                    height=image_size[1],
                    action_repeat=action_repeat,
                )
                for i in range(num_envs)
            ]
            envs = gym.vector.SyncVectorEnv(env_fns)
            envs = NumpyToTorch(envs, device)
            return envs
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")
    elif env_id.startswith("humanoid_bench/"):
        from .humanoid_bench_env import HumanoidBenchEnv

        env_fns = [
            lambda: HumanoidBenchEnv(
                env_id.replace("humanoid_bench/", ""), seed + i * SEED_SPACING, image_size, obs_type
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

        envs = IsaacGymEnv(env_id, num_envs, seed=seed)
        return envs
    else:  # gymnasium envs
        env_fns = [lambda: gym.make(env_id) for _ in range(num_envs)]
        envs = gym.vector.SyncVectorEnv(env_fns)
        envs = NumpyToTorch(envs, device)
        return envs
