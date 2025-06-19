"""Unified environment factory for creating different types of environments."""

import gymnasium as gym

from src.wrapper.numpy_to_torch_wrapper import NumpyToTorch


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
    env_id: str, num_envs: int, capture_video: bool = False, run_name: str = "", device: str = "cuda", **kwargs
):
    """
    Create a vectorized environment.

    Args:
        env_id: Environment identifier
        num_envs: Number of parallel environments
        capture_video: Whether to record video
        run_name: Name for video recording
        device: Device to run on (for IsaacLab)
        **kwargs: Additional environment arguments

    Returns:
        Vectorized environment
    """
    if env_id.startswith("dm_control/"):
        env_fns = [make_env(env_id, i, capture_video and i == 0, run_name, **kwargs) for i in range(num_envs)]
        envs = gym.vector.SyncVectorEnv(env_fns)
        envs = NumpyToTorch(envs, device)
        return envs
    elif env_id.startswith("Isaac-"):
        from .isaaclab_env import IsaacLabEnv

        envs = IsaacLabEnv(env_id, num_envs, seed=kwargs.get("seed", 0))  # TODO: seed
        return envs
    elif env_id.startswith("isaacgymenv/"):
        from .isaacgym_env import IsaacGymEnv

        envs = IsaacGymEnv(env_id, num_envs, seed=kwargs.get("seed", 0))
        return envs
    else:
        raise ValueError(f"Unknown environment: {env_id}")
