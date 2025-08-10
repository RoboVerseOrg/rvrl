import torch

from rvrl.envs.maniskill_env import ManiskillVecEnv

print("rgb")

env = ManiskillVecEnv("TurnFaucet-v1", num_envs=4, seed=0, device="cuda", obs_mode="rgb", image_size=(64, 64))
print(env.single_action_space)
print(env.action_space)
print(env.single_observation_space)
print(env.observation_space)
print(env.reset()[0].shape)
print(env.step(torch.randn(env.action_space.shape))[0].shape)
env.close()

print()
print("proprio")

env = ManiskillVecEnv("PickCube-v1", num_envs=4, seed=0, device="cuda", obs_mode="proprio")
print(env.single_action_space)
print(env.action_space)
print(env.single_observation_space)
print(env.observation_space)
print(env.reset()[0].shape)
print(env.step(torch.randn(env.action_space.shape))[0].shape)
env.close()
