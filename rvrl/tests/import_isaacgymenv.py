import rootutils

rootutils.setup_root(__file__, pythonpath=True)
from rvrl.envs.isaacgym_env import IsaacGymEnv

env = IsaacGymEnv(task_name="Cartpole", num_envs=1, seed=0, device="cuda")
print(env.envs)
