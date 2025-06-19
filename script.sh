## DMControl Tasks
python ppo.py --env-id=dm_control/cartpole-balance-v0 --seed 2 --num-envs=4 --lr=1e-3 --num_steps=2048 --total_timesteps=1000000

## IsaacGymEnv Tasks
python ppo.py --env-id=isaacgymenv/Cartpole --seed 2 --num-envs=512 --lr=3e-4 --num_steps=16 --total_timesteps=1500000

## IsaacLab Tasks
python ppo.py --env-id=isaaclab/Isaac-Reach-Franka-v0 --seed 1 --num-envs=32 --lr=1e-3 --num_steps=360 --total_timesteps=3000000
