# RVRL

## Installation

Main environment:
```bash
conda create -n rvrl python=3.11 -y && conda activate rvrl
uv pip install -e ".[dmc,maniskill]"
```

Humanoid-bench environment:
```bash
conda activate rvrl
uv pip install -e ".[humanoid_bench]"
cd third_party
git clone --depth 1 https://github.com/Fisher-Wang/humanoid-bench && cd humanoid-bench
uv pip install -e .
cd ../..
```

IsaacLab environment:
```bash
conda create -n rvrl_lab python=3.10 -y && conda activate rvrl_lab
uv pip install -e ".[isaaclab]"
cd third_party
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab210 && cd IsaacLab210
sed -i '/^EXTRAS_REQUIRE = {/,/^}$/c\EXTRAS_REQUIRE = {\n    "sb3": [],\n    "skrl": [],\n    "rl-games": [],\n    "rsl-rl": [],\n}' source/isaaclab_rl/setup.py
sed -i 's/if platform\.system() == "Linux":/if False:/' source/isaaclab_mimic/setup.py
./isaaclab.sh -i
cd ../..
```

IsaacGym environment:
```bash
conda create -n rvrl_gym python=3.8 -y && conda activate rvrl_gym
cd third_party
wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
find isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
uv pip install isaacgym/python
git clone --depth 1 https://github.com/isaac-sim/IsaacGymEnvs && cd IsaacGymEnvs
uv pip install -e .
cd ../..
uv pip install networkx==2.1
```

## Usage
see [script.sh](./script.sh)
