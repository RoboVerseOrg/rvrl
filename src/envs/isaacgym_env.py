import hydra
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict


class IsaacGymEnv:
    def __init__(self, task_name: str, num_envs: int = 1, seed: int = 0, device: str = "cuda"):
        with hydra.initialize(config_path="../../refs/minimal-stable-PPO/configs"):
            cfg = hydra.compose(config_name="config", overrides=[f"task={task_name}"])
        cfg.sim_device = device
        cfg.rl_device = device
        cfg.seed = seed

        self.envs = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task),
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=False,
            force_render=True,
        )
