import gymnasium as gym
from . import agents
from .UnitreeA1_Walkenv_Cfg import UnitreeA1WalkEnvCfg

# 注册宇树a1机器人的普通行走仿真环境
gym.register(
    id = "Isaaclab-UnitreeA1-Walk-v0",
    entry_point = "omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs = {
        "env_cfg_entry_point": UnitreeA1WalkEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.UnitreeA1WalkPPORunnerCfg,
    },
)