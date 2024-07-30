"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import ext_template.tasks  # noqa: F401

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

Unitree_package_path = "scripts"
if Unitree_package_path not in sys.path:
    sys.path.append(Unitree_package_path)

import unitree_isaaclab_env # 导入自定义的Unitree训练的环境的包，这会执行unitree_isaaclab_env/__init__.py文件，自动注册自定义的Unitree的环境

# 这些配置是在Pytorch中优化CUDA和cuDNN的配置
# TF32是在NVIDIA Ampere架构GPU上引入的新浮点格式，前两个设为true，则会允许cuda和cudnn使用TF32格式，可以显著提高计算性能，但会带来轻微的数值精度损失，但在深度学习任务中通常不会有太大影响
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 控制cudnn是否使用确定性算法，如果设置为True，保证每次运行的结果是一样的，但会降低性能；如果为False，每次算法的运行结果可能会有所不同，但通常会更快
torch.backends.cudnn.deterministic = False
# cudnn是否启用基准测试模式。设为false则不启用，使用默认的算法运行卷积操作；设为true则启用，为每一个卷积层运行基准测试，以选择最优的算法提高性能
# 这个过程在初次运行时会有一定的时间开销，但是在后续的运行中会提高性能
torch.backends.cudnn.benchmark = False


def main(Pre_trained_model_path : str  = ""):
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    # NOTE: 从这里可以看出，RSL-RL的训练结果会被保存在logs/rsl_rl/experiment_name文件夹下，按照时间戳命名
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # 注意，os.path.abspath会将指定路径转化为绝对路径，是根据运行脚本的终端路径来转化的（而不是脚本文件所在的路径）
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume and args_cli.pre_trained:
        raise ValueError("Cannot resume training and load pre-trained model at the same time.")
    elif agent_cfg.resume and not args_cli.pre_trained:
        # get path to previous checkpoint
        # get_checkpoint_path是用来获取之前训练的模型的路径，并选出最新的模型和训练参数并返回
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    if args_cli.pre_trained and not agent_cfg.resume: # NOTE: 如果设置了使用预训练模型并且传入了预训练模型的路径
        load_dict = torch.load(Pre_trained_model_path)
        runner.alg.actor_critic.load_state_dict(load_dict["actor_critic"])

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # NOTE: 我对rsl-rl的on_policy_runner.py文件做了修改，给learn函数添加了一个参数only_positive_rewards，其默认为false
    # 如果only_positive_rewards设为True，则如果与环境交互的total reward为负数时，会将其clip到0；如果为False，则不会clip
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True, only_positive_rewards=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
