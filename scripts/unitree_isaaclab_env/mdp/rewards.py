import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.assets.asset_base import AssetBase
from omni.isaac.lab.sensors import RayCaster

def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: list[float]) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    thredhold: [lower, upper], the time threshold for the feet to be considered in the air.
    if the feet are in the air for more than upper or less than lower, the reward is negative.
    if the feet are in the air for less than upper and more than lower, the reward is positive.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # 注意区分step_dt和sim dt的区别，step_dt是sim_dt*decimation
    # compute_first_contact是用来计算哪些body在最近这个step_dt内第一次接触到地面
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # last_air_time是用来计算contact传感器上一次腾空的时间
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 当腾空时间大于threshold时，为正数，否则为负数
    bool_over_upper = last_air_time > threshold[1]
    over_lower = last_air_time - threshold[0]
    over_lower[bool_over_upper] = 0.0
    over_lower_down_upper = over_lower + bool_over_upper*(threshold[1] - last_air_time)
    # 只计算刚刚接触到地面的足的腾空时间
    reward = torch.sum(over_lower_down_upper * first_contact, dim=1)
    # no reward for zero command
    # 如果速度命令的L2范数小于0.1，则reward为0。因为如果速度命令小于0.1，那么机器人其实不需要迈步
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_current_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """pnealize feet spent too long time in the air using L2-kernel.
    
    if time of feet in the air is greater than threshold, the reward is negative.
    else zero reward.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    bool_over_threshold = current_air_time > threshold
    reward = torch.sum((current_air_time - threshold) * bool_over_threshold, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_current_ground_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """pnealize feet spent too long time on the ground using L2-kernel.
    
    if time of feet on the ground is greater than threshold, the reward is negative.
    else zero reward.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    bool_over_threshold = current_contact_time > threshold
    reward = torch.sum((current_contact_time - threshold) * bool_over_threshold, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward * 2 # 调整数量级

def feet_hight_reward(env: ManagerBasedRLEnv, sensor_cfg: list[SceneEntityCfg], command_name: str) -> torch.Tensor:
    
    # 奖励足的腾空高度，需要对每个足都设置RayCasterCfg，但经过我的测试来看，这个奖励并不是很好用

    sensor_data_allenv = torch.zeros(env.num_envs).to(env.device)
    for item in sensor_cfg:
        sensor: RayCaster = env.scene.sensors[item.name]
        sensor_data = sensor.data.pos_w[:,2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]
        sensor_data = torch.sum(sensor_data, dim=1)
        sensor_data_allenv += sensor_data
    sensor_data_allenv *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return sensor_data_allenv / len(sensor_cfg) * 100 # 对数量级做了调整
