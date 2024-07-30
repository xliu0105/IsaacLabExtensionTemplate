import torch
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab.utils.math as math_utils

# 定义一个获取机器人的加速度的函数
# NOTE: 由于加速度数据非常不稳定，而且可能会有突发的大幅度变化，因此在使用的时候对其进行数值缩放是非常重要的
def get_body_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Acceleration of all bodies. Shape is (num_instances, 6)."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_acc_world = asset.data.body_acc_w # 这个返回的shape是(num_instances, num_bodies, 6)
    root_acc_w = torch.squeeze(body_acc_world[:,asset_cfg.body_ids,:]) # 只取机器人root的加速度，并将维度转化为(num_instances, 6)
    root_acc_b = torch.empty_like(root_acc_w) # 创建一个和root_acc_w相同shape的tensor
    root_acc_b[:,:3] = math_utils.quat_rotate_inverse(asset.data.root_quat_w, root_acc_w[:,:3]) # 将加速度转换到机器人的坐标系下
    root_acc_b[:,3:] = math_utils.quat_rotate_inverse(asset.data.root_quat_w, root_acc_w[:,3:]) # 将加速度转换到机器人的坐标系下
    return root_acc_b / 10.0 # 这里对加速度数据进行了数值缩放