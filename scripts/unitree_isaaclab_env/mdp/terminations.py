import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.envs import ManagerBasedRLEnv


# 如果机器人的base接触到地面
def base_contact_ground_func(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    filterd_contact_ground = env.scene[asset_cfg.name].data.force_matrix_w # 正常来说这个返回的shape应该是(N,1,1,3)，N应该是环境的数量
    filterd_contact_ground = torch.squeeze(filterd_contact_ground) # squeeze后的shape应该是(N,3)
    filterd_contact_ground = torch.sum(filterd_contact_ground,dim = 1) # sum后的shape
    return filterd_contact_ground > (torch.zeros_like(filterd_contact_ground) + 1e-3) # 返回一个bool值，表示是否与地面接触