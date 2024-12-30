import math
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CommandTermCfg as ComdTermCfg
from omni.isaac.lab.managers import CommandTerm as ComdTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils import configclass
from . import mdp
from omni.isaac.lab.assets import Articulation, RigidObject
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 导入宇树a1机器人的配置
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG

# 定义SceneCfg，正常来说，四足机器人还需要有一个IMU传感器，可是Isaac lab没有提供IMU传感器的配置，所以在观测中加入IMU能够获取的信息作为替代
@configclass
class UnitreeA1WalkSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

    # IMPORTANT: 这个replace是定义在@configclass的修饰器中的，这里报错不需要理会
    robot : ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/UnitreeA1")

    # NOTE: 注意，如果要用contact sensor，注意要在ArticulationCfg的配置中把activate_contact_sensors设为True
    # NOTE: GPU报警告，无法处理与地面的接触filter
    base_contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/UnitreeA1/trunk",
                                           filter_prim_paths_expr =["/World/GroundPlane/GroundPlane/CollisionPlane"]) # 注意，这个传感器是用来检测机器人的root是否与地面接触的
    # NOTE: 由于这个传感器需要用来计算足的腾空时间，因此需要设置track_air_time = True
    foot_contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/UnitreeA1/.*foot",track_air_time = True, debug_vis = False, history_length = 3)
    whole_body_contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/UnitreeA1/.*", debug_vis = False, history_length = 3)

    # height_scanner_FL = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/UnitreeA1/FL_foot",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.01, 0.01]),
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.08)), attach_yaw_only=True, debug_vis=False, mesh_prim_paths=["/World/GroundPlane"],
    # )
    # height_scanner_FR = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/UnitreeA1/FR_foot",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.01, 0.01]),
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.08)), attach_yaw_only=True, debug_vis=False, mesh_prim_paths=["/World/GroundPlane"],
    # )
    # height_scanner_RL = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/UnitreeA1/RL_foot",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.01, 0.01]),
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.08)), attach_yaw_only=True, debug_vis=False, mesh_prim_paths=["/World/GroundPlane"],
    # )
    # height_scanner_RR = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/UnitreeA1/RR_foot",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.01, 0.01]),
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.08)), attach_yaw_only=True, debug_vis=False, mesh_prim_paths=["/World/GroundPlane"],
    # )


# 定义ObservationsCfg
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        Vel_command = ObsTerm(func=mdp.generated_commands,params={"command_name":"VelocityCommand"}) # 机器人的速度指令，Shape is (num_instances, 3)
        joint_pos = ObsTerm(func=mdp.joint_pos) # 关节position，Shape is (num_instances, num_joints)
        joint_vel = ObsTerm(func=mdp.joint_vel) # 关节velocity，Shape is (num_instances, num_joints)
        # 正常来说，不能直接从IMU中获取速度信息，必须通过状态估计才行
        # IMU_base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # root在自身坐标系下的线速度，Shape is (num_instances, 3)
        IMU_base_acc = ObsTerm(func=mdp.get_body_acc_lin, params={"asset_cfg":
                                                              SceneEntityCfg(name="robot",body_names="trunk")}) # 机器人在自身坐标系的加速度，Shape is (num_instances, 6)
        IMU_base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # root在自身坐标系下的角速度，Shape is (num_instances, 3)
        prev_action = ObsTerm(func=mdp.last_action, params={"action_name":"joint_pos"}) # 上一个action，Shape is (num_instances, num_joints)
        project_grav = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg":SceneEntityCfg(name="robot")}) # 机器人的重力投影，Shape is (num_instances, 3)
        # 这里的last_action是获取上一次的action，注意，是actor网络输出的action，而不是传入给环境经过缩放+偏移后的action

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# 定义CommandsCfg
@configclass
class CommandsCfg:
    # 只设置了速度的command，没有设置heading的command
    # TODO: 其实在下面这个配置中，我认为heading_control_stiffness参数和rel_heading_envs参数是不需要配置的，因为设置了heading_command
    VelocityCommand = mdp.UniformVelocityCommandCfg(ranges = mdp.UniformVelocityCommandCfg.Ranges(
                                                    lin_vel_x = (-0.6,0.6),lin_vel_y = (-0.6,0.6),ang_vel_z = (-0.8,0.8),heading = (-math.pi,math.pi)),
                                                    heading_command = False, debug_vis = True, resampling_time_range = (10,10),
                                                    asset_name = "robot", rel_standing_envs = 0.1,
                                                    heading_control_stiffness = 0.5, rel_heading_envs = 0.0)
    

# 定义RewardsCfg
@configclass
class RewardsCfg:
    # IMPORTANT: 在mdp中的rewards中定义了很多reward或者pnalty的函数，考虑到RL的训练是基于reward的，因此需要注意weight的正负号设计
    # 让机器人能够按照指定的在自身坐标系下的速度和方向行走
    track_lin_vel_xy_reward = RewTerm(func = mdp.track_lin_vel_xy_exp, weight = 4, params={"std":math.sqrt(0.25),"command_name":"VelocityCommand"})
    track_ang_vel_z_reward = RewTerm(func = mdp.track_ang_vel_z_exp, weight = 2, params={"std":math.sqrt(0.25),"command_name":"VelocityCommand"})
    # 惩罚在z轴方向上的速度以及x和y的角速度
    pnealize_linvel_z_l2 = RewTerm(func = mdp.lin_vel_z_l2, weight = -2) # 注意权重是负数
    pnealize_angvel_xy_l2 = RewTerm(func = mdp.ang_vel_xy_l2, weight = -0.05) # 注意权重是负数
    # 惩罚关节的速度、加速度和关节力矩
    # pnealize_joint_vel_l2 = RewTerm(func = mdp.joint_vel_l2, weight = -0.001) # 注意权重是负数
    pnealize_joint_acc_l2 = RewTerm(func = mdp.joint_acc_l2, weight = -2.5e-7) # 注意权重是负数
    pnealize_joint_effort_l2 = RewTerm(func = mdp.joint_torques_l2, weight = -2e-4) # 注意权重是负数
    # 奖励机器人足的腾空时间长度，如果超过threshold，就会有奖励，低于threshold相当于有惩罚
    feet_air_time_reward = RewTerm(func = mdp.feet_air_time, weight = 2, 
                                   params={"threshold":[0.5,1.0],"sensor_cfg":SceneEntityCfg(name="foot_contact_sensor",body_names=".*foot"),
                                           "command_name":"VelocityCommand"})
    pnealize_base_orientation = RewTerm(func = mdp.flat_orientation_l2, weight = -5.0)
    pnealize_joint_offset_default = RewTerm(func = mdp.joint_deviation_l1, weight = -0.005,
                                            params={"asset_cfg":SceneEntityCfg(name = "robot",joint_names = [".*_hip_joint",".*_thigh_joint"])})
    # 惩罚机器人的控制action变化率
    pnealize_action_change_l2 = RewTerm(func = mdp.action_rate_l2, weight = -0.01) # 注意权重是负数
    pnealize_joint_pos_limit = RewTerm(func = mdp.joint_pos_limits, weight = -10) # 注意权重是负数
    # 惩罚机器人碰撞
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-0.1,
                                 params={"sensor_cfg": SceneEntityCfg("whole_body_contact_sensor", body_names=[".*calf",".*thigh"]), "threshold": 100.0})
    
    # NOTE: 如果在奖励函数中引入这个由雷达传感器获取的高度信息，使用rsl-rl训练过程中会报一个错误：normal expects all elements of std >= 0.0
    # 要解决这个问题，可以参考这个回答：https://github.com/leggedrobotics/rsl_rl/issues/33#issuecomment-2254092684
    # 奖励机器人的抬腿高度
    # reward_foot_hight = RewTerm(func = mdp.feet_hight_reward, weight = 0.0008, 
    #                               params = {"sensor_cfg":[SceneEntityCfg(name="height_scanner_FL"),SceneEntityCfg(name="height_scanner_FR"),
    #                                                       SceneEntityCfg(name="height_scanner_RL"),SceneEntityCfg(name="height_scanner_RR")],
    #                                         "command_name":"VelocityCommand"})

    pnealize_foot_contact_ground_time = RewTerm(func = mdp.feet_current_ground_time, weight = -20,
                                                params = {"sensor_cfg":SceneEntityCfg(name="foot_contact_sensor",body_names=".*foot"),
                                                          "command_name":"VelocityCommand", "threshold":1})


# 定义ActionsCfg
@configclass
class ActionsCfg:
    # IMPORTANT: 在JointPositionActionCfg中，scale非常重要，这个scale是用来控制action的幅度的，如果scale过大，那么机器人的动作就会非常大，这样会导致训练很难收敛
    joint_pos = mdp.JointPositionActionCfg(asset_name = "robot", joint_names=[".*_hip_joint",".*_thigh_joint",".*_calf_joint"],
                                           debug_vis=False, scale=0.25)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func = mdp.time_out, time_out=True) # 这个是设置timeout的terminations

    base_contact_ground = DoneTerm(func = mdp.base_contact_ground_func,
                                   params = {"asset_cfg":SceneEntityCfg(name="base_contact_sensor", body_names="trunk")}) # 这个是设置机器人root是否与地面接触的terminations

    base_bad_orientation = DoneTerm(func = mdp.bad_orientation, params={"limit_angle":math.pi/3}) # 如果机器人的roll或者pitch超过了limit_angle，就会终止
    base_hight = DoneTerm(func = mdp.root_height_below_minimum, params={"minimum_height":0.06}) # 如果机器人的高度低于height，就会终止

@configclass
class CurriculumCfg:
    modify_track_lin_vel_xy_reward = CurrTerm(func = mdp.modify_reward_weight, 
                                              params = {"term_name":"track_lin_vel_xy_reward","weight":1,"num_steps":800})
    modify_track_ang_vel_z_reward = CurrTerm(func = mdp.modify_reward_weight,
                                                params = {"term_name":"track_ang_vel_z_reward","weight":0.5,"num_steps":800})


@configclass
class EventCfg:
    reset_A1_position = EventTerm(func = mdp.reset_root_state_uniform, mode = "reset", 
                                  params = {"pose_range":{"x":(-0.5,0.5), "y":(-0.5,0.5), "z":(-0.2,0.4), "roll":(-math.pi/20,math.pi/20),
                                                          "pitch":(-math.pi/20,math.pi/20), "yaw":(-math.pi/12,math.pi/12)},
                                            "velocity_range": {"x":(-0.2,0.2), "y":(-0.2,0.2), "z":(-0.2,0.2), "yaw":(-math.pi/24,math.pi/24),
                                                               "pitch":(-math.pi/24,math.pi/24), "roll":(-math.pi/24,math.pi/24)}})
    
    reset_A1_joints_position = EventTerm(func = mdp.reset_joints_by_offset, mode = "reset",
                                        params = {"position_range":(-0.1,0.1), "velocity_range":(-0.07,0.07)})

@configclass
class UnitreeA1WalkEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self) -> None:
        self.episode_length_s = 15.0 # 一个episode的长度，单位是秒
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
    
    scene: UnitreeA1WalkSceneCfg = UnitreeA1WalkSceneCfg(num_envs = 512, env_spacing= 2.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventCfg = EventCfg()