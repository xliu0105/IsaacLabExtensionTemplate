from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UnitreeA1WalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 # 这个应该是multi-step的步数
    max_iterations = 1000 # 训练的最大迭代次数
    save_interval = 30 # 多少次迭代保存一次模型
    experiment_name = "UnitreeA1Walk"
    empirical_normalization = False
    resume = False # IMPORTANT: 是否resume训练
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128,64,32],
        critic_hidden_dims=[128,64,32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )