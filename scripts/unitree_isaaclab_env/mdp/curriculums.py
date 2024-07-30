from omni.isaac.lab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
from typing import TYPE_CHECKING


def reward_weight_adapt(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, steps_length: int,
                        max_iterations : int | None = None ,weight_descent: bool = True):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight to multiple the reward term weight.
        steps_length: How many steps to adjust the reward weight for.
        weight_descent: If True, the weight will decrease linearly from the current weight to the target weight. if False, the weight will increase linearly from the current weight to the target weight.
    """
    if(max_iterations is None or (env.common_step_counter / steps_length) <= max_iterations):
        if (env.common_step_counter % steps_length) == 0:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            if weight_descent:
                term_cfg.weight *= (1-weight)
            else:
                term_cfg.weight *= (1+weight)
            env.reward_manager.set_term_cfg(term_name, term_cfg)
            print(f"Set term {term_name} weight to {term_cfg.weight}")