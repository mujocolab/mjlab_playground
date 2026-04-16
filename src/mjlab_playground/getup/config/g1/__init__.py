from mjlab.tasks.registry import register_mjlab_task

from mjlab_playground.getup.rl import GetupOnPolicyRunner

from .env_cfgs import unitree_g1_getup_env_cfg
from .rl_cfg import unitree_g1_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-Unitree-G1",
  env_cfg=unitree_g1_getup_env_cfg(),
  play_env_cfg=unitree_g1_getup_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_ppo_runner_cfg(),
  runner_cls=GetupOnPolicyRunner,
)
