"""Getup task configuration.

This module provides a factory function to create a base getup task config.
Robot-specific configurations call the factory and customize as needed.

Adapted from MuJoCo Playground (https://github.com/google-deepmind/mujoco_playground/).

References:
  Zakka et al., "MuJoCo Playground", 2025. https://arxiv.org/abs/2502.08844
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_playground.getup import mdp
from mjlab_playground.getup.mdp.actions import SettleRelativeJointPositionActionCfg


def make_getup_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base getup (fall recovery) task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.03, n_max=0.03),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": SettleRelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.6,
      settle_steps=25,  # 0.5s settle at 50Hz, matching playground.
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_fallen_or_standing": EventTermCfg(
      func=mdp.reset_fallen_or_standing,
      mode="reset",
      params={
        "fall_probability": 0.6,
        "fall_height": 0.5,
        "velocity_range": 0.5,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "getup_success": MetricsTermCfg(
      func=mdp.getup_success,
      reduce="last",
      params={
        "desired_height": 0.275,
        "height_tolerance": 0.02,
        "orientation_threshold": 0.05,
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "orientation": RewardTermCfg(
      func=mdp.orientation_reward,
      weight=1.0,
    ),
    "torso_height": RewardTermCfg(
      func=mdp.torso_height_reward,
      weight=1.0,
      params={"desired_height": 0.275},
    ),
    "posture": RewardTermCfg(
      func=mdp.gated_posture_reward,
      weight=1.0,
      params={
        "orientation_threshold": 0.01,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std": {},  # Set per-robot.
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=0.0),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }

  ##
  # Curriculum
  ##

  # Curriculum (1.5K iterations):
  #   Phase 1 (0 to 500):     Learn to stand up, minimal penalties.
  #   Phase 2 (500 to 900):   Gentle smoothing.
  #   Phase 3 (900 to 1200):  Moderate tightening.
  #   Phase 4 (1200 to 1500): Final deployment quality with 400W cap.
  curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 500 * 24, "weight": -0.05},
          {"step": 900 * 24, "weight": -0.08},
          {"step": 1200 * 24, "weight": -0.1},
        ],
      },
    ),
    "joint_vel_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_l2",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 500 * 24, "weight": -0.005},
          {"step": 900 * 24, "weight": -0.008},
          {"step": 1200 * 24, "weight": -0.01},
        ],
      },
    ),
    "energy_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 500 * 24, "params": {"threshold": 1000.0}},
          {"step": 900 * 24, "params": {"threshold": 700.0}},
          {"step": 1200 * 24, "params": {"threshold": 400.0}},
        ],
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "energy": TerminationTermCfg(
      func=mdp.energy_termination,
      params={"threshold": float("inf")},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=100,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=6.0,
  )
