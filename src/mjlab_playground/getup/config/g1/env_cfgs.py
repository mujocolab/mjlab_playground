"""Unitree G1 getup environment configuration."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  ACTUATOR_4010,
  ACTUATOR_5020,
  ACTUATOR_7520_14,
  ACTUATOR_7520_22,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_playground.getup import mdp
from mjlab_playground.getup.getup_env_cfg import make_getup_env_cfg
from mjlab_playground.getup.mdp.actions import SettleRelativeJointPositionActionCfg

##
# Actuator config at 5 Hz.
##

_NATURAL_FREQ = 5.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0


def _kp(act) -> float:
  return act.reflected_inertia * _NATURAL_FREQ**2


def _kv(act) -> float:
  return 2.0 * _DAMPING_RATIO * act.reflected_inertia * _NATURAL_FREQ


_G1_ARTICULATION_5HZ = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
      target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
      ),
      stiffness=_kp(ACTUATOR_5020),
      damping=_kv(ACTUATOR_5020),
      effort_limit=ACTUATOR_5020.effort_limit,
      armature=ACTUATOR_5020.reflected_inertia,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
      stiffness=_kp(ACTUATOR_7520_14),
      damping=_kv(ACTUATOR_7520_14),
      effort_limit=ACTUATOR_7520_14.effort_limit,
      armature=ACTUATOR_7520_14.reflected_inertia,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
      stiffness=_kp(ACTUATOR_7520_22),
      damping=_kv(ACTUATOR_7520_22),
      effort_limit=ACTUATOR_7520_22.effort_limit,
      armature=ACTUATOR_7520_22.reflected_inertia,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
      stiffness=_kp(ACTUATOR_4010),
      damping=_kv(ACTUATOR_4010),
      effort_limit=ACTUATOR_4010.effort_limit,
      armature=ACTUATOR_4010.reflected_inertia,
    ),
    # Waist pitch/roll and ankles are 4-bar linkages with 2x 5020 actuators.
    BuiltinPositionActuatorCfg(
      target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
      stiffness=_kp(ACTUATOR_5020) * 2,
      damping=_kv(ACTUATOR_5020) * 2,
      effort_limit=ACTUATOR_5020.effort_limit * 2,
      armature=ACTUATOR_5020.reflected_inertia * 2,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
      stiffness=_kp(ACTUATOR_5020) * 2,
      damping=_kv(ACTUATOR_5020) * 2,
      effort_limit=ACTUATOR_5020.effort_limit * 2,
      armature=ACTUATOR_5020.reflected_inertia * 2,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

##
# Heights.
##

# Derived from home keyframe.
_TORSO_HEIGHT = 0.804  # torso_link xpos z (body frame origin, not COM)
_PELVIS_HEIGHT = 0.760  # pelvis xpos z (body frame origin, not COM)


def unitree_g1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.sim.nconmax = 75
  cfg.sim.mujoco.cone = "pyramidal"
  cfg.sim.mujoco.impratio = 1.0

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.articulation = _G1_ARTICULATION_5HZ
  cfg.scene.entities = {"robot": robot_cfg}

  # Self-collision sensor (pelvis is G1 root).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Torso + waist (pelvis) height. Waist reward prevents "sitting" local minimum where
  # torso_link is high but pelvis stays near the ground.
  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso_link",)
  )
  cfg.rewards["waist_height"] = RewardTermCfg(
    func=mdp.height_reward,
    weight=1.0,
    params={
      "desired_height": _PELVIS_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
    },
  )
  cfg.metrics["getup_success"].params["desired_height"] = _PELVIS_HEIGHT
  cfg.metrics["getup_success"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso_link",)
  )

  # Per-joint posture std: tight hips, medium knees and ankles, loose arms and waist.
  cfg.rewards["posture"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(".*",), body_names=("torso_link",)
  )
  cfg.rewards["posture"].params["std"] = {
    r".*_hip_roll_joint": 0.08,
    r".*_hip_yaw_joint": 0.08,
    r".*_hip_pitch_joint": 0.12,
    r".*_knee_joint": 0.15,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.2,
    r"(waist_.*|.*_shoulder.*|.*_elbow.*|.*_wrist.*)": 0.5,
  }

  cfg.rewards["orientation"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso_link",)
  )

  # Override projected_gravity to use torso_link instead of pelvis (root).
  # G1's 3-DOF waist decouples pelvis and torso orientation — the pelvis gravity
  # signal is misleading; the policy needs to see torso uprightness directly.
  _torso_cfg = SceneEntityCfg("robot", body_names=("torso_link",))
  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.body_projected_gravity,
    params={"asset_cfg": _torso_cfg},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  cfg.observations["critic"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.body_projected_gravity,
    params={"asset_cfg": _torso_cfg},
  )

  cfg.viewer.body_name = "torso_link"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso_link",)
  )

  # # G1 has 7 foot collision geoms per foot (left/right_foot1-7_collision).
  # foot_geom_names = tuple(
  #   f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  # )
  cfg.events["geom_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  # cfg.events["foot_friction_spin"] = EventTermCfg(
  #   mode="startup",
  #   func=envs_mdp.dr.geom_friction,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
  #     "operation": "abs",
  #     "distribution": "log_uniform",
  #     "axes": [1],
  #     "ranges": (1e-4, 2e-2),
  #     "shared_random": True,
  #   },
  # )
  # cfg.events["foot_friction_roll"] = EventTermCfg(
  #   mode="startup",
  #   func=envs_mdp.dr.geom_friction,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
  #     "operation": "abs",
  #     "distribution": "log_uniform",
  #     "axes": [2],
  #     "ranges": (1e-5, 5e-3),
  #     "shared_random": True,
  #   },
  # )

  # G1 needs more clearance when placed fallen.
  cfg.events["reset_fallen_or_standing"].params["fall_height"] = 0.8
  cfg.events["reset_fallen_or_standing"].params["joint_range_scale"] = 0.5

  assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
  cfg.actions["joint_pos"].settle_steps = 30
  cfg.terminations["energy"].params["settle_steps"] = 30

  # # joint_torques_l2 sums squared actuator forces — at ~30 Nm average across 29 joints
  # # the raw value is ~26000, so weight must be small.
  # cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=0.0)

  # Smoothness curriculum: mirrors T1 shape but delayed ~200 iters to give G1 more time
  # to learn getup before penalties kick in. action_rate raw value is ~67 at weight=-0.01
  # (29 joints vs T1's fewer), so keep final weight at -0.1 matching T1 — policy adapts.
  # cfg.curriculum = {
  #   "action_rate_weight": CurriculumTermCfg(
  #     func=mdp.reward_curriculum,
  #     params={
  #       "reward_name": "action_rate_l2",
  #       "stages": [
  #         {"step": 0, "weight": -0.01},
  #         {"step": 800 * 24, "weight": -0.05},
  #         {"step": 1200 * 24, "weight": -0.08},
  #         {"step": 1600 * 24, "weight": -0.1},
  #       ],
  #     },
  #   ),
  #   "joint_vel_weight": CurriculumTermCfg(
  #     func=mdp.reward_curriculum,
  #     params={
  #       "reward_name": "joint_vel_l2",
  #       "stages": [
  #         {"step": 0, "weight": 0.0},
  #         {"step": 1000 * 24, "weight": -0.005},
  #         {"step": 1400 * 24, "weight": -0.008},
  #         {"step": 1800 * 24, "weight": -0.01},
  #       ],
  #     },
  #   ),
  # }
  # cfg.curriculum = {
  #   "action_rate_weight": CurriculumTermCfg(
  #     func=mdp.reward_curriculum,
  #     params={
  #       "reward_name": "action_rate_l2",
  #       "stages": [
  #         {"step": 0,          "weight": -0.001},
  #         {"step": 1500 * 24,  "weight": -0.003},
  #         {"step": 2500 * 24,  "weight": -0.006},
  #         {"step": 3500 * 24,  "weight": -0.01},
  #         {"step": 4000 * 24,  "weight": -0.05},
  #       ],
  #     },
  #   ),
  #   "joint_vel_weight": CurriculumTermCfg(
  #     func=mdp.reward_curriculum,
  #     params={
  #       "reward_name": "joint_vel_l2",
  #       "stages": [
  #         {"step": 0,          "weight": -0.0},
  #         {"step": 2000 * 24,  "weight": -0.005},
  #         {"step": 2500 * 24,  "weight": -0.008},
  #         {"step": 3000 * 24,  "weight": -0.01},
  #         {"step": 3500 * 24,  "weight": -0.1},
  #       ],
  #     },
  #   ),
  # }
  cfg.curriculum = {}
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["joint_vel_l2"].weight = -0.0
  cfg.rewards["joint_vel_hinge"] = RewardTermCfg(
    func=mdp.joint_vel_hinge,
    weight=-0.1,
    params={"threshold": 2.0},  # rad/s
  )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
