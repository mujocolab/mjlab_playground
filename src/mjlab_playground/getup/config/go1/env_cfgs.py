"""Unitree Go1 getup environment configuration."""

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_playground.getup import mdp
from mjlab_playground.getup.getup_env_cfg import make_getup_env_cfg

_TORSO_HEIGHT = 0.275


def unitree_go1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 getup task configuration."""
  cfg = make_getup_env_cfg()

  robot_cfg = get_go1_robot_cfg()

  # Can't have frictionless body contacts for getup, so upgrade all
  # bodies to condim=3 and feet to condim=6.
  _foot_regex = "^[FR][LR]_foot_collision$"
  robot_cfg.collisions = (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      solref=(0.01, 1),
      condim={_foot_regex: 6, ".*_collision": 3},
      friction={_foot_regex: (1, 5e-3, 5e-4), ".*_collision": (0.6,)},
      priority=1,
    ),
  )

  cfg.scene.entities = {"robot": robot_cfg}

  cfg.sim.njmax = 200

  # Self-collision sensor (history_length matches decimation=4).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Per-joint posture std: tight hips (prevent splay), medium thighs,
  # looser calves.
  cfg.rewards["posture"].params["std"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }

  cfg.viewer.body_name = "trunk"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  foot_geom_names = tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL"))
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
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
