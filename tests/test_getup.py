"""Tests for the getup task configurations."""

from __future__ import annotations

import mujoco
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from mjlab_playground.getup.config.go1.env_cfgs import unitree_go1_getup_env_cfg
from mjlab_playground.getup.config.t1.env_cfgs import booster_t1_getup_env_cfg

_NUM_ENVS = 5


# Helpers.


def _collision_geom_priorities(model: mujoco.MjModel) -> dict[str, int]:
  """Return {geom_name: priority} for all geoms ending in '_collision'."""
  result = {}
  for i in range(model.ngeom):
    name = model.geom(i).name
    if name.endswith("_collision"):
      result[name] = model.geom(i).priority
  return result


def _get_friction(
  env: ManagerBasedRlEnv,
  geom_names: tuple[str, ...],
) -> torch.Tensor:
  """Return geom_friction for the given geoms. Shape: [num_envs, num_geoms, 3]."""
  entity = env.scene["robot"]
  cfg = SceneEntityCfg("robot", geom_names=geom_names)
  cfg.resolve(env.scene)
  global_ids = entity.indexing.geom_ids[cfg.geom_ids]
  return env.sim.model.geom_friction[:, global_ids]


def _assert_varied_across_worlds(values: torch.Tensor, label: str) -> None:
  """Assert that not all worlds have the same value (DR produced variation)."""
  assert values.shape[0] >= 2, "Need at least 2 envs to check variation"
  all_same = torch.all(values[0] == values[1:])
  assert not all_same, f"{label}: all {values.shape[0]} worlds got identical values"


# Fixtures.


@pytest.fixture(scope="module")
def t1_model() -> mujoco.MjModel:
  cfg = booster_t1_getup_env_cfg()
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, "cpu")
  return scene.compile()


@pytest.fixture(scope="module")
def go1_model() -> mujoco.MjModel:
  cfg = unitree_go1_getup_env_cfg()
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, "cpu")
  return scene.compile()


@pytest.fixture(scope="module")
def t1_env() -> ManagerBasedRlEnv:
  cfg = booster_t1_getup_env_cfg()
  cfg.scene.num_envs = _NUM_ENVS
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()
  return env


@pytest.fixture(scope="module")
def go1_env() -> ManagerBasedRlEnv:
  cfg = unitree_go1_getup_env_cfg()
  cfg.scene.num_envs = _NUM_ENVS
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()
  return env


# Collision priority.


def test_t1_collision_priority(t1_model: mujoco.MjModel) -> None:
  priorities = _collision_geom_priorities(t1_model)
  assert len(priorities) > 0, "No collision geoms found for T1"
  for name, priority in priorities.items():
    assert priority == 1, f"T1 geom {name!r} has priority={priority}, expected 1"


def test_go1_collision_priority(go1_model: mujoco.MjModel) -> None:
  priorities = _collision_geom_priorities(go1_model)
  assert len(priorities) > 0, "No collision geoms found for Go1"
  for name, priority in priorities.items():
    assert priority == 1, f"Go1 geom {name!r} has priority={priority}, expected 1"


# Friction domain randomization.


def test_t1_friction_dr(t1_env: ManagerBasedRlEnv) -> None:
  all_friction = _get_friction(t1_env, (".*_collision",))
  _assert_varied_across_worlds(all_friction[:, :, 0], "T1 slide friction (axis 0)")

  foot_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )
  foot_friction = _get_friction(t1_env, foot_names)
  _assert_varied_across_worlds(foot_friction[:, :, 1], "T1 foot spin friction (axis 1)")
  _assert_varied_across_worlds(foot_friction[:, :, 2], "T1 foot roll friction (axis 2)")


def test_go1_friction_dr(go1_env: ManagerBasedRlEnv) -> None:
  all_friction = _get_friction(go1_env, (".*_collision",))
  _assert_varied_across_worlds(all_friction[:, :, 0], "Go1 slide friction (axis 0)")

  foot_names = tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL"))
  foot_friction = _get_friction(go1_env, foot_names)
  _assert_varied_across_worlds(
    foot_friction[:, :, 1], "Go1 foot spin friction (axis 1)"
  )
  _assert_varied_across_worlds(
    foot_friction[:, :, 2], "Go1 foot roll friction (axis 2)"
  )
