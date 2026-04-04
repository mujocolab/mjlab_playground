"""Reward functions for the getup task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Target gravity vector when upright (gravity points down in body frame).
_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


def orientation_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for upright orientation.

  Returns exp(-2 * ||up - projected_gravity||^2).
  """
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  up = _UP_VEC.to(gravity.device)
  error = torch.sum(torch.square(up - gravity), dim=-1)
  return torch.exp(-2.0 * error)


def body_height_reward(
  env: ManagerBasedRlEnv,
  desired_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for raising a specific body to the desired height.

  Like torso_height_reward but targets a named body via asset_cfg.body_ids
  instead of the root link. Use asset_cfg=SceneEntityCfg("robot", body_names=("Waist",))
  to target e.g. the waist.

  Returns a value in [0, 1].
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
  clamped = torch.clamp(height, max=desired_height)
  return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


def torso_height_reward(
  env: ManagerBasedRlEnv,
  desired_height: float = 0.275,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for raising the torso to the desired height.

  Returns a value in [0, 1].
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  clamped = torch.clamp(height, max=desired_height)
  return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


def _is_upright(asset: Entity, orientation_threshold: float) -> torch.Tensor:
  gravity = asset.data.projected_gravity_b
  up = _UP_VEC.to(gravity.device)
  error = torch.sum(torch.square(up - gravity), dim=-1)
  return (error < orientation_threshold).float()


def _is_at_desired_height(
  asset: Entity, desired_height: float, height_tolerance: float
) -> torch.Tensor:
  height = asset.data.root_link_pos_w[:, 2]
  clamped = torch.clamp(height, max=desired_height)
  error = desired_height - clamped
  return (error < height_tolerance).float()


class gated_posture_reward:
  """Reward for returning to default pose, gated on being upright.

  Uses per-joint standard deviations for fine-grained control over which
  joints matter most. Implemented as a class to resolve the std dict once.

  Returns is_upright * exp(-mean(error^2 / std^2)).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: dict[str, float],
    orientation_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del std  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    gate = _is_upright(asset, orientation_threshold)
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return gate * torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


class getup_success:
  """Binary success metric: 1 once the robot has stood up, 0 otherwise.

  Latches to 1.0 when the standing condition is first met. Use with
  ``reduce="last"`` in MetricsTermCfg so the manager reports the final
  value at episode end, giving a clean 0-to-1 success rate.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._stood_up = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._stood_up[:] = 0.0
    else:
      self._stood_up[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    desired_height: float = 0.275,
    height_tolerance: float = 0.02,
    orientation_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    standing = _is_upright(asset, orientation_threshold) * _is_at_desired_height(
      asset, desired_height, height_tolerance
    )
    self._stood_up = torch.maximum(self._stood_up, standing)
    return self._stood_up


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  Counts substeps where any contact force exceeds *force_threshold*
  using the sensor's force history.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.sum(dim=-1).float()
