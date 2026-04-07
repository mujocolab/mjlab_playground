"""Reward functions for the getup task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost  # noqa: F401
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Projected gravity in body frame when upright.
_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


def joint_vel_hinge(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint velocities exceeding a threshold (hinge/relu penalty).

  Unlike joint_vel_l2 which penalizes all motion, this is zero below `threshold`
  and quadratic on the excess above it. This lets the policy move freely at normal
  speeds while strongly penalizing violent velocity spikes.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  excess = torch.clamp(torch.abs(vel) - threshold, min=0.0)
  return torch.sum(excess**2, dim=-1)


def orientation_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for upright orientation.

  If asset_cfg has body_names, uses that body's orientation. Otherwise falls
  back to the root body's projected_gravity_b.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_names is not None:
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(
      env.num_envs, -1
    )
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    quat = asset.data.body_link_quat_w[:, body_ids[0]]  # (num_envs, 4)
    gravity = quat_apply_inverse(quat, gravity_w)
  else:
    gravity = asset.data.projected_gravity_b
  up = _UP_VEC.to(gravity.device)
  error = torch.sum(torch.square(up - gravity), dim=-1)
  return torch.exp(-2.0 * error)


def height_reward(
  env: ManagerBasedRlEnv,
  desired_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for raising a body to the desired height."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
  clamped = torch.clamp(height, max=desired_height)
  return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


def _is_upright(
  asset: Entity,
  orientation_threshold: float,
  body_ids: list[int] | None = None,
) -> torch.Tensor:
  if body_ids is not None:
    quat = asset.data.body_link_quat_w[:, body_ids[0]]
    gravity = quat_apply_inverse(
      quat, torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(quat.shape[0], -1)
    )
  else:
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
  """Reward for returning to default pose, gated on being upright."""

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
    body_ids = asset_cfg.body_ids if asset_cfg.body_names is not None else None
    assert body_ids is None or isinstance(body_ids, list)
    gate = _is_upright(asset, orientation_threshold, body_ids)
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return gate * torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


class getup_success:
  """Binary success metric: 1 once the robot has stood up, 0 otherwise."""

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
    body_ids = asset_cfg.body_ids if asset_cfg.body_names is not None else None
    assert body_ids is None or isinstance(body_ids, list)
    standing = _is_upright(
      asset, orientation_threshold, body_ids
    ) * _is_at_desired_height(asset, desired_height, height_tolerance)
    self._stood_up = torch.maximum(self._stood_up, standing)
    return self._stood_up
