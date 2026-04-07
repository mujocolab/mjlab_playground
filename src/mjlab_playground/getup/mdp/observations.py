"""Observation functions for the getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gravity vector projected into a specific body's frame.

  Unlike the built-in projected_gravity which always uses the root body,
  this function supports any body via asset_cfg.body_names.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  assert isinstance(body_ids, list)
  gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
  quat = asset.data.body_link_quat_w[:, body_ids[0]]
  return quat_apply_inverse(quat, gravity_w)
