"""Reset events for the getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_fallen_or_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fall_probability: float = 0.6,
  fall_height: float = 0.5,
  velocity_range: float = 0.5,
  joint_range_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robots to either a random fallen configuration or standing.

  With ``fall_probability``, the robot is placed at ``fall_height`` with a random
  orientation, random joint positions across the full joint range, and random root
  velocities. Otherwise it starts in the default standing pose.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    fall_probability: Probability of starting in a fallen configuration.
    fall_height: Height (m) to place the robot when fallen.
    velocity_range: Root velocity sampled uniformly in [-range, range].
    joint_range_scale: Scale factor in (0, 1] applied to the soft joint range
      symmetrically around the midpoint. 1.0 = full range; 0.5 = half range.
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  n = len(env_ids)
  asset: Entity = env.scene[asset_cfg.name]

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  # Sample fall mask and store it so the action term can skip settle for standing envs.
  fall_mask = torch.rand(n, device=env.device) < fall_probability
  if "settle_mask" not in env.extras:
    env.extras["settle_mask"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
  env.extras["settle_mask"][env_ids] = fall_mask

  # Root state.
  root_states = default_root_state[env_ids].clone()

  # Fallen: random quaternion, fixed height, random velocities.
  random_quat = torch.randn(n, 4, device=env.device)
  random_quat = F.normalize(random_quat, dim=-1)

  fallen_positions = env.scene.env_origins[env_ids].clone()
  fallen_positions[:, 2] += fall_height

  fallen_velocities = sample_uniform(
    -velocity_range, velocity_range, (n, 6), env.device
  )

  # Standing: default state offset to env origin with a small z bump to avoid ground
  # penetration.
  standing_positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
  standing_positions[:, 2] += 0.02

  mask = fall_mask.unsqueeze(-1)
  positions = torch.where(mask, fallen_positions, standing_positions)
  orientations = torch.where(mask, random_quat, root_states[:, 3:7])
  velocities = torch.where(mask, fallen_velocities, root_states[:, 7:13])

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

  # Joint state.
  joint_limits = soft_joint_pos_limits[env_ids]
  mid = (joint_limits[..., 0] + joint_limits[..., 1]) * 0.5
  half = (joint_limits[..., 1] - joint_limits[..., 0]) * 0.5 * joint_range_scale
  random_joint_pos = sample_uniform(mid - half, mid + half, mid.shape, env.device)

  joint_pos = torch.where(mask, random_joint_pos, default_joint_pos[env_ids].clone())
  joint_vel = torch.where(
    mask,
    sample_uniform(-velocity_range, velocity_range, joint_pos.shape, env.device),
    default_joint_vel[env_ids].clone(),
  )

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
