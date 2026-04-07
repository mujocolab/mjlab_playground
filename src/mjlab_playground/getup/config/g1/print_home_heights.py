"""Print torso_link and pelvis z-heights at the home keyframe."""

import mujoco
from mjlab.scene import Scene

from mjlab_playground.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg

cfg = unitree_g1_getup_env_cfg()
model = Scene(cfg.scene, device="cpu").compile()
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

for body_name in ("pelvis", "torso_link"):
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{body_name}")
  print(f"{body_name}: z = {data.xpos[bid, 2]:.6f}")
