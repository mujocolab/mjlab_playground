"""Booster T1 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

T1_XML: Path = Path(__file__).parent / "xmls" / "t1.xml"
assert T1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(T1_XML))


##
# Actuator config.
# Reference: https://booster.feishu.cn/wiki/JGZAwk8CUi5m6nklgxMcp2KlnVe
##

_rpm = lambda r: r * 2 * 3.14159265 / 60  # noqa: E731

NECK_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(18e-6, 10),
  velocity_limit=_rpm(400),
  effort_limit=7.0,
)

ARM_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(21.8e-6, 36),
  velocity_limit=_rpm(89),
  effort_limit=36.0,
)

WAIST_HIP_ROLL_YAW_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(76.5e-6, 25),
  velocity_limit=_rpm(70),
  effort_limit=40.0,
)

HIP_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(161.7e-6, 18),
  velocity_limit=_rpm(157),
  effort_limit=55.0,
)

KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(196.3e-6, 18),
  velocity_limit=_rpm(140),
  effort_limit=65.0,
)

ANKLE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(26.2e-6, 36),
  velocity_limit=_rpm(117),
  effort_limit=50.0,
)

NATURAL_FREQ = 5.0 * 2.0 * 3.14159265  # 5 Hz
DAMPING_RATIO = 2.0


def _kp(act: ElectricActuator) -> float:
  return act.reflected_inertia * NATURAL_FREQ**2


def _kv(act: ElectricActuator) -> float:
  return 2.0 * DAMPING_RATIO * act.reflected_inertia * NATURAL_FREQ


T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=_kp(NECK_ACTUATOR),
  damping=_kv(NECK_ACTUATOR),
  effort_limit=NECK_ACTUATOR.effort_limit,
  armature=NECK_ACTUATOR.reflected_inertia,
)

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  stiffness=_kp(ARM_ACTUATOR),
  damping=_kv(ARM_ACTUATOR),
  effort_limit=ARM_ACTUATOR.effort_limit,
  armature=ARM_ACTUATOR.reflected_inertia,
)

T1_ACTUATOR_WAIST_HIP_ROLL_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("Waist", ".*_Hip_Roll", ".*_Hip_Yaw"),
  stiffness=_kp(WAIST_HIP_ROLL_YAW_ACTUATOR),
  damping=_kv(WAIST_HIP_ROLL_YAW_ACTUATOR),
  effort_limit=WAIST_HIP_ROLL_YAW_ACTUATOR.effort_limit,
  armature=WAIST_HIP_ROLL_YAW_ACTUATOR.reflected_inertia,
)

T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  stiffness=_kp(HIP_PITCH_ACTUATOR),
  damping=_kv(HIP_PITCH_ACTUATOR),
  effort_limit=HIP_PITCH_ACTUATOR.effort_limit,
  armature=HIP_PITCH_ACTUATOR.reflected_inertia,
)

T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  stiffness=_kp(KNEE_ACTUATOR),
  damping=_kv(KNEE_ACTUATOR),
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

T1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll"),
  stiffness=_kp(ANKLE_ACTUATOR),
  damping=_kv(ANKLE_ACTUATOR),
  effort_limit=ANKLE_ACTUATOR.effort_limit,
  armature=ANKLE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.665),
  joint_pos={
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(left|right)_foot\d+_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 3},
  friction={_foot_regex: (1, 5e-3, 5e-4), ".*_collision": (0.6,)},
  priority=1,
)

##
# Final config.
##

T1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T1_ACTUATOR_NECK,
    T1_ACTUATOR_ARM,
    T1_ACTUATOR_WAIST_HIP_ROLL_YAW,
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_KNEE,
    T1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t1_robot_cfg() -> EntityCfg:
  """Get a fresh T1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=T1_ARTICULATION,
  )


T1_ACTION_SCALE: dict[str, float] = {}
for a in T1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    T1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_t1_robot_cfg())

  viewer.launch(robot.spec.compile())
