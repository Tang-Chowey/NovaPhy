"""Configuration and constants for the FR3 IK demo."""

from dataclasses import dataclass
import os

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_demos_dir = os.path.dirname(_module_dir)

FRANKA_URDF_PATH = os.path.join(
    _demos_dir, "data", "franka_emika_panda", "urdf", "fr3_franka_hand.urdf"
)
FRANKA_EE_LINK_NAME = "fr3_hand_tcp"
FRANKA_SHOULDER_LINK_NAME = "fr3_link1"
FRANKA_ARM_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

# A typical non-singular FR3 seed in radians.
FRANKA_ARM_Q_INIT = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.65,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.0,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.5,
    "fr3_joint7": 0.75,
}

# Use exact EE frame from URDF link, no extra offset.
EE_LOCAL_OFFSET = np.zeros(3, dtype=np.float32)

# Gripper visuals at TCP (same orientation as last link / EE frame). Local +Z points back toward
# the wrist (-Z is approach). The last link mesh ends on the z=0 plane here; the disc sits on that
# face and extends along -Z (outward), not into the link along +Z.
GRIPPER_DISC_RADIUS = 0.030
GRIPPER_DISC_HALF_THICK = 0.0036
# Disc z in [-2*half_thick, 0]: inner face flush with link tip at TCP; outer face toward workspace.
GRIPPER_CLAW_SEP_Y = GRIPPER_DISC_RADIUS * 0.5
GRIPPER_CLAW_HALF_LEN_Z = 0.018
GRIPPER_CLAW_HALF_THICK_XY = 0.0032
# Draggable IK gizmo lies this far along local -Z past TCP; IK drives TCP at gizmo + GIZMO_TCP_OFFSET * EE +Z.
GIZMO_TCP_OFFSET = GRIPPER_CLAW_HALF_LEN_Z + GRIPPER_DISC_HALF_THICK

# Polyscope draggable IK target sphere mesh radius (m).
IK_ARM_TARGET_SPHERE_RADIUS = 0.00005

# Initial target in FR3 workspace (z-up convention).
DEFAULT_TARGET_POS = np.array([0.42, 0.0, 0.48], dtype=np.float32)

# Fallback joint limits (used only if metadata limits are unavailable).
JOINT_ANGLE_MIN = -np.pi
JOINT_ANGLE_MAX = np.pi

# IK (position-only)
IK_TOL_POS = 2e-4
IK_MAX_ITER = 120
IK_LAMBDA = 0.02
IK_N_TRIES = 4
IK_PERTURB = 0.4

# Anti-jitter around target (m): when EE is already close and target change is tiny,
# skip re-solving IK to avoid hopping between near-equivalent joint solutions.
IK_HOLD_ERR_ENTER = 0.004
IK_HOLD_ERR_EXIT = 0.006
IK_HOLD_TARGET_DELTA = 8e-4
IK_COST_TIE_TOL = 1.5e-3
IK_BENCH_SUCCESS_TOL = 4e-3
IK_BENCH_SAFE_RADIUS_RATIO = 0.9
IK_BENCH_MIN_RADIUS_RATIO = 0.2

# Motion / timing
ANIMATION_DT = 1.0 / 60.0
DEFAULT_SMOOTH_ALPHA = 0.22
IK_MIN_INTERVAL_S = 1.0 / 45.0

# World-to-visual: FK gives link frame at each joint; segment geometry is local.
R_IDENTITY = np.eye(3, dtype=np.float64)

# Env keys (centralized to avoid typo-prone string literals).
ENV_IK_ARM_VIS = "NOVAPHY_IK_ARM_VIS"
ENV_IK_ARM_DAE_APPLY_NODE_MATS = "NOVAPHY_IK_ARM_DAE_APPLY_NODE_MATS"
ENV_IK_ARM_DAE_FORCE_SCALE_ONLY = "NOVAPHY_IK_ARM_DAE_FORCE_SCALE_ONLY"
ENV_IK_ARM_DAE_AUTO_SCALE_ONLY = "NOVAPHY_IK_ARM_DAE_AUTO_SCALE_ONLY"
ENV_IK_ARM_DAE_ALLOW_FULL_NODE_TRANSFORM = "NOVAPHY_IK_ARM_DAE_ALLOW_FULL_NODE_TRANSFORM"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Internal benchmark tuning values with CLI-overridable defaults."""

    workspace_probe_samples: int = 256
    max_generate_floor: int = 4000
    max_generate_per_target: int = 100
    seed_perturb_std: float = 0.55


def parse_env_flag(name: str, default: bool = False) -> bool:
    """Parse common true/false env values in one place."""
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return bool(default)


__all__ = [
    "ANIMATION_DT",
    "BenchmarkConfig",
    "DEFAULT_SMOOTH_ALPHA",
    "DEFAULT_TARGET_POS",
    "EE_LOCAL_OFFSET",
    "ENV_IK_ARM_DAE_ALLOW_FULL_NODE_TRANSFORM",
    "ENV_IK_ARM_DAE_APPLY_NODE_MATS",
    "ENV_IK_ARM_DAE_AUTO_SCALE_ONLY",
    "ENV_IK_ARM_DAE_FORCE_SCALE_ONLY",
    "ENV_IK_ARM_VIS",
    "FRANKA_ARM_JOINT_NAMES",
    "FRANKA_ARM_Q_INIT",
    "FRANKA_EE_LINK_NAME",
    "FRANKA_SHOULDER_LINK_NAME",
    "FRANKA_URDF_PATH",
    "GIZMO_TCP_OFFSET",
    "IK_BENCH_MIN_RADIUS_RATIO",
    "IK_BENCH_SAFE_RADIUS_RATIO",
    "IK_ARM_TARGET_SPHERE_RADIUS",
    "IK_BENCH_SUCCESS_TOL",
    "IK_COST_TIE_TOL",
    "IK_HOLD_ERR_ENTER",
    "IK_HOLD_ERR_EXIT",
    "IK_HOLD_TARGET_DELTA",
    "IK_LAMBDA",
    "IK_MAX_ITER",
    "IK_MIN_INTERVAL_S",
    "IK_N_TRIES",
    "IK_PERTURB",
    "IK_TOL_POS",
    "JOINT_ANGLE_MAX",
    "JOINT_ANGLE_MIN",
    "R_IDENTITY",
    "parse_env_flag",
]
