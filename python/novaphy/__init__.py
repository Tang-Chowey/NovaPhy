"""Python package entrypoint for NovaPhy.

Exports pybind11-backed physics engine APIs, including rigid-body simulation,
collision detection, articulated-body dynamics, and math/spatial utilities.
All physical quantities follow SI units (MKS) unless otherwise noted.
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

def _add_dll_directories():
    # CUDA runtime
    if _sys.platform == "win32":
        _cuda_path = _os.environ.get("CUDA_PATH", "")
        if _cuda_path:
            _cuda_bin = _Path(_cuda_path) / "bin"
            if _cuda_bin.is_dir():
                _os.add_dll_directory(str(_cuda_bin))

_add_dll_directories()

from novaphy._core import (
    version,
    has_ipc,
    # Math types
    Transform,
    SpatialTransform,
    # Math functions
    skew,
    spatial_cross_motion,
    spatial_cross_force,
    spatial_inertia_matrix,
    deg2rad,
    rad2deg,
    # Device / Control
    DeviceType,
    Device,
    JointTargetMode,
    JointDrive,
    Control,
    # Core types
    ShapeType,
    RigidBody,
    CollisionShape,
    AABB,
    ContactPoint,
    Site,
    # Collision
    BroadPhasePair,
    SweepAndPrune,
    collide_shapes,
    PerformanceMonitor,
    PerformanceMetric,
    PerformancePhaseStat,
    # Simulation
    ModelBuilder,
    Model,
    CollisionFilterPair,
    SolverSettings,
    SolverBase,
    SolverSequentialImpulse,
    SimState,
    World,
    # Articulated bodies
    JointType,
    Joint,
    Articulation,
    ArticulatedSolver,
    XPBDSolverSettings,
    XPBDStepStats,
    XPBDSolver,
    # Featherstone algorithms
    forward_kinematics,
    inverse_dynamics,
    mass_matrix_crba,
    forward_dynamics,
    # Multibody solver settings (for World constructor)
    MultiBodySolverSettings,
    forward_link_velocities,
    # Fluid simulation
    SPHKernels,
    FluidMaterial,
    FluidBlockDef,
    ParticleState,
    SpatialHashGrid,
    generate_fluid_block,
    PBFSettings,
    PBFSolver,
    BoundaryParticle,
    sample_model_boundaries,
    UrdfGeometryType,
    UrdfGeometry,
    UrdfVisual,
    UrdfCollision,
    UrdfInertial,
    UrdfLink,
    UrdfJoint,
    UrdfModelData,
    UrdfImportOptions,
    SceneLinkMetadata,
    SceneJointMetadata,
    SceneShapeMetadata,
    SceneBuildMetadata,
    UsdAnimationTrack,
    UsdPrim,
    UsdStageData,
    SceneBuildResult,
    FeatureCheckItem,
    FeatureCheckReport,
    RecordedKeyframe,
    RecordedCollisionEvent,
    RecordedConstraintReaction,
    UrdfParser,
    OpenUsdImporter,
    SceneBuilderEngine,
    SimulationExporter,
    FeatureCompletenessChecker,
    batch_transform_vertices,
    # VBD/AVBD
    VBDConfig,
    VBDWorld,
    VbdBackend,
)

# Optional IPC support (requires CUDA + libuipc)
try:
    from novaphy._core import IPCConfig, IPCWorld
except ImportError:
    pass

# Phase-3 backward-compatible wrappers
from novaphy._compat import ArticulatedWorld, FluidWorld  # noqa: E402

__version__ = version()

__all__ = [
    "version",
    "has_ipc",
    "Transform",
    "SpatialTransform",
    "skew",
    "spatial_cross_motion",
    "spatial_cross_force",
    "spatial_inertia_matrix",
    "deg2rad",
    "rad2deg",
    "DeviceType",
    "Device",
    "JointTargetMode",
    "JointDrive",
    "Control",
    "ShapeType",
    "RigidBody",
    "CollisionShape",
    "AABB",
    "ContactPoint",
    "Site",
    "BroadPhasePair",
    "SweepAndPrune",
    "collide_shapes",
    "PerformanceMonitor",
    "PerformanceMetric",
    "PerformancePhaseStat",
    "ModelBuilder",
    "Model",
    "CollisionFilterPair",
    "SolverSettings",
    "SolverBase",
    "SolverSequentialImpulse",
    "SimState",
    "World",
    "JointType",
    "Joint",
    "Articulation",
    "ArticulatedSolver",
    "XPBDSolverSettings",
    "XPBDStepStats",
    "XPBDSolver",
    "forward_kinematics",
    "inverse_dynamics",
    "mass_matrix_crba",
    "forward_dynamics",
    "MultiBodySolverSettings",
    "forward_link_velocities",
    "SPHKernels",
    "FluidMaterial",
    "FluidBlockDef",
    "ParticleState",
    "SpatialHashGrid",
    "generate_fluid_block",
    "PBFSettings",
    "PBFSolver",
    "BoundaryParticle",
    "sample_model_boundaries",
    "UrdfGeometryType",
    "UrdfGeometry",
    "UrdfVisual",
    "UrdfCollision",
    "UrdfInertial",
    "UrdfLink",
    "UrdfJoint",
    "UrdfModelData",
    "UrdfImportOptions",
    "SceneLinkMetadata",
    "SceneJointMetadata",
    "SceneShapeMetadata",
    "SceneBuildMetadata",
    "UsdAnimationTrack",
    "UsdPrim",
    "UsdStageData",
    "SceneBuildResult",
    "FeatureCheckItem",
    "FeatureCheckReport",
    "RecordedKeyframe",
    "RecordedCollisionEvent",
    "RecordedConstraintReaction",
    "UrdfParser",
    "OpenUsdImporter",
    "SceneBuilderEngine",
    "SimulationExporter",
    "FeatureCompletenessChecker",
    "batch_transform_vertices",
    # VBD/AVBD
    "VBDConfig",
    "VBDWorld",
    "VbdBackend",
    # Phase-3 compat wrappers
    "ArticulatedWorld",
    "FluidWorld",
    "sensors",
]

# Sensor framework (Python module)
from . import sensors  # noqa: E402

# Conditionally export IPC symbols
if has_ipc():
    __all__ += ["IPCConfig", "IPCWorld"]
