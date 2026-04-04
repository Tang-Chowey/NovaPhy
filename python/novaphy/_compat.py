"""Backward-compatible wrappers for Phase-3 unified World.

``ArticulatedWorld`` and ``FluidWorld`` are thin Python wrappers around the
unified ``novaphy.World`` class.  They preserve the original API surface so
that existing tests and demos continue to work without modification.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np

# Imports from _core are deferred inside each class to avoid circular import
# at module load time.


class FluidWorld:
    """Backward-compatible wrapper: ``FluidWorld(model, fluid_blocks, ...)``

    Internally builds a unified ``World`` after injecting the given fluid
    blocks into ``model.fluid_blocks``.
    """

    def __init__(
        self,
        model,
        fluid_blocks: Optional[Sequence] = None,
        solver_settings=None,
        pbf_settings=None,
        boundary_extent: float = 1.0,
    ) -> None:
        from novaphy._core import (
            PBFSettings,
            SolverSettings,
            World,
            XPBDSolverSettings,
        )

        # Store original fluid_blocks to restore later and avoid mutating caller's model
        original_fluid_blocks = model.fluid_blocks
        try:
            if fluid_blocks:
                model.fluid_blocks = list(fluid_blocks)

            self._world: World = World(
                model,
                solver_settings=solver_settings or SolverSettings(),
                xpbd_settings=XPBDSolverSettings(),
                pbf_settings=pbf_settings or PBFSettings(),
                fluid_boundary_extent=boundary_extent,
            )
        finally:
            # Restore original fluid_blocks to avoid side effects on caller's model
            model.fluid_blocks = original_fluid_blocks

    # ---------- fluid helpers ----------

    @property
    def fluid_state(self):
        return self._world.state.fluid_state

    @property
    def num_particles(self) -> int:
        return self._world.state.fluid_state.num_particles

    @property
    def boundary_particles(self):
        return self._world.boundary_particles

    @property
    def num_boundary_particles(self) -> int:
        return len(self._world.boundary_particles)

    # ---------- simulation ----------

    def step(self, dt: float) -> None:
        self._world.step(dt)

    def set_gravity(self, gravity) -> None:
        self._world.set_gravity(gravity)

    # ---------- pass-through attributes ----------

    @property
    def gravity(self):
        return self._world.gravity

    @property
    def state(self):
        return self._world.state

    @property
    def contacts(self):
        return self._world.contacts

    @property
    def model(self):
        return self._world.model

    @property
    def performance_monitor(self):
        return self._world.performance_monitor

    def apply_force(self, body_index: int, force) -> None:
        self._world.apply_force(body_index, force)

    def apply_torque(self, body_index: int, torque) -> None:
        self._world.apply_torque(body_index, torque)


class ArticulatedWorld:
    """Backward-compatible wrapper: ``ArticulatedWorld(scene, xpbd_settings)``

    Internally builds a unified ``World`` with the articulation embedded in
    ``model.articulations``.

    Exposes name-based joint control helpers (``set_joint_positions``,
    ``set_target_positions``, ``set_default_drive_gains``, etc.) on top of the
    low-level ``Control`` / ``SimState`` API.
    """

    def __init__(self, scene, xpbd_settings=None) -> None:
        from novaphy._core import (
            Control,
            JointDrive,
            SolverSettings,
            World,
            XPBDSolverSettings,
        )

        self._scene = scene
        meta = scene.metadata

        # Build the controllable joint list in URDF-authored order (from meta.joints).
        # meta.dof_joint_names is in qd/topological order which may differ from URDF order.
        self._dof_joint_names: List[str] = []    # unique controllable joint names (URDF order)
        self._dof_q_starts: List[int] = []       # q-vector start index per joint
        self._dof_qd_indices: List[int] = []     # qd-vector index (first DOF) per joint
        self._dof_link_indices: List[int] = []   # articulation link/joint index per joint

        for jm in meta.joints:
            # Skip auto-generated root and fixed (zero-DOF) joints
            if jm.joint_name.startswith("__") or jm.num_qd == 0:
                continue
            # Skip collapsed joints (they have no q in the articulation)
            if jm.collapsed:
                continue
            self._dof_joint_names.append(jm.joint_name)
            self._dof_q_starts.append(jm.q_start)
            self._dof_qd_indices.append(jm.qd_start)
            self._dof_link_indices.append(jm.articulation_index)

        # Embed articulation into model
        scene.model.articulations = [scene.articulation]

        xpbd = xpbd_settings or XPBDSolverSettings()
        self._xpbd_settings = xpbd

        # Keep track of extra static shapes (e.g. ground planes added later).
        # Store the original shapes list so rebuilds always start from scratch.
        self._extra_shapes: list = []
        self._original_shapes: list = list(scene.model.shapes)

        self._world: World = self._build_world(scene.model, xpbd)

        # Restore initial articulation state
        if scene.initial_q is not None:
            self._world.state.set_q(0, scene.initial_q)
        if scene.initial_qd is not None:
            self._world.state.set_qd(0, scene.initial_qd)

        # Build Control with one JointDrive per LINK (indexed by link/joint index).
        # Non-controllable links (e.g. free root) keep Off drives by default.
        total_qd = scene.articulation.total_qd()
        n_links = scene.articulation.num_links()
        self._control: Control = Control()
        self._control.joint_forces = np.zeros(total_qd, dtype=np.float32)
        self._control.joint_drives = [JointDrive() for _ in range(n_links)]

    # ---------- internal helpers ----------

    def _build_world(self, model, xpbd_settings):
        from novaphy._core import SolverSettings, World

        # Always reset to the original shapes list to avoid duplication on rebuilds.
        model.shapes = list(self._original_shapes) + list(self._extra_shapes)

        return World(
            model,
            solver_settings=SolverSettings(),
            xpbd_settings=xpbd_settings,
        )

    def _rebuild_world(self) -> None:
        """Rebuild the World after model modifications (e.g. add_ground_plane)."""
        q = np.array(self._world.state.q[0], dtype=np.float32) if self._world.state.q else None
        qd = np.array(self._world.state.qd[0], dtype=np.float32) if self._world.state.qd else None

        scene = self._scene
        scene.model.articulations = [scene.articulation]
        self._world = self._build_world(scene.model, self._xpbd_settings)

        if q is not None:
            self._world.state.set_q(0, q)
        if qd is not None:
            self._world.state.set_qd(0, qd)

    # ---------- joint name API ----------

    @property
    def joint_names(self) -> List[str]:
        return list(self._dof_joint_names)

    @property
    def q(self) -> np.ndarray:
        q_list = self._world.state.q
        return np.array(q_list[0], dtype=np.float32) if q_list else np.zeros(0, dtype=np.float32)

    @property
    def qd(self) -> np.ndarray:
        qd_list = self._world.state.qd
        return np.array(qd_list[0], dtype=np.float32) if qd_list else np.zeros(0, dtype=np.float32)

    @property
    def joint_positions(self) -> Dict[str, float]:
        q = self.q
        return {
            name: float(q[self._dof_q_starts[i]])
            for i, name in enumerate(self._dof_joint_names)
        }

    def set_joint_positions(self, positions: Dict[str, float]) -> None:
        for name in positions:
            if name not in self._dof_joint_names:
                raise ValueError(f"Unknown joint name: '{name}'")
        q = self.q.copy()
        for name, val in positions.items():
            i = self._dof_joint_names.index(name)
            q[self._dof_q_starts[i]] = float(val)
        self._world.state.set_q(0, q)

    def set_default_drive_gains(self, stiffness: float, damping: float) -> None:
        from novaphy._core import JointDrive

        drives = list(self._control.joint_drives)
        for i, name in enumerate(self._dof_joint_names):
            link_idx = self._dof_link_indices[i]
            old = drives[link_idx]
            new_drive = JointDrive()
            new_drive.mode = old.mode
            new_drive.target_position = old.target_position
            new_drive.target_velocity = old.target_velocity
            new_drive.stiffness = stiffness
            new_drive.damping = damping
            new_drive.force_limit = old.force_limit
            drives[link_idx] = new_drive
        self._control.joint_drives = drives

    def set_target_positions(self, targets: Dict[str, float]) -> None:
        from novaphy._core import JointDrive, JointTargetMode

        for name in targets:
            if name not in self._dof_joint_names:
                raise ValueError(f"Unknown joint name: '{name}'")

        drives = list(self._control.joint_drives)
        for name, val in targets.items():
            i = self._dof_joint_names.index(name)
            link_idx = self._dof_link_indices[i]
            old = drives[link_idx]
            new_drive = JointDrive()
            new_drive.mode = JointTargetMode.TargetPosition
            new_drive.target_position = float(val)
            new_drive.stiffness = old.stiffness
            new_drive.damping = old.damping
            new_drive.force_limit = old.force_limit
            drives[link_idx] = new_drive
        self._control.joint_drives = drives

    # ---------- static world modifications ----------

    def add_ground_plane(self, normal, offset: float) -> None:
        from novaphy._core import CollisionShape

        plane = CollisionShape.make_plane(normal, float(offset))
        self._extra_shapes.append(plane)
        self._rebuild_world()

    # ---------- simulation ----------

    def step(self, dt: float) -> None:
        self._world.step_with_control(self._world.state, self._control, dt)

    def set_gravity(self, gravity) -> None:
        self._world.set_gravity(gravity)

    # ---------- pass-through attributes ----------

    @property
    def solver(self):
        return self._world.xpbd_solver

    @property
    def contacts(self):
        return self._world.contacts

    @property
    def state(self):
        return self._world.state

    @property
    def model(self):
        return self._world.model
