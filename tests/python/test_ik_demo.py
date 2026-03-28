"""Tests for the IK arm demo and IK solver module."""

import sys
from pathlib import Path

import numpy as np
import novaphy


def _ensure_repo_paths():
    root = Path(__file__).resolve().parents[2]
    demos = root / "demos"
    for p in (str(root), str(demos)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_ik_solver():
    _ensure_repo_paths()
    from ik import ik_solver

    return ik_solver


def _franka_setup():
    _ensure_repo_paths()
    from ik_arm_demo.scene_setup import build_franka_ik_setup

    return build_franka_ik_setup()


def test_ik_solver_two_link():
    """IK solver converges for a simple 2R planar arm."""
    ik = _load_ik_solver()
    art = novaphy.Articulation()
    j0 = novaphy.Joint()
    j0.type = novaphy.JointType.Revolute
    j0.axis = np.array([0, 0, 1], dtype=np.float32)
    j0.parent = -1
    j0.parent_to_joint = novaphy.Transform.identity()
    j1 = novaphy.Joint()
    j1.type = novaphy.JointType.Revolute
    j1.axis = np.array([0, 0, 1], dtype=np.float32)
    j1.parent = 0
    j1.parent_to_joint = novaphy.Transform.from_translation(
        np.array([0, -0.5, 0], dtype=np.float32)
    )
    art.joints = [j0, j1]
    b0 = novaphy.RigidBody()
    b0.mass = 1.0
    b0.com = np.array([0, -0.25, 0], dtype=np.float32)
    b0.inertia = np.eye(3, dtype=np.float32) * 0.01
    b1 = novaphy.RigidBody()
    b1.mass = 1.0
    b1.com = np.array([0, -0.25, 0], dtype=np.float32)
    b1.inertia = np.eye(3, dtype=np.float32) * 0.01
    art.bodies = [b0, b1]
    art.build_spatial_inertias()

    q0 = np.zeros(2, dtype=np.float32)
    target = np.array([0.35, -0.35, 0.0])
    q_out, converged = ik.solve_ik(art, q0, target, max_iter=80)
    p = ik.get_ee_position(art, q_out)
    err = np.linalg.norm(np.asarray(p, dtype=np.float64) - target)
    assert err < 0.03, "IK should converge to within 3 cm"
    assert art.total_q() == 2
    assert q_out.shape == (2,)


def test_ik_demo_headless():
    """FR3 demo runs headless with default target and reaches within tolerance."""
    _ensure_repo_paths()
    from ik_arm_demo.runtime import _run_headless_ik_once

    q, converged, err = _run_headless_ik_once(max_iter=150)
    assert q.ndim == 1 and q.size > 0
    assert err < 0.12, "Headless IK should reach default target within 12 cm"


def test_solve_ik_pose_best_of_seeds_runs():
    ik = _load_ik_solver()
    _ensure_repo_paths()
    from ik_arm_demo.config import EE_LOCAL_OFFSET

    setup = _franka_setup()
    art = setup["art"]
    nq = art.total_q()
    q0 = np.asarray(setup["q_init"], dtype=np.float32).ravel()
    cy, sy = np.cos(0.1), np.sin(0.1)
    R = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    p = np.array([0.38, 0.2, 0.15], dtype=np.float64)
    ee_idx = int(setup["ee_link_index"])
    lim = setup["joint_limits"]
    q_sol, conv = ik.solve_ik_pose_best_of_seeds(
        art,
        q0,
        p,
        R,
        joint_limits=lim,
        ee_link_index=ee_idx,
        local_offset=EE_LOCAL_OFFSET,
        n_tries=3,
        max_iter=100,
        tol_pos=5e-4,
        tol_orient=5e-2,
    )
    assert q_sol.shape == (nq,)
    p_ee = ik.get_ee_position(art, q_sol, ee_link_index=ee_idx, local_offset=EE_LOCAL_OFFSET)
    assert np.linalg.norm(np.asarray(p_ee, dtype=np.float64) - p) < 0.15
