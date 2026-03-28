"""Core IK utilities for the FR3 IK demo."""

import os
import sys

import numpy as np

_ik_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_ik_pkg_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ik import ik_solver

from .config import (
    EE_LOCAL_OFFSET,
    GIZMO_TCP_OFFSET,
    IK_COST_TIE_TOL,
    IK_LAMBDA,
    IK_N_TRIES,
    IK_PERTURB,
    IK_TOL_POS,
)


def as_vec3(value, *, dtype, name: str = "value"):
    """Coerce ``value`` to a length-3 vector of the requested dtype (copy)."""
    arr = np.asarray(value, dtype=dtype).ravel()
    if arr.size < 3:
        raise ValueError(f"{name}: expected at least 3 elements, got {arr.size}")
    return arr[:3].copy()


def _as_vec3_f64(value, *, name: str = "value"):
    return as_vec3(value, dtype=np.float64, name=name)


def _as_vec3_f32(value, *, name: str = "value"):
    return as_vec3(value, dtype=np.float32, name=name)


def _vec3_changed(a, b, eps=1e-6):
    try:
        aa = _as_vec3_f64(a, name="a")
        bb = _as_vec3_f64(b, name="b")
    except ValueError:
        return True
    return bool(np.linalg.norm(aa - bb) > float(eps))


def _tcp_target_world_from_gizmo(art, q, gizmo_world_xyz, ee_link_index=-1):
    """World-space TCP goal for position IK given the draggable gizmo position."""
    R = ik_solver.get_ee_rotation(
        art, np.asarray(q, dtype=np.float32), ee_link_index=ee_link_index
    )
    g = _as_vec3_f64(gizmo_world_xyz, name="gizmo_world_xyz")
    return g + float(GIZMO_TCP_OFFSET) * R[:, 2]


def ee_world_xyz_m(art, q, *, ee_link_index: int) -> np.ndarray:
    """TCP / EE position in world frame (m), float64 length-3."""
    p = ik_solver.get_ee_position(
        art,
        np.asarray(q, dtype=np.float32),
        ee_link_index=int(ee_link_index),
        local_offset=EE_LOCAL_OFFSET,
    )
    return _as_vec3_f64(p, name="ee_world")


def tcp_position_error_m(art, q, target_xyz_world, *, ee_link_index: int) -> float:
    """Scalar position error ||EE(q) - target|| in meters (demo EE frame and offset)."""
    e = ee_world_xyz_m(art, q, ee_link_index=ee_link_index)
    t = _as_vec3_f64(target_xyz_world, name="target_world")
    return float(np.linalg.norm(e - t))


def _solve_ik_position_best_of_seeds(
    art,
    q0,
    target_position,
    joint_limits=None,
    local_offset=None,
    prefer_q=None,
    cost_tie_tol=1e-3,
    n_tries=4,
    perturb_scale=0.35,
    **solve_kwargs,
):
    """Run position IK from several perturbed seeds; pick the best EE distance.

    For each try, ``solve_ik`` runs from a clipped perturbation of ``q0``. The winning
    solution minimizes ‖EE(q) − target‖; if two costs are within ``cost_tie_tol``, the
    joint vector closer to ``prefer_q`` is kept (reduces visible discontinuities).
    """
    nq = art.total_q()
    ee_link_index = int(solve_kwargs.get("ee_link_index", -1))
    q0 = np.asarray(q0, dtype=np.float64).ravel()
    if q0.size != nq:
        raise ValueError("q0 length must match art.total_q()")

    if joint_limits is not None:
        q_lo = np.asarray(joint_limits[0], dtype=np.float64).ravel()
        q_hi = np.asarray(joint_limits[1], dtype=np.float64).ravel()
    else:
        q_lo = np.full(nq, -np.inf)
        q_hi = np.full(nq, np.inf)

    tgt = np.asarray(target_position, dtype=np.float64).ravel()
    dirs_base = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, -1.0, 1.0],
            [0.5, -0.7, 0.9, -0.4, 0.6, -0.3],
        ],
        dtype=np.float64,
    )

    best_q = q0.astype(np.float32)
    best_cost = np.inf
    best_conv = False
    prefer_q = (
        np.asarray(prefer_q, dtype=np.float64).ravel().copy()
        if prefer_q is not None
        else np.asarray(q0, dtype=np.float64).ravel().copy()
    )
    tie_tol = float(max(0.0, cost_tie_tol))
    n_tries = int(max(1, n_tries))
    for i in range(n_tries):
        row = dirs_base[i % dirs_base.shape[0]]
        if nq > row.size:
            rep = int(np.ceil(float(nq) / float(row.size)))
            row = np.tile(row, rep)[:nq]
        else:
            row = row[:nq]
        q_seed = np.clip(q0 + perturb_scale * row, q_lo, q_hi).astype(np.float32)
        q_sol, conv = ik_solver.solve_ik(
            art,
            q_seed,
            tgt,
            joint_limits=joint_limits,
            local_offset=local_offset,
            **solve_kwargs,
        )
        p = ik_solver.get_ee_position(
            art, q_sol, ee_link_index=ee_link_index, local_offset=local_offset
        )
        cost = float(np.linalg.norm(np.asarray(p, dtype=np.float64) - tgt))
        if cost < (best_cost - tie_tol):
            best_cost = cost
            best_q = q_sol
            best_conv = bool(conv)
        elif abs(cost - best_cost) <= tie_tol:
            curr_d = float(
                np.linalg.norm(np.asarray(q_sol, dtype=np.float64).ravel() - prefer_q)
            )
            best_d = float(
                np.linalg.norm(np.asarray(best_q, dtype=np.float64).ravel() - prefer_q)
            )
            if curr_d < best_d:
                best_q = q_sol
                best_conv = bool(conv)
    return best_q, best_conv


def _build_ik_solve_kwargs(ee_link_index: int, max_iter: int):
    return {
        "ee_link_index": int(ee_link_index),
        "local_offset": EE_LOCAL_OFFSET,
        "cost_tie_tol": IK_COST_TIE_TOL,
        "n_tries": IK_N_TRIES,
        "perturb_scale": IK_PERTURB,
        "tol": IK_TOL_POS,
        "max_iter": int(max_iter),
        "lambda_damp": IK_LAMBDA,
        "step_size": 1.0,
    }


def _solve_position_ik_goal(
    art,
    q_seed,
    target_position,
    joint_limits,
    ee_link_index: int,
    max_iter: int,
    prefer_q=None,
):
    solve_kwargs = _build_ik_solve_kwargs(ee_link_index=ee_link_index, max_iter=max_iter)
    return _solve_ik_position_best_of_seeds(
        art,
        q_seed,
        target_position,
        joint_limits=joint_limits,
        prefer_q=prefer_q,
        **solve_kwargs,
    )


__all__ = ["as_vec3", "ee_world_xyz_m", "tcp_position_error_m"]
