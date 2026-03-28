"""Inverse kinematics solver for NovaPhy Articulation.

Supports position-only IK and full pose IK (position + orientation).
Uses damped least-squares with Levenberg-Marquardt adaptive damping and
angular-velocity (geometric) Jacobian for orientation. Finite-difference
Jacobian computation supports Revolute and Slide joints. No UI dependencies.

Key design choices borrowed from newton/newton/_src/sim/ik/:
  - World-frame orientation error via quaternion (robust at 180 deg)
  - Geometric (angular velocity) Jacobian for orientation rows
  - Levenberg-Marquardt adaptive damping (accept/reject with lambda update)

For a 6 x nv pose Jacobian, vertically stack ``compute_jacobian_position`` and
``compute_jacobian_angular_velocity`` (same convention as ``solve_ik_pose``).
"""

import numpy as np
import novaphy
from novaphy.viz import quat_to_rotation_matrix as _quat_xyzw_to_rotation_matrix


def _quat_to_rotation_matrix(quat_xyzw):
    """(4,) xyzw quaternion -> (3,3) rotation matrix (delegates to novaphy.viz)."""
    return _quat_xyzw_to_rotation_matrix(np.asarray(quat_xyzw, dtype=np.float32))


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw convention, matching NovaPhy)
# ---------------------------------------------------------------------------


def _mat_to_quat(R):
    """Convert 3x3 rotation matrix to unit quaternion [x, y, z, w].

    Uses Shepperd's method for numerical stability across all rotations.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_conj(q):
    """Quaternion conjugate (= inverse for unit quaternions). [x,y,z,w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul(q1, q2):
    """Hamilton product of two [x,y,z,w] quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _rotation_to_axis_angle(R):
    """Extract axis * angle (3-vector) from a 3x3 rotation matrix.

    Robust near angle = 0 and angle = pi (uses symmetric-part
    extraction when sin(angle) is near zero).
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    sin_angle = np.sin(angle)
    if np.abs(sin_angle) > 1e-6:
        axis = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=np.float64,
        )
        axis = axis / (2.0 * sin_angle)
    else:
        # Near pi: R ≈ 2*n*n^T - I, so (R+I)/2 ≈ n*n^T
        S = (R + np.eye(3, dtype=np.float64)) / 2.0
        norms = np.array([np.linalg.norm(S[:, k]) for k in range(3)])
        k = int(np.argmax(norms))
        if norms[k] < 1e-10:
            return np.zeros(3, dtype=np.float64)
        axis = S[:, k] / norms[k]
    return axis * angle


# ---------------------------------------------------------------------------
# Forward kinematics helpers
# ---------------------------------------------------------------------------


def get_ee_position(art, q, ee_link_index=-1, local_offset=None):
    """End-effector position in world frame.

    Args:
        art: NovaPhy Articulation.
        q: Generalized position vector, shape (art.total_q(),).
        ee_link_index: Link index for end-effector; -1 means last link.
        local_offset: Optional (3,) offset in end-effector link frame.

    Returns:
        (3,) float32 array, world position.
    """
    transforms = novaphy.forward_kinematics(art, np.asarray(q, dtype=np.float32))
    idx = ee_link_index if ee_link_index >= 0 else (len(transforms) + ee_link_index)
    t = transforms[idx]
    pos = np.array(t.position, dtype=np.float32)
    if local_offset is not None:
        offset = np.asarray(local_offset, dtype=np.float32)
        if offset.size == 3 and np.any(np.abs(offset) > 1e-9):
            R = _quat_to_rotation_matrix(t.rotation)
            pos = pos + (R @ offset)
    return pos


def get_ee_rotation(art, q, ee_link_index=-1):
    """End-effector orientation in world frame as (3, 3) rotation matrix."""
    transforms = novaphy.forward_kinematics(art, np.asarray(q, dtype=np.float32))
    idx = ee_link_index if ee_link_index >= 0 else (len(transforms) + ee_link_index)
    t = transforms[idx]
    R = _quat_to_rotation_matrix(t.rotation)
    return np.asarray(R, dtype=np.float64)


# ---------------------------------------------------------------------------
# Orientation error
# ---------------------------------------------------------------------------


def rotation_error_axis_angle(R_current, R_target):
    """World-frame orientation error as axis-angle (3-vector).

    Computes the rotation needed in world frame to take R_current to R_target:
        R_target = exp([result]_x) @ R_current

    Uses quaternion algebra, which is robust at 180 degrees (unlike the
    matrix-log approach that divides by sin(angle)).

    Reference: Newton's IKObjectiveRotation uses the same quaternion-based
    error with atan2 for the angle, avoiding the sin(angle) singularity.
    """
    R_current = np.asarray(R_current, dtype=np.float64)
    R_target = np.asarray(R_target, dtype=np.float64)

    q_cur = _mat_to_quat(R_current)
    q_tgt = _mat_to_quat(R_target)
    q_err = _quat_mul(q_tgt, _quat_conj(q_cur))

    # Canonicalize to shorter arc when NOT near 180 degrees.
    # At exactly 180 degrees (w ≈ 0) the natural quaternion sign from
    # the product already encodes the correct rotation direction.
    if q_err[3] < 0 and np.abs(q_err[3]) > 1e-3:
        q_err = -q_err

    v = q_err[:3]
    v_norm = np.linalg.norm(v)
    angle = 2.0 * np.arctan2(v_norm, q_err[3])

    if v_norm < 1e-10:
        return np.zeros(3, dtype=np.float64)

    axis = v / v_norm
    return axis * angle


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------


def compute_jacobian_position(art, q, ee_link_index=-1, local_offset=None, eps=1e-6):
    """Position Jacobian (3 x nv) via forward finite differences.

    J such that d(ee_pos)/dt ≈ J @ qd.
    """
    nv = art.total_qd()
    nq = art.total_q()
    q = np.asarray(q, dtype=np.float64).ravel()
    if q.size != nq:
        raise ValueError("q length must match art.total_q()")

    p0 = get_ee_position(art, q.astype(np.float32), ee_link_index, local_offset)
    p0 = np.asarray(p0, dtype=np.float64).ravel()

    J = np.zeros((3, nv), dtype=np.float64)
    q_work = q.copy()
    col = 0
    for link in range(art.num_links()):
        joint = art.joints[link]
        nqd_i = joint.num_qd()
        qi = art.q_start(link)
        for k in range(nqd_i):
            if joint.type == novaphy.JointType.Revolute or joint.type == novaphy.JointType.Slide:
                q_work[qi + k] = q[qi + k] + eps
                p_plus = get_ee_position(art, q_work.astype(np.float32), ee_link_index, local_offset)
                q_work[qi + k] = q[qi + k]
                J[:, col] = (np.asarray(p_plus, dtype=np.float64).ravel() - p0) / eps
            col += 1
    return J


def compute_jacobian_angular_velocity(art, q, ee_link_index=-1, eps=1e-6):
    """Angular-velocity (geometric) Jacobian (3 x nv) in world frame.

    Each column j gives the world-frame angular velocity of the end-effector
    caused by a unit velocity of joint j:
        J_omega[:, j] = axis_angle(R(q + eps*e_j) @ R(q)^T) / eps

    This is the correct Jacobian for orientation-constrained IK, as opposed
    to d(axis_angle_error)/dq which conflates the task derivative with the
    error derivative and fails at large errors.

    Reference: Newton's analytic Jacobian computes the angular part of the
    motion sub-space (omega column of the spatial velocity twist), which is
    the same quantity.
    """
    nv = art.total_qd()
    nq = art.total_q()
    q = np.asarray(q, dtype=np.float64).ravel()
    if q.size != nq:
        raise ValueError("q length must match art.total_q()")

    R0 = get_ee_rotation(art, q, ee_link_index)

    J = np.zeros((3, nv), dtype=np.float64)
    q_work = q.copy()
    col = 0
    for link in range(art.num_links()):
        joint = art.joints[link]
        nqd_i = joint.num_qd()
        qi = art.q_start(link)
        for k in range(nqd_i):
            if joint.type == novaphy.JointType.Revolute or joint.type == novaphy.JointType.Slide:
                q_work[qi + k] = q[qi + k] + eps
                R_plus = get_ee_rotation(art, q_work.astype(np.float32), ee_link_index)
                q_work[qi + k] = q[qi + k]
                R_delta = np.asarray(R_plus, dtype=np.float64) @ R0.T
                J[:, col] = _rotation_to_axis_angle(R_delta) / eps
            col += 1
    return J


def _pose_cost(art, q, target_position, target_orientation, ee_link_index, local_offset, weight_orientation):
    p = np.asarray(get_ee_position(art, q.astype(np.float32), ee_link_index, local_offset), dtype=np.float64)
    R = get_ee_rotation(art, q, ee_link_index)
    e_p = target_position - p
    e_o = rotation_error_axis_angle(R, target_orientation)
    w = float(weight_orientation)
    return float(np.dot(e_p, e_p) + w * w * np.dot(e_o, e_o))


# ---------------------------------------------------------------------------
# IK solvers
# ---------------------------------------------------------------------------


def solve_ik(
    art,
    q0,
    target_ee,
    joint_limits=None,
    ee_link_index=-1,
    local_offset=None,
    tol=1e-4,
    max_iter=100,
    lambda_damp=0.01,
    step_size=1.0,
):
    """Position-only IK with Levenberg-Marquardt adaptive damping.

    Uses normal-equation form (J^T J + lambda I) dq = J^T e with J of shape (3, nv),
    nv = ``art.total_qd()``. This matches ``art.total_q()`` for typical fixed-base
    serial arms; models with nq != nv are not supported here.
    """
    target_ee = np.asarray(target_ee, dtype=np.float64).ravel()
    if target_ee.size != 3:
        raise ValueError("target_ee must be length 3")

    q = np.asarray(q0, dtype=np.float64).ravel().copy()
    nq = art.total_q()
    if q.size != nq:
        raise ValueError("q0 length must match art.total_q()")

    if joint_limits is not None:
        q_min = np.asarray(joint_limits[0], dtype=np.float64).ravel()
        q_max = np.asarray(joint_limits[1], dtype=np.float64).ravel()
    else:
        q_min = np.full(nq, -np.inf, dtype=np.float64)
        q_max = np.full(nq, np.inf, dtype=np.float64)

    lam = float(lambda_damp)

    for _ in range(max_iter):
        p = get_ee_position(art, q.astype(np.float32), ee_link_index, local_offset)
        p = np.asarray(p, dtype=np.float64).ravel()
        e = target_ee - p
        err = np.linalg.norm(e)
        if err < tol:
            return q.astype(np.float32), True

        J = compute_jacobian_position(art, q, ee_link_index, local_offset)
        A = J.T @ J + lam * np.eye(nq, dtype=np.float64)
        b = J.T @ e
        try:
            dq = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return q.astype(np.float32), False

        q_new = np.clip(q + step_size * dq, q_min, q_max)

        p_new = get_ee_position(art, q_new.astype(np.float32), ee_link_index, local_offset)
        err_new = np.linalg.norm(target_ee - np.asarray(p_new, dtype=np.float64))
        if err_new < err:
            q = q_new
            lam = max(lam / 2.0, 1e-7)
        else:
            lam = min(lam * 3.0, 1e4)

    return q.astype(np.float32), False


def solve_ik_pose(
    art,
    q0,
    target_position,
    target_orientation,
    joint_limits=None,
    ee_link_index=-1,
    local_offset=None,
    tol_pos=1e-4,
    tol_orient=1e-2,
    max_iter=150,
    lambda_damp=0.01,
    step_size=1.0,
    weight_orientation=1.0,
):
    """Pose IK (position + orientation) with LM adaptive damping.

    Uses the angular-velocity Jacobian for orientation rows (not
    d(axis_angle_error)/dq), making convergence robust for large
    orientation errors including near-180-degree cases.

    The normal-equation form (J^T J + lambda I) dq = J^T e provides
    an n x n system (n = number of DOFs), which is smaller and better
    conditioned than the m x m DLS form when m > n (overconstrained).

    Reference: Newton's IKOptimizerLM uses the same approach with
    tiled Cholesky factorization of J^T J + lambda I.
    """
    target_position = np.asarray(target_position, dtype=np.float64).ravel()
    target_orientation = np.asarray(target_orientation, dtype=np.float64)
    if target_position.size != 3:
        raise ValueError("target_position must be length 3")
    if target_orientation.shape != (3, 3):
        raise ValueError("target_orientation must be (3, 3)")

    q = np.asarray(q0, dtype=np.float64).ravel().copy()
    nq = art.total_q()
    if q.size != nq:
        raise ValueError("q0 length must match art.total_q()")

    if joint_limits is not None:
        q_min = np.asarray(joint_limits[0], dtype=np.float64).ravel()
        q_max = np.asarray(joint_limits[1], dtype=np.float64).ravel()
    else:
        q_min = np.full(nq, -np.inf, dtype=np.float64)
        q_max = np.full(nq, np.inf, dtype=np.float64)

    lam = float(lambda_damp)
    w = float(weight_orientation)

    for _ in range(max_iter):
        p = get_ee_position(art, q.astype(np.float32), ee_link_index, local_offset)
        p = np.asarray(p, dtype=np.float64).ravel()
        R = get_ee_rotation(art, q, ee_link_index)

        e_p = target_position - p
        e_o = rotation_error_axis_angle(R, target_orientation)
        err_pos = np.linalg.norm(e_p)
        err_orient = np.linalg.norm(e_o)
        if err_pos < tol_pos and err_orient < tol_orient:
            return q.astype(np.float32), True

        J_pos = compute_jacobian_position(art, q, ee_link_index, local_offset)
        J_omega = compute_jacobian_angular_velocity(art, q, ee_link_index)

        J_w = np.vstack([J_pos, w * J_omega])
        e = np.concatenate([e_p, w * e_o])

        A = J_w.T @ J_w + lam * np.eye(nq, dtype=np.float64)
        b = J_w.T @ e
        try:
            dq = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return q.astype(np.float32), False

        q_new = np.clip(q + step_size * dq, q_min, q_max)

        # Adaptive damping: accept step if cost decreases
        p_new = get_ee_position(art, q_new.astype(np.float32), ee_link_index, local_offset)
        R_new = get_ee_rotation(art, q_new, ee_link_index)
        e_p_new = target_position - np.asarray(p_new, dtype=np.float64)
        e_o_new = rotation_error_axis_angle(R_new, target_orientation)
        e_new = np.concatenate([e_p_new, w * e_o_new])

        cost_old = np.dot(e, e)
        cost_new = np.dot(e_new, e_new)

        if cost_new < cost_old:
            q = q_new
            lam = max(lam / 2.0, 1e-7)
        else:
            lam = min(lam * 3.0, 1e4)

    return q.astype(np.float32), False


def solve_ik_pose_best_of_seeds(
    art,
    q0,
    target_position,
    target_orientation,
    joint_limits=None,
    ee_link_index=-1,
    local_offset=None,
    n_tries=4,
    perturb_scale=0.35,
    **solve_kwargs,
):
    """Run :func:`solve_ik_pose` from several deterministic seeds; return lowest-cost result.

    Useful for 6R arms when a single LM run stalls in a poor local minimum.
    """
    nq = art.total_q()
    q0 = np.asarray(q0, dtype=np.float64).ravel()
    if q0.size != nq:
        raise ValueError("q0 length must match art.total_q()")

    if joint_limits is not None:
        q_lo = np.asarray(joint_limits[0], dtype=np.float64).ravel()
        q_hi = np.asarray(joint_limits[1], dtype=np.float64).ravel()
    else:
        q_lo = np.full(nq, -np.inf)
        q_hi = np.full(nq, np.inf)

    w = float(solve_kwargs.get("weight_orientation", 1.0))
    best_q = q0.astype(np.float32)
    best_cost = np.inf
    best_conv = False

    # Deterministic "random" offsets (no numpy RNG state)
    dirs = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1, 1],
            [1, 1, -1, -1, 0, 0],
            [-1, 0, 1, 0, -1, 1],
            [0.5, -0.7, 0.9, -0.4, 0.6, -0.3],
        ],
        dtype=np.float64,
    )

    n_tries = int(max(1, n_tries))
    for i in range(n_tries):
        row = dirs[i % dirs.shape[0]]
        if nq > row.size:
            rep = int(np.ceil(float(nq) / float(row.size)))
            row = np.tile(row, rep)[:nq]
        else:
            row = row[:nq]
        q_seed = np.clip(q0 + perturb_scale * row, q_lo, q_hi).astype(np.float32)
        q_sol, conv = solve_ik_pose(
            art,
            q_seed,
            target_position,
            target_orientation,
            joint_limits=joint_limits,
            ee_link_index=ee_link_index,
            local_offset=local_offset,
            **solve_kwargs,
        )
        cost = _pose_cost(
            art,
            q_sol,
            np.asarray(target_position, dtype=np.float64),
            np.asarray(target_orientation, dtype=np.float64),
            ee_link_index,
            local_offset,
            w,
        )
        if cost < best_cost:
            best_cost = cost
            best_q = q_sol
            best_conv = bool(conv)

    return best_q, best_conv


__all__ = [
    "compute_jacobian_angular_velocity",
    "compute_jacobian_position",
    "get_ee_position",
    "get_ee_rotation",
    "rotation_error_axis_angle",
    "solve_ik",
    "solve_ik_pose",
    "solve_ik_pose_best_of_seeds",
]

