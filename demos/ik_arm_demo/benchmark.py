"""Benchmark module for FR3 IK demo."""

import time

import numpy as np
import novaphy

from ik import ik_solver

from .config import (
    EE_LOCAL_OFFSET,
    FRANKA_SHOULDER_LINK_NAME,
    IK_BENCH_MIN_RADIUS_RATIO,
    IK_BENCH_SAFE_RADIUS_RATIO,
    IK_BENCH_SUCCESS_TOL,
    IK_MAX_ITER,
    BenchmarkConfig,
)
from .ik_logic import _solve_position_ik_goal, tcp_position_error_m
from .scene_setup import build_franka_ik_setup, find_articulation_link_index_by_name


def _format_stat_line(name, values, scale=1.0, unit=""):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return "%s: n/a" % name
    v = arr * float(scale)
    return (
        "%s: mean=%.4f%s, median=%.4f%s, p95=%.4f%s, p99=%.4f%s, max=%.4f%s"
        % (
            name,
            float(np.mean(v)),
            unit,
            float(np.median(v)),
            unit,
            float(np.percentile(v, 95)),
            unit,
            float(np.percentile(v, 99)),
            unit,
            float(np.max(v)),
            unit,
        )
    )


def run_benchmark(
    n_targets=300,
    n_seeds=4,
    max_iter=None,
    success_tol=IK_BENCH_SUCCESS_TOL,
    rng_seed=0,
    safe_radius_ratio=IK_BENCH_SAFE_RADIUS_RATIO,
    min_radius_ratio=IK_BENCH_MIN_RADIUS_RATIO,
    benchmark_cfg: BenchmarkConfig | None = None,
):
    """Offline IK benchmark: many reachable targets + many initial seeds.

    Estimates reachable arm reach by probing random joint poses (max EE distance from
    shoulder). Targets are rejection-sampled so EE positions lie in an annulus
    [r_min, r_safe]. Each target is solved from several perturbed seeds; errors and
    timings are aggregated for reporting.
    """
    cfg = benchmark_cfg if benchmark_cfg is not None else BenchmarkConfig()
    setup = build_franka_ik_setup()
    art = setup["art"]
    ee_link_index = int(setup["ee_link_index"])
    metadata = setup["scene"].metadata
    nq = art.total_q()
    q_ref = np.asarray(setup["q_init"], dtype=np.float32).ravel().copy()
    joint_limits = setup["joint_limits"]
    q_lo, q_hi = (
        np.asarray(joint_limits[0], dtype=np.float64),
        np.asarray(joint_limits[1], dtype=np.float64),
    )
    iters = IK_MAX_ITER if max_iter is None else int(max_iter)
    n_targets_requested = int(max(1, n_targets))
    n_seeds = int(max(1, n_seeds))
    success_tol = float(max(0.0, success_tol))
    safe_radius_ratio = float(np.clip(safe_radius_ratio, 0.1, 1.0))
    min_radius_ratio = float(np.clip(min_radius_ratio, 0.0, safe_radius_ratio))
    rng = np.random.default_rng(int(rng_seed))

    shoulder_idx = find_articulation_link_index_by_name(metadata, FRANKA_SHOULDER_LINK_NAME)
    if shoulder_idx < 0:
        shoulder_idx = 0
    tf_ref = novaphy.forward_kinematics(art, q_ref)
    shoulder_origin = np.asarray(tf_ref[shoulder_idx].position, dtype=np.float64).ravel()

    r_probe = []
    for _ in range(int(max(1, cfg.workspace_probe_samples))):
        q_probe = rng.uniform(q_lo, q_hi, size=nq).astype(np.float32)
        p_probe = ik_solver.get_ee_position(
            art,
            q_probe,
            ee_link_index=ee_link_index,
            local_offset=EE_LOCAL_OFFSET,
        )
        p_probe = np.asarray(p_probe, dtype=np.float64).ravel()
        r_probe.append(float(np.linalg.norm(p_probe - shoulder_origin)))
    r_max = float(max(r_probe) if r_probe else 1.0)
    r_safe = safe_radius_ratio * r_max
    r_min = min_radius_ratio * r_max
    if r_min >= r_safe:
        r_min = 0.0

    targets = []
    generated = 0
    max_generate = max(
        int(max(1, cfg.max_generate_floor)),
        int(max(1, cfg.max_generate_per_target)) * n_targets_requested,
    )
    while len(targets) < n_targets_requested and generated < max_generate:
        q_rand = rng.uniform(q_lo, q_hi, size=nq).astype(np.float32)
        p_i = ik_solver.get_ee_position(
            art,
            q_rand,
            ee_link_index=ee_link_index,
            local_offset=EE_LOCAL_OFFSET,
        )
        p_i = np.asarray(p_i, dtype=np.float64).ravel()
        generated += 1
        r_i = float(np.linalg.norm(p_i - shoulder_origin))
        if (r_i <= r_safe) and (r_i >= r_min):
            targets.append(p_i)

    if len(targets) < n_targets_requested:
        print(
            "warning: only sampled %d targets inside [r_min, r_safe] (requested %d). "
            "Increase max attempts, decrease min radius ratio, or increase safe radius ratio."
            % (len(targets), n_targets_requested)
        )
    targets = np.asarray(targets, dtype=np.float64)
    n_targets_eff = int(targets.shape[0])
    if n_targets_eff == 0:
        print("benchmark aborted: no target found inside [r_min, r_safe].")
        return {
            "n_targets": 0,
            "n_targets_eff": 0,
            "n_targets_requested": n_targets_requested,
            "n_seeds": n_seeds,
            "total_cases": 0,
            "success_rate": 0.0,
            "converged_rate": 0.0,
            "pos_errs_m": np.asarray([], dtype=np.float64),
            "solve_times_ms": np.asarray([], dtype=np.float64),
        }

    pos_errs = []
    converged_flags = []
    solve_times_ms = []
    total_cases = n_targets_eff * n_seeds
    case_idx = 0
    for i in range(n_targets_eff):
        p_tgt = targets[i]
        for _ in range(n_seeds):
            case_idx += 1
            q_seed = np.clip(
                q_ref.astype(np.float64)
                + rng.normal(loc=0.0, scale=float(cfg.seed_perturb_std), size=nq),
                q_lo,
                q_hi,
            ).astype(np.float32)
            t0 = time.perf_counter()
            q_sol, conv = _solve_position_ik_goal(
                art,
                q_seed,
                p_tgt,
                joint_limits=joint_limits,
                ee_link_index=ee_link_index,
                max_iter=iters,
                prefer_q=q_seed,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            err = tcp_position_error_m(art, q_sol, p_tgt, ee_link_index=ee_link_index)
            pos_errs.append(err)
            converged_flags.append(bool(conv))
            solve_times_ms.append(float(dt_ms))
            if case_idx % 200 == 0 or case_idx == total_cases:
                print("benchmark progress: %d/%d" % (case_idx, total_cases))

    pos_errs = np.asarray(pos_errs, dtype=np.float64)
    conv = np.asarray(converged_flags, dtype=bool)
    t_ms = np.asarray(solve_times_ms, dtype=np.float64)
    success = pos_errs < success_tol

    print("")
    print("===== IK Offline Benchmark =====")
    print(
        "targets=%d (requested=%d), seeds_per_target=%d, total_cases=%d, max_iter=%d, success_tol=%.6f m, rng_seed=%d"
        % (
            n_targets_eff,
            n_targets_requested,
            n_seeds,
            total_cases,
            iters,
            success_tol,
            int(rng_seed),
        )
    )
    print(
        "reachable sampling: shoulder_origin=(%.3f, %.3f, %.3f) m, R_max=%.3f m, "
        "R_min=%.3f m (ratio=%.2f), R_safe=%.3f m (ratio=%.2f), accepted=%d/%d"
        % (
            shoulder_origin[0],
            shoulder_origin[1],
            shoulder_origin[2],
            r_max,
            r_min,
            min_radius_ratio,
            r_safe,
            safe_radius_ratio,
            n_targets_eff,
            generated,
        )
    )
    print(
        "success_rate(err<tol)=%.2f%%, converged_rate=%.2f%%"
        % (100.0 * float(np.mean(success)), 100.0 * float(np.mean(conv)))
    )
    print(_format_stat_line("position_error", pos_errs, scale=1000.0, unit=" mm"))
    print(_format_stat_line("solve_time", t_ms, scale=1.0, unit=" ms"))
    print("================================")
    return {
        "n_targets": n_targets_eff,
        "n_targets_eff": n_targets_eff,
        "n_targets_requested": n_targets_requested,
        "n_seeds": n_seeds,
        "total_cases": total_cases,
        "safe_radius_ratio": safe_radius_ratio,
        "min_radius_ratio": min_radius_ratio,
        "r_max_m": r_max,
        "r_min_m": r_min,
        "r_safe_m": r_safe,
        "success_rate": float(np.mean(success)),
        "converged_rate": float(np.mean(conv)),
        "pos_errs_m": pos_errs,
        "solve_times_ms": t_ms,
    }


__all__ = ["run_benchmark"]
