"""Argparse helpers shared by ``demo_ik_arm.py``."""

import argparse

from .config import (
    IK_BENCH_MIN_RADIUS_RATIO,
    IK_BENCH_SAFE_RADIUS_RATIO,
    IK_BENCH_SUCCESS_TOL,
    BenchmarkConfig,
)


def add_benchmark_cli_arguments(parser: argparse.ArgumentParser, bench_defaults: BenchmarkConfig) -> None:
    """Register all benchmark-related flags (used when ``--benchmark`` is set)."""
    parser.add_argument("--bench-targets", type=int, default=300, help="Number of benchmark targets")
    parser.add_argument("--bench-seeds", type=int, default=4, help="Seeds per benchmark target")
    parser.add_argument(
        "--bench-success-tol",
        type=float,
        default=IK_BENCH_SUCCESS_TOL,
        help="Benchmark success threshold on position error (m)",
    )
    parser.add_argument(
        "--bench-safe-radius-ratio",
        type=float,
        default=IK_BENCH_SAFE_RADIUS_RATIO,
        help="Safe reachable radius ratio in (0,1]",
    )
    parser.add_argument(
        "--bench-min-radius-ratio",
        type=float,
        default=IK_BENCH_MIN_RADIUS_RATIO,
        help="Min reachable radius ratio in [0,safe]",
    )
    parser.add_argument(
        "--bench-workspace-probes",
        type=int,
        default=bench_defaults.workspace_probe_samples,
        help="Workspace probe samples for benchmark reachability",
    )
    parser.add_argument(
        "--bench-max-generate-floor",
        type=int,
        default=bench_defaults.max_generate_floor,
        help="Minimum rejection-sampling attempts for benchmark targets",
    )
    parser.add_argument(
        "--bench-max-generate-per-target",
        type=int,
        default=bench_defaults.max_generate_per_target,
        help="Additional rejection-sampling attempts per benchmark target",
    )
    parser.add_argument(
        "--bench-seed-perturb-std",
        type=float,
        default=bench_defaults.seed_perturb_std,
        help="Std dev (rad) for benchmark random seed perturbation",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for benchmark sampling")


def benchmark_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Build a frozen ``BenchmarkConfig`` from parsed CLI values (with basic sanitization)."""
    return BenchmarkConfig(
        workspace_probe_samples=int(max(1, args.bench_workspace_probes)),
        max_generate_floor=int(max(1, args.bench_max_generate_floor)),
        max_generate_per_target=int(max(1, args.bench_max_generate_per_target)),
        seed_perturb_std=float(max(0.0, args.bench_seed_perturb_std)),
    )


__all__ = ["add_benchmark_cli_arguments", "benchmark_config_from_args"]
