"""Entry point for the NovaPhy FR3 (Franka Research 3) inverse kinematics (IK) demo.

Loads the FR3 arm in NovaPhy and solves position IK for the tool center point (TCP).
In interactive mode, drag the target sphere in Polyscope for live solves; in headless
mode, solve once for a given world-space target; or run an offline benchmark over
sampled targets and seeds (success rate and errors).

Dependencies and paths
----------------------
- Requires ``novaphy``; interactive mode also needs ``polyscope``.
- The script prepends the repository root and ``demos/`` to ``sys.path`` so ``ik`` and
  ``ik_arm_demo`` import correctly. Run from the repo root (examples below), or from
  ``demos/`` as ``python demo_ik_arm.py``.

Usage (examples)
----------------
From the ``NovaPhy/`` repository root::

    # Interactive Polyscope demo: drag the target, watch IK and motion
    python demos/demo_ik_arm.py

    # Headless: world positions printed in meters, position error in millimeters
    python demos/demo_ik_arm.py --headless --target-pos 0.5 0.0 0.4

    # Offline benchmark: sample reachable workspace, multiple seeds per target
    python demos/demo_ik_arm.py --benchmark

    # Override IK max iterations (applies to interactive, headless, and benchmark)
    python demos/demo_ik_arm.py --max-iter 80

    # Benchmark tuning: number of targets, seeds per target, RNG seed, etc.
    python demos/demo_ik_arm.py --benchmark --bench-targets 500 --bench-seeds 8 --seed 42

Full CLI help::

    python demos/demo_ik_arm.py -h
"""

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from ik_arm_demo.benchmark import run_benchmark
from ik_arm_demo.cli_args import add_benchmark_cli_arguments, benchmark_config_from_args
from ik_arm_demo.config import BenchmarkConfig
from ik_arm_demo.runtime import run_runtime


def _build_argument_parser() -> argparse.ArgumentParser:
    bench_defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description="NovaPhy FR3 IK interactive demo")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--benchmark", action="store_true", help="Run offline IK benchmark")
    parser.add_argument(
        "--target-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Target TCP position for --headless",
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Override IK maximum iterations")
    add_benchmark_cli_arguments(parser, bench_defaults)
    return parser


def main():
    args = _build_argument_parser().parse_args()
    bench_cfg = benchmark_config_from_args(args)
    if args.benchmark:
        run_benchmark(
            n_targets=args.bench_targets,
            n_seeds=args.bench_seeds,
            max_iter=args.max_iter,
            success_tol=args.bench_success_tol,
            rng_seed=args.seed,
            safe_radius_ratio=args.bench_safe_radius_ratio,
            min_radius_ratio=args.bench_min_radius_ratio,
            benchmark_cfg=bench_cfg,
        )
    else:
        run_runtime(
            headless=bool(args.headless),
            max_iter=args.max_iter,
            target_pos=args.target_pos,
        )


if __name__ == "__main__":
    main()
