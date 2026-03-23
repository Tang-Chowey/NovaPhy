"""Demo: Newton-style basic URDF quadruped using NovaPhy ArticulatedWorld."""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import novaphy


DEFAULT_STANDING_POSE: Dict[str, float] = {
    "LF_HAA": 0.2,
    "LF_HFE": 0.4,
    "LF_KFE": -0.6,
    "RF_HAA": -0.2,
    "RF_HFE": -0.4,
    "RF_KFE": 0.6,
    "LH_HAA": -0.2,
    "LH_HFE": 0.4,
    "LH_KFE": -0.6,
    "RH_HAA": 0.2,
    "RH_HFE": -0.4,
    "RH_KFE": 0.6,
}


@dataclass
class DemoConfig:
    urdf_path: str = "demos/data/quadruped.urdf"
    output_dir: str = "build/demo_basic_urdf"
    fps: int = 100
    frames: int = 400
    solver_substeps: int = 10
    solver_iterations: int = 6
    root_height: float = 0.7
    gravity_z: float = -9.81
    drive_stiffness: float = 2000.0
    drive_damping: float = 1.0

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)


def _import_options(config: DemoConfig) -> novaphy.UrdfImportOptions:
    options = novaphy.UrdfImportOptions()
    options.floating_base = True
    options.enable_self_collisions = False
    options.collapse_fixed_joints = True
    options.ignore_inertial_definitions = True
    options.root_transform = novaphy.Transform.from_translation(
        np.array([0.0, 0.0, config.root_height], dtype=np.float32)
    )
    return options


def build_world(config: DemoConfig) -> Tuple[novaphy.SceneBuildResult, novaphy.ArticulatedWorld]:
    parser = novaphy.UrdfParser()
    urdf_model = parser.parse_file(Path(config.urdf_path))
    scene = novaphy.SceneBuilderEngine().build_from_urdf(urdf_model, _import_options(config))

    solver_settings = novaphy.XPBDSolverSettings()
    solver_settings.substeps = int(config.solver_substeps)
    solver_settings.iterations = int(config.solver_iterations)
    world = novaphy.ArticulatedWorld(scene, solver_settings)
    world.add_ground_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0)
    world.set_gravity(np.array([0.0, 0.0, config.gravity_z], dtype=np.float32))
    world.set_default_drive_gains(config.drive_stiffness, config.drive_damping)

    missing = [name for name in DEFAULT_STANDING_POSE if name not in world.joint_names]
    if missing:
        raise ValueError(f"Quadruped scene is missing expected joint names: {missing}")

    world.set_joint_positions(DEFAULT_STANDING_POSE)
    world.set_target_positions(DEFAULT_STANDING_POSE)
    return scene, world


def _root_height(scene: novaphy.SceneBuildResult, world: novaphy.ArticulatedWorld) -> float:
    root_index = scene.metadata.root_articulation_index
    q_start = scene.articulation.q_start(root_index)
    return float(world.q[q_start + 2])


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_demo(config: DemoConfig) -> Dict[str, str]:
    scene, world = build_world(config)

    joint_rows: List[Dict[str, float]] = []
    contact_rows: List[Dict[str, float]] = []
    ground_only_contacts = True
    contact_steps = 0
    peak_abs_qd = 0.0
    final_root_height = _root_height(scene, world)

    for frame in range(config.frames):
        time_seconds = frame * config.dt
        world.step(config.dt)
        final_root_height = _root_height(scene, world)
        qd = np.array(world.qd, dtype=np.float32)
        if qd.size:
            peak_abs_qd = max(peak_abs_qd, float(np.max(np.abs(qd))))

        row = {
            "frame": frame,
            "time": time_seconds,
            "root_z": final_root_height,
        }
        row.update({name: float(world.joint_positions.get(name, 0.0)) for name in world.joint_names})
        joint_rows.append(row)

        if world.contacts:
            contact_steps += 1
        for contact in world.contacts:
            ground_only_contacts = ground_only_contacts and (contact.body_a < 0 or contact.body_b < 0)
            contact_rows.append(
                {
                    "frame": frame,
                    "time": time_seconds,
                    "body_a": int(contact.body_a),
                    "body_b": int(contact.body_b),
                    "px": float(contact.position[0]),
                    "py": float(contact.position[1]),
                    "pz": float(contact.position[2]),
                    "nx": float(contact.normal[0]),
                    "ny": float(contact.normal[1]),
                    "nz": float(contact.normal[2]),
                    "penetration": float(contact.penetration),
                }
            )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    joint_trace_path = output_dir / "joint_trace.csv"
    contact_trace_path = output_dir / "contact_trace.csv"
    config_path = output_dir / "config.json"

    final_qd = np.array(world.qd, dtype=np.float32)
    summary = {
        "joint_names": list(world.joint_names),
        "final_root_height": final_root_height,
        "final_max_abs_qd": float(np.max(np.abs(final_qd))) if final_qd.size else 0.0,
        "peak_abs_qd": peak_abs_qd,
        "ground_only_contacts": ground_only_contacts,
        "contact_steps": contact_steps,
        "filtered_link_pair_count": len(scene.metadata.filtered_link_pairs),
        "warnings": list(scene.warnings),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    _write_csv(joint_trace_path, joint_rows)
    _write_csv(contact_trace_path, contact_rows)

    return {
        "summary_json": str(summary_path),
        "joint_trace_csv": str(joint_trace_path),
        "contact_trace_csv": str(contact_trace_path),
        "config_json": str(config_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("NovaPhy Newton-style basic URDF demo")
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--root-height", type=float, default=None)
    parser.add_argument("--solver-substeps", type=int, default=None)
    parser.add_argument("--solver-iterations", type=int, default=None)
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> DemoConfig:
    config = DemoConfig()
    if args.urdf:
        config.urdf_path = args.urdf
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.frames is not None:
        config.frames = args.frames
    if args.fps is not None:
        config.fps = args.fps
    if args.root_height is not None:
        config.root_height = args.root_height
    if args.solver_substeps is not None:
        config.solver_substeps = args.solver_substeps
    if args.solver_iterations is not None:
        config.solver_iterations = args.solver_iterations
    return config


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    outputs = run_demo(config)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
