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

try:
    import polyscope as ps
    import polyscope.imgui as psim
    HAS_POLYSCOPE = True
except ImportError:
    ps = None
    psim = None
    HAS_POLYSCOPE = False

from novaphy.viz import make_box_mesh, make_sphere_mesh, make_cylinder_mesh, make_ground_plane_mesh, quat_to_rotation_matrix


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
    root_height: float = 0.45
    gravity_z: float = -9.81
    drive_stiffness: float = 2000.0
    drive_damping: float = 1.0
    gui: bool = False
    steps_per_frame: int = 2

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


def run_gui(config: DemoConfig) -> None:
    """Run the URDF demo with interactive Polyscope visualization."""
    if not HAS_POLYSCOPE:
        raise ImportError(
            "polyscope is required for GUI mode. Install with: pip install polyscope"
        )

    scene, world = build_world(config)
    articulation = scene.articulation
    model = scene.model

    # --- Polyscope init ---
    ps.init()
    ps.set_program_name("NovaPhy - Basic URDF Quadruped")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("tile_reflection")
    ps.look_at((3.0, 3.0, 2.0), (0.0, 0.0, 0.4))

    # --- Build mesh data for each collision shape ---
    # Each entry: (name, local_verts, faces, body_index, shape_local_transform)
    shape_meshes = []

    for i, shape in enumerate(model.shapes):
        stype = shape.type.name
        name = f"shape_{i}"

        if stype == "Box":
            he = np.array(shape.box_half_extents, dtype=np.float32)
            verts, faces = make_box_mesh(he)
            shape_meshes.append((name, verts, faces, shape.body_index, shape.local_transform))
        elif stype == "Sphere":
            verts, faces = make_sphere_mesh(shape.sphere_radius)
            shape_meshes.append((name, verts, faces, shape.body_index, shape.local_transform))
        elif stype == "Cylinder":
            verts, faces = make_cylinder_mesh(shape.cylinder_radius, shape.cylinder_half_length)
            shape_meshes.append((name, verts, faces, shape.body_index, shape.local_transform))
        elif stype == "Plane":
            verts, faces = make_ground_plane_mesh(50.0, shape.plane_offset, up="z")
            gm = ps.register_surface_mesh(name, verts, faces)
            gm.set_color((0.55, 0.55, 0.55))
            gm.set_edge_width(0.5)
            # Ground doesn't need transform updates

    # Assign pleasant colors to link shapes
    body_colors = [
        (0.20, 0.55, 0.85),  # blue
        (0.85, 0.45, 0.20),  # orange
        (0.30, 0.75, 0.45),  # green
        (0.75, 0.30, 0.60),  # magenta
        (0.50, 0.70, 0.20),  # lime
        (0.90, 0.65, 0.15),  # gold
        (0.35, 0.55, 0.75),  # steel blue
        (0.70, 0.35, 0.35),  # rust
    ]

    def update_meshes():
        """Use FK to get link transforms and update Polyscope meshes."""
        q_vec = np.array(world.q, dtype=np.float32)
        transforms = novaphy.forward_kinematics(articulation, q_vec)

        for name, local_verts, faces, body_idx, shape_local_tf in shape_meshes:
            if body_idx < 0 or body_idx >= len(transforms):
                continue
            # Compose: world_tf = link_transform * shape_local_transform
            link_tf = transforms[body_idx]
            world_tf = link_tf * shape_local_tf
            pos = np.array(world_tf.position, dtype=np.float32)
            rot = quat_to_rotation_matrix(
                np.array(world_tf.rotation, dtype=np.float32)
            )
            world_verts = (local_verts @ rot.T) + pos

            if ps.has_surface_mesh(name):
                ps.get_surface_mesh(name).update_vertex_positions(world_verts)
            else:
                sm = ps.register_surface_mesh(name, world_verts, faces)
                color = body_colors[body_idx % len(body_colors)]
                sm.set_color(color)
                sm.set_smooth_shade(True)

    # Initial render
    update_meshes()

    # --- GUI state ---
    gui_state = {
        "paused": False,
        "step_once": False,
        "frame": 0,
    }

    def callback():
        # --- ImGui controls ---
        _, gui_state["paused"] = psim.Checkbox("Paused", gui_state["paused"])
        psim.SameLine()
        if psim.Button("Step"):
            gui_state["step_once"] = True
        psim.SameLine()
        if psim.Button("Reset"):
            # Rebuild the world from scratch
            nonlocal scene, world, articulation, model
            scene, world = build_world(config)
            articulation = scene.articulation
            model = scene.model
            gui_state["frame"] = 0

        root_z = _root_height(scene, world)
        psim.TextUnformatted(f"Frame: {gui_state['frame']}")
        psim.TextUnformatted(f"Root height: {root_z:.4f} m")

        # Show joint positions
        if psim.TreeNode("Joint Positions"):
            for jname in world.joint_names:
                val = world.joint_positions.get(jname, 0.0)
                psim.TextUnformatted(f"  {jname}: {val:.4f} rad")
            psim.TreePop()

        # --- Step simulation ---
        if not gui_state["paused"] or gui_state["step_once"]:
            for _ in range(config.steps_per_frame):
                world.step(config.dt)
                gui_state["frame"] += 1
            gui_state["step_once"] = False

        update_meshes()

    ps.set_user_callback(callback)
    ps.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("NovaPhy Newton-style basic URDF demo")
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--root-height", type=float, default=None)
    parser.add_argument("--solver-substeps", type=int, default=None)
    parser.add_argument("--solver-iterations", type=int, default=None)
    parser.add_argument("--gui", action="store_true", help="Run with Polyscope GUI")
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
    if args.gui:
        config.gui = True
    return config


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    if config.gui:
        run_gui(config)
    else:
        outputs = run_demo(config)
        print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
