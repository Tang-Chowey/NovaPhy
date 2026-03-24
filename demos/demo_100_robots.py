"""Demo: 100 quadruped robots in a 10×10 grid on a shared ground plane."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

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

from novaphy.viz import (
    make_box_mesh, make_sphere_mesh, make_cylinder_mesh,
    make_ground_plane_mesh, quat_to_rotation_matrix,
)


DEFAULT_STANDING_POSE = {
    "LF_HAA": 0.2,  "LF_HFE": 0.4,  "LF_KFE": -0.6,
    "RF_HAA": -0.2, "RF_HFE": -0.4, "RF_KFE": 0.6,
    "LH_HAA": -0.2, "LH_HFE": 0.4,  "LH_KFE": -0.6,
    "RH_HAA": 0.2,  "RH_HFE": -0.4, "RH_KFE": 0.6,
}

# Grid layout
N_ROWS = 10
N_COLS = 10
SPACING = 1.5  # meters between robot centers

# Physics
ROOT_HEIGHT = 0.45
DRIVE_STIFFNESS = 2000.0
DRIVE_DAMPING = 1.0
SOLVER_SUBSTEPS = 10
SOLVER_ITERATIONS = 6
DT = 1.0 / 100.0
STEPS_PER_FRAME = 2
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)


def _build_one_robot(urdf_model, row: int, col: int):
    """Build a single ArticulatedWorld positioned in the grid."""
    x = (col - (N_COLS - 1) / 2.0) * SPACING
    y = (row - (N_ROWS - 1) / 2.0) * SPACING

    options = novaphy.UrdfImportOptions()
    options.floating_base = True
    options.enable_self_collisions = False
    options.collapse_fixed_joints = True
    options.ignore_inertial_definitions = True
    options.root_transform = novaphy.Transform.from_translation(
        np.array([x, y, ROOT_HEIGHT], dtype=np.float32)
    )

    scene = novaphy.SceneBuilderEngine().build_from_urdf(urdf_model, options)

    solver_settings = novaphy.XPBDSolverSettings()
    solver_settings.substeps = SOLVER_SUBSTEPS
    solver_settings.iterations = SOLVER_ITERATIONS

    world = novaphy.ArticulatedWorld(scene, solver_settings)
    world.add_ground_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0)
    world.set_gravity(GRAVITY)
    world.set_default_drive_gains(DRIVE_STIFFNESS, DRIVE_DAMPING)
    world.set_joint_positions(DEFAULT_STANDING_POSE)
    world.set_target_positions(DEFAULT_STANDING_POSE)

    return scene, world


def run_gui(urdf_path: str):
    """Run the 100-robot demo with Polyscope GUI."""
    if not HAS_POLYSCOPE:
        raise ImportError("polyscope is required. Install with: pip install polyscope")

    # Parse URDF once
    parser = novaphy.UrdfParser()
    urdf_model = parser.parse_file(Path(urdf_path))

    # Build all robots
    print(f"Building {N_ROWS * N_COLS} robots...")
    robots: List[Tuple] = []  # (scene, world, robot_id)
    for row in range(N_ROWS):
        for col in range(N_COLS):
            robot_id = row * N_COLS + col
            scene, world = _build_one_robot(urdf_model, row, col)
            robots.append((scene, world, robot_id))
    print(f"All {len(robots)} robots built.")

    # --- Polyscope init ---
    ps.init()
    ps.set_program_name(f"NovaPhy - {len(robots)} Quadrupeds")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("tile_reflection")
    ps.look_at(
        (N_COLS * SPACING * 0.8, N_ROWS * SPACING * 0.8, N_ROWS * SPACING * 0.5),
        (0.0, 0.0, 0.0),
    )

    # --- Ground mesh ---
    ground_size = max(N_ROWS, N_COLS) * SPACING
    gv, gf = make_ground_plane_mesh(ground_size, 0.0, up="z")
    gm = ps.register_surface_mesh("ground_plane", gv, gf)
    gm.set_color((0.55, 0.55, 0.55))
    gm.set_edge_width(0.5)

    # --- Build mesh data per robot ---
    # Each entry: (ps_name, local_verts, faces, body_index, shape_local_tf, robot_idx)
    all_shape_meshes = []
    body_colors = [
        (0.20, 0.55, 0.85), (0.85, 0.45, 0.20), (0.30, 0.75, 0.45),
        (0.75, 0.30, 0.60), (0.50, 0.70, 0.20), (0.90, 0.65, 0.15),
        (0.35, 0.55, 0.75), (0.70, 0.35, 0.35),
    ]

    for scene, world, robot_id in robots:
        model = scene.model
        for i, shape in enumerate(model.shapes):
            stype = shape.type.name
            ps_name = f"r{robot_id}_s{i}"

            if stype == "Box":
                he = np.array(shape.box_half_extents, dtype=np.float32)
                verts, faces = make_box_mesh(he)
            elif stype == "Sphere":
                verts, faces = make_sphere_mesh(shape.sphere_radius, n_lat=8, n_lon=12)
            elif stype == "Cylinder":
                verts, faces = make_cylinder_mesh(shape.cylinder_radius, shape.cylinder_half_length, n_segments=12)
            elif stype == "Plane":
                continue  # shared ground already registered
            else:
                continue

            all_shape_meshes.append(
                (ps_name, verts, faces, shape.body_index, shape.local_transform, robot_id)
            )

    print(f"Total shape meshes: {len(all_shape_meshes)}")

    def update_all_meshes():
        """Update all robot meshes from FK."""
        for scene, world, robot_id in robots:
            articulation = scene.articulation
            q_vec = np.array(world.q, dtype=np.float32)
            transforms = novaphy.forward_kinematics(articulation, q_vec)

            for ps_name, local_verts, faces, body_idx, shape_local_tf, rid in all_shape_meshes:
                if rid != robot_id:
                    continue
                if body_idx < 0 or body_idx >= len(transforms):
                    continue

                link_tf = transforms[body_idx]
                world_tf = link_tf * shape_local_tf
                pos = np.array(world_tf.position, dtype=np.float32)
                rot = quat_to_rotation_matrix(
                    np.array(world_tf.rotation, dtype=np.float32)
                )
                world_verts = (local_verts @ rot.T) + pos

                if ps.has_surface_mesh(ps_name):
                    ps.get_surface_mesh(ps_name).update_vertex_positions(world_verts)
                else:
                    sm = ps.register_surface_mesh(ps_name, world_verts, faces)
                    color = body_colors[body_idx % len(body_colors)]
                    sm.set_color(color)
                    sm.set_smooth_shade(True)

    # Initial render
    update_all_meshes()

    # --- GUI state ---
    gui_state = {"paused": False, "step_once": False, "frame": 0}

    def callback():
        _, gui_state["paused"] = psim.Checkbox("Paused", gui_state["paused"])
        psim.SameLine()
        if psim.Button("Step"):
            gui_state["step_once"] = True

        psim.TextUnformatted(f"Frame: {gui_state['frame']}")
        psim.TextUnformatted(f"Robots: {len(robots)}")

        if not gui_state["paused"] or gui_state["step_once"]:
            for _ in range(STEPS_PER_FRAME):
                for scene, world, robot_id in robots:
                    world.step(DT)
                gui_state["frame"] += 1
            gui_state["step_once"] = False

        update_all_meshes()

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser("NovaPhy 100-robot grid demo")
    parser.add_argument("--urdf", type=str, default="demos/data/quadruped.urdf")
    parser.add_argument("--gui", action="store_true", default=True)
    args = parser.parse_args()
    run_gui(args.urdf)


if __name__ == "__main__":
    main()
