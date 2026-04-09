"""Demo: Quadruped robot(s) using ModelBuilder -> Model -> World API.

Defaults to 1 robot; use --rows/--cols to scale up.

Key features:
  - Each robot gets its own World (same structure as demo_100_robots.py)
  - Joint drives use torque-level PID via Control.articulation_joint_forces
  - Collision shapes from URDF must be manually re-attached (leaf links only)
  - Joint name → index mapping extracted from SceneBuildMetadata
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import novaphy as nv

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STANDING_POSE = {
    # Grounded equilibrium angles: computed by forcing body level + stationary
    # while running physics, so PD torques are near zero at steady state.
    # This eliminates the structural forward pitch caused by non-equilibrium targets.
    "LF_HAA": +0.1995, "LF_HFE": +0.3855, "LF_KFE": -0.5952,
    "RF_HAA": -0.1995, "RF_HFE": -0.3855, "RF_KFE": +0.5952,
    "LH_HAA": -0.2030, "LH_HFE": +0.4308, "LH_KFE": -0.5901,
    "RH_HAA": +0.2030, "RH_HFE": -0.4308, "RH_KFE": +0.5901,
}

SPACING = 1.5        # metres between robot centres
ROOT_HEIGHT = 0.5    # initial drop height above ground

# Simulation
DT = 1.0 / 60.0
STEPS_PER_FRAME = 50
SIM_DT = DT / STEPS_PER_FRAME
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)

# Torque-level PD gains (applied via Control before each step)
# These are physical units: KP [N·m/rad], KD [N·m·s/rad], MAX_TORQUE [N·m]
# Torque PD is far more stable than velocity-level PGS motors for floating-base:
#   tau = KP * (target - q) + KD * (0 - qd)
KP = 80.0
KD = 3.0
MAX_TORQUE = 8000

# Body-orientation PID gains (applied via free-joint torque BEFORE world.step()).
# Torque correction feeds through ABA+PGS in the same timestep so contact friction
# automatically counteracts the resulting foot motion — no persistent forward drift.
#
# The structural forward pitch (~2°) is a steady-state ABA bias from CoM offset.
# The integral term (KI_BODY) accumulates pitch error and feeds forward a torque
# that exactly cancels this bias, driving equilibrium pitch → 0°.
#
# Units: KP_BODY [N·m/rad], KD_BODY [N·m·s/rad], KI_BODY [N·m/(rad·s)]
KP_BODY = 5000.0              # pitch/roll stiffness
KD_BODY = 500.0               # pitch/roll damping
KI_BODY = 5000.0              # pitch integral gain (removes steady-state offset)
MAX_ORIENTATION_TORQUE = 2000.0  # PD+I torque cap [N·m]
MAX_INTEGRAL_TAU = 250.0      # integral contribution cap (anti-windup) [N·m]

# PGS solver (contacts + joint limits only — no motor constraints)
SOLVER_ITERATIONS = 100
ERP = 0.2


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_joint_maps(metadata):
    """Return (name→q_index, name→qd_index, name→link_index) from SceneBuildMetadata."""
    q_idx = {}
    qd_idx = {}
    link_idx = {}
    for jm in metadata.joints:
        if jm.articulation_index >= 0 and jm.num_qd > 0 and jm.num_q == 1:
            q_idx[jm.joint_name]    = jm.q_start
            qd_idx[jm.joint_name]   = jm.qd_start
            link_idx[jm.joint_name] = jm.articulation_index
    return q_idx, qd_idx, link_idx


def _make_initial_q(base_q, x, y, q_map):
    """Copy base_q, set root XY position, apply standing pose."""
    q0 = np.array(base_q, dtype=np.float32).copy()
    q0[0] = x
    q0[1] = y
    q0[2] = ROOT_HEIGHT
    q0[3:7] = [0.0, 0.0, 0.0, 1.0]   # identity quaternion (xyzw)
    for name, angle in DEFAULT_STANDING_POSE.items():
        if name in q_map:
            q0[q_map[name]] = angle
    return q0


def _leaf_link_indices(articulation) -> set:
    """Return link indices that have no children (leaves of the kinematic tree).

    For a quadruped: these are the shin/lower-leg links that actually touch
    the ground. Using only leaf shapes avoids spurious intra-robot collisions.
    """
    parent_set = {j.parent for j in articulation.joints if j.parent >= 0}
    return {i for i in range(articulation.num_links()) if i not in parent_set}


def _quat_to_rpy(q_xyzw):
    """Extract roll/pitch from quaternion (xyzw convention, z-up)."""
    x, y, z, w = q_xyzw
    # roll (rotation about x)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    # pitch (rotation about y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return roll, pitch


def _clone_shape_for_art(shape, art_idx, link_idx):
    """Clone a URDF collision shape, tagging it to an articulation/link."""
    stype = shape.type.name
    if stype == "Box":
        return nv.CollisionShape.make_box(
            np.array(shape.box_half_extents, dtype=np.float32),
            0, shape.local_transform,
            shape.friction, shape.restitution,
            art_idx=art_idx, link_idx=link_idx)
    elif stype == "Sphere":
        return nv.CollisionShape.make_sphere(
            shape.sphere_radius, 0, shape.local_transform,
            shape.friction, shape.restitution,
            art_idx=art_idx, link_idx=link_idx)
    elif stype == "Cylinder":
        return nv.CollisionShape.make_cylinder(
            shape.cylinder_radius, shape.cylinder_half_length,
            0, shape.local_transform,
            shape.friction, shape.restitution,
            art_idx=art_idx, link_idx=link_idx)
    else:
        raise ValueError(f"Unsupported shape type for articulation collider: {stype}")


def _build_one_world(articulation, foot_shapes, x, y, q_map):
    """Build one World with a single quadruped robot at position (x, y)."""
    articulation.linear_damping = 0.02
    articulation.angular_damping = 0.02

    builder = nv.ModelBuilder()
    builder.set_gravity(GRAVITY)
    model = builder.build()
    model.articulations = [articulation]

    shapes = []

    # Ground plane (body_index=-1 → static)
    gnd = nv.CollisionShape.make_plane(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0)
    gnd.friction = 0.6
    gnd.restitution = 0.05
    shapes.append(gnd)

    # Foot colliders (art_idx=0 since this world has only one robot)
    for shape in foot_shapes:
        shapes.append(_clone_shape_for_art(shape, art_idx=0, link_idx=shape.body_index))

    model.shapes = shapes

    solver = nv.MultiBodySolverSettings()
    solver.num_iterations = SOLVER_ITERATIONS
    solver.erp = ERP

    world = nv.World(model, multibody_settings=solver)

    # Initial state
    art_total_q = articulation.total_q()
    q0  = _make_initial_q([0.0] * art_total_q, x, y, q_map)
    qd0 = np.zeros(articulation.total_qd(), dtype=np.float32)
    world.state.set_q(0, q0)
    world.state.set_qd(0, qd0)

    return world


def compute_pd_torques(world, articulation, robot_id, q_map, qd_map,
                       art_total_qd, pitch_integrals):
    """Compute joint PD torques + PID orientation correction and return tau.

    Joint PD:  tau = KP*(q_target - q) + KD*(0 - qd)
    Orientation (free-joint DOFs 0=roll, 1=pitch):
        tau[1] = KP_BODY*pitch + KI_BODY*∫pitch dt - KD_BODY*omega_y
        tau[0] = KP_BODY*roll                       - KD_BODY*omega_x

    Torque correction feeds through ABA+PGS in the same timestep so contact
    friction automatically counteracts resulting foot motion (no drift).
    The integral term removes the ~2° steady-state forward pitch (ABA CoM bias).

    pitch_integrals: dict {robot_id: float}, updated in-place each call.

    Convention (z-up, x-forward, base_body=true right-multiply conjugate):
      pitch > 0 => nose DOWN => tau[1] > 0 raises nose  => +KP_BODY*pitch
      roll  > 0 => left-up   => tau[0] > 0 rights body  => +KP_BODY*roll
    """
    q  = np.array(world.state.q[0],  dtype=np.float32)
    qd = np.array(world.state.qd[0], dtype=np.float32)
    tau = np.zeros(art_total_qd, dtype=np.float32)

    for name, target in DEFAULT_STANDING_POSE.items():
        if name not in q_map:
            continue
        qi  = q_map[name]
        di  = qd_map[name]
        pos_err = target - q[qi]
        vel_err = -qd[di]
        t = KP * pos_err + KD * vel_err
        tau[di] = float(np.clip(t, -MAX_TORQUE, MAX_TORQUE))

    # Orientation PID via free-joint torque (DOF indices 0=roll, 1=pitch).
    roll, pitch = _quat_to_rpy(q[3:7])

    # Integral: accumulate pitch error; cap to prevent windup.
    pitch_integrals[robot_id] += SIM_DT * float(pitch)
    integral_tau = float(np.clip(KI_BODY * pitch_integrals[robot_id],
                                  -MAX_INTEGRAL_TAU, MAX_INTEGRAL_TAU))

    tau[1] = float(np.clip(KP_BODY * pitch - KD_BODY * qd[1] + integral_tau,
                           -MAX_ORIENTATION_TORQUE, MAX_ORIENTATION_TORQUE))
    tau[0] = float(np.clip(KP_BODY * roll  - KD_BODY * qd[0],
                           -MAX_ORIENTATION_TORQUE, MAX_ORIENTATION_TORQUE))

    return tau


def build_robots(urdf_path: str, n_rows: int, n_cols: int):
    """
    Build n_rows*n_cols quadruped robots, each in its own World.

    Returns (robots, scene, q_map, qd_map)
    where robots = list of (world, articulation, robot_id)
    """
    # --- Parse URDF once ---
    parser = nv.UrdfParser()
    urdf_model = parser.parse_file(Path(urdf_path))

    options = nv.UrdfImportOptions()
    options.floating_base = True
    options.enable_self_collisions = False
    options.collapse_fixed_joints = True
    options.ignore_inertial_definitions = True

    scene = nv.SceneBuilderEngine().build_from_urdf(urdf_model, options)
    articulation = scene.articulation
    q_map, qd_map, link_map = _build_joint_maps(scene.metadata)

    if not q_map:
        raise RuntimeError(
            "No named joints found in URDF metadata. "
            "Check that the URDF has non-fixed joints with names."
        )

    # Only attach collision shapes on leaf links (feet/shins).
    foot_links  = _leaf_link_indices(articulation)
    foot_shapes = [s for s in scene.model.shapes
                   if s.type.name != "Plane" and s.body_index in foot_links]

    # --- Build one world per robot ---
    robots = []
    for row in range(n_rows):
        for col in range(n_cols):
            robot_id = row * n_cols + col
            x = (col - (n_cols - 1) / 2.0) * SPACING
            y = (row - (n_rows - 1) / 2.0) * SPACING

            world = _build_one_world(articulation, foot_shapes, x, y, q_map)
            robots.append((world, articulation, robot_id))

    print(f"Built {len(robots)} robot(s)  "
          f"foot_links={sorted(foot_links)}  "
          f"foot_shapes={len(foot_shapes)}  "
          f"controlled_joints={sum(1 for n in DEFAULT_STANDING_POSE if n in q_map)}")
    return robots, scene, q_map, qd_map


# ---------------------------------------------------------------------------
# Headless run
# ---------------------------------------------------------------------------

def run_headless(urdf_path: str, n_rows: int, n_cols: int, num_steps: int = 500):
    print(f"NovaPhy World (multibody) — {n_rows}×{n_cols} robots (headless)")
    robots, scene, q_map, qd_map = build_robots(urdf_path, n_rows, n_cols)
    art_total_qd = scene.articulation.total_qd()
    pitch_integrals = {robot_id: 0.0 for _, _, robot_id in robots}

    for step in range(num_steps):
        for world, articulation, robot_id in robots:
            tau = compute_pd_torques(world, articulation, robot_id,
                                     q_map, qd_map, art_total_qd, pitch_integrals)
            control = nv.Control()
            control.articulation_joint_forces = [tau]
            world.step_with_control(world.state, control, SIM_DT)

    for world, articulation, robot_id in robots:
        q = np.array(world.state.q[0], dtype=np.float32)
        z = q[2]
        assert not np.isnan(q).any(), f"Robot {robot_id}: NaN in q"
        assert z > -0.2, f"Robot {robot_id}: fell through ground (z={z:.3f})"

    world0, _, _ = robots[0]
    q = np.array(world0.state.q[0], dtype=np.float32)
    print(f"Robot 0 final z={q[2]:.3f}  |q|={np.linalg.norm(q):.3f}")
    print("Headless run passed.")


# ---------------------------------------------------------------------------
# GUI run
# ---------------------------------------------------------------------------

def run_gui(urdf_path: str, n_rows: int, n_cols: int):
    if not HAS_POLYSCOPE:
        raise ImportError("polyscope required: pip install polyscope")

    robots, scene, q_map, qd_map = build_robots(urdf_path, n_rows, n_cols)
    art_total_qd = scene.articulation.total_qd()
    pitch_integrals = {robot_id: 0.0 for _, _, robot_id in robots}

    # --- Polyscope init ---
    ps.init()
    ps.set_program_name(f"NovaPhy World (multibody) — {len(robots)} Quadruped(s)")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("tile_reflection")
    ground_size = max(n_rows, n_cols) * SPACING + 2.0
    ps.look_at(
        (ground_size * 0.8, ground_size * 0.8, ground_size * 0.5),
        (0.0, 0.0, 0.0),
    )

    # Ground mesh
    gv, gf = make_ground_plane_mesh(ground_size, 0.0, up="z")
    gm = ps.register_surface_mesh("ground_plane", gv, gf)
    gm.set_color((0.55, 0.55, 0.55))
    gm.set_edge_width(0.5)

    # Build per-robot mesh descriptors from URDF shapes
    # (ps_name, local_verts, faces, articulation_link_index, shape_local_tf, robot_id)
    all_shape_descs = []
    body_colors = [
        (0.20, 0.55, 0.85), (0.85, 0.45, 0.20), (0.30, 0.75, 0.45),
        (0.75, 0.30, 0.60), (0.50, 0.70, 0.20), (0.90, 0.65, 0.15),
        (0.35, 0.55, 0.75), (0.70, 0.35, 0.35),
    ]

    for _, _, robot_id in robots:
        for i, shape in enumerate(scene.model.shapes):
            stype = shape.type.name
            if stype == "Box":
                he = np.array(shape.box_half_extents, dtype=np.float32)
                verts, faces = make_box_mesh(he)
            elif stype == "Sphere":
                verts, faces = make_sphere_mesh(shape.sphere_radius, n_lat=8, n_lon=12)
            elif stype == "Cylinder":
                verts, faces = make_cylinder_mesh(
                    shape.cylinder_radius, shape.cylinder_half_length, n_segments=12)
            elif stype == "Plane":
                continue
            else:
                continue

            all_shape_descs.append((
                f"r{robot_id}_s{i}",
                verts,
                faces,
                shape.body_index,
                shape.local_transform,
                robot_id,
            ))

    print(f"Total shape meshes: {len(all_shape_descs)}")

    def update_meshes():
        for world, articulation, robot_id in robots:
            transforms = nv.forward_kinematics(articulation, world.state.q[0])

            for ps_name, local_verts, faces, link_idx, local_tf, rid in all_shape_descs:
                if rid != robot_id:
                    continue
                if link_idx < 0 or link_idx >= len(transforms):
                    continue

                link_tf = transforms[link_idx]
                world_tf = link_tf * local_tf
                pos = np.array(world_tf.position, dtype=np.float32)
                rot = quat_to_rotation_matrix(
                    np.array(world_tf.rotation, dtype=np.float32))
                world_verts = (local_verts @ rot.T) + pos

                if ps.has_surface_mesh(ps_name):
                    ps.get_surface_mesh(ps_name).update_vertex_positions(world_verts)
                else:
                    sm = ps.register_surface_mesh(ps_name, world_verts, faces)
                    sm.set_color(body_colors[link_idx % len(body_colors)])
                    sm.set_smooth_shade(True)

    update_meshes()

    gui = {"paused": True, "step_once": False, "frame": 0}

    def callback():
        _, gui["paused"] = psim.Checkbox("Paused", gui["paused"])
        psim.SameLine()
        if psim.Button("Step"):
            gui["step_once"] = True

        world0, _, _ = robots[0]
        q0 = np.array(world0.state.q[0], dtype=np.float32)
        total_contacts = sum(len(w.multibody_contacts) for w, _, _ in robots)
        psim.TextUnformatted(f"Frame: {gui['frame']}")
        psim.TextUnformatted(f"Robots: {len(robots)}")
        psim.TextUnformatted(f"Robot0 z={q0[2]:.3f}")
        psim.TextUnformatted(f"Contacts: {total_contacts}")

        if not gui["paused"] or gui["step_once"]:
            for _ in range(STEPS_PER_FRAME):
                for world, articulation, robot_id in robots:
                    tau = compute_pd_torques(world, articulation, robot_id,
                                             q_map, qd_map, art_total_qd,
                                             pitch_integrals)
                    control = nv.Control()
                    control.articulation_joint_forces = [tau]
                    world.step_with_control(world.state, control, SIM_DT)
            gui["frame"] += 1
            gui["step_once"] = False

        update_meshes()

    ps.set_user_callback(callback)
    ps.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("NovaPhy World (multibody) robot demo")
    parser.add_argument("--urdf", default="demos/data/quadruped.urdf")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of steps for headless mode")
    args = parser.parse_args()

    if args.headless or not HAS_POLYSCOPE:
        run_headless(args.urdf, args.rows, args.cols, args.steps)
    else:
        run_gui(args.urdf, args.rows, args.cols)


if __name__ == "__main__":
    main()
