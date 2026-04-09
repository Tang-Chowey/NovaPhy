"""Demo: Motor-driven arm sweeping rigid-body boxes off a shelf.

Scene
-----
- Ground plane at y = 0
- Pedestal + rotating arm: 2-link articulation
    Link 0: Fixed joint anchor/pedestal at (0, 1.3, 0)
    Link 1: Revolute Y (vertical axis) — arm extends 1.1 m along +X at q=0
- Motor drives the arm at -1.5 rad/s (sweeps from +X toward +Z)
  Implemented as torque-level PD: tau = stiffness*(0 - q) + damping*(target_vel - qd)
- Static shelf surface at height y ≈ 1.12 (box shape registered as static shape)
- 7 rigid boxes resting on the shelf, in a row along Z

Physics sequence
----------------
1. Motor accelerates arm from rest.
2. At t ≈ 0.3 s arm tip reaches the first box; arm sweeps through the row.
3. Boxes fly off the shelf and land on the ground.
4. Arm continues rotating, possibly interacting with fallen boxes.

New API features exercised
-----------------------------------------------------
- Torque-level motor via Control.articulation_joint_forces
- Revolute joint with Y axis (vertical axis; horizontal sweep in XZ plane)
- Static box CollisionShape used as shelf surface (local_transform = world pose)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import novaphy as nv

try:
    import polyscope as ps
    import polyscope.imgui as psim
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False

from novaphy.viz import make_box_mesh, make_ground_plane_mesh, quat_to_rotation_matrix

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

PIVOT_Y    = 1.30      # arm pivot height (m)
ARM_L      = 1.10      # arm full length (m)
ARM_HALF   = ARM_L / 2.0
ARM_HY     = 0.07      # arm half-height (vertical)
ARM_HZ     = 0.05      # arm half-depth (into page)
ARM_MASS   = 1.2       # kg

MOTOR_VEL  = -5.5     # rad/s  (negative → sweep from +X toward +Z)
MOTOR_KP   = 0.0      # position stiffness (0 = pure velocity drive)
MOTOR_KD   = 5.0     # velocity damping (Nm·s/rad)

SHELF_HX   = 0.70      # shelf half-extent along X
SHELF_HY   = 0.03      # shelf half-height
SHELF_HZ   = 0.65      # shelf half-extent along Z
SHELF_CX   = 0.55      # shelf centre X
SHELF_CY   = 1.12      # shelf centre Y  (top at 1.15)
SHELF_CZ   = 0.55      # shelf centre Z

BOX_H      = 0.07      # box half-extent (14 cm cubes)
BOX_MASS   = 0.25      # kg
N_BOXES    = 7
BOX_Z_START = 0.10     # first box Z
BOX_Z_STEP  = 0.13     # centre-to-centre spacing in Z
BOX_X       = 0.55     # box X position (centre of shelf along X)
BOX_Y       = SHELF_CY + SHELF_HY + BOX_H   # box centre Y (resting on shelf top)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_inertia(mass, hx, hy, hz):
    lx, ly, lz = 2*hx, 2*hy, 2*hz
    return np.array([
        mass/12*(ly**2+lz**2),
        mass/12*(lx**2+lz**2),
        mass/12*(lx**2+ly**2),
    ], dtype=np.float32)


def _compute_motor_torque(q, qd, target_vel, kp, kd):
    """PD velocity motor: tau = kp*(0 - q) + kd*(target_vel - qd)."""
    return kp * (0.0 - q) + kd * (target_vel - qd)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_world():
    builder = nv.ModelBuilder()

    solver_settings = nv.MultiBodySolverSettings()
    solver_settings.num_iterations = 100
    solver_settings.erp            = 0.2
    solver_settings.linear_slop    = 0.002
    solver_settings.restitution_threshold = 0.4

    gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
    builder.set_gravity(gravity)

    # ---- Ground ----
    gnd = nv.CollisionShape.make_plane(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0)
    gnd.friction    = 0.6
    gnd.restitution = 0.1
    builder.add_shape(gnd)

    # ---- Shelf — static box (local_transform = world pose) ----
    shelf_shape = nv.CollisionShape.make_box(
        np.array([SHELF_HX, SHELF_HY, SHELF_HZ], dtype=np.float32), -1)
    shelf_tf = nv.Transform()
    shelf_tf.position = np.array([SHELF_CX, SHELF_CY, SHELF_CZ], dtype=np.float32)
    shelf_shape.local_transform = shelf_tf
    shelf_shape.friction    = 0.6
    shelf_shape.restitution = 0.05
    builder.add_shape(shelf_shape)

    # ----------------------------------------------------------------
    # Arm articulation
    # Link 0: Fixed anchor (pedestal)
    # Link 1: Revolute Y (sweeps in XZ plane)
    # ----------------------------------------------------------------
    art = nv.Articulation()
    joints_list = []
    bodies_list = []

    # Link 0 — Fixed pedestal
    j0 = nv.Joint()
    j0.type   = nv.JointType.Fixed
    j0.parent = -1
    tf0 = nv.Transform()
    tf0.position = np.array([0.0, PIVOT_Y, 0.0], dtype=np.float32)
    j0.parent_to_joint = tf0
    joints_list.append(j0)
    b0 = nv.RigidBody()
    b0.mass    = 1.0e6
    b0.inertia = np.diag([1e6, 1e6, 1e6]).astype(np.float32)
    b0.com     = np.zeros(3, dtype=np.float32)
    bodies_list.append(b0)

    # Link 1 — Revolute Y arm
    j1 = nv.Joint()
    j1.type   = nv.JointType.Revolute
    j1.axis   = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # Y axis: horizontal sweep
    j1.parent = 0
    ptj1 = nv.Transform()
    ptj1.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # joint at anchor
    j1.parent_to_joint = ptj1
    joints_list.append(j1)
    b1 = nv.RigidBody()
    b1.mass    = ARM_MASS
    b1.com     = np.array([ARM_HALF, 0.0, 0.0], dtype=np.float32)
    b1.inertia = np.diag(
        _box_inertia(ARM_MASS, ARM_HALF, ARM_HY, ARM_HZ)
    ).astype(np.float32)
    bodies_list.append(b1)

    art.joints = joints_list
    art.bodies = bodies_list
    art.angular_damping = 0.05
    art.linear_damping  = 0.02
    art.build_spatial_inertias()

    art_idx = builder.add_articulation(art)

    # Pedestal collision shape (visual pillar from y=0 to PIVOT_Y)
    ped_half_y = PIVOT_Y / 2.0
    ped_tf = nv.Transform()
    ped_tf.position = np.array([0.0, -ped_half_y, 0.0], dtype=np.float32)
    ped_shape = nv.CollisionShape.make_box(
        np.array([0.08, ped_half_y, 0.08], dtype=np.float32),
        0, ped_tf, friction=0.5, restitution=0.1,
        art_idx=art_idx, link_idx=0)
    builder.add_shape(ped_shape)

    # Arm collision shape (box along +X from pivot)
    arm_tf = nv.Transform()
    arm_tf.position = np.array([ARM_HALF, 0.0, 0.0], dtype=np.float32)
    arm_shape = nv.CollisionShape.make_box(
        np.array([ARM_HALF, ARM_HY, ARM_HZ], dtype=np.float32),
        0, arm_tf, friction=0.4, restitution=0.1,
        art_idx=art_idx, link_idx=1)
    builder.add_shape(arm_shape)

    # ----------------------------------------------------------------
    # Rigid boxes on shelf
    # ----------------------------------------------------------------
    box_body_indices = []
    for i in range(N_BOXES):
        bz = BOX_Z_START + i * BOX_Z_STEP

        rb = nv.RigidBody()
        rb.mass = BOX_MASS
        rb.inertia = np.diag(_box_inertia(BOX_MASS, BOX_H, BOX_H, BOX_H)).astype(np.float32)
        rb.linear_damping  = 0.01
        rb.angular_damping = 0.04

        tf = nv.Transform()
        tf.position = np.array([BOX_X, BOX_Y, bz], dtype=np.float32)

        body_idx = builder.add_body(rb, tf)
        box_body_indices.append(body_idx)

        sh = nv.CollisionShape.make_box(
            np.array([BOX_H, BOX_H, BOX_H], dtype=np.float32),
            body_idx, friction=0.50, restitution=0.20)
        builder.add_shape(sh)

    # Build model and world
    model = builder.build()
    world = nv.World(model, multibody_settings=solver_settings)
    world.set_gravity(gravity)

    # Initial state: arm pointing in +X direction, at rest
    q0  = np.zeros(1, dtype=np.float32)
    qd0 = np.zeros(1, dtype=np.float32)
    world.state.set_q(art_idx, q0)
    world.state.set_qd(art_idx, qd0)

    return world, art, art_idx, box_body_indices


# ---------------------------------------------------------------------------
# Headless run + sanity checks
# ---------------------------------------------------------------------------

def run():
    print("=" * 62)
    print("NovaPhy: Motor Arm Demo – Revolute Y + Torque Motor")
    print(f"  Arm: L={ARM_L} m, pivot y={PIVOT_Y} m, motor ω={MOTOR_VEL} r/s")
    print(f"  Shelf at y={SHELF_CY:.2f} m,  {N_BOXES} boxes on top")
    print("=" * 62)

    world, art, art_idx, box_body_indices = build_world()

    dt        = 1.0 / 60.0
    num_steps = int(5.0 / dt)

    initial_box_y = [float(np.array(world.state.transforms[bi].position)[1])
                     for bi in box_body_indices]

    print(f"\n{'Step':>5}  {'t':>6}  {'q_arm':>9}  {'qdot':>8}  "
          f"{'box0_y':>8}  {'box_knocked':>11}  {'nc':>5}")
    print("-" * 65)

    first_sweep_t   = None
    boxes_knocked   = 0
    peak_box_height = [iy for iy in initial_box_y]

    for step in range(num_steps):
        # Compute motor torque and step with control
        q_arm  = float(np.array(world.state.q[art_idx])[0])
        qd_arm = float(np.array(world.state.qd[art_idx])[0])
        tau = _compute_motor_torque(q_arm, qd_arm, MOTOR_VEL, MOTOR_KP, MOTOR_KD)

        control = nv.Control()
        control.articulation_joint_forces = [np.array([tau], dtype=np.float32)]
        world.step_with_control(world.state, control, dt)

        t   = (step + 1) * dt
        q   = np.array(world.state.q[art_idx])
        qd  = np.array(world.state.qd[art_idx])
        nc  = len(world.multibody_contacts) + len(world.contacts)

        for i, bi in enumerate(box_body_indices):
            by = float(np.array(world.state.transforms[bi].position)[1])
            if by > peak_box_height[i]:
                peak_box_height[i] = by

        knocked_now = sum(
            1 for i, bi in enumerate(box_body_indices)
            if float(np.array(world.state.transforms[bi].position)[1]) < initial_box_y[i] - 0.05
        )
        if knocked_now > boxes_knocked:
            boxes_knocked = knocked_now
            if first_sweep_t is None and knocked_now >= 1:
                first_sweep_t = t
                print(f"\n  >>> Arm swept first box off shelf at t={t:.4f}s  "
                      f"q={np.degrees(q[0]):+.1f}°  nc={nc}\n")

        b0_y = float(np.array(world.state.transforms[box_body_indices[0]].position)[1])
        if step % int(0.5 / dt) == 0 or step == num_steps - 1:
            print(f"{step:5d}  {t:6.3f}  "
                  f"{np.degrees(q[0]):+9.2f}  {qd[0]:+8.3f}  "
                  f"{b0_y:8.3f}  {knocked_now:11d}  {nc:5d}")

    q_f = np.array(world.state.q[art_idx])
    qd_f = np.array(world.state.qd[art_idx])
    print("\n" + "=" * 62)
    print("Final state (t=5s)")
    print("=" * 62)
    print(f"  Arm angle: {np.degrees(q_f[0]):+.2f}°   speed: {qd_f[0]:+.3f} r/s")
    for i, bi in enumerate(box_body_indices):
        p = np.array(world.state.transforms[bi].position)
        print(f"  box[{i}]: pos=({p[0]:+.3f}, {p[1]:.3f}, {p[2]:+.3f})")

    print("\nSanity checks:")

    assert not np.isnan(q_f).any(), "FAIL: NaN in arm state"
    print("  [PASS] No NaN in arm state")

    assert abs(qd_f[0]) > 0.1, \
        f"FAIL: Arm not rotating (ω={qd_f[0]:+.3f} r/s)"
    print(f"  [PASS] Arm rotating at ω={qd_f[0]:+.3f} r/s")

    assert first_sweep_t is not None, "FAIL: No boxes knocked off shelf"
    print(f"  [PASS] First box swept off at t={first_sweep_t:.4f}s")

    assert boxes_knocked >= N_BOXES // 2, \
        f"FAIL: Only {boxes_knocked}/{N_BOXES} boxes knocked off"
    print(f"  [PASS] {boxes_knocked}/{N_BOXES} boxes knocked off shelf")

    for i, bi in enumerate(box_body_indices):
        p = np.array(world.state.transforms[bi].position)
        assert not np.isnan(p).any(), f"FAIL: NaN in box {i}"
    print("  [PASS] No NaN in box positions")

    print("\nDemo completed successfully.")


# ---------------------------------------------------------------------------
# Interactive Polyscope visualization
# ---------------------------------------------------------------------------

def run_visual():
    world, art, art_idx, box_body_indices = build_world()

    ps.init()
    ps.set_program_name("NovaPhy – Motor Arm Shelf Sweep")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")

    # Ground
    gv, gf = make_ground_plane_mesh(size=10.0, offset=0.0)
    ps.register_surface_mesh("ground", gv, gf).set_color((0.42, 0.42, 0.42))

    # Shelf visual (static — never moves)
    shelf_v0, shelf_f = make_box_mesh(
        np.array([SHELF_HX, SHELF_HY, SHELF_HZ], dtype=np.float32))
    ps_shelf = ps.register_surface_mesh(
        "shelf",
        shelf_v0 + np.array([SHELF_CX, SHELF_CY, SHELF_CZ], dtype=np.float32),
        shelf_f)
    ps_shelf.set_color((0.60, 0.48, 0.32))   # wood brown

    # Pedestal column visual (static — never moves)
    ped_half_y = PIVOT_Y / 2.0
    ped_v0, ped_f = make_box_mesh(
        np.array([0.08, ped_half_y, 0.08], dtype=np.float32))
    ps.register_surface_mesh(
        "pedestal",
        ped_v0 + np.array([0.0, ped_half_y, 0.0], dtype=np.float32),
        ped_f).set_color((0.28, 0.28, 0.28))

    # Arm mesh
    arm_v0, arm_f = make_box_mesh(
        np.array([ARM_HALF, ARM_HY, ARM_HZ], dtype=np.float32))
    ps_arm = ps.register_surface_mesh("arm", arm_v0, arm_f)
    ps_arm.set_color((0.75, 0.75, 0.80))   # metallic grey

    # Box meshes
    box_v0, box_f = make_box_mesh(
        np.array([BOX_H, BOX_H, BOX_H], dtype=np.float32))
    BOX_COLORS = [
        (0.92, 0.25, 0.25),
        (0.92, 0.60, 0.15),
        (0.85, 0.85, 0.15),
        (0.30, 0.80, 0.30),
        (0.20, 0.70, 0.90),
        (0.55, 0.25, 0.85),
        (0.90, 0.40, 0.65),
    ]
    ps_boxes = []
    for i in range(N_BOXES):
        m = ps.register_surface_mesh(f"box_{i}", box_v0, box_f)
        m.set_color(BOX_COLORS[i % len(BOX_COLORS)])
        ps_boxes.append(m)

    dt       = 1.0 / 60.0
    substeps = 10
    sim_dt   = dt / substeps

    paused   = [True]
    show_cp  = [True]
    step_ctr = [0]

    _arm_offset = np.array([ARM_HALF, 0.0, 0.0], dtype=np.float32)

    def _update_meshes():
        q = np.array(world.state.q[art_idx])
        tfs = nv.forward_kinematics(art, q)

        # Arm (link 1)
        tf1  = tfs[1]
        pos  = np.array(tf1.position, dtype=np.float32)
        rot  = quat_to_rotation_matrix(np.array(tf1.rotation, dtype=np.float32))
        centre = pos + rot @ _arm_offset
        ps_arm.update_vertex_positions((arm_v0 @ rot.T) + centre)

        # Boxes
        for i, bi in enumerate(box_body_indices):
            tf  = world.state.transforms[bi]
            pos = np.array(tf.position, dtype=np.float32)
            rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
            ps_boxes[i].update_vertex_positions((box_v0 @ rot.T) + pos)

        # Contacts
        if show_cp[0]:
            contacts = list(world.multibody_contacts) + list(world.contacts)
            if len(contacts) > 0:
                cp_pos = np.array([np.array(c.position) for c in contacts],
                                  dtype=np.float32)
                pc = ps.register_point_cloud("contacts", cp_pos)
                pc.set_radius(0.012, relative=False)
                pc.set_color((1.0, 1.0, 0.0))
            elif ps.has_point_cloud("contacts"):
                ps.remove_point_cloud("contacts")
        elif ps.has_point_cloud("contacts"):
            ps.remove_point_cloud("contacts")

    _update_meshes()

    def callback():
        psim.PushItemWidth(155)
        psim.TextUnformatted("=== Motor Arm ===")
        _, paused[0]  = psim.Checkbox("Pause", paused[0])
        psim.SameLine()
        if psim.Button("Step") and paused[0]:
            q_arm  = float(np.array(world.state.q[art_idx])[0])
            qd_arm = float(np.array(world.state.qd[art_idx])[0])
            tau = _compute_motor_torque(q_arm, qd_arm, MOTOR_VEL, MOTOR_KP, MOTOR_KD)
            control = nv.Control()
            control.articulation_joint_forces = [np.array([tau], dtype=np.float32)]
            world.step_with_control(world.state, control, sim_dt)
            step_ctr[0] += 1
            _update_meshes()
        _, show_cp[0] = psim.Checkbox("Contacts", show_cp[0])
        psim.Separator()

        t   = step_ctr[0] * sim_dt
        q   = np.array(world.state.q[art_idx])
        qd  = np.array(world.state.qd[art_idx])
        nc  = len(world.multibody_contacts) + len(world.contacts)

        psim.TextUnformatted(f"t = {t:.3f} s    contacts = {nc}")
        psim.TextUnformatted(f"Arm  q  = {np.degrees(q[0]):+7.2f}°")
        psim.TextUnformatted(f"Arm  ω  = {qd[0]:+7.3f} r/s  (target {MOTOR_VEL})")
        psim.Separator()
        knocked = sum(
            1 for i, bi in enumerate(box_body_indices)
            if float(np.array(world.state.transforms[bi].position)[1]) < BOX_Y - 0.05
        )
        psim.TextUnformatted(f"Boxes knocked: {knocked}/{N_BOXES}")
        psim.Separator()
        for i, bi in enumerate(box_body_indices):
            by = float(np.array(world.state.transforms[bi].position)[1])
            psim.TextUnformatted(f"  b{i}: y={by:.3f}")
        psim.PopItemWidth()

        if not paused[0]:
            for _ in range(substeps):
                q_arm  = float(np.array(world.state.q[art_idx])[0])
                qd_arm = float(np.array(world.state.qd[art_idx])[0])
                tau = _compute_motor_torque(q_arm, qd_arm, MOTOR_VEL, MOTOR_KP, MOTOR_KD)
                control = nv.Control()
                control.articulation_joint_forces = [np.array([tau], dtype=np.float32)]
                world.step_with_control(world.state, control, sim_dt)
            step_ctr[0] += substeps
            _update_meshes()

    ps.set_user_callback(callback)
    ps.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--headless" in sys.argv or not HAS_POLYSCOPE:
        run()
    else:
        run_visual()
