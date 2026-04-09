"""Demo: Multi-link chain wrecking ball smashing a rigid-body tower.

Scene
-----
- Ground plane at y = 0
- 10-link chain pendulum anchored at (0, 10, 0):
    Link 0: Fixed anchor (very heavy)
    Links 1-9: Revolute Z, thin rods 0.6 m each
    Link 10: Revolute Z, heavy sphere ball R=0.4 m (4 kg)
- Tower: 3 columns (along Z) × 5 rows, rigid boxes at x = 1 m
- Initial chain angle q = [π/3, 0, 0, ..., 0] — chain pulled to the LEFT side

Physics sequence
----------------
1. Ball released from the left side; entire chain swings right under gravity.
2. Near t ≈ 1.0 s the ball sweeps through the tower, scattering boxes.
3. Boxes topple/fly off and land on the ground.
4. Chain continues to oscillate with damping.

Contact paths exercised
-----------------------
- link  → rb      [case 4d] (ball smashes tower boxes)
- rb    → rb      [case 4e] (boxes knock each other)
- rb    → static  [case 4c] (boxes land on ground)
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

from novaphy.viz import (make_box_mesh, make_sphere_mesh,
                         make_ground_plane_mesh, quat_to_rotation_matrix)

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

ANCHOR_Y   = 3       # pivot height (m)

ROD_HALF   = 0.35     # half-length of each rod segment (= 0.6 m total per rod)
ROD_MASS   = 0.05       # kg per rod link
ROD_THICK  = 0.025     # half-thickness of rod shape (visual/collision)

BALL_R     = 0.4      # sphere radius (m)
BALL_MASS  = 1.5       # kg

# Chain structure: Fixed + 4 Revolute
N_REVOLUTE = 4                 # links 1-4 (rods)
CHAIN_LEN  = 2 * ROD_HALF * N_REVOLUTE + 0.0   # distance from anchor to ball ≈ 6.0 m

INIT_Q0    = np.pi / 3.0        # 60° — first revolute angle at release

# Joint limits (prevent chain from folding back on itself)
JOINT_LOWER_LIMIT = -np.pi
JOINT_UPPER_LIMIT = np.pi

# Tower
TOWER_X    = 1       # x position of tower (right of pivot)
BOX_H      = 0.15      # half-ext of each cube box (m) → 0.30 m cubes
BOX_MASS   = 0.6       # kg
N_COLS_Z   = 3         # columns along Z
N_ROWS_Y   = 5         # rows stacked vertically
TOWER_Z_SPACING = 0.38 # centre-to-centre spacing in Z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_inertia(mass, hx, hy, hz):
    lx, ly, lz = 2*hx, 2*hy, 2*hz
    return np.array([
        mass/12.0*(ly**2+lz**2),
        mass/12.0*(lx**2+lz**2),
        mass/12.0*(lx**2+ly**2),
    ], dtype=np.float32)


def _rod_inertia(mass, half_len, thick):
    """Inertia of a thin rod about its centre (COM)."""
    l, t = 2*half_len, 2*thick
    return np.array([
        mass/12.0*(t**2+t**2),
        mass/12.0*(l**2+t**2),
        mass/12.0*(l**2+t**2),
    ], dtype=np.float32)


def _sphere_inertia(mass, r):
    I = 2.0/5.0 * mass * r**2
    return np.array([I, I, I], dtype=np.float32)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_world():
    builder = nv.ModelBuilder()

    solver_settings = nv.MultiBodySolverSettings()
    solver_settings.num_iterations = 50
    solver_settings.erp            = 0.2
    solver_settings.linear_slop    = 0.002
    solver_settings.restitution_threshold = 0.5

    gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
    builder.set_gravity(gravity)

    # ---- Ground ----
    gnd = nv.CollisionShape.make_plane(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0)
    gnd.friction    = 0.6
    gnd.restitution = 0.1
    builder.add_shape(gnd)

    # ----------------------------------------------------------------
    # Chain pendulum articulation
    # Link 0: Fixed (heavy anchor)
    # Links 1-3: Revolute Z (rods)
    # Link 4: Revolute Z (ball)
    # ----------------------------------------------------------------
    art = nv.Articulation()
    joints_list = []
    bodies_list = []

    # Link 0 — Fixed anchor
    j0 = nv.Joint()
    j0.type   = nv.JointType.Fixed
    j0.parent = -1
    tf0 = nv.Transform()
    tf0.position = np.array([0.0, ANCHOR_Y, 0.0], dtype=np.float32)
    j0.parent_to_joint = tf0
    joints_list.append(j0)
    b0 = nv.RigidBody()
    b0.mass    = 1.0e6
    b0.inertia = np.diag([1.0e6, 1.0e6, 1.0e6]).astype(np.float32)
    b0.com     = np.zeros(3, dtype=np.float32)
    bodies_list.append(b0)

    # Links 1-(N_REVOLUTE-1) — thin revolute rods
    for i in range(1, N_REVOLUTE):
        j = nv.Joint()
        j.type   = nv.JointType.Revolute
        j.axis   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        j.parent = i - 1
        j.limit_enabled = True
        j.lower_limit = float(JOINT_LOWER_LIMIT)
        j.upper_limit = float(JOINT_UPPER_LIMIT)
        ptj = nv.Transform()
        if i == 1:
            ptj.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # joint at anchor
        else:
            ptj.position = np.array([0.0, -2.0 * ROD_HALF, 0.0], dtype=np.float32)
        j.parent_to_joint = ptj
        joints_list.append(j)

        b = nv.RigidBody()
        b.mass    = ROD_MASS
        b.com     = np.array([0.0, -ROD_HALF, 0.0], dtype=np.float32)
        b.inertia = np.diag(_rod_inertia(ROD_MASS, ROD_HALF, ROD_THICK)).astype(np.float32)
        bodies_list.append(b)

    # Last link — sphere ball
    j_last = nv.Joint()
    j_last.type   = nv.JointType.Revolute
    j_last.axis   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    j_last.parent = N_REVOLUTE - 1
    j_last.limit_enabled = True
    j_last.lower_limit = float(JOINT_LOWER_LIMIT)
    j_last.upper_limit = float(JOINT_UPPER_LIMIT)
    ptj_last = nv.Transform()
    ptj_last.position = np.array([0.0, -2.0 * ROD_HALF, 0.0], dtype=np.float32)
    j_last.parent_to_joint = ptj_last
    joints_list.append(j_last)
    b_last = nv.RigidBody()
    b_last.mass    = BALL_MASS
    b_last.com     = np.zeros(3, dtype=np.float32)
    b_last.inertia = np.diag(_sphere_inertia(BALL_MASS, BALL_R)).astype(np.float32)
    bodies_list.append(b_last)

    art.joints = joints_list
    art.bodies = bodies_list
    art.angular_damping = 0.02
    art.linear_damping  = 0.01
    art.build_spatial_inertias()

    art_idx = builder.add_articulation(art)

    # Collision shapes for rods (links 1 to N_REVOLUTE-1)
    for link_i in range(1, N_REVOLUTE):
        rod_tf = nv.Transform()
        rod_tf.position = np.array([0.0, -ROD_HALF, 0.0], dtype=np.float32)
        rod_shape = nv.CollisionShape.make_box(
            np.array([ROD_THICK, ROD_HALF, ROD_THICK], dtype=np.float32),
            0, rod_tf, friction=0.3, restitution=0.1,
            art_idx=art_idx, link_idx=link_i)
        builder.add_shape(rod_shape)

    # Ball (last link) — sphere
    ball_shape = nv.CollisionShape.make_sphere(BALL_R, 0,
        friction=0.5, restitution=0.3,
        art_idx=art_idx, link_idx=N_REVOLUTE)
    builder.add_shape(ball_shape)

    # ----------------------------------------------------------------
    # Tower of rigid boxes
    # 3 columns (Z) × 5 rows (Y)
    # ----------------------------------------------------------------
    z_positions = [
        (k - (N_COLS_Z - 1) / 2.0) * TOWER_Z_SPACING
        for k in range(N_COLS_Z)
    ]
    box_body_indices = []

    for row in range(N_ROWS_Y):
        for col_z in z_positions:
            bx = TOWER_X
            by = BOX_H + row * 2.0 * BOX_H
            bz = col_z

            rb = nv.RigidBody()
            rb.mass = BOX_MASS
            rb.inertia = np.diag(_box_inertia(BOX_MASS, BOX_H, BOX_H, BOX_H)).astype(np.float32)
            rb.linear_damping  = 0.01
            rb.angular_damping = 0.04

            tf = nv.Transform()
            tf.position = np.array([bx, by, bz], dtype=np.float32)

            body_idx = builder.add_body(rb, tf)
            box_body_indices.append(body_idx)

            sh = nv.CollisionShape.make_box(
                np.array([BOX_H, BOX_H, BOX_H], dtype=np.float32),
                body_idx, friction=0.55, restitution=0.25)
            builder.add_shape(sh)

    # Build model and world
    model = builder.build()
    world = nv.World(model, multibody_settings=solver_settings)
    world.set_gravity(gravity)

    # Initial state: only first joint at INIT_Q0, others at 0, qd = 0
    q0  = np.zeros(N_REVOLUTE, dtype=np.float32)
    q0[0] = INIT_Q0
    qd0 = np.zeros(N_REVOLUTE, dtype=np.float32)
    world.state.set_q(art_idx, q0)
    world.state.set_qd(art_idx, qd0)

    return world, art, art_idx, box_body_indices


# ---------------------------------------------------------------------------
# Headless run + sanity checks
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("NovaPhy: Wrecking Ball Chain Demo")
    print(f"  Chain: Fixed anchor + {N_REVOLUTE} revolute links")
    print(f"  Ball:  sphere R={BALL_R} m, mass={BALL_MASS} kg")
    print(f"  Tower: {N_COLS_Z}×{N_ROWS_Y} rigid boxes at x={TOWER_X} m")
    print("=" * 60)

    world, art, art_idx, box_body_indices = build_world()

    dt        = 1.0 / 120.0
    num_steps = int(8.0 / dt)

    print(f"\n{'Step':>5}  {'t':>6}  {'q1':>8}  {'q2':>8}  {'q_last':>8}  "
          f"{'ball_x':>8}  {'ball_y':>8}  {'nc':>5}")
    print("-" * 68)

    first_impact_t     = None
    peak_box_scatter   = 0.0
    initial_box_pos    = [np.array(world.state.transforms[bi].position).copy()
                          for bi in box_body_indices]
    BASE_CONTACTS      = 0

    for step in range(num_steps):
        world.step(dt)
        t  = (step + 1) * dt
        q  = np.array(world.state.q[art_idx])
        nc = len(world.multibody_contacts) + len(world.contacts)

        # FK to get ball world position (last link)
        tfs     = nv.forward_kinematics(art, q)
        ball_tf = tfs[N_REVOLUTE]
        ball_pos = np.array(ball_tf.position)

        if first_impact_t is None and ball_pos[0] > TOWER_X - BALL_R - BOX_H:
            if nc > BASE_CONTACTS + 3:
                first_impact_t = t
                print(f"\n  >>> IMPACT at t={t:.4f}s  "
                      f"ball=({ball_pos[0]:.3f}, {ball_pos[1]:.3f})  nc={nc}\n")

        # Track max box displacement
        for i, (bi, ip) in enumerate(zip(box_body_indices, initial_box_pos)):
            p = np.array(world.state.transforms[bi].position)
            disp = float(np.linalg.norm(p[[0, 2]] - ip[[0, 2]]))
            if disp > peak_box_scatter:
                peak_box_scatter = disp

        if step % int(0.5 / dt) == 0 or step == num_steps - 1:
            print(f"{step:5d}  {t:6.3f}  "
                  f"{np.degrees(q[0]):+8.2f}  {np.degrees(q[1]):+8.2f}  "
                  f"{np.degrees(q[-1]):+8.2f}  "
                  f"{ball_pos[0]:8.3f}  {ball_pos[1]:8.3f}  {nc:5d}")

    q_f = np.array(world.state.q[art_idx])
    print("\n" + "=" * 60)
    print("Final state (t=8s)")
    print("=" * 60)
    print(f"  Chain angles: q1={np.degrees(q_f[0]):+.2f}°  "
          f"q2={np.degrees(q_f[1]):+.2f}°  q{N_REVOLUTE}={np.degrees(q_f[-1]):+.2f}°")
    print(f"  Peak box scatter: {peak_box_scatter:.3f} m")
    settled = sum(1 for bi in box_body_indices
                  if abs(np.array(world.state.linear_velocities[bi])[1]) < 0.1)
    print(f"  Boxes settled: {settled}/{len(box_body_indices)}")

    print("\nSanity checks:")

    assert not np.isnan(q_f).any(), "FAIL: NaN in chain state"
    print("  [PASS] No NaN in chain state")

    for i, bi in enumerate(box_body_indices):
        p = np.array(world.state.transforms[bi].position)
        assert not np.isnan(p).any(), f"FAIL: NaN in box {i}"
    print("  [PASS] No NaN in box positions")

    assert first_impact_t is not None, "FAIL: Ball never reached the tower"
    print(f"  [PASS] Ball reached tower at t={first_impact_t:.4f}s")

    assert peak_box_scatter > 0.2, \
        f"FAIL: Boxes not scattered (max={peak_box_scatter:.3f} m)"
    print(f"  [PASS] Boxes scattered: max horizontal disp = {peak_box_scatter:.3f} m")

    for i, bi in enumerate(box_body_indices):
        p = np.array(world.state.transforms[bi].position)
        assert p[1] >= -1.0, f"FAIL: Box {i} fell through ground y={p[1]:.3f}"
    print("  [PASS] All boxes above y=-1.0 (no ground penetration explosion)")

    print("\nDemo completed successfully.")


# ---------------------------------------------------------------------------
# Interactive Polyscope visualization
# ---------------------------------------------------------------------------

def run_visual():
    world, art, art_idx, box_body_indices = build_world()

    ps.init()
    ps.set_program_name("NovaPhy – Wrecking Ball Chain")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")

    # Ground
    gv, gf = make_ground_plane_mesh(size=14.0, offset=0.0)
    ps.register_surface_mesh("ground", gv, gf).set_color((0.42, 0.42, 0.42))

    # Anchor visual (small cube)
    anc_v, anc_f = make_box_mesh(np.array([0.08, 0.08, 0.08], dtype=np.float32))
    ps_anc = ps.register_surface_mesh("anchor", anc_v + np.array([0, ANCHOR_Y, 0]),
                                      anc_f)
    ps_anc.set_color((0.3, 0.3, 0.3))

    # Rod meshes (links 1 to N_REVOLUTE-1)
    rod_v0, rod_f = make_box_mesh(
        np.array([ROD_THICK, ROD_HALF, ROD_THICK], dtype=np.float32))
    ps_rods = []
    for i in range(N_REVOLUTE - 1):
        t = i / max(N_REVOLUTE - 2, 1)
        col = (0.50 + 0.15 * t, 0.40 + 0.15 * t, 0.25 + 0.10 * t)
        m = ps.register_surface_mesh(f"rod_{i+1}", rod_v0, rod_f)
        m.set_color(col)
        ps_rods.append(m)

    # Ball (last link) — sphere mesh
    ball_v0, ball_f = make_sphere_mesh(BALL_R, n_lat=16, n_lon=32)
    ps_ball = ps.register_surface_mesh("ball", ball_v0, ball_f)
    ps_ball.set_color((0.15, 0.15, 0.15))   # dark iron

    # Tower boxes
    box_v0, box_f = make_box_mesh(np.array([BOX_H, BOX_H, BOX_H], dtype=np.float32))
    ps_boxes = []
    for i in range(len(box_body_indices)):
        row = i // N_COLS_Z
        t_c = row / max(N_ROWS_Y - 1, 1)
        color = (0.9, 0.3 + 0.5 * t_c, 0.1)
        m = ps.register_surface_mesh(f"box_{i}", box_v0, box_f)
        m.set_color(color)
        ps_boxes.append(m)

    dt       = 1.0 / 60.0
    substeps = 10
    sim_dt   = dt / substeps

    paused   = [True]
    show_cp  = [False]
    step_ctr = [0]

    _rod_offset = np.array([0.0, -ROD_HALF, 0.0], dtype=np.float32)

    def _update_meshes():
        q = np.array(world.state.q[art_idx])
        tfs = nv.forward_kinematics(art, q)

        # Rods
        for i, ps_rod in enumerate(ps_rods):
            tf  = tfs[i + 1]
            pos = np.array(tf.position, dtype=np.float32)
            rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
            centre = pos + rot @ _rod_offset
            ps_rod.update_vertex_positions((rod_v0 @ rot.T) + centre)

        # Ball (last link)
        tf_ball = tfs[N_REVOLUTE]
        pos_b   = np.array(tf_ball.position, dtype=np.float32)
        rot_b   = quat_to_rotation_matrix(np.array(tf_ball.rotation, dtype=np.float32))
        ps_ball.update_vertex_positions((ball_v0 @ rot_b.T) + pos_b)

        # Tower boxes
        for i, bi in enumerate(box_body_indices):
            tf  = world.state.transforms[bi]
            pos = np.array(tf.position, dtype=np.float32)
            rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
            ps_boxes[i].update_vertex_positions((box_v0 @ rot.T) + pos)

        # Contact points
        if show_cp[0]:
            contacts = list(world.multibody_contacts) + list(world.contacts)
            if len(contacts) > 0:
                cp_pos = np.array([np.array(c.position) for c in contacts],
                                  dtype=np.float32)
                pc = ps.register_point_cloud("contacts", cp_pos)
                pc.set_radius(0.015, relative=False)
                pc.set_color((1.0, 1.0, 0.0))
            elif ps.has_point_cloud("contacts"):
                ps.remove_point_cloud("contacts")
        elif ps.has_point_cloud("contacts"):
            ps.remove_point_cloud("contacts")

    _update_meshes()

    def callback():
        psim.PushItemWidth(160)
        psim.TextUnformatted("=== Wrecking Ball ===")
        _, paused[0]  = psim.Checkbox("Pause", paused[0])
        psim.SameLine()
        if psim.Button("Step") and paused[0]:
            world.step(sim_dt)
            step_ctr[0] += 1
            _update_meshes()
        _, show_cp[0] = psim.Checkbox("Contacts", show_cp[0])
        psim.Separator()

        t  = step_ctr[0] * sim_dt
        q  = np.array(world.state.q[art_idx])
        nc = len(world.multibody_contacts) + len(world.contacts)

        tfs = nv.forward_kinematics(art, q)
        ball_pos = np.array(tfs[N_REVOLUTE].position)

        psim.TextUnformatted(f"t = {t:.3f} s    contacts = {nc}")
        psim.Separator()
        psim.TextUnformatted("Chain angles (deg)")
        psim.TextUnformatted(f"  θ1={np.degrees(q[0]):+6.1f}")
        psim.TextUnformatted(f"  θ2={np.degrees(q[1]):+6.1f}")
        psim.TextUnformatted(f"  θ{N_REVOLUTE}={np.degrees(q[-1]):+6.1f}")
        psim.Separator()
        psim.TextUnformatted("Ball position")
        psim.TextUnformatted(f"  ({ball_pos[0]:+.3f}, {ball_pos[1]:+.3f}, {ball_pos[2]:+.3f})")
        psim.Separator()
        psim.TextUnformatted(f"Boxes: {len(box_body_indices)}")
        psim.PopItemWidth()

        if not paused[0]:
            for _ in range(substeps):
                world.step(sim_dt)
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
