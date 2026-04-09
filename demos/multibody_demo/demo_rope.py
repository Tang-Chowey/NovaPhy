"""Demo: Multi-link hinged rope/chain falling from 90 degrees (horizontal).

Scene
-----
- Ground plane at y = 0
- 15-link chain anchored at (0, 8, 0):
    Link 0: Fixed anchor (very heavy)
    Links 1-14: Revolute Z, thin rod segments 0.25 m each
- Initial chain angle q = [π/2, 0, 0, ..., 0] — chain starts HORIZONTAL (90°)

Physics sequence
----------------
1. Rope released from horizontal position under gravity.
2. Chain swings down, converting potential to kinetic energy.
3. Rope oscillates with damping, gradually settling.
4. Links may collide with each other during motion.

This demo demonstrates:
- Multi-body dynamics with many revolute joints
- Energy conservation (mostly) during swing
- Chain behavior under gravity
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

from novaphy.viz import (make_box_mesh, make_ground_plane_mesh,
                         quat_to_rotation_matrix)

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

ANCHOR_Y   = 8.0       # pivot height (m)

LINK_HALF  = 0.125     # half-length of each link segment (= 0.25 m total per link)
LINK_MASS  = 0.05      # kg per link
LINK_THICK = 0.05     # half-thickness of link shape (visual/collision)

# Chain structure: Fixed + 15 Revolute
N_LINKS    = 15        # number of revolute links (rope segments)
CHAIN_LEN  = 2 * LINK_HALF * N_LINKS  # total length ≈ 3.75 m

INIT_ANGLE = np.pi / 2.0  # 90° — horizontal release


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_inertia(mass, half_len, thick):
    """Inertia of a thin rod about its centre (COM)."""
    l, t = 2*half_len, 2*thick
    return np.array([
        mass/12.0*(t**2+t**2),
        mass/12.0*(l**2+t**2),
        mass/12.0*(l**2+t**2),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_world():
    builder = nv.ModelBuilder()

    solver_settings = nv.MultiBodySolverSettings()
    solver_settings.num_iterations = 50
    solver_settings.erp            = 0.2
    solver_settings.cfm            = 1e-5
    solver_settings.linear_slop    = 0.001
    solver_settings.restitution_threshold = 0.3

    gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
    builder.set_gravity(gravity)

    # ---- Ground ----
    gnd = nv.CollisionShape.make_plane(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0)
    gnd.friction    = 0.5
    gnd.restitution = 0.1
    builder.add_shape(gnd)

    # ----------------------------------------------------------------
    # Rope articulation
    # Link 0: Fixed (anchor)
    # Links 1-15: Revolute Z (rope segments)
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

    # Links 1-15 — revolute rope segments
    for i in range(1, N_LINKS + 1):
        j = nv.Joint()
        j.type   = nv.JointType.Revolute
        j.axis   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        j.parent = i - 1
        ptj = nv.Transform()
        if i == 1:
            ptj.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # joint at anchor
        else:
            ptj.position = np.array([0.0, -2.0 * LINK_HALF, 0.0], dtype=np.float32)
        j.parent_to_joint = ptj
        joints_list.append(j)

        b = nv.RigidBody()
        b.mass    = LINK_MASS
        b.com     = np.array([0.0, -LINK_HALF, 0.0], dtype=np.float32)
        b.inertia = np.diag(_link_inertia(LINK_MASS, LINK_HALF, LINK_THICK)).astype(np.float32)
        bodies_list.append(b)

    art.joints = joints_list
    art.bodies = bodies_list
    art.linear_damping  = 0.04
    art.angular_damping = 0.04
    art.build_spatial_inertias()

    art_idx = builder.add_articulation(art)

    # Collision shapes for each rope link (box collider offset to link COM)
    for link_i in range(1, N_LINKS + 1):
        link_tf = nv.Transform()
        link_tf.position = np.array([0.0, -LINK_HALF, 0.0], dtype=np.float32)
        link_shape = nv.CollisionShape.make_box(
            np.array([LINK_THICK, LINK_HALF, LINK_THICK], dtype=np.float32),
            0, link_tf, friction=0.5, restitution=0.0,
            art_idx=art_idx, link_idx=link_i)
        builder.add_shape(link_shape)

    # Build model and world
    model = builder.build()
    world = nv.World(model, multibody_settings=solver_settings)
    world.set_gravity(gravity)

    # Initial state: only first joint at INIT_ANGLE (90° = horizontal), others at 0, qd = 0
    q0  = np.zeros(N_LINKS, dtype=np.float32)
    q0[0] = INIT_ANGLE
    qd0 = np.zeros(N_LINKS, dtype=np.float32)
    world.state.set_q(art_idx, q0)
    world.state.set_qd(art_idx, qd0)

    return world, art, art_idx


# ---------------------------------------------------------------------------
# Headless run + sanity checks
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("NovaPhy: Hinged Rope Chain Demo")
    print(f"  Rope: Fixed anchor + {N_LINKS} revolute links")
    print(f"  Chain length: {CHAIN_LEN:.3f} m")
    print(f"  Initial angle: {np.degrees(INIT_ANGLE):.1f}° (horizontal)")
    print("=" * 60)

    world, art, art_idx = build_world()

    dt        = 1.0 / 240.0
    num_steps = int(10.0 / dt)

    print(f"\n{'Step':>5}  {'t':>6}  {'q1':>8}  {'q2':>8}  {'q_last':>8}  "
          f"{'tip_x':>8}  {'tip_y':>8}")
    print("-" * 57)

    min_tip_y  = float('inf')
    min_link_y = float('inf')

    for step in range(num_steps):
        world.step(dt)
        t  = (step + 1) * dt
        q  = np.array(world.state.q[art_idx])

        # FK to get tip world position (last link)
        tfs    = nv.forward_kinematics(art, q)
        tip_tf = tfs[N_LINKS]
        tip_pos = np.array(tip_tf.position)

        min_tip_y = min(min_tip_y, tip_pos[1])
        for li in range(1, N_LINKS + 1):
            min_link_y = min(min_link_y, np.array(tfs[li].position)[1])

        if step % int(0.25 / dt) == 0 or step == num_steps - 1:
            print(f"{step:5d}  {t:6.3f}  "
                  f"{np.degrees(q[0]):+8.2f}  {np.degrees(q[1]):+8.2f}  "
                  f"{np.degrees(q[-1]):+8.2f}  "
                  f"{tip_pos[0]:8.3f}  {tip_pos[1]:8.3f}")

    q_f = np.array(world.state.q[art_idx])
    print("\n" + "=" * 60)
    print("Final state (t=10s)")
    print("=" * 60)
    print(f"  Rope angles: q1={np.degrees(q_f[0]):+.2f}°  "
          f"q2={np.degrees(q_f[1]):+.2f}°  q{N_LINKS}={np.degrees(q_f[-1]):+.2f}°")

    tfs_f = nv.forward_kinematics(art, q_f)
    tip_pos_f = np.array(tfs_f[N_LINKS].position)
    print(f"  Tip position: ({tip_pos_f[0]:+.3f}, {tip_pos_f[1]:+.3f}, {tip_pos_f[2]:+.3f})")

    print("\nSanity checks:")

    assert not np.isnan(q_f).any(), "FAIL: NaN in rope state"
    print("  [PASS] No NaN in rope state")

    assert min_tip_y < ANCHOR_Y - 1.0, \
        f"FAIL: Tip never swung down (min_y={min_tip_y:.3f})"
    print(f"  [PASS] Tip swung down from y={ANCHOR_Y:.3f} to min y={min_tip_y:.3f}")

    assert min_link_y > -0.1, \
        f"FAIL: Link penetrated ground (min_link_y={min_link_y:.3f})"
    print(f"  [PASS] No ground penetration (min_link_y={min_link_y:.3f})")

    for i in range(N_LINKS + 1):
        pos = np.array(tfs_f[i].position)
        assert np.linalg.norm(pos) < 100.0, \
            f"FAIL: Link {i} position exploded: {pos}"
    print("  [PASS] All links within reasonable bounds")

    print("\nDemo completed successfully.")


# ---------------------------------------------------------------------------
# Interactive Polyscope visualization
# ---------------------------------------------------------------------------

def run_visual():
    world, art, art_idx = build_world()

    ps.init()
    ps.set_program_name("NovaPhy – Hinged Rope Chain")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")

    # Ground
    gv, gf = make_ground_plane_mesh(size=10.0, offset=0.0)
    ps.register_surface_mesh("ground", gv, gf).set_color((0.42, 0.42, 0.42))

    # Anchor visual (small cube)
    anc_v, anc_f = make_box_mesh(np.array([0.08, 0.08, 0.08], dtype=np.float32))
    ps_anc = ps.register_surface_mesh("anchor", anc_v + np.array([0, ANCHOR_Y, 0]),
                                      anc_f)
    ps_anc.set_color((0.3, 0.3, 0.3))

    # Link meshes
    link_v0, link_f = make_box_mesh(
        np.array([LINK_THICK, LINK_HALF, LINK_THICK], dtype=np.float32))
    ps_links = []
    for i in range(N_LINKS):
        t = i / max(N_LINKS - 1, 1)
        col = (0.2 + 0.3 * t, 0.5 + 0.2 * t, 0.8 - 0.3 * t)
        m = ps.register_surface_mesh(f"link_{i+1}", link_v0, link_f)
        m.set_color(col)
        ps_links.append(m)

    dt       = 1.0 / 60.0
    substeps = 150
    sim_dt   = dt / substeps

    paused   = [True]
    show_cp  = [False]
    step_ctr = [0]

    _link_offset = np.array([0.0, -LINK_HALF, 0.0], dtype=np.float32)

    def _update_meshes():
        q = np.array(world.state.q[art_idx])
        tfs = nv.forward_kinematics(art, q)

        for i, ps_link in enumerate(ps_links):
            tf  = tfs[i + 1]
            pos = np.array(tf.position, dtype=np.float32)
            rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
            centre = pos + rot @ _link_offset
            ps_link.update_vertex_positions((link_v0 @ rot.T) + centre)

        if show_cp[0]:
            contacts = world.multibody_contacts
            if len(contacts) > 0:
                cp_pos = np.array([np.array(c.position) for c in contacts],
                                  dtype=np.float32)
                pc = ps.register_point_cloud("contacts", cp_pos)
                pc.set_radius(0.01, relative=False)
                pc.set_color((1.0, 1.0, 0.0))
            elif ps.has_point_cloud("contacts"):
                ps.remove_point_cloud("contacts")
        elif ps.has_point_cloud("contacts"):
            ps.remove_point_cloud("contacts")

    _update_meshes()

    def callback():
        psim.PushItemWidth(180)
        psim.TextUnformatted("=== Hinged Rope ===")
        _, paused[0]  = psim.Checkbox("Pause", paused[0])
        psim.SameLine()
        if psim.Button("Step") and paused[0]:
            for _ in range(substeps):
                world.step(sim_dt)
            step_ctr[0] += substeps
            _update_meshes()
        _, show_cp[0] = psim.Checkbox("Contacts", show_cp[0])
        psim.Separator()

        t  = step_ctr[0] * sim_dt
        q  = np.array(world.state.q[art_idx])
        nc = len(world.multibody_contacts)

        tfs = nv.forward_kinematics(art, q)
        tip_pos = np.array(tfs[N_LINKS].position)

        psim.TextUnformatted(f"t = {t:.3f} s    contacts = {nc}")
        psim.Separator()
        psim.TextUnformatted("Rope angles (deg)")
        psim.TextUnformatted(f"  θ1={np.degrees(q[0]):+6.1f}")
        psim.TextUnformatted(f"  θ2={np.degrees(q[1]):+6.1f}")
        psim.TextUnformatted(f"  θ{N_LINKS}={np.degrees(q[-1]):+6.1f}")
        psim.Separator()
        psim.TextUnformatted("Tip position")
        psim.TextUnformatted(f"  ({tip_pos[0]:+.3f}, {tip_pos[1]:+.3f}, {tip_pos[2]:+.3f})")
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
