"""Demo: Seesaw launcher – revolute-joint multibody colliding with rigid bodies.

Scene
-----
- Ground plane at y = 0
- Seesaw plank: 1-link revolute-joint multibody (2 kg, 2 m span × 0.1 m tall × 0.6 m deep)
  pivoted at (0, 0.5, 0), free to rotate around the Z axis
- 4 light rigid boxes (0.2 kg, 0.13 m cube) placed side-by-side on the left half at rest
- 1 heavy rigid box (5 kg, 0.16 m cube) dropped from y = 2.3 at x = +0.72

Physics sequence
----------------
1. Light boxes settle onto the seesaw surface (~0.1 s).
2. Heavy box free-falls for ~0.55 s and strikes the right side of the seesaw.
3. The revolute joint transfers angular impulse through the plank.
4. The left end swings upward, launching all four light boxes.
5. The seesaw right end strikes the ground, limiting rotation.
6. Light boxes fly off and land on the ground; the heavy box settles on the seesaw.

Contact paths exercised
-----------------------
- rb → multibody   (heavy box → seesaw):  body_a = rb,   body_b = link  [case 4d]
- multibody → rb   (seesaw → light boxes): body_a = rb,   body_b = link  [case 4d]
- rb → static      (all boxes → ground):  body_a = static, body_b = rb   [case 4c]
- rb → rb          (box-box collisions):  body_a/b = rb  [case 4e]
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
# Helpers
# ---------------------------------------------------------------------------

def box_inertia(mass, hx, hy, hz):
    lx, ly, lz = 2 * hx, 2 * hy, 2 * hz
    return np.array([
        mass / 12.0 * (ly**2 + lz**2),
        mass / 12.0 * (lx**2 + lz**2),
        mass / 12.0 * (lx**2 + ly**2),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

# Scene constants (shared by headless and visual)
PIVOT_Y    = 0.5          # pivot height above ground (m)
PLANK_HX   = 1.0          # seesaw half-length
PLANK_HY   = 0.05         # seesaw half-height
PLANK_HZ   = 0.30         # seesaw half-depth
PLANK_MASS = 2.0          # kg

LIGHT_H    = 0.065        # light box half-extent (m)
LIGHT_MASS = 0.20         # kg
N_LIGHT    = 4

HEAVY_H    = 0.08         # heavy box half-extent (m)
HEAVY_MASS = 5.0          # kg
HEAVY_X    = 0.72         # drop x (right of pivot)
HEAVY_Y0   = 2.3          # drop height (m)

# Expected contact height (plank top + heavy box bottom)
IMPACT_Y = PIVOT_Y + PLANK_HY + HEAVY_H


def build_world():
    """Construct World with seesaw + light boxes + heavy box."""
    builder = nv.ModelBuilder()

    solver_settings = nv.MultiBodySolverSettings()
    solver_settings.num_iterations  = 40
    solver_settings.erp             = 0.2
    solver_settings.linear_slop     = 0.003
    solver_settings.restitution_threshold = 0.3

    gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
    builder.set_gravity(gravity)

    # ---- Ground plane ----
    gnd = nv.CollisionShape.make_plane(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0)
    gnd.friction = 0.5
    gnd.restitution = 0.1
    builder.add_shape(gnd)

    # ---- Seesaw: 1-link revolute-joint multibody ----
    art = nv.Articulation()
    j = nv.Joint()
    j.type   = nv.JointType.Revolute
    j.axis   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    j.parent = -1
    pivot_tf = nv.Transform()
    pivot_tf.position = np.array([0.0, PIVOT_Y, 0.0], dtype=np.float32)
    j.parent_to_joint = pivot_tf
    art.joints = [j]

    plank_rb = nv.RigidBody()
    plank_rb.mass    = PLANK_MASS
    plank_rb.inertia = np.diag(
        box_inertia(PLANK_MASS, PLANK_HX, PLANK_HY, PLANK_HZ)
    ).astype(np.float32)
    plank_rb.com = np.zeros(3, dtype=np.float32)
    art.bodies = [plank_rb]
    art.build_spatial_inertias()

    art_idx = builder.add_articulation(art)

    plank_shape = nv.CollisionShape.make_box(
        np.array([PLANK_HX, PLANK_HY, PLANK_HZ], dtype=np.float32), 0,
        friction=0.45, restitution=0.15,
        art_idx=art_idx, link_idx=0)
    builder.add_shape(plank_shape)

    # ---- Light boxes: 4 rigid bodies on left half ----
    light_body_indices = []
    lx_positions = [-0.22, -0.42, -0.62, -0.80]
    light_y0 = PIVOT_Y + PLANK_HY + LIGHT_H + 0.003  # just above plank

    for i, lx in enumerate(lx_positions):
        rb = nv.RigidBody()
        rb.mass = LIGHT_MASS
        rb.inertia = np.diag(box_inertia(LIGHT_MASS, LIGHT_H, LIGHT_H, LIGHT_H)).astype(np.float32)
        rb.linear_damping  = 0.01
        rb.angular_damping = 0.05

        t = nv.Transform()
        t.position = np.array([lx, light_y0, 0.0], dtype=np.float32)
        body_idx = builder.add_body(rb, t)
        light_body_indices.append(body_idx)

        lb_shape = nv.CollisionShape.make_box(
            np.array([LIGHT_H, LIGHT_H, LIGHT_H], dtype=np.float32), body_idx,
            friction=0.45, restitution=0.25)
        builder.add_shape(lb_shape)

    # ---- Heavy box: dropped on right side ----
    rb_heavy = nv.RigidBody()
    rb_heavy.mass = HEAVY_MASS
    rb_heavy.inertia = np.diag(box_inertia(HEAVY_MASS, HEAVY_H, HEAVY_H, HEAVY_H)).astype(np.float32)
    rb_heavy.linear_damping  = 0.01
    rb_heavy.angular_damping = 0.05

    th = nv.Transform()
    th.position = np.array([HEAVY_X, HEAVY_Y0, 0.0], dtype=np.float32)
    heavy_body_idx = builder.add_body(rb_heavy, th)

    heavy_shape = nv.CollisionShape.make_box(
        np.array([HEAVY_H, HEAVY_H, HEAVY_H], dtype=np.float32), heavy_body_idx,
        friction=0.45, restitution=0.20)
    builder.add_shape(heavy_shape)

    # Build model and world
    model = builder.build()
    world = nv.World(model, multibody_settings=solver_settings)
    world.set_gravity(gravity)

    # Initial state for seesaw articulation
    world.state.set_q(art_idx, np.zeros(1, dtype=np.float32))
    world.state.set_qd(art_idx, np.zeros(1, dtype=np.float32))

    return world, art, art_idx, light_body_indices, heavy_body_idx


# ---------------------------------------------------------------------------
# Headless simulation + sanity checks
# ---------------------------------------------------------------------------

def run():
    print("=" * 64)
    print("NovaPhy: Seesaw Launcher – Revolute Multibody + Rigid Bodies")
    print("=" * 64)

    world, art, art_idx, light_body_indices, heavy_body_idx = build_world()

    t_impact_theory = np.sqrt(2.0 * (HEAVY_Y0 - IMPACT_Y) / 9.81)
    print(f"\nPivot at y={PIVOT_Y:.2f}m | Plank half-span={PLANK_HX:.2f}m | "
          f"mass={PLANK_MASS:.1f}kg")
    print(f"Light boxes: {N_LIGHT} × {LIGHT_MASS:.2f}kg, half-ext={LIGHT_H:.3f}m")
    print(f"Heavy box: {HEAVY_MASS:.1f}kg at ({HEAVY_X:.2f}, {HEAVY_Y0:.2f})")
    print(f"Theory impact at t ≈ {t_impact_theory:.3f}s\n")

    initial_light_y = [np.array(world.state.transforms[bi].position)[1]
                       for bi in light_body_indices]

    dt        = 1.0 / 480.0
    num_steps = int(5.0 / dt)   # 5 seconds

    print(f"{'Step':>6}  {'t':>6}  {'theta':>8}  {'omega':>8}  "
          f"{'heavy_y':>8}  {'n_cont':>6}  {'peak_lb_y':>9}")
    print("-" * 68)

    peak_theta    = 0.0
    peak_light_y  = [iy for iy in initial_light_y]
    impact_t      = None
    IMPACT_DETECT_Y = IMPACT_Y + 0.05

    for step in range(num_steps):
        world.step(dt)
        t = (step + 1) * dt

        theta  = np.array(world.state.q[art_idx])[0]
        omega  = np.array(world.state.qd[art_idx])[0]
        h_pos  = np.array(world.state.transforms[heavy_body_idx].position)
        n_cont = len(world.multibody_contacts) + len(world.contacts)

        if theta < peak_theta:
            peak_theta = theta

        for i, bi in enumerate(light_body_indices):
            ly = np.array(world.state.transforms[bi].position)[1]
            if ly > peak_light_y[i]:
                peak_light_y[i] = ly

        if impact_t is None and h_pos[1] <= IMPACT_DETECT_Y:
            impact_t = t
            print(f"\n  >>> IMPACT at t={t:.4f}s (theory {t_impact_theory:.3f}s, "
                  f"err {abs(t - t_impact_theory)*1000:.0f} ms)\n")

        if step % int(0.25 / dt) == 0 or step == num_steps - 1:
            max_lby = max(np.array(world.state.transforms[bi].position)[1]
                         for bi in light_body_indices)
            print(f"{step:6d}  {t:6.3f}  {theta:+8.4f}  {omega:+8.3f}  "
                  f"{h_pos[1]:8.4f}  {n_cont:6d}  {max_lby:9.4f}")

    # ---- Final state ----
    print("\n" + "=" * 64)
    print("Final state (t=5s)")
    print("=" * 64)
    q_s   = np.array(world.state.q[art_idx])
    qd_s  = np.array(world.state.qd[art_idx])
    hpos  = np.array(world.state.transforms[heavy_body_idx].position)
    print(f"  Seesaw angle  : {q_s[0]:+.4f} rad  ({np.degrees(q_s[0]):+.2f}°)")
    print(f"  Seesaw omega  : {qd_s[0]:+.4f} rad/s")
    print(f"  Heavy box pos : {hpos}")
    for i, bi in enumerate(light_body_indices):
        lp = np.array(world.state.transforms[bi].position)
        print(f"  LightBox[{i}] pos: [{lp[0]:+.3f}, {lp[1]:.4f}, {lp[2]:+.3f}]")

    # ---- Sanity checks ----
    print("\nSanity checks:")

    assert impact_t is not None, "FAIL: Heavy box never contacted seesaw or light boxes"
    print(f"  [PASS] Impact detected at t={impact_t:.4f}s")

    assert peak_theta < -0.15, \
        f"FAIL: Seesaw did not rotate significantly (peak θ={peak_theta:.4f} rad)"
    print(f"  [PASS] Seesaw peak angle = {peak_theta:+.4f} rad  ({np.degrees(peak_theta):+.1f}°)")

    max_launch = max(peak_light_y[i] - initial_light_y[i] for i in range(N_LIGHT))
    assert max_launch > 0.08, \
        f"FAIL: Light boxes not launched (max rise = {max_launch:.3f} m)"
    print(f"  [PASS] Light boxes launched: max rise = {max_launch:.3f} m")

    for i, bi in enumerate(light_body_indices):
        lp = np.array(world.state.transforms[bi].position)
        assert not np.isnan(lp).any(), f"FAIL: LightBox[{i}] position is NaN"
    assert not np.isnan(np.array(world.state.q[art_idx])).any(), "FAIL: Seesaw state is NaN"
    assert not np.isnan(np.array(world.state.transforms[heavy_body_idx].position)).any(), \
        "FAIL: Heavy box is NaN"
    print(f"  [PASS] No NaN in any body")

    print("\nDemo completed successfully.")


# ---------------------------------------------------------------------------
# Interactive Polyscope visualization
# ---------------------------------------------------------------------------

def run_visual():
    world, art, art_idx, light_body_indices, heavy_body_idx = build_world()

    # ---- Polyscope setup ----
    ps.init()
    ps.set_program_name("NovaPhy – Seesaw Launcher")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")

    # Ground
    gv, gf = make_ground_plane_mesh(size=14.0, offset=0.0)
    ps.register_surface_mesh("ground", gv, gf).set_color((0.40, 0.40, 0.40))

    # Pivot support – thin upright box (visual only, no collision)
    piv_v, piv_f = make_box_mesh(np.array([0.04, PIVOT_Y, 0.04], dtype=np.float32))
    ps_pivot = ps.register_surface_mesh("pivot", piv_v + np.array([0, PIVOT_Y, 0]),
                                        piv_f)
    ps_pivot.set_color((0.25, 0.25, 0.25))

    # Seesaw plank
    plank_v0, plank_f = make_box_mesh(
        np.array([PLANK_HX, PLANK_HY, PLANK_HZ], dtype=np.float32))
    ps_seesaw = ps.register_surface_mesh("seesaw", plank_v0, plank_f)
    ps_seesaw.set_color((0.85, 0.65, 0.20))   # gold

    # Light boxes (one mesh per box)
    lb_v0, lb_f = make_box_mesh(np.array([LIGHT_H, LIGHT_H, LIGHT_H], dtype=np.float32))
    LIGHT_COLORS = [
        (0.20, 0.75, 0.45),   # green
        (0.20, 0.65, 0.85),   # cyan-blue
        (0.70, 0.35, 0.80),   # purple
        (0.90, 0.55, 0.15),   # orange
    ]
    ps_lights = []
    for i in range(N_LIGHT):
        m = ps.register_surface_mesh(f"light_{i}", lb_v0, lb_f)
        m.set_color(LIGHT_COLORS[i % len(LIGHT_COLORS)])
        ps_lights.append(m)

    # Heavy box
    hv_v0, hv_f = make_box_mesh(
        np.array([HEAVY_H, HEAVY_H, HEAVY_H], dtype=np.float32))
    ps_heavy = ps.register_surface_mesh("heavy", hv_v0, hv_f)
    ps_heavy.set_color((0.90, 0.20, 0.20))    # red

    # ---- Sim state ----
    dt       = 1.0 / 60.0
    substeps = 8
    sim_dt   = dt / substeps

    paused    = [True]
    show_cp   = [True]
    step_ctr  = [0]

    def _apply_transform(mesh, verts_local, tf):
        pos = np.array(tf.position, dtype=np.float32)
        rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
        mesh.update_vertex_positions((verts_local @ rot.T) + pos)

    def _update_meshes():
        # Seesaw (multibody FK)
        q = np.array(world.state.q[art_idx])
        tfs = nv.forward_kinematics(art, q)
        _apply_transform(ps_seesaw, plank_v0, tfs[0])

        # Light boxes
        for i, bi in enumerate(light_body_indices):
            _apply_transform(ps_lights[i], lb_v0, world.state.transforms[bi])

        # Heavy box
        _apply_transform(ps_heavy, hv_v0, world.state.transforms[heavy_body_idx])

        # Contact points
        if show_cp[0]:
            contacts = list(world.multibody_contacts) + list(world.contacts)
            if len(contacts) > 0:
                cp_pos = np.array([np.array(c.position) for c in contacts],
                                  dtype=np.float32)
                pc = ps.register_point_cloud("contacts", cp_pos)
                pc.set_radius(0.018, relative=False)
                pc.set_color((1.0, 1.0, 0.0))
            elif ps.has_point_cloud("contacts"):
                ps.remove_point_cloud("contacts")
        elif ps.has_point_cloud("contacts"):
            ps.remove_point_cloud("contacts")

    _update_meshes()

    def callback():
        psim.PushItemWidth(140)

        psim.TextUnformatted("=== Seesaw Launcher ===")
        _, paused[0] = psim.Checkbox("Pause", paused[0])
        psim.SameLine()
        if psim.Button("Step") and paused[0]:
            world.step(sim_dt)
            step_ctr[0] += 1
            _update_meshes()
        _, show_cp[0] = psim.Checkbox("Show contacts", show_cp[0])

        psim.Separator()
        t  = step_ctr[0] * sim_dt
        th = float(np.array(world.state.q[art_idx])[0])
        om = float(np.array(world.state.qd[art_idx])[0])
        nc = len(world.multibody_contacts) + len(world.contacts)

        psim.TextUnformatted(f"t = {t:.3f} s    contacts = {nc}")
        psim.TextUnformatted(f"Seesaw  θ = {np.degrees(th):+7.2f}°   ω = {om:+6.2f} r/s")

        psim.Separator()
        psim.TextUnformatted("Light boxes (y)")
        for i, bi in enumerate(light_body_indices):
            ly = float(np.array(world.state.transforms[bi].position)[1])
            lv = float(np.array(world.state.linear_velocities[bi])[1])
            psim.TextUnformatted(f"  [{i}]  y={ly:.3f}  vy={lv:+.2f}")

        psim.Separator()
        hpos = np.array(world.state.transforms[heavy_body_idx].position)
        hvel = np.array(world.state.linear_velocities[heavy_body_idx])
        psim.TextUnformatted(f"Heavy box")
        psim.TextUnformatted(f"  pos ({hpos[0]:+.3f}, {hpos[1]:.3f}, {hpos[2]:+.3f})")
        psim.TextUnformatted(f"  vel ({hvel[0]:+.3f}, {hvel[1]:+.3f}, {hvel[2]:+.3f})")

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
