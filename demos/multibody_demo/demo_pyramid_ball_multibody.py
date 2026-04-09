"""Demo: 4-3-2-1 box pyramid on a ground plane with a sphere projectile.

Demonstrates stable stacking with multiple contact constraints,
and dynamic collision response when a sphere hits the pyramid.

This version uses the ModelBuilder -> Model -> World API with the multibody solver.
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

BOX_HALF    = np.array([0.5, 0.5, 0.5], dtype=np.float32)
BOX_MASS    = 1.0
BOX_SPACING = 1.05  # slight gap between boxes

SPHERE_RADIUS = 0.4
SPHERE_MASS   = 3.0


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


def _sphere_inertia(mass, r):
    I = 2.0/5.0 * mass * r**2
    return np.array([I, I, I], dtype=np.float32)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_world():
    builder = nv.ModelBuilder()
    builder.set_gravity(np.array([0.0, -9.81, 0.0], dtype=np.float32))

    # Ground plane (body_index=-1 → static)
    gnd = nv.CollisionShape.make_plane(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0)
    gnd.friction = 0.6
    gnd.restitution = 0.1
    builder.add_shape(gnd)

    # Build pyramid layers: 4-3-2-1
    layers = [4, 3, 2, 1]
    box_indices = []

    for layer_idx, count in enumerate(layers):
        y = 0.5 + layer_idx * 1.0
        start_x = -(count - 1) * BOX_SPACING / 2.0
        for i in range(count):
            x = start_x + i * BOX_SPACING

            rb = nv.RigidBody()
            rb.mass = BOX_MASS
            rb.inertia = np.diag(
                _box_inertia(BOX_MASS, BOX_HALF[0], BOX_HALF[1], BOX_HALF[2])
            ).astype(np.float32)
            rb.linear_damping = 0.01
            rb.angular_damping = 0.02

            tf = nv.Transform()
            tf.position = np.array([x, y, 0.0], dtype=np.float32)

            body_idx = builder.add_body(rb, tf)

            sh = nv.CollisionShape.make_box(BOX_HALF, body_idx,
                                            friction=0.6, restitution=0.1)
            builder.add_shape(sh)
            box_indices.append(body_idx)

    # Sphere projectile
    rb = nv.RigidBody()
    rb.mass = SPHERE_MASS
    rb.inertia = np.diag(
        _sphere_inertia(SPHERE_MASS, SPHERE_RADIUS)
    ).astype(np.float32)
    rb.linear_damping = 0.0
    rb.angular_damping = 0.01

    tf = nv.Transform()
    tf.position = np.array([-4.0, 0.4, 0.0], dtype=np.float32)

    ball_idx = builder.add_body(rb, tf)

    sh = nv.CollisionShape.make_sphere(SPHERE_RADIUS, ball_idx,
                                       friction=0.3, restitution=0.5)
    builder.add_shape(sh)

    # Build model and create world with multibody solver settings
    model = builder.build()

    sol = nv.MultiBodySolverSettings()
    sol.num_iterations = 50
    sol.erp = 0.2
    sol.linear_slop = 0.002

    world = nv.World(model, multibody_settings=sol)

    # Give the ball an initial velocity toward the pyramid
    world.state.set_linear_velocity(
        ball_idx, np.array([25.0, 0.0, 0.0], dtype=np.float32))

    return world, box_indices, ball_idx


# ---------------------------------------------------------------------------
# Headless run + sanity checks
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("NovaPhy: Pyramid Ball Demo (ModelBuilder -> World)")
    print("  4-3-2-1 box pyramid with sphere projectile")
    print("=" * 60)

    world, box_indices, ball_idx = build_world()

    dt = 1.0 / 60.0
    num_steps = int(5.0 / dt)

    print(f"\n{'Step':>5}  {'t':>6}  {'ball_x':>8}  {'ball_y':>8}  {'nc':>5}")
    print("-" * 40)

    for step in range(num_steps):
        world.step(dt)
        t = (step + 1) * dt

        ball_pos = np.array(world.state.transforms[ball_idx].position)
        nc = len(world.multibody_contacts)

        if step % int(0.5 / dt) == 0 or step == num_steps - 1:
            print(f"{step:5d}  {t:6.3f}  {ball_pos[0]:8.3f}  {ball_pos[1]:8.3f}  {nc:5d}")

    print("\n" + "=" * 60)
    print("Final state (t=5s)")
    print("=" * 60)

    ball_pos_f = np.array(world.state.transforms[ball_idx].position)
    print(f"  Ball position: ({ball_pos_f[0]:.3f}, {ball_pos_f[1]:.3f}, {ball_pos_f[2]:.3f})")

    fallen = sum(1 for idx in box_indices
                 if np.array(world.state.transforms[idx].position)[0] > 2.0)
    print(f"  Boxes knocked over: {fallen}/{len(box_indices)}")

    print("\nSanity checks:")

    assert not np.isnan(ball_pos_f).any(), "FAIL: NaN in ball position"
    print("  [PASS] No NaN in ball position")

    for i, idx in enumerate(box_indices):
        p = np.array(world.state.transforms[idx].position)
        assert not np.isnan(p).any(), f"FAIL: NaN in box {i}"
    print("  [PASS] No NaN in box positions")

    assert ball_pos_f[0] > -2.0, f"FAIL: Ball didn't move much x={ball_pos_f[0]:.3f}"
    print(f"  [PASS] Ball moved into pyramid (x={ball_pos_f[0]:.3f})")

    print("\nDemo completed successfully.")


# ---------------------------------------------------------------------------
# Interactive Polyscope visualization
# ---------------------------------------------------------------------------

def run_visual():
    world, box_indices, ball_idx = build_world()

    ps.init()
    ps.set_program_name("NovaPhy – Pyramid Ball")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")

    # Ground
    gv, gf = make_ground_plane_mesh(size=14.0, offset=0.0)
    ps.register_surface_mesh("ground", gv, gf).set_color((0.42, 0.42, 0.42))

    # Box meshes
    box_v0, box_f = make_box_mesh(BOX_HALF)
    ps_boxes = []
    for i in range(len(box_indices)):
        layer = i // 4
        t = layer / 3.0
        color = (0.7 - 0.2 * t, 0.5 + 0.2 * t, 0.3 + 0.1 * t)
        m = ps.register_surface_mesh(f"box_{i}", box_v0, box_f)
        m.set_smooth_shade(True)
        ps_boxes.append(m)

    # Ball mesh
    ball_v0, ball_f = make_sphere_mesh(SPHERE_RADIUS, n_lat=16, n_lon=32)
    ps_ball = ps.register_surface_mesh("ball", ball_v0, ball_f)
    ps_ball.set_smooth_shade(True)
    ps_ball.set_color((0.8, 0.2, 0.2))

    dt = 1.0 / 60.0
    substeps = 10
    sim_dt = dt / substeps

    paused = [True]
    show_cp = [False]
    step_ctr = [0]

    def _update_meshes():
        # Boxes
        for i, idx in enumerate(box_indices):
            tf = world.state.transforms[idx]
            pos = np.array(tf.position, dtype=np.float32)
            rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
            ps_boxes[i].update_vertex_positions((box_v0 @ rot.T) + pos)

        # Ball
        tf = world.state.transforms[ball_idx]
        pos = np.array(tf.position, dtype=np.float32)
        rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
        ps_ball.update_vertex_positions((ball_v0 @ rot.T) + pos)

        # Contact points
        if show_cp[0]:
            contacts = world.multibody_contacts
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
        psim.TextUnformatted("=== Pyramid Ball ===")
        _, paused[0] = psim.Checkbox("Pause", paused[0])
        psim.SameLine()
        if psim.Button("Step") and paused[0]:
            world.step(sim_dt)
            step_ctr[0] += 1
            _update_meshes()
        _, show_cp[0] = psim.Checkbox("Contacts", show_cp[0])
        psim.Separator()

        t = step_ctr[0] * sim_dt
        ball_pos = np.array(world.state.transforms[ball_idx].position)
        nc = len(world.multibody_contacts)

        psim.TextUnformatted(f"t = {t:.3f} s    contacts = {nc}")
        psim.Separator()
        psim.TextUnformatted("Ball position")
        psim.TextUnformatted(f"  ({ball_pos[0]:+.3f}, {ball_pos[1]:+.3f}, {ball_pos[2]:+.3f})")
        psim.Separator()
        psim.TextUnformatted(f"Boxes: {len(box_indices)}")
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
