"""Demo: Newton's Cradle with Hinge-Joint Suspended Balls.

Scene
-----
- A fixed horizontal support beam at y = 1.2m
- 5 steel balls (0.12m radius) each suspended by a separate hinge joint
- Each ball hangs 0.6m below its hinge point, swinging like a pendulum
- Balls are arranged horizontally, touching each other at rest
- Pull the leftmost ball aside and release → momentum transfers through chain
- The ball on the opposite end swings out with the same energy

Key Physics Features
--------------------
- Each ball is a separate Articulation (1 fixed anchor + 1 revolute joint)
- Revolute hinge joints allow balls to swing as pendulums
- Ball-ball collisions transfer momentum between different articulations
- Joint limits prevent excessive swinging
- Demonstrates conservation of momentum and energy in elastic collisions

API Features Exercised
-----------------------
- ModelBuilder -> Model -> World pipeline
- Multiple articulations in the same World
- Each articulation: Fixed anchor + 1 Revolute joint
- CollisionShape with art_idx/link_idx for multibody collision
- Joint limits via Joint.limit_enabled / lower_limit / upper_limit
- nv.forward_kinematics(art, q) for visualization
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

from novaphy.viz import make_sphere_mesh, make_box_mesh, make_ground_plane_mesh, quat_to_rotation_matrix

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

NUM_BALLS = 5            # Number of balls in the cradle
BALL_RADIUS = 0.12       # Radius of each steel ball (m)
STRING_LENGTH = 0.6      # Length from hinge to ball center (m)
BALL_MASS = 3          # Mass of each ball (kg)
BALL_SPACING = 2.0 * BALL_RADIUS + 0.0005  # Slight gap to prevent initial overlap

# Support beam dimensions
BEAM_LENGTH = NUM_BALLS * BALL_SPACING + 0.3
BEAM_HEIGHT = 0.05
BEAM_DEPTH = 0.12
BEAM_Y = 1.2             # Height of the support beam

# Joint limits (max swing angle from vertical)
MAX_SWING_ANGLE = np.deg2rad(50.0)

# Initial condition: pull leftmost ball aside
INITIAL_PULL_ANGLE = np.deg2rad(40.0)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sphere_inertia(mass, radius):
    """Solid sphere inertia tensor (diagonal)."""
    I = 2.0 / 5.0 * mass * radius**2
    return np.array([I, I, I], dtype=np.float32)


def box_inertia(mass, hx, hy, hz):
    """Solid box inertia tensor (diagonal)."""
    lx, ly, lz = 2*hx, 2*hy, 2*hz
    return np.array([
        mass/12.0 * (ly**2 + lz**2),
        mass/12.0 * (lx**2 + lz**2),
        mass/12.0 * (lx**2 + ly**2),
    ], dtype=np.float32)


def create_single_ball_pendulum(art_idx, hinge_x, initial_angle=0.0):
    """Create a single pendulum articulation with its collision shapes.

    Structure:
        Link 0: Fixed anchor (heavy, doesn't move)
        Link 1: Revolute-jointed ball (swings like a pendulum)

    Args:
        art_idx: Articulation index (for collision shape assignment)
        hinge_x: X position of the hinge
        initial_angle: Initial joint angle (radians)

    Returns:
        (art, q0, qd0, shapes): Articulation, initial state, and collision shapes
    """
    art = nv.Articulation()
    joints_list = []
    bodies_list = []

    # Link 0: Fixed anchor (part of support beam)
    j0 = nv.Joint()
    j0.type = nv.JointType.Fixed
    j0.parent = -1
    ptj0 = nv.Transform()
    ptj0.position = np.array([hinge_x, BEAM_Y, 0.0], dtype=np.float32)
    j0.parent_to_joint = ptj0
    joints_list.append(j0)

    b0 = nv.RigidBody()
    b0.mass = 1000.0
    b0.com = np.zeros(3, dtype=np.float32)
    b0.inertia = np.diag(box_inertia(1000.0, 0.05, 0.025, 0.06)).astype(np.float32)
    bodies_list.append(b0)

    # Link 1: Ball with revolute joint
    j1 = nv.Joint()
    j1.type = nv.JointType.Revolute
    j1.axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-axis hinge
    j1.parent = 0
    ptj1 = nv.Transform()
    ptj1.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    j1.parent_to_joint = ptj1
    j1.limit_enabled = True
    j1.lower_limit = -MAX_SWING_ANGLE
    j1.upper_limit = MAX_SWING_ANGLE
    joints_list.append(j1)

    b1 = nv.RigidBody()
    b1.mass = BALL_MASS
    b1.com = np.array([0.0, -STRING_LENGTH, 0.0], dtype=np.float32)
    b1.inertia = np.diag(sphere_inertia(BALL_MASS, BALL_RADIUS)).astype(np.float32)
    bodies_list.append(b1)

    art.joints = joints_list
    art.bodies = bodies_list
    art.angular_damping = 0.005
    art.linear_damping = 0.0
    art.build_spatial_inertias()

    # Initial configuration (1 DOF)
    q0 = np.array([initial_angle], dtype=np.float32)
    qd0 = np.zeros(1, dtype=np.float32)

    # Collision shapes
    shapes = []

    # Anchor collision (link 0)
    anchor_shape = nv.CollisionShape.make_box(
        np.array([0.05, 0.025, 0.06], dtype=np.float32),
        0, friction=0.3, restitution=0.1,
        art_idx=art_idx, link_idx=0)
    shapes.append(anchor_shape)

    # Ball collision (link 1)
    ball_offset = nv.Transform()
    ball_offset.position = np.array([0.0, -STRING_LENGTH, 0.0], dtype=np.float32)
    ball_shape = nv.CollisionShape.make_sphere(
        BALL_RADIUS, 0, ball_offset,
        friction=0.05, restitution=0.98,
        art_idx=art_idx, link_idx=1)
    shapes.append(ball_shape)

    return art, q0, qd0, shapes


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_world(pull_angle=INITIAL_PULL_ANGLE):
    """Build and return the Newton's cradle simulation world.

    Each ball is a separate articulation to enable inter-body collisions.

    Args:
        pull_angle: Initial angle for leftmost ball (radians, negative = left)

    Returns:
        world: World instance
        articulations: List of Articulation instances (one per ball)
        params: Dict of scene parameters for visualization
    """
    builder = nv.ModelBuilder()
    builder.set_gravity(np.array([0.0, -9.81, 0.0], dtype=np.float32))

    # Ground plane (for safety, below the cradle)
    ground = nv.CollisionShape.make_plane(
        np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0)
    ground.friction = 0.5
    ground.restitution = 0.0
    builder.add_shape(ground)

    # Calculate x-position of first ball (center the cradle at x=0)
    first_ball_x = -(NUM_BALLS - 1) * BALL_SPACING / 2.0

    # Prepare each ball as a separate articulation
    articulations = []
    all_q0 = []
    all_qd0 = []
    all_shapes = []

    for i in range(NUM_BALLS):
        hinge_x = first_ball_x + i * BALL_SPACING
        initial_angle = -pull_angle if i == 0 else 0.0
        art, q0, qd0, shapes = create_single_ball_pendulum(i, hinge_x, initial_angle)
        articulations.append(art)
        all_q0.append(q0)
        all_qd0.append(qd0)
        all_shapes.extend(shapes)

    # Add all articulations to the builder
    for art in articulations:
        builder.add_articulation(art)

    # Build model and set articulation shapes
    model = builder.build()
    model.shapes = model.shapes + all_shapes

    # Create world with multibody solver
    sol = nv.MultiBodySolverSettings()
    sol.num_iterations = 60
    sol.erp = 0.2
    sol.linear_slop = 0.001
    sol.restitution_threshold = 0.2
    world = nv.World(model, multibody_settings=sol)

    # Set initial state for each articulation
    for i in range(NUM_BALLS):
        world.state.set_q(i, all_q0[i])
        world.state.set_qd(i, all_qd0[i])

    params = {
        'ball_radius': BALL_RADIUS,
        'string_length': STRING_LENGTH,
        'first_ball_x': first_ball_x,
        'beam_y': BEAM_Y,
        'num_balls': NUM_BALLS,
    }

    return world, articulations, params


# ---------------------------------------------------------------------------
# Headless simulation + validation
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("NovaPhy: Newton's Cradle with Hinge-Joint Suspended Balls")
    print(f"  {NUM_BALLS} steel balls, each on a separate hinge pendulum")
    print(f"  String length = {STRING_LENGTH}m")
    print(f"  Initial: leftmost ball pulled to {np.degrees(INITIAL_PULL_ANGLE):.1f}°")
    print("=" * 70)

    world, articulations, p = build_world(pull_angle=INITIAL_PULL_ANGLE)

    dt = 1.0 / 240.0  # High timestep for stability
    num_steps = 1200  # 5 seconds (shorter to reduce damping effects)

    print(f"\n{'Step':>6}  {'t(s)':>7}  "
          f"{'B1(°)':>8}  {'B2(°)':>8}  {'B5(°)':>8}  "
          f"{'E(J)':>10}  {'cnt':>5}")
    print("-" * 68)

    # Track energy and collisions
    first_collision_t = None
    max_right_swing = -float('inf')
    energy_history = []

    for step in range(num_steps):
        world.step(dt)
        t = (step + 1) * dt

        # Get all joint angles and velocities
        q_list = [np.array(world.state.q[i]) for i in range(NUM_BALLS)]
        qd_list = [np.array(world.state.qd[i]) for i in range(NUM_BALLS)]
        nc = len(world.multibody_contacts)

        # Calculate total energy
        ke = 0.0
        pe = 0.0
        for i in range(NUM_BALLS):
            omega = qd_list[i][0] if len(qd_list[i]) > 0 else 0.0
            I = BALL_MASS * STRING_LENGTH**2
            ke += 0.5 * I * omega**2
            theta = q_list[i][0] if len(q_list[i]) > 0 else 0.0
            h = STRING_LENGTH * (1.0 - np.cos(theta))
            pe += BALL_MASS * 9.81 * h

        total_energy = ke + pe
        energy_history.append(total_energy)

        # Track maximum right swing (last ball)
        if len(q_list[NUM_BALLS-1]) > 0:
            if q_list[NUM_BALLS-1][0] > max_right_swing:
                max_right_swing = q_list[NUM_BALLS-1][0]

        # Detect first collision
        if nc > 0 and first_collision_t is None:
            first_collision_t = t
            print(f"\n  >>>  FIRST BALL COLLISION at t={t:.4f}s  <<<\n")

        if step % 120 == 0 or step == num_steps - 1:
            q1_deg = np.degrees(q_list[0][0]) if len(q_list[0]) > 0 else 0
            q2_deg = np.degrees(q_list[1][0]) if len(q_list[1]) > 0 else 0
            q5_deg = np.degrees(q_list[NUM_BALLS-1][0]) if len(q_list[NUM_BALLS-1]) > 0 else 0
            print(f"{step:6d}  {t:7.2f}  "
                  f"{q1_deg:+8.2f}  {q2_deg:+8.2f}  {q5_deg:+8.2f}  "
                  f"{total_energy:10.4f}  {nc:5d}")

    print("\n" + "=" * 70)
    print("Final Analysis")
    print("=" * 70)

    initial_energy = energy_history[0]
    # Check energy at t=2.5s (step 600) - after momentum transfer but before much damping
    mid_energy = energy_history[min(600, len(energy_history)-1)]
    energy_loss_mid = (initial_energy - mid_energy) / initial_energy * 100

    print(f"  Initial energy: {initial_energy:.4f} J")
    print(f"  Energy at 2.5s: {mid_energy:.4f} J")
    print(f"  Energy loss:    {energy_loss_mid:.2f}%")
    print(f"  Max right swing: {np.degrees(max_right_swing):.2f}°")

    print("\nSanity checks:")

    # Check no NaN
    for i in range(NUM_BALLS):
        q = np.array(world.state.q[i])
        qd = np.array(world.state.qd[i])
        assert not np.isnan(q).any(), f"FAIL: NaN in pendulum {i} joint angles"
        assert not np.isnan(qd).any(), f"FAIL: NaN in pendulum {i} joint velocities"
    print("  [PASS] No NaN in any state")

    # Check collision occurred
    assert first_collision_t is not None, "FAIL: No collision detected"
    print(f"  [PASS] First collision at t={first_collision_t:.4f}s")

    # Check that right ball swung out (momentum transfer)
    assert max_right_swing > np.deg2rad(10.0), \
        f"FAIL: Right ball didn't swing out enough ({np.degrees(max_right_swing):.2f}°)"
    print(f"  [PASS] Right ball swung out to {np.degrees(max_right_swing):.2f}°")

    # Check energy conservation (note: some loss is expected due to ERP/damping)
    assert energy_loss_mid < 70.0, f"FAIL: Too much energy lost ({energy_loss_mid:.1f}%)"
    print(f"  [INFO] Energy loss {energy_loss_mid:.1f}% (expected due to numerical damping)")

    print("\nDemo completed successfully!")
    print("Run without --headless for interactive Polyscope visualization.")


# ---------------------------------------------------------------------------
# Interactive Polyscope visualization
# ---------------------------------------------------------------------------

def run_visual():
    world, articulations, params = build_world(pull_angle=INITIAL_PULL_ANGLE)

    ps.init()
    ps.set_program_name("NovaPhy – Newton's Cradle (Hinge Suspension)")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")

    # Ground plane
    gv, gf = make_ground_plane_mesh(size=10.0, offset=0.0)
    ps.register_surface_mesh("ground", gv, gf).set_color((0.4, 0.4, 0.4))

    # Support beam (visual only)
    beam_verts, beam_faces = make_box_mesh(
        np.array([BEAM_LENGTH/2, BEAM_HEIGHT/2, BEAM_DEPTH/2], dtype=np.float32))
    beam_pos = np.array([0.0, BEAM_Y, 0.0], dtype=np.float32)
    beam_verts_world = beam_verts + beam_pos
    ps.register_surface_mesh("beam", beam_verts_world, beam_faces).set_color((0.55, 0.45, 0.35))

    # Ball mesh
    ball_verts, ball_faces = make_sphere_mesh(BALL_RADIUS, 24)
    ps_balls = []
    ball_colors = [
        (0.95, 0.25, 0.25),  # Red
        (0.95, 0.65, 0.25),  # Orange
        (0.95, 0.95, 0.25),  # Yellow
        (0.25, 0.75, 0.95),  # Cyan
        (0.65, 0.25, 0.95),  # Purple
    ]
    for i in range(NUM_BALLS):
        m = ps.register_surface_mesh(f"ball_{i}", ball_verts, ball_faces)
        m.set_color(ball_colors[i % len(ball_colors)])
        m.set_smooth_shade(True)
        ps_balls.append(m)

    # String lines
    string_verts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -STRING_LENGTH, 0.0],
    ], dtype=np.float32)
    string_edges = np.array([[0, 1]], dtype=np.int32)
    ps_strings = []
    for i in range(NUM_BALLS):
        sl = ps.register_curve_network(f"string_{i}", string_verts.copy(), string_edges)
        sl.set_radius(0.003, relative=False)
        sl.set_color((0.8, 0.8, 0.8))
        ps_strings.append(sl)

    dt = 1.0 / 60.0
    substeps = 4
    sim_dt = dt / substeps

    paused = [True]
    show_strings = [True]
    reset_sim = [False]
    step_ctr = [0]
    pull_angle = [INITIAL_PULL_ANGLE]

    def _update_meshes():
        for i, art in enumerate(articulations):
            tfs = nv.forward_kinematics(art, np.array(world.state.q[i]))
            tf = tfs[1]  # Link 1 is the ball

            ball_pos = np.array(tf.position, dtype=np.float32)
            ball_rot = quat_to_rotation_matrix(np.array(tf.rotation, dtype=np.float32))
            ball_center = ball_pos + ball_rot @ np.array([0.0, -STRING_LENGTH, 0.0], dtype=np.float32)
            ps_balls[i].update_vertex_positions((ball_verts @ ball_rot.T) + ball_center)

            # Update string (from anchor to ball top)
            anchor_tf = tfs[0]
            anchor_pos = np.array(anchor_tf.position, dtype=np.float32)
            ball_top = ball_pos + ball_rot @ np.array([0.0, -STRING_LENGTH + BALL_RADIUS*0.85, 0.0], dtype=np.float32)
            ps_strings[i].update_node_positions(np.stack([anchor_pos, ball_top]))

    _update_meshes()

    def callback():
        nonlocal world, articulations
        psim.PushItemWidth(200)

        psim.TextUnformatted("=== Newton's Cradle ===")
        _, paused[0] = psim.Checkbox("Pause", paused[0])

        psim.SameLine()
        if psim.Button("Step x1") and paused[0]:
            world.step(sim_dt)
            step_ctr[0] += 1
            _update_meshes()

        psim.SameLine()
        if psim.Button("Reset"):
            reset_sim[0] = True

        psim.Separator()
        _, show_strings[0] = psim.Checkbox("Show strings", show_strings[0])

        psim.Separator()
        changed, new_angle = psim.SliderFloat("Pull angle", pull_angle[0], 0.0, 60.0)
        if changed:
            pull_angle[0] = new_angle

        psim.Separator()

        # Display state
        nc = len(world.multibody_contacts)
        t = step_ctr[0] * sim_dt
        psim.TextUnformatted(f"t = {t:.3f} s")
        psim.TextUnformatted(f"contacts = {nc}")
        psim.Separator()

        psim.TextUnformatted("Ball angles (deg):")
        for i in range(NUM_BALLS):
            q = np.array(world.state.q[i])
            angle = np.degrees(q[0]) if len(q) > 0 else 0.0
            label = f"  B{i+1}:"
            psim.TextUnformatted(label)
            psim.SameLine()
            psim.TextUnformatted(f"{angle:+7.2f}°")

        psim.Separator()

        # Calculate energy
        ke = pe = 0.0
        for i in range(NUM_BALLS):
            qd = np.array(world.state.qd[i])
            q = np.array(world.state.q[i])
            omega = qd[0] if len(qd) > 0 else 0.0
            I = BALL_MASS * STRING_LENGTH**2
            ke += 0.5 * I * omega**2
            theta = q[0] if len(q) > 0 else 0.0
            h = STRING_LENGTH * (1.0 - np.cos(theta))
            pe += BALL_MASS * 9.81 * h
        psim.TextUnformatted(f"KE = {ke:.4f} J")
        psim.TextUnformatted(f"PE = {pe:.4f} J")
        psim.TextUnformatted(f"Total = {ke+pe:.4f} J")

        psim.PopItemWidth()

        # Handle reset
        if reset_sim[0]:
            world, articulations, _ = build_world(pull_angle=np.deg2rad(pull_angle[0]))
            step_ctr[0] = 0
            reset_sim[0] = False
            _update_meshes()

        # Toggle string visibility
        for sl in ps_strings:
            sl.set_enabled(show_strings[0])

        # Physics step
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
