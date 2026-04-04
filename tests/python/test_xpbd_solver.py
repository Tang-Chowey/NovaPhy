import numpy as np
import numpy.testing as npt

import novaphy


def _make_free_body_articulation():
    art = novaphy.Articulation()

    joint = novaphy.Joint()
    joint.type = novaphy.JointType.Free
    joint.parent = -1
    art.joints = [joint]

    body = novaphy.RigidBody()
    body.mass = 2.0
    body.inertia = np.eye(3, dtype=np.float32) * 0.5
    art.bodies = [body]
    art.build_spatial_inertias()
    return art


def _make_slide_articulation(axis):
    art = novaphy.Articulation()

    joint = novaphy.Joint()
    joint.type = novaphy.JointType.Slide
    joint.parent = -1
    joint.axis = np.array(axis, dtype=np.float32)
    art.joints = [joint]

    body = novaphy.RigidBody()
    body.mass = 1.0
    body.inertia = np.eye(3, dtype=np.float32) * 0.1
    art.bodies = [body]
    art.build_spatial_inertias()
    return art


def _make_revolute_articulation(limit_enabled=False, lower=-0.25, upper=0.25):
    art = novaphy.Articulation()

    joint = novaphy.Joint()
    joint.type = novaphy.JointType.Revolute
    joint.parent = -1
    joint.axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    joint.limit_enabled = limit_enabled
    joint.lower_limit = lower
    joint.upper_limit = upper
    art.joints = [joint]

    body = novaphy.RigidBody()
    body.mass = 1.0
    body.com = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    body.inertia = np.eye(3, dtype=np.float32)
    art.bodies = [body]
    art.build_spatial_inertias()
    return art


def _make_double_revolute_chain():
    art = novaphy.Articulation()

    j0 = novaphy.Joint()
    j0.type = novaphy.JointType.Revolute
    j0.parent = -1
    j0.axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    j1 = novaphy.Joint()
    j1.type = novaphy.JointType.Revolute
    j1.parent = 0
    j1.axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    j1.parent_to_joint = novaphy.Transform.from_translation(np.array([0.0, -1.0, 0.0], dtype=np.float32))

    art.joints = [j0, j1]

    b0 = novaphy.RigidBody()
    b0.mass = 1.0
    b0.com = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    b0.inertia = np.eye(3, dtype=np.float32)

    b1 = novaphy.RigidBody()
    b1.mass = 1.0
    b1.com = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    b1.inertia = np.eye(3, dtype=np.float32)

    art.bodies = [b0, b1]
    art.build_spatial_inertias()
    return art


def test_xpbd_solver_reports_stats_and_preserves_dimensions():
    art = _make_slide_articulation([0.0, 0.0, 1.0])
    q = np.zeros(1, dtype=np.float32)
    qd = np.zeros(1, dtype=np.float32)
    tau = np.zeros(1, dtype=np.float32)

    settings = novaphy.XPBDSolverSettings()
    settings.substeps = 4
    settings.iterations = 3
    solver = novaphy.XPBDSolver(settings)
    q_new, qd_new = solver.step(
        art,
        q,
        qd,
        tau,
        np.array([0.0, 0.0, -9.81], dtype=np.float32),
        1.0 / 60.0,
    )

    assert q_new.shape == q.shape
    assert qd_new.shape == qd.shape
    assert solver.last_stats.substeps == 4
    assert solver.last_stats.iterations == 3
    assert solver.last_stats.projected_constraints == 0


def test_xpbd_solver_normalizes_free_joint_quaternion():
    art = _make_free_body_articulation()
    q = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    qd = np.zeros(6, dtype=np.float32)
    tau = np.zeros(6, dtype=np.float32)

    solver = novaphy.XPBDSolver()
    q_new, _ = solver.step(
        art,
        q,
        qd,
        tau,
        np.zeros(3, dtype=np.float32),
        1.0 / 120.0,
    )

    npt.assert_allclose(np.linalg.norm(q_new[3:7]), 1.0, atol=1e-5)


def test_xpbd_solver_free_root_moves_under_gravity():
    art = _make_free_body_articulation()
    q = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    qd = np.zeros(6, dtype=np.float32)
    tau = np.zeros(6, dtype=np.float32)

    settings = novaphy.XPBDSolverSettings()
    settings.substeps = 5
    solver = novaphy.XPBDSolver(settings)

    q_new, qd_new = solver.step(
        art,
        q,
        qd,
        tau,
        np.array([0.0, 0.0, -9.81], dtype=np.float32),
        1.0 / 30.0,
    )

    assert q_new[2] < q[2]
    assert qd_new[5] < 0.0


def test_xpbd_solver_respects_revolute_limits():
    art = _make_revolute_articulation(limit_enabled=True)
    q = np.array([0.5], dtype=np.float32)
    qd = np.array([1.0], dtype=np.float32)
    tau = np.zeros(1, dtype=np.float32)

    solver = novaphy.XPBDSolver()
    q_new, qd_new = solver.step(
        art,
        q,
        qd,
        tau,
        np.zeros(3, dtype=np.float32),
        1.0 / 60.0,
    )

    assert q_new[0] <= 0.25 + 1.0e-6
    assert qd_new[0] <= 1.0e-6
    assert solver.last_stats.projected_constraints > 0


def test_xpbd_solver_target_position_drive_converges():
    art = _make_revolute_articulation(limit_enabled=False)
    q = np.array([0.0], dtype=np.float32)
    qd = np.array([0.0], dtype=np.float32)
    tau = np.zeros(1, dtype=np.float32)

    drive = novaphy.JointDrive()
    drive.mode = novaphy.JointTargetMode.TargetPosition
    drive.target_position = 0.6
    drive.stiffness = 120.0
    drive.damping = 8.0
    control = novaphy.Control()
    control.joint_drives = [drive]

    solver = novaphy.XPBDSolver()
    for _ in range(60):
        q, qd = solver.step(
            art,
            q,
            qd,
            tau,
            np.zeros(3, dtype=np.float32),
            1.0 / 120.0,
            control,
        )

    assert abs(q[0] - 0.6) < 0.05
    assert abs(qd[0]) < 0.5


def test_xpbd_solver_double_pendulum_stays_finite_under_gravity():
    art = _make_double_revolute_chain()
    q = np.array([0.3, -0.2], dtype=np.float32)
    qd = np.zeros(2, dtype=np.float32)
    tau = np.zeros(2, dtype=np.float32)

    solver = novaphy.XPBDSolver()
    for _ in range(120):
        q, qd = solver.step(
            art,
            q,
            qd,
            tau,
            np.array([0.0, -9.81, 0.0], dtype=np.float32),
            1.0 / 120.0,
        )

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(qd))


def test_world_routes_joint_forces_per_articulation():
    art_a = _make_slide_articulation([1.0, 0.0, 0.0])
    art_b = _make_slide_articulation([1.0, 0.0, 0.0])

    model = novaphy.ModelBuilder().build()
    model.gravity = np.zeros(3, dtype=np.float32)
    model.articulations = [art_a, art_b]

    world = novaphy.World(model)

    control = novaphy.Control()
    control.articulation_joint_forces = [
        np.array([2.0], dtype=np.float32),
        np.array([-3.0], dtype=np.float32),
    ]

    world.step_with_control(world.state, control, 1.0 / 60.0)

    assert world.state.qd[0][0] > 0.0
    assert world.state.qd[1][0] < 0.0
