import numpy as np

import novaphy


def _make_ground_plane(offset=0.0):
    return novaphy.CollisionShape.make_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), offset, 0.8, 0.0)


def _make_free_box_scene(initial_height=0.1):
    art = novaphy.Articulation()
    joint = novaphy.Joint()
    joint.type = novaphy.JointType.Free
    joint.parent = -1
    art.joints = [joint]

    body = novaphy.RigidBody.from_box(1.0, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    art.bodies = [body]
    art.build_spatial_inertias()

    builder = novaphy.ModelBuilder()
    body_index = builder.add_body(
        body,
        novaphy.Transform.from_translation(np.array([0.0, 0.0, initial_height], dtype=np.float32)),
    )
    builder.add_shape(novaphy.CollisionShape.make_box(np.array([0.2, 0.2, 0.2], dtype=np.float32), body_index))
    model = builder.build()

    q = np.array([0.0, 0.0, initial_height, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    qd = np.zeros(6, dtype=np.float32)
    return art, model, q, qd


def _make_revolute_ground_scene():
    art = novaphy.Articulation()
    joint = novaphy.Joint()
    joint.type = novaphy.JointType.Revolute
    joint.parent = -1
    joint.axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    art.joints = [joint]

    body = novaphy.RigidBody.from_box(1.0, np.array([0.05, 0.05, 0.5], dtype=np.float32))
    body.com = np.array([0.0, 0.0, -0.5], dtype=np.float32)
    art.bodies = [body]
    art.build_spatial_inertias()

    builder = novaphy.ModelBuilder()
    body_index = builder.add_body(body, novaphy.Transform.identity())
    local = novaphy.Transform.from_translation(np.array([0.0, 0.0, -0.5], dtype=np.float32))
    builder.add_shape(novaphy.CollisionShape.make_box(np.array([0.05, 0.05, 0.5], dtype=np.float32), body_index, local))
    model = builder.build()

    q = np.array([0.0], dtype=np.float32)
    qd = np.zeros(1, dtype=np.float32)
    return art, model, q, qd


def test_xpbd_step_with_contacts_keeps_free_box_above_ground():
    art, model, q, qd = _make_free_box_scene()
    tau = np.zeros(6, dtype=np.float32)
    solver = novaphy.XPBDSolver()

    ever_contact = False
    contacts = []
    for _ in range(30):
        q, qd, contacts = solver.step_with_contacts(
            art,
            model,
            [_make_ground_plane()],
            q,
            qd,
            tau,
            np.array([0.0, 0.0, -9.81], dtype=np.float32),
            1.0 / 120.0,
        )
        ever_contact = ever_contact or bool(contacts)

    assert ever_contact
    assert q[2] >= 0.18
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(qd))


def test_xpbd_step_with_contacts_projects_revolute_link_out_of_ground():
    art, model, q, qd = _make_revolute_ground_scene()
    tau = np.zeros(1, dtype=np.float32)
    solver_settings = novaphy.XPBDSolverSettings()
    solver_settings.contact_relaxation = 0.6
    solver_settings.friction_damping = 0.2
    solver = novaphy.XPBDSolver(solver_settings)

    ever_contact = False
    contacts = []
    for _ in range(20):
        q, qd, contacts = solver.step_with_contacts(
            art,
            model,
            [_make_ground_plane(offset=-0.2)],
            q,
            qd,
            tau,
            np.zeros(3, dtype=np.float32),
            1.0 / 120.0,
        )
        ever_contact = ever_contact or bool(contacts)

    assert ever_contact
    assert abs(q[0]) > 0.01
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(qd))
