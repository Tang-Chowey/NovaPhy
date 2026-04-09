"""Tests for NovaPhy sensor framework."""

import numpy as np
import numpy.testing as npt
import pytest
import novaphy
from novaphy.sensors import (
    SensorIMU, SensorContact, SensorFrameTransform,
    resolve_sites, compute_site_world_transforms,
)


# --- Site utility tests ---

def test_resolve_sites_by_index():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b = builder.add_shape_box(half)
    builder.add_site(b, label="a")
    builder.add_site(b, label="b")
    model = builder.build()
    assert resolve_sites(model, [0, 1]) == [0, 1]


def test_resolve_sites_by_pattern():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b = builder.add_shape_box(half)
    builder.add_site(b, label="imu_left")
    builder.add_site(b, label="imu_right")
    builder.add_site(b, label="foot_left")
    model = builder.build()
    result = resolve_sites(model, "imu_*")
    assert result == [0, 1]


def test_resolve_sites_multi_pattern():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b = builder.add_shape_box(half)
    builder.add_site(b, label="imu_0")
    builder.add_site(b, label="foot_0")
    builder.add_site(b, label="cam_0")
    model = builder.build()
    result = resolve_sites(model, ["imu_*", "cam_*"])
    assert result == [0, 2]


def test_compute_site_world_transforms_free_body():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([1.0, 2.0, 3.0], dtype=np.float32)))
    offset_tf = novaphy.Transform.from_translation(
        np.array([0.5, 0.0, 0.0], dtype=np.float32))
    builder.add_site(b, offset_tf, "s0")
    model = builder.build()
    world = novaphy.World(model)
    tfs = compute_site_world_transforms(model, world.state)
    npt.assert_allclose(tfs[0].position, [1.5, 2.0, 3.0], atol=1e-5)


# --- SensorIMU tests ---

def _make_imu_scene():
    """One box above ground with an IMU site at body center."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 5.0, 0.0], dtype=np.float32)),
    )
    builder.add_site(body_idx, label="imu_center")
    model = builder.build()
    return model, novaphy.World(model)


def test_imu_default_construction():
    model, world = _make_imu_scene()
    imu = SensorIMU(model, sites="imu_*")
    assert imu.num_sensors == 1
    assert imu.accelerometer.shape == (1, 3)
    assert imu.gyroscope.shape == (1, 3)


def test_imu_freefall_reads_zero():
    """In free fall (no contacts), accelerometer should read ~0."""
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    builder.add_site(body_idx, label="imu_0")
    model = builder.build()
    world = novaphy.World(model)

    imu = SensorIMU(model, sites="imu_0")
    dt = 1.0 / 120.0

    world.step(dt)
    imu.update(world.state, dt)

    world.step(dt)
    imu.update(world.state, dt)

    # In free fall: coordinate acceleration = g, so specific force = g - g = 0
    npt.assert_allclose(imu.accelerometer[0], [0, 0, 0], atol=0.5)


def test_imu_resting_reads_g_up():
    """At rest on ground, accelerometer should read +g upward."""
    model, world = _make_imu_scene()
    imu = SensorIMU(model, sites="imu_center")
    dt = 1.0 / 120.0

    for _ in range(500):
        world.step(dt)

    world.step(dt)
    imu.update(world.state, dt)
    world.step(dt)
    imu.update(world.state, dt)

    npt.assert_allclose(imu.accelerometer[0, 1], 9.81, atol=1.5)
    npt.assert_allclose(imu.accelerometer[0, 0], 0.0, atol=1.0)
    npt.assert_allclose(imu.accelerometer[0, 2], 0.0, atol=1.0)


def test_imu_gyroscope_zero_when_not_rotating():
    model, world = _make_imu_scene()
    imu = SensorIMU(model, sites="imu_center")
    dt = 1.0 / 120.0
    world.step(dt)
    imu.update(world.state, dt)
    npt.assert_allclose(imu.gyroscope[0], [0, 0, 0], atol=0.01)


# --- SensorContact tests ---

def test_contact_sensor_has_contacts_on_ground():
    """A box at rest on a plane should register contacts."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 2.0, 0.0], dtype=np.float32)),
        density=1000.0,
    )
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    for _ in range(500):
        world.step(dt)

    sensor = SensorContact(model, body_indices=[body_idx])
    sensor.update(world.state, world.contacts, dt)
    assert sensor.num_contacts[0] > 0, "Should have contacts"


def test_contact_sensor_no_contacts_in_freefall():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    for _ in range(10):
        world.step(dt)

    sensor = SensorContact(model, body_indices=[body_idx])
    sensor.update(world.state, world.contacts, dt)
    assert sensor.num_contacts[0] == 0


# --- SensorFrameTransform tests ---

def test_frame_transform_identity_when_same_site():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(half)
    builder.add_site(body_idx, label="origin")
    model = builder.build()
    world = novaphy.World(model)

    sensor = SensorFrameTransform(model,
                                  target_sites="origin",
                                  reference_sites="origin")
    sensor.update(world.state)
    npt.assert_allclose(sensor.positions[0], [0, 0, 0], atol=1e-6)
    npt.assert_allclose(sensor.orientations[0], [0, 0, 0, 1], atol=1e-6)


def test_frame_transform_measures_offset():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(half)

    tf_a = novaphy.Transform.identity()
    tf_b = novaphy.Transform.from_translation(
        np.array([1.0, 0.0, 0.0], dtype=np.float32))
    builder.add_site(body_idx, tf_a, "ref")
    builder.add_site(body_idx, tf_b, "target")
    model = builder.build()
    world = novaphy.World(model)

    sensor = SensorFrameTransform(model,
                                  target_sites="target",
                                  reference_sites="ref")
    sensor.update(world.state)
    npt.assert_allclose(sensor.positions[0], [1, 0, 0], atol=1e-5)


def test_frame_transform_two_bodies():
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b0 = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        is_static=True,
    )
    b1 = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([3.0, 0.0, 0.0], dtype=np.float32)),
        is_static=True,
    )
    builder.add_site(b0, label="base")
    builder.add_site(b1, label="end")
    model = builder.build()
    world = novaphy.World(model)

    sensor = SensorFrameTransform(model,
                                  target_sites="end",
                                  reference_sites="base")
    sensor.update(world.state)
    npt.assert_allclose(sensor.positions[0], [3, 0, 0], atol=1e-5)


# --- Phase 1: SensorContact force/torque acceptance tests ---

def test_contact_force_resting_box_balances_gravity():
    """At rest on ground, contact force on the box should ≈ m*g upward."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 2.0, 0.0], dtype=np.float32)),
        density=1000.0,
    )
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    for _ in range(500):
        world.step(dt)

    sensor = SensorContact(model, body_indices=[body_idx])
    sensor.update(world.state, world.contacts, dt)

    mass = model.bodies[body_idx].mass
    expected_fy = mass * 9.81
    assert sensor.num_contacts[0] > 0
    npt.assert_allclose(sensor.forces[0, 1], expected_fy, rtol=0.15)
    npt.assert_allclose(sensor.forces[0, 0], 0.0, atol=expected_fy * 0.1)
    npt.assert_allclose(sensor.forces[0, 2], 0.0, atol=expected_fy * 0.1)


def test_contact_force_freefall_is_zero():
    """In free fall (no contacts), forces should be zero."""
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    for _ in range(10):
        world.step(dt)

    sensor = SensorContact(model, body_indices=[body_idx])
    sensor.update(world.state, world.contacts, dt)
    npt.assert_allclose(sensor.forces[0], [0, 0, 0], atol=1e-6)


def test_contact_impulse_field_exposed():
    """Verify that contact_impulse is accessible from Python."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 2.0, 0.0], dtype=np.float32)),
        density=1000.0,
    )
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    for _ in range(500):
        world.step(dt)

    contacts = world.contacts
    assert len(contacts) > 0
    cp = contacts[0]
    impulse = np.array(cp.contact_impulse, dtype=np.float32)
    assert impulse.shape == (3,)
    assert np.linalg.norm(impulse) > 0, "Resting contact should have nonzero impulse"


def test_contact_impulse_fields_are_read_only():
    """Solver-owned impulse fields should be read-only from Python."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 2.0, 0.0], dtype=np.float32)),
        density=1000.0,
    )
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    for _ in range(500):
        world.step(dt)

    cp = world.contacts[0]
    with pytest.raises(AttributeError):
        cp.contact_impulse = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(AttributeError):
        cp.accumulated_normal_impulse = 1.0


def test_contact_sensor_rejects_non_positive_dt():
    """A non-positive dt should raise instead of silently returning zero force."""
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    model = builder.build()
    world = novaphy.World(model)

    sensor = SensorContact(model, body_indices=[body_idx])
    with pytest.raises(ValueError, match="dt must be positive"):
        sensor.update(world.state, world.contacts, 0.0)


# --- Phase 2: IMU offset site velocity tests ---

def test_imu_offset_site_different_velocity_during_rotation():
    """Two sites at different offsets on a spinning body should report
    different linear velocities (and hence different accelerometer)."""
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    tf_center = novaphy.Transform.identity()
    tf_offset = novaphy.Transform.from_translation(
        np.array([1.0, 0.0, 0.0], dtype=np.float32))
    builder.add_site(body_idx, tf_center, "center")
    builder.add_site(body_idx, tf_offset, "offset")
    model = builder.build()
    world = novaphy.World(model)

    world.state.set_angular_velocity(body_idx,
                                     np.array([0.0, 5.0, 0.0], dtype=np.float32))

    imu_c = SensorIMU(model, sites="center")
    imu_o = SensorIMU(model, sites="offset")
    dt = 1.0 / 240.0

    world.step(dt)
    imu_c.update(world.state, dt)
    imu_o.update(world.state, dt)
    world.step(dt)
    imu_c.update(world.state, dt)
    imu_o.update(world.state, dt)

    acc_c = np.linalg.norm(imu_c.accelerometer[0])
    acc_o = np.linalg.norm(imu_o.accelerometer[0])
    assert acc_o > acc_c + 0.1, \
        f"Offset site should have larger acceleration: center={acc_c:.3f}, offset={acc_o:.3f}"


# --- Phase 3: Articulation IMU tests ---

def test_articulation_imu_nonzero_gyroscope():
    """Articulation IMU should read non-zero gyroscope when joint is moving."""
    art = novaphy.Articulation()
    joint = novaphy.Joint()
    joint.type = novaphy.JointType.Revolute
    joint.axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    joint.parent = -1
    joint.parent_to_joint = novaphy.Transform.identity()
    art.joints = [joint]

    body = novaphy.RigidBody.from_box(1.0, np.array([0.3, 0.3, 0.3], dtype=np.float32))
    art.bodies = [body]
    art.build_spatial_inertias()

    builder = novaphy.ModelBuilder()
    builder.set_gravity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    builder.add_site_on_link(0, 0, label="art_imu")
    model = builder.build()
    model.articulations = [art]

    state = novaphy.SimState.from_model(model)
    state.set_q(0, np.array([0.0], dtype=np.float32))
    state.set_qd(0, np.array([3.0], dtype=np.float32))

    imu = SensorIMU(model, sites="art_imu")
    imu.update(state, dt=1.0 / 120.0)

    assert np.linalg.norm(imu.gyroscope[0]) > 1.0, \
        f"Articulation gyroscope should be non-zero, got {imu.gyroscope[0]}"


# --- Phase 4: Unified update API tests ---

def test_sensor_update_signatures():
    """Verify all sensors can be updated with their documented signatures."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 2.0, 0.0], dtype=np.float32)),
    )
    builder.add_site(body_idx, label="s0")
    model = builder.build()
    world = novaphy.World(model)
    dt = 1.0 / 120.0
    world.step(dt)

    imu = SensorIMU(model, sites="s0")
    imu.update(world.state, dt)

    contact = SensorContact(model, body_indices=[body_idx])
    contact.update(world.state, world.contacts, dt)

    frame = SensorFrameTransform(model,
                                 target_sites="s0",
                                 reference_sites="s0")
    frame.update(world.state)


# --- Phase 5: FrameTransform velocity + add_site_on_link ---

def test_frame_transform_relative_velocity():
    """Two bodies moving apart should show nonzero relative linear velocity."""
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    b0 = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    b1 = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([3.0, 10.0, 0.0], dtype=np.float32)),
    )
    builder.add_site(b0, label="ref")
    builder.add_site(b1, label="target")
    model = builder.build()
    world = novaphy.World(model)

    world.state.set_linear_velocity(
        b0, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    world.state.set_linear_velocity(
        b1, np.array([2.0, 0.0, 0.0], dtype=np.float32))

    sensor = SensorFrameTransform(model,
                                  target_sites="target",
                                  reference_sites="ref")
    sensor.update(world.state)

    npt.assert_allclose(sensor.linear_velocities[0, 0], 2.0, atol=0.1)
    npt.assert_allclose(sensor.linear_velocities[0, 1], 0.0, atol=0.1)
    npt.assert_allclose(sensor.linear_velocities[0, 2], 0.0, atol=0.1)


def test_add_site_on_link():
    """Verify add_site_on_link creates a valid articulation site."""
    builder = novaphy.ModelBuilder()
    si = builder.add_site_on_link(0, 0, label="link0_imu")
    model = builder.build()
    assert model.sites[si].articulation_index == 0
    assert model.sites[si].link_index == 0
    assert model.sites[si].label == "link0_imu"
    assert model.sites[si].is_articulation_site()


# --- Phase 6: dt validation and transport term tests ---

def test_imu_rejects_non_positive_dt():
    """SensorIMU.update() should raise ValueError for dt <= 0."""
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(half)
    builder.add_site(body_idx, label="s0")
    model = builder.build()
    world = novaphy.World(model)

    imu = SensorIMU(model, sites="s0")
    with pytest.raises(ValueError, match="dt must be positive"):
        imu.update(world.state, dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        imu.update(world.state, dt=-0.01)


def test_frame_transform_zero_relative_velocity_on_same_rotating_body():
    """Two sites fixed on the same rotating body must have zero relative linear velocity.

    This verifies the transport term (-omega_ref x r_rel) is applied correctly.
    Without it, sites at different offsets would report spurious relative velocity.
    """
    builder = novaphy.ModelBuilder()
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body_idx = builder.add_shape_box(
        half,
        novaphy.Transform.from_translation(
            np.array([0.0, 10.0, 0.0], dtype=np.float32)),
    )
    tf_ref = novaphy.Transform.identity()
    tf_tgt = novaphy.Transform.from_translation(
        np.array([1.0, 0.0, 0.0], dtype=np.float32))
    builder.add_site(body_idx, tf_ref, "ref")
    builder.add_site(body_idx, tf_tgt, "tgt")
    model = builder.build()
    world = novaphy.World(model)

    world.state.set_angular_velocity(
        body_idx, np.array([0.0, 5.0, 0.0], dtype=np.float32))

    sensor = SensorFrameTransform(model,
                                  target_sites="tgt",
                                  reference_sites="ref")
    sensor.update(world.state)

    npt.assert_allclose(sensor.linear_velocities[0], [0, 0, 0], atol=1e-4,
                        err_msg="Sites on the same rigid body must have zero relative linear velocity")
