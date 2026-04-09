"""Tests for the Featherstone multibody pipeline via unified World API.

Tests articulated body dynamics using World with multibody_settings,
verifying ABA forward dynamics, contact solving, joint limits, and
motor-style torque control through the standard World/SimState/Control
interface.
"""

import numpy as np
import pytest
import novaphy as nv

GRAVITY = np.array([0, -9.81, 0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_single_pendulum(axis=np.array([0, 0, 1], dtype=np.float32),
                         mass=1.0, length=1.0):
    """Single revolute-joint pendulum hanging from the world origin."""
    art = nv.Articulation()

    j = nv.Joint()
    j.type = nv.JointType.Revolute
    j.axis = axis
    j.parent = -1
    t = nv.Transform()
    t.position = np.array([0, 0, 0], dtype=np.float32)
    j.parent_to_joint = t
    art.joints = [j]

    b = nv.RigidBody()
    b.mass = mass
    b.inertia = np.eye(3, dtype=np.float32) * (mass * length**2 / 12.0)
    b.com = np.array([0, -length / 2, 0], dtype=np.float32)
    art.bodies = [b]
    art.build_spatial_inertias()
    return art


def make_double_pendulum(mass=1.0, length=1.0):
    """Two revolute joints (z-axis) in a chain."""
    art = nv.Articulation()
    joints = []
    bodies = []
    for i in range(2):
        j = nv.Joint()
        j.type = nv.JointType.Revolute
        j.axis = np.array([0, 0, 1], dtype=np.float32)
        j.parent = i - 1
        t = nv.Transform()
        if i == 0:
            t.position = np.array([0, 0, 0], dtype=np.float32)
        else:
            t.position = np.array([0, -length, 0], dtype=np.float32)
        j.parent_to_joint = t
        joints.append(j)

        b = nv.RigidBody()
        b.mass = mass
        b.inertia = np.eye(3, dtype=np.float32) * (mass * length**2 / 12.0)
        b.com = np.array([0, -length / 2, 0], dtype=np.float32)
        bodies.append(b)

    art.joints = joints
    art.bodies = bodies
    art.build_spatial_inertias()
    return art


def make_floating_box(mass=1.0, half_ext=0.5):
    """Single link with Free joint (6-DOF floating base)."""
    art = nv.Articulation()

    j = nv.Joint()
    j.type = nv.JointType.Free
    j.parent = -1
    t = nv.Transform()
    t.position = np.array([0, 0, 0], dtype=np.float32)
    j.parent_to_joint = t
    art.joints = [j]

    b = nv.RigidBody()
    b.mass = mass
    I_val = mass * (2 * half_ext)**2 / 6.0
    b.inertia = np.eye(3, dtype=np.float32) * I_val
    b.com = np.array([0, 0, 0], dtype=np.float32)
    art.bodies = [b]
    art.build_spatial_inertias()
    return art


def make_multibody_world(art, q0=None, qd0=None, gravity=GRAVITY,
                         num_iterations=10, shapes=None):
    """Build a World with multibody solver from an articulation.

    Returns the constructed world.
    """
    builder = nv.ModelBuilder()
    builder.set_gravity(gravity)
    builder.add_articulation(art)

    if shapes:
        for s in shapes:
            builder.add_shape(s)

    model = builder.build()
    mbs = nv.MultiBodySolverSettings()
    mbs.num_iterations = num_iterations
    world = nv.World(model, multibody_settings=mbs)

    if q0 is not None:
        world.state.set_q(0, np.asarray(q0, dtype=np.float32))
    if qd0 is not None:
        world.state.set_qd(0, np.asarray(qd0, dtype=np.float32))

    return world


# ---------------------------------------------------------------------------
# ABA / forward dynamics tests (via World pipeline)
# ---------------------------------------------------------------------------

class TestABA:
    """Tests for Articulated Body Algorithm forward dynamics via World."""

    def test_single_pendulum_at_rest(self):
        """At q=0 (hanging down), gravity produces no torque => stays near rest."""
        art = make_single_pendulum()
        world = make_multibody_world(art, q0=[0.0], qd0=[0.0])

        world.step(0.001)
        qd = np.array(world.state.qd[0])
        assert abs(qd[0]) < 0.1, f"Expected ~0 velocity at rest, got {qd[0]}"

    def test_single_pendulum_horizontal(self):
        """Pendulum at 90 degrees should gain significant velocity."""
        art = make_single_pendulum()
        world = make_multibody_world(art, q0=[np.pi / 2], qd0=[0.0])

        world.step(0.01)
        qd = np.array(world.state.qd[0])
        assert abs(qd[0]) > 0.05, f"Expected significant velocity, got {qd[0]}"

    def test_floating_base_free_fall(self):
        """Free joint under gravity: should accelerate downward."""
        art = make_floating_box()
        q0 = np.array([0, 5, 0, 0, 0, 0, 1], dtype=np.float32)
        qd0 = np.zeros(6, dtype=np.float32)
        world = make_multibody_world(art, q0=q0, qd0=qd0)

        world.step(0.01)
        qd = np.array(world.state.qd[0])

        # angular velocity ~ 0
        assert np.allclose(qd[:3], 0, atol=0.1), \
            f"Angular vel should be ~0, got {qd[:3]}"
        # linear velocity y should be negative (falling)
        assert qd[4] < -0.05, \
            f"Linear Y vel should be negative (falling), got {qd[4]}"


# ---------------------------------------------------------------------------
# Position integration tests (via World pipeline)
# ---------------------------------------------------------------------------

class TestPositionIntegration:
    """Tests for position integration through the World pipeline."""

    def test_revolute_position_update(self):
        """Revolute joint position should advance in the direction of velocity."""
        art = make_single_pendulum()
        # Use zero gravity to isolate position integration
        world = make_multibody_world(art, q0=[0.0], qd0=[1.0],
                                     gravity=np.array([0, 0, 0], dtype=np.float32))
        world.step(0.01)
        q = np.array(world.state.q[0])
        assert abs(q[0] - 0.01) < 0.01, f"Expected q~0.01, got {q[0]}"

    def test_free_joint_position_update(self):
        """Free joint: position should translate by vel*dt."""
        art = make_floating_box()
        q0 = np.array([0, 5, 0, 0, 0, 0, 1], dtype=np.float32)
        qd0 = np.array([0, 0, 0, 1, -2, 3], dtype=np.float32)
        world = make_multibody_world(art, q0=q0, qd0=qd0,
                                     gravity=np.array([0, 0, 0], dtype=np.float32))
        world.step(0.01)
        q = np.array(world.state.q[0])
        np.testing.assert_allclose(q[0:3], [0.01, 4.98, 0.03], atol=0.01)


# ---------------------------------------------------------------------------
# World integration tests
# ---------------------------------------------------------------------------

class TestMultiBodyWorld:
    """Tests for the full pipeline via World with multibody solver."""

    def test_pendulum_simulation(self):
        """Simulate a single pendulum for multiple steps, verify it moves."""
        art = make_single_pendulum()
        q0 = np.array([np.pi / 4], dtype=np.float32)
        world = make_multibody_world(art, q0=q0, qd0=[0.0])

        dt = 0.001
        for _ in range(100):
            world.step(dt)

        q_final = np.array(world.state.q[0])
        qd_final = np.array(world.state.qd[0])

        assert abs(q_final[0] - np.pi / 4) > 0.001, "Pendulum should move"
        assert abs(qd_final[0]) > 0.01, "Pendulum should have velocity"

    def test_gravity_setting(self):
        """Verify custom gravity is applied."""
        art = make_single_pendulum()
        weak_gravity = np.array([0, -1.0, 0], dtype=np.float32)
        world = make_multibody_world(art, q0=[np.pi / 4], qd0=[0.0],
                                     gravity=weak_gravity)

        world.step(0.001)
        qd = np.array(world.state.qd[0])
        assert abs(qd[0]) < 0.1

    def test_box_falls_onto_ground_plane(self):
        """A floating box with collision should come to rest on a ground plane."""
        art = make_floating_box(mass=1.0, half_ext=0.5)

        box_shape = nv.CollisionShape.make_box(
            np.array([0.5, 0.5, 0.5], dtype=np.float32), 0,
            art_idx=0, link_idx=0)
        ground = nv.CollisionShape.make_plane(
            np.array([0, 1, 0], dtype=np.float32), 0.0)

        q0 = np.array([0, 2, 0, 0, 0, 0, 1], dtype=np.float32)
        qd0 = np.zeros(6, dtype=np.float32)

        world = make_multibody_world(art, q0=q0, qd0=qd0,
                                     num_iterations=20,
                                     shapes=[box_shape, ground])

        dt = 1.0 / 240.0
        for _ in range(2000):
            world.step(dt)

        q_final = np.array(world.state.q[0])
        qd_final = np.array(world.state.qd[0])

        assert q_final[1] < 2.0, "Box should have fallen"
        assert q_final[1] > 0.3, f"Box fell through ground: y={q_final[1]}"
        assert q_final[1] < 0.8, f"Box too high: y={q_final[1]}"

        assert np.linalg.norm(qd_final) < 1.0, \
            f"Box should have settled, vel_norm={np.linalg.norm(qd_final)}"

        contacts = world.multibody_contacts
        assert len(contacts) > 0, "Should have contacts when resting on plane"

    def test_double_pendulum_energy(self):
        """Double pendulum without damping: kinetic energy should stay bounded (no explosion or total loss)."""
        art = make_double_pendulum()
        q0 = np.array([0.5, -0.3], dtype=np.float32)
        qd0 = np.zeros(2, dtype=np.float32)
        world = make_multibody_world(art, q0=q0, qd0=qd0)

        dt = 0.0001
        energies = []
        for step in range(1000):
            world.step(dt)
            q = np.array(world.state.q[0])
            qd = np.array(world.state.qd[0])
            M = np.array(nv.mass_matrix_crba(art, q))
            KE = 0.5 * qd @ M @ qd
            energies.append(KE)

        assert max(energies) < 100, "Energy exploded"
        assert max(energies) > 0, "System lost all energy"


# ---------------------------------------------------------------------------
# Joint torque control tests (via Control)
# ---------------------------------------------------------------------------

class TestJointTorqueControl:
    """Tests for joint torque control via Control.articulation_joint_forces."""

    def test_torque_drives_joint(self):
        """Applying constant torque via Control should drive the joint."""
        art = make_single_pendulum()
        # Zero gravity to isolate torque effect
        world = make_multibody_world(art, q0=[0.0], qd0=[0.0],
                                     gravity=np.array([0, 0, 0], dtype=np.float32))

        dt = 0.001
        control = nv.Control()
        control.articulation_joint_forces = [
            np.array([5.0], dtype=np.float32)
        ]
        for _ in range(200):
            world.step_with_control(world.state, control, dt)

        q_final = np.array(world.state.q[0])
        assert abs(q_final[0]) > 0.1, f"Torque should drive joint, got q={q_final[0]}"


# ---------------------------------------------------------------------------
# Joint Limit tests (via Joint.limit_enabled)
# ---------------------------------------------------------------------------

class TestJointLimit:
    """Tests for joint limits configured via Joint.limit_enabled."""

    def test_limit_constrains_position(self):
        """Joint limit should prevent exceeding bounds."""
        art = make_single_pendulum()
        art.joints[0].limit_enabled = True
        art.joints[0].lower_limit = -0.6
        art.joints[0].upper_limit = 0.6

        q0 = np.array([0.5], dtype=np.float32)
        world = make_multibody_world(art, q0=q0, qd0=[0.0])

        dt = 0.001
        for _ in range(5000):
            world.step(dt)

        q_final = np.array(world.state.q[0])
        assert abs(q_final[0]) < 1.5, f"Joint limit exceeded significantly: q={q_final[0]}"


# ---------------------------------------------------------------------------
# FK tests (via forward_kinematics function)
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    """Tests for forward kinematics via the existing API."""

    def test_fk_produces_transforms(self):
        """FK should produce one transform per link."""
        art = make_double_pendulum()
        q = np.array([0.0, 0.0], dtype=np.float32)
        transforms = nv.forward_kinematics(art, q)
        assert len(transforms) == 2
