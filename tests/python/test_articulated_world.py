import numpy as np
import pytest

import novaphy


def _make_link(name: str, mass: float, half_extents, collision_offset=(0.0, 0.0, 0.0)):
    link = novaphy.UrdfLink()
    link.name = name

    inertial = novaphy.UrdfInertial()
    inertial.mass = float(mass)
    inertial.inertia = np.eye(3, dtype=np.float32) * max(float(mass) * 0.05, 1.0e-3)
    link.inertial = inertial

    collision = novaphy.UrdfCollision()
    collision.origin = novaphy.Transform.from_translation(np.array(collision_offset, dtype=np.float32))
    geometry = novaphy.UrdfGeometry()
    geometry.type = novaphy.UrdfGeometryType.Box
    geometry.size = np.array(half_extents, dtype=np.float32) * 2.0
    collision.geometry = geometry
    link.collisions = [collision]
    return link


def _build_single_hip_scene(root_height: float = 0.6):
    model = novaphy.UrdfModelData()
    model.name = "single_hip"
    model.links = [
        _make_link("base", 5.0, [0.25, 0.12, 0.12]),
        _make_link("leg", 1.0, [0.05, 0.05, 0.3], collision_offset=(0.0, 0.0, -0.3)),
    ]

    hip = novaphy.UrdfJoint()
    hip.name = "hip"
    hip.type = "revolute"
    hip.parent_link = "base"
    hip.child_link = "leg"
    hip.origin = novaphy.Transform.from_translation(np.array([0.0, 0.0, -0.05], dtype=np.float32))
    hip.axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    hip.lower_limit = -1.0
    hip.upper_limit = 1.0
    model.joints = [hip]

    options = novaphy.UrdfImportOptions()
    options.floating_base = True
    options.enable_self_collisions = False
    options.collapse_fixed_joints = True
    options.ignore_inertial_definitions = True
    options.root_transform = novaphy.Transform.from_translation(
        np.array([0.0, 0.0, root_height], dtype=np.float32)
    )

    scene = novaphy.SceneBuilderEngine().build_from_urdf(model, options)
    assert scene.articulation.total_q() == 8
    assert scene.articulation.total_qd() == 7
    return scene


def test_articulated_world_exposes_joint_names_and_name_based_targets():
    scene = _build_single_hip_scene(root_height=0.7)
    solver_settings = novaphy.XPBDSolverSettings()
    solver_settings.substeps = 8
    world = novaphy.ArticulatedWorld(scene, solver_settings)

    assert world.joint_names == ["hip"]

    world.set_joint_positions({"hip": 0.1})
    assert abs(float(world.joint_positions["hip"]) - 0.1) < 1.0e-6

    world.set_gravity(np.zeros(3, dtype=np.float32))
    world.set_default_drive_gains(80.0, 6.0)
    world.set_target_positions({"hip": 0.4})

    for _ in range(60):
        world.step(1.0 / 120.0)

    assert abs(float(world.q[-1]) - 0.4) < 0.05
    assert world.solver.last_stats.projected_constraints > 0


def test_articulated_world_reports_contacts_with_static_ground():
    scene = _build_single_hip_scene(root_height=0.08)
    world = novaphy.ArticulatedWorld(scene)
    world.add_ground_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0)
    world.set_gravity(np.array([0.0, 0.0, -9.81], dtype=np.float32))

    ever_contact = False
    for _ in range(20):
        world.step(1.0 / 120.0)
        ever_contact = ever_contact or bool(world.contacts) or world.solver.last_stats.contact_count > 0

    assert ever_contact
    assert np.all(np.isfinite(world.q))
    assert np.all(np.isfinite(world.qd))


def test_articulated_world_unknown_joint_name_raises():
    scene = _build_single_hip_scene()
    world = novaphy.ArticulatedWorld(scene)

    with pytest.raises(ValueError, match="Unknown joint name"):
        world.set_target_positions({"knee": 0.2})
