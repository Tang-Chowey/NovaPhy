"""Tests for NovaPhy collision detection (broadphase + narrowphase)."""

import numpy as np
import numpy.testing as npt
import novaphy


# --- Broadphase Tests ---

def test_broadphase_no_overlap():
    sap = novaphy.SweepAndPrune()
    aabbs = [
        novaphy.AABB(np.array([-1, -1, -1], dtype=np.float32),
                     np.array([0, 0, 0], dtype=np.float32)),
        novaphy.AABB(np.array([2, 2, 2], dtype=np.float32),
                     np.array([3, 3, 3], dtype=np.float32)),
    ]
    sap.update(aabbs, [False, False])
    assert len(sap.get_pairs()) == 0


def test_broadphase_overlap():
    sap = novaphy.SweepAndPrune()
    aabbs = [
        novaphy.AABB(np.array([-1, -1, -1], dtype=np.float32),
                     np.array([1, 1, 1], dtype=np.float32)),
        novaphy.AABB(np.array([0, 0, 0], dtype=np.float32),
                     np.array([2, 2, 2], dtype=np.float32)),
    ]
    sap.update(aabbs, [False, False])
    pairs = sap.get_pairs()
    assert len(pairs) == 1
    assert pairs[0].body_a == 0
    assert pairs[0].body_b == 1


def test_broadphase_static_static_skip():
    sap = novaphy.SweepAndPrune()
    aabbs = [
        novaphy.AABB(np.array([-1, -1, -1], dtype=np.float32),
                     np.array([1, 1, 1], dtype=np.float32)),
        novaphy.AABB(np.array([0, 0, 0], dtype=np.float32),
                     np.array([2, 2, 2], dtype=np.float32)),
    ]
    sap.update(aabbs, [True, True])
    assert len(sap.get_pairs()) == 0


# --- Narrowphase Tests ---

def test_sphere_sphere_collision():
    a = novaphy.CollisionShape()
    a.type = novaphy.ShapeType.Sphere
    a.sphere_radius = 1.0
    a.body_index = 0

    b = novaphy.CollisionShape()
    b.type = novaphy.ShapeType.Sphere
    b.sphere_radius = 1.0
    b.body_index = 1

    ta = novaphy.Transform.from_translation(np.array([0, 0, 0], dtype=np.float32))
    tb = novaphy.Transform.from_translation(np.array([1.5, 0, 0], dtype=np.float32))

    hit, contacts = novaphy.collide_shapes(a, ta, b, tb)
    assert hit is True
    assert len(contacts) == 1
    assert contacts[0].penetration > 0
    npt.assert_allclose(contacts[0].normal, [1, 0, 0], atol=1e-5)


def test_sphere_sphere_no_collision():
    a = novaphy.CollisionShape()
    a.type = novaphy.ShapeType.Sphere
    a.sphere_radius = 0.5
    a.body_index = 0

    b = novaphy.CollisionShape()
    b.type = novaphy.ShapeType.Sphere
    b.sphere_radius = 0.5
    b.body_index = 1

    ta = novaphy.Transform.from_translation(np.array([0, 0, 0], dtype=np.float32))
    tb = novaphy.Transform.from_translation(np.array([2, 0, 0], dtype=np.float32))

    hit, contacts = novaphy.collide_shapes(a, ta, b, tb)
    assert hit is False
    assert len(contacts) == 0


def test_sphere_plane_collision():
    sphere = novaphy.CollisionShape()
    sphere.type = novaphy.ShapeType.Sphere
    sphere.sphere_radius = 1.0
    sphere.body_index = 0

    plane = novaphy.CollisionShape.make_plane(
        np.array([0, 1, 0], dtype=np.float32), 0.0)

    ts = novaphy.Transform.from_translation(np.array([0, 0.5, 0], dtype=np.float32))
    tp = novaphy.Transform.identity()

    hit, contacts = novaphy.collide_shapes(sphere, ts, plane, tp)
    assert hit is True
    assert len(contacts) == 1
    npt.assert_allclose(contacts[0].penetration, 0.5, atol=1e-5)


def test_box_plane_collision():
    box = novaphy.CollisionShape.make_box(
        np.array([0.5, 0.5, 0.5], dtype=np.float32), 0)

    plane = novaphy.CollisionShape.make_plane(
        np.array([0, 1, 0], dtype=np.float32), 0.0)

    # Box slightly penetrating the ground plane (center at y=0.4, bottom corners at y=-0.1)
    tb = novaphy.Transform.from_translation(np.array([0, 0.4, 0], dtype=np.float32))
    tp = novaphy.Transform.identity()

    hit, contacts = novaphy.collide_shapes(box, tb, plane, tp)
    # Bottom 4 corners should penetrate the plane
    assert hit is True
    assert len(contacts) == 4  # 4 bottom corners below y=0
    for c in contacts:
        assert c.penetration > 0


def test_body_from_box():
    body = novaphy.RigidBody.from_box(2.0, np.array([1, 1, 1], dtype=np.float32))
    assert abs(body.mass - 2.0) < 1e-6
    assert not body.is_static()


def test_body_static():
    body = novaphy.RigidBody.make_static()
    assert body.is_static()
    assert abs(body.inv_mass()) < 1e-6
