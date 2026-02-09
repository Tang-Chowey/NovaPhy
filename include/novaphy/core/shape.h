#pragma once

#include "novaphy/core/aabb.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

/// Collision shape types
enum class ShapeType { Box, Sphere, Plane };

/// Box: defined by half-extents
struct BoxShape {
    Vec3f half_extents = Vec3f(0.5f, 0.5f, 0.5f);
};

/// Sphere: defined by radius
struct SphereShape {
    float radius = 0.5f;
};

/// Infinite plane: defined by normal and distance from origin
struct PlaneShape {
    Vec3f normal = Vec3f(0.0f, 1.0f, 0.0f);
    float offset = 0.0f;  // distance from origin along normal
};

/// A collision shape attached to a body
struct CollisionShape {
    ShapeType type = ShapeType::Box;

    BoxShape box;
    SphereShape sphere;
    PlaneShape plane;

    Transform local_transform = Transform::identity();  // shape offset in body frame
    float friction = 0.5f;
    float restitution = 0.3f;

    int body_index = -1;  // which body this shape belongs to

    /// Compute world-space AABB given the body's world transform
    AABB compute_aabb(const Transform& body_transform) const {
        Transform world = body_transform * local_transform;
        switch (type) {
            case ShapeType::Box:
                return AABB::from_oriented_box(box.half_extents, world);
            case ShapeType::Sphere:
                return AABB::from_sphere(world.position, sphere.radius);
            case ShapeType::Plane:
                // Planes are infinite; use a very large AABB
                return AABB(Vec3f::Constant(-1e6f), Vec3f::Constant(1e6f));
        }
        return AABB();
    }

    /// Create a box collision shape
    static CollisionShape make_box(const Vec3f& half_extents, int body_idx,
                                   const Transform& local = Transform::identity(),
                                   float friction = 0.5f, float restitution = 0.3f) {
        CollisionShape s;
        s.type = ShapeType::Box;
        s.box.half_extents = half_extents;
        s.body_index = body_idx;
        s.local_transform = local;
        s.friction = friction;
        s.restitution = restitution;
        return s;
    }

    /// Create a sphere collision shape
    static CollisionShape make_sphere(float radius, int body_idx,
                                      const Transform& local = Transform::identity(),
                                      float friction = 0.5f, float restitution = 0.3f) {
        CollisionShape s;
        s.type = ShapeType::Sphere;
        s.sphere.radius = radius;
        s.body_index = body_idx;
        s.local_transform = local;
        s.friction = friction;
        s.restitution = restitution;
        return s;
    }

    /// Create an infinite ground plane shape
    static CollisionShape make_plane(const Vec3f& normal, float offset,
                                     float friction = 0.5f, float restitution = 0.0f) {
        CollisionShape s;
        s.type = ShapeType::Plane;
        s.plane.normal = normal.normalized();
        s.plane.offset = offset;
        s.body_index = -1;  // planes are typically world-owned
        s.friction = friction;
        s.restitution = restitution;
        return s;
    }
};

}  // namespace novaphy
