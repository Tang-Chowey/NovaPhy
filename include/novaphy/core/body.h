#pragma once

#include "novaphy/math/math_types.h"
#include "novaphy/math/spatial.h"

namespace novaphy {

/// Rigid body: mass properties and physical state.
struct RigidBody {
    float mass = 1.0f;
    Mat3f inertia = Mat3f::Identity();  // body-frame inertia tensor (about COM)
    Vec3f com = Vec3f::Zero();          // center of mass in body frame

    /// Inverse mass (0 for static bodies)
    float inv_mass() const { return mass > 0.0f ? 1.0f / mass : 0.0f; }

    /// Inverse inertia tensor (zero for static bodies)
    Mat3f inv_inertia() const {
        if (mass <= 0.0f) return Mat3f::Zero();
        return inertia.inverse();
    }

    /// Is this body static (infinite mass)?
    bool is_static() const { return mass <= 0.0f; }

    /// Build 6x6 spatial inertia matrix at the body frame origin
    SpatialMatrix spatial_inertia() const {
        return spatial_inertia_matrix(mass, com, inertia);
    }

    /// Create a RigidBody with box inertia
    static RigidBody from_box(float m, const Vec3f& half_extents) {
        RigidBody b;
        b.mass = m;
        float w = 2.0f * half_extents.x();
        float h = 2.0f * half_extents.y();
        float d = 2.0f * half_extents.z();
        float c = m / 12.0f;
        b.inertia = Mat3f::Zero();
        b.inertia(0, 0) = c * (h * h + d * d);
        b.inertia(1, 1) = c * (w * w + d * d);
        b.inertia(2, 2) = c * (w * w + h * h);
        return b;
    }

    /// Create a RigidBody with sphere inertia
    static RigidBody from_sphere(float m, float radius) {
        RigidBody b;
        b.mass = m;
        float I = (2.0f / 5.0f) * m * radius * radius;
        b.inertia = Mat3f::Identity() * I;
        return b;
    }

    /// Create a static (immovable) body
    static RigidBody make_static() {
        RigidBody b;
        b.mass = 0.0f;
        b.inertia = Mat3f::Zero();
        return b;
    }
};

}  // namespace novaphy
