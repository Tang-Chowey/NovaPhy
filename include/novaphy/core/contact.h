#pragma once

#include "novaphy/math/math_types.h"
#include <vector>

namespace novaphy {

/// A single contact point between two bodies
struct ContactPoint {
    Vec3f position = Vec3f::Zero();   // world-space contact point
    Vec3f normal = Vec3f::Zero();     // contact normal (from body_a toward body_b)
    float penetration = 0.0f;         // positive = overlapping

    int body_a = -1;                  // body index (or -1 for world)
    int body_b = -1;                  // body index (or -1 for world)
    int shape_a = -1;                 // shape index
    int shape_b = -1;                 // shape index

    float friction = 0.5f;            // combined friction coefficient
    float restitution = 0.0f;         // combined restitution coefficient

    // Solver cache (warm starting)
    float accumulated_normal_impulse = 0.0f;
    float accumulated_tangent_impulse_1 = 0.0f;
    float accumulated_tangent_impulse_2 = 0.0f;
};

/// A contact manifold: a set of contact points between the same pair of shapes
struct ContactManifold {
    int shape_a = -1;
    int shape_b = -1;
    int body_a = -1;
    int body_b = -1;
    std::vector<ContactPoint> points;
};

/// Combine friction coefficients (geometric mean)
inline float combine_friction(float a, float b) {
    return std::sqrt(a * b);
}

/// Combine restitution coefficients (maximum)
inline float combine_restitution(float a, float b) {
    return std::max(a, b);
}

}  // namespace novaphy
