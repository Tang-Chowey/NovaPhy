/**
 * @file model_builder.cpp
 * @brief Builder utilities for assembling simulation models.
 */
#include "novaphy/core/model_builder.h"

#include <cmath>

#include "novaphy/core/model.h"

namespace novaphy {

int ModelBuilder::add_body(const RigidBody& body, const Transform& transform) {
    int idx = static_cast<int>(bodies_.size());
    bodies_.push_back(body);
    initial_transforms_.push_back(transform);
    return idx;
}

int ModelBuilder::add_shape(const CollisionShape& shape) {
    int idx = static_cast<int>(shapes_.size());
    shapes_.push_back(shape);
    return idx;
}

int ModelBuilder::add_ground_plane(float y, float friction, float restitution) {
    CollisionShape plane = CollisionShape::make_plane(
        Vec3f(0.0f, 1.0f, 0.0f), y, friction, restitution);
    return add_shape(plane);
}

// ---- Newton-style convenience API ----

int ModelBuilder::add_shape_box(const Vec3f& half_extents,
                                const Transform& transform,
                                float density,
                                float friction,
                                float restitution,
                                bool is_static) {
    RigidBody body;
    if (is_static) {
        body = RigidBody::make_static();
    } else {
        // volume = 8 * hx * hy * hz
        float volume = 8.0f * half_extents.x() * half_extents.y() * half_extents.z();
        float mass = density * volume;
        body = RigidBody::from_box(mass, half_extents);
    }
    int body_idx = add_body(body, transform);
    add_shape(CollisionShape::make_box(half_extents, body_idx,
                                        Transform::identity(), friction, restitution));
    return body_idx;
}

int ModelBuilder::add_shape_sphere(float radius,
                                   const Transform& transform,
                                   float density,
                                   float friction,
                                   float restitution,
                                   bool is_static) {
    RigidBody body;
    if (is_static) {
        body = RigidBody::make_static();
    } else {
        // volume = (4/3) * pi * r^3
        constexpr float pi = 3.14159265358979323846f;
        float volume = (4.0f / 3.0f) * pi * radius * radius * radius;
        float mass = density * volume;
        body = RigidBody::from_sphere(mass, radius);
    }
    int body_idx = add_body(body, transform);
    add_shape(CollisionShape::make_sphere(radius, body_idx,
                                           Transform::identity(), friction, restitution));
    return body_idx;
}

int ModelBuilder::add_shape_cylinder(float radius, float half_length,
                                     const Transform& transform,
                                     float density,
                                     float friction,
                                     float restitution,
                                     bool is_static) {
    RigidBody body;
    if (is_static) {
        body = RigidBody::make_static();
    } else {
        // volume = pi * r^2 * 2 * half_length
        constexpr float pi = 3.14159265358979323846f;
        float volume = pi * radius * radius * 2.0f * half_length;
        float mass = density * volume;
        body = RigidBody::from_cylinder(mass, radius, 2.0f * half_length);
    }
    int body_idx = add_body(body, transform);
    add_shape(CollisionShape::make_cylinder(radius, half_length, body_idx,
                                             Transform::identity(), friction, restitution));
    return body_idx;
}

int ModelBuilder::add_shape_to_body(int body_index, const CollisionShape& shape) {
    CollisionShape s = shape;
    s.body_index = body_index;
    return add_shape(s);
}

void ModelBuilder::add_collision_filter(int shape_a, int shape_b) {
    CollisionFilterPair pair;
    pair.shape_a = std::min(shape_a, shape_b);
    pair.shape_b = std::max(shape_a, shape_b);
    collision_filter_pairs_.push_back(pair);
}

Model ModelBuilder::build() const {
    Model m;
    m.bodies = bodies_;
    m.initial_transforms = initial_transforms_;
    m.shapes = shapes_;
    m.collision_filter_pairs = collision_filter_pairs_;
    m.gravity = gravity_;
    return m;
}

}  // namespace novaphy
