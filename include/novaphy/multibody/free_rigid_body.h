/*
 * NovaPhy Physics Engine
 * Free rigid body (maximal coordinates) for the multibody solver pipeline.
 *
 * Represents one dynamic/static entry from Model.bodies when using MultiBodySolver.
 * Constructed with direct mass/inertia parameters (no ConstructionInfo).
 */

#pragma once

#include "novaphy/math/math_types.h"

namespace novaphy {

/**
 * @brief Dynamic rigid body with velocity, forces, and integration.
 *
 * Used for free (non-articulated) bodies in the same World as Featherstone
 * articulations. Three regimes:
 * - Dynamic: positive mass, driven by forces and contacts
 * - Static: zero mass, immovable
 * - Kinematic: reserved; currently treated like static for integration skips
 */
class FreeRigidBody {
public:
    /**
     * @brief Construct a rigid body with direct parameters.
     * @param mass Body mass (kg). 0 = static.
     * @param local_inertia Inertia tensor diagonal (kg*m^2).
     * @param start_transform Initial world transform.
     * @param linear_damping Linear velocity damping coefficient.
     * @param angular_damping Angular velocity damping coefficient.
     */
    FreeRigidBody(float mass, const Vec3f& local_inertia,
                  const Transform& start_transform = Transform::identity(),
                  float linear_damping = 0.0f,
                  float angular_damping = 0.0f);

    ~FreeRigidBody() = default;

    const Transform& world_transform() const { return world_transform_; }
    void set_world_transform(const Transform& t) { world_transform_ = t; }
    void translate(const Vec3f& v) { world_transform_.position += v; }

    float mass() const { return mass_; }
    float inv_mass() const { return inv_mass_; }

    void set_mass_props(float mass, const Vec3f& inertia);

    const Mat3f& inv_inertia_tensor_world() const { return inv_inertia_tensor_world_; }

    void update_inertia_tensor();

    const Vec3f& linear_velocity() const { return linear_velocity_; }
    void set_linear_velocity(const Vec3f& v) { linear_velocity_ = v; }

    const Vec3f& angular_velocity() const { return angular_velocity_; }
    void set_angular_velocity(const Vec3f& v) { angular_velocity_ = v; }

    void clear_forces();
    void apply_force(const Vec3f& f) { total_force_ += f; }
    void apply_torque(const Vec3f& t) { total_torque_ += t; }

    void set_gravity(const Vec3f& g) { gravity_acceleration_ = g; }
    void apply_gravity();

    void apply_damping(float dt);

    void integrate_velocities(float dt);
    void predict_integrated_transform(float dt, Transform& predicted) const;
    void proceed_to_transform(const Transform& new_trans);

    Vec3f& delta_linear_velocity() { return delta_linear_velocity_; }
    Vec3f& delta_angular_velocity() { return delta_angular_velocity_; }

    const Vec3f& center_of_mass_position() const { return world_transform_.position; }

private:
    bool is_static_or_kinematic() const { return inv_mass_ <= 0.0f; }

    Transform world_transform_ = Transform::identity();

    float mass_ = 0.0f;
    float inv_mass_ = 0.0f;
    Vec3f inv_inertia_local_ = Vec3f::Zero();
    Mat3f inv_inertia_tensor_world_ = Mat3f::Zero();

    Vec3f linear_velocity_ = Vec3f::Zero();
    Vec3f angular_velocity_ = Vec3f::Zero();

    Vec3f total_force_ = Vec3f::Zero();
    Vec3f total_torque_ = Vec3f::Zero();
    Vec3f gravity_acceleration_ = Vec3f::Zero();

    float linear_damping_ = 0.0f;
    float angular_damping_ = 0.0f;

    Vec3f delta_linear_velocity_ = Vec3f::Zero();
    Vec3f delta_angular_velocity_ = Vec3f::Zero();
};

}  // namespace novaphy
