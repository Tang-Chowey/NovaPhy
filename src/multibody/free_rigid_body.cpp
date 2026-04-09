/**
 * @file free_rigid_body.cpp
 * @brief Free rigid body state integration for the multibody solver pipeline.
 *
 * @details Mass/inertia setup, force accumulation, damping, velocity integration,
 * and pose prediction for Model.bodies entries handled alongside Featherstone
 * articulations in MultiBodySolver.
 */
#include "novaphy/multibody/free_rigid_body.h"

#include <cmath>

namespace novaphy {

FreeRigidBody::FreeRigidBody(float mass, const Vec3f& local_inertia,
                             const Transform& start_transform,
                             float linear_damping, float angular_damping)
    : linear_damping_(linear_damping), angular_damping_(angular_damping) {
    world_transform_ = start_transform;
    set_mass_props(mass, local_inertia);
    update_inertia_tensor();
}

void FreeRigidBody::set_mass_props(float mass, const Vec3f& inertia) {
    if (mass > 0.0f) {
        mass_ = mass;
        inv_mass_ = 1.0f / mass;
        inv_inertia_local_ = Vec3f(
            inertia.x() > 0.0f ? 1.0f / inertia.x() : 0.0f,
            inertia.y() > 0.0f ? 1.0f / inertia.y() : 0.0f,
            inertia.z() > 0.0f ? 1.0f / inertia.z() : 0.0f
        );
    } else {
        mass_ = 0.0f;
        inv_mass_ = 0.0f;
        inv_inertia_local_ = Vec3f::Zero();
    }
}

void FreeRigidBody::update_inertia_tensor() {
    const Mat3f rot = world_transform_.rotation_matrix();
    const Mat3f rot_t = rot.transpose();
    inv_inertia_tensor_world_ = rot * inv_inertia_local_.asDiagonal() * rot_t;
}

void FreeRigidBody::clear_forces() {
    total_force_ = Vec3f::Zero();
    total_torque_ = Vec3f::Zero();
}

void FreeRigidBody::apply_gravity() {
    if (is_static_or_kinematic()) return;
    total_force_ += gravity_acceleration_ * mass_;
}

void FreeRigidBody::apply_damping(float dt) {
    linear_velocity_ *= std::pow(1.0f - linear_damping_, dt);
    angular_velocity_ *= std::pow(1.0f - angular_damping_, dt);
}

void FreeRigidBody::integrate_velocities(float dt) {
    if (is_static_or_kinematic()) return;

    linear_velocity_ += (total_force_ * inv_mass_) * dt;
    angular_velocity_ += (inv_inertia_tensor_world_ * total_torque_) * dt;
}

void FreeRigidBody::predict_integrated_transform(float dt, Transform& predicted) const {
    predicted = world_transform_;
    predicted.position += linear_velocity_ * dt;

    Vec3f axis = angular_velocity_;
    float angle = axis.norm();
    if (angle > 1e-8f) {
        axis /= angle;
        angle *= dt;
        Quatf delta_q(Eigen::AngleAxisf(angle, axis));
        predicted.rotation = delta_q * predicted.rotation;
        predicted.rotation.normalize();
    }
}

void FreeRigidBody::proceed_to_transform(const Transform& new_trans) {
    world_transform_ = new_trans;
    update_inertia_tensor();
}

}  // namespace novaphy
