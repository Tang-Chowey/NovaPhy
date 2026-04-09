/**
 * @file multibody_constraint.cpp
 * @brief Joint motor and joint limit constraint implementations.
 */
#include "novaphy/multibody/multibody_constraint.h"

#include <algorithm>
#include <cmath>

namespace novaphy {

// ---------------------------------------------------------------------------
// MultiBodyJointMotor
// ---------------------------------------------------------------------------

MultiBodyJointMotor::MultiBodyJointMotor(MultiBody* body, int link,
                                         float desired_velocity,
                                         float max_impulse)
    : desired_velocity_(desired_velocity), max_impulse_(max_impulse) {
    body_a_ = body;
    link_a_ = link;
}

void MultiBodyJointMotor::create_constraint_rows(
    std::vector<MultiBodySolverConstraint>& rows,
    MultiBodyConstraintSolverData& data,
    float dt) {
    if (!body_a_ || link_a_ < 0) return;

    const auto& model = body_a_->model();
    const auto& joint = model.joints[link_a_];
    const int nv_i = joint.num_qd();
    if (nv_i == 0) return;  // Fixed joint

    const int ndof = 6 + body_a_->num_dofs();
    const int dof_offset = model.qd_start(link_a_);
    const int q_offset = model.q_start(link_a_);

    // Current state
    float current_pos = body_a_->q()(q_offset);
    float current_vel = body_a_->qd()(dof_offset);

    // Target joint acceleration proxy: ERP position correction + velocity tracking (PD-style).
    float pos_error = desired_position_ - current_pos;
    float pos_term = erp_ * pos_error / dt;
    float vel_error = desired_velocity_ - current_vel;
    float rhs = kp_ * pos_term + current_vel + kd_ * vel_error;
    rhs = std::clamp(rhs, -rhs_clamp_, rhs_clamp_);

    // Allocate
    int jac_index = data.allocate_jacobian(ndof);
    int vel_index;
    if (body_a_->companion_id < 0) {
        vel_index = data.allocate_delta_velocities(ndof);
        body_a_->companion_id = vel_index;
    } else {
        vel_index = body_a_->companion_id;
    }

    // Jacobian: identity at this joint's DOF
    float* jac = &data.jacobians[jac_index];
    for (int i = 0; i < ndof; ++i) jac[i] = 0.0f;
    jac[6 + dof_offset] = 1.0f;

    // Compute unit impulse response
    float* unit_response = &data.delta_velocities_unit_impulse[jac_index];
    body_a_->calc_acceleration_deltas(jac, unit_response);

    // Effective mass denominator
    float denom = 0.0f;
    for (int i = 0; i < ndof; ++i) {
        denom += jac[i] * unit_response[i];
    }
    if (denom < 1e-10f) return;

    MultiBodySolverConstraint c;
    c.multi_body_a = body_a_;
    c.link_a = link_a_;
    c.delta_vel_a_index = vel_index;
    c.jac_a_index = jac_index;
    c.jac_diag_ab_inv = 1.0f / denom;
    c.rhs = rhs;
    c.cfm = 0.0f;
    c.lower_limit = -max_impulse_;
    c.upper_limit = max_impulse_;
    c.applied_impulse = 0.0f;

    rows.push_back(c);
}

// ---------------------------------------------------------------------------
// MultiBodyJointLimit
// ---------------------------------------------------------------------------

MultiBodyJointLimit::MultiBodyJointLimit(MultiBody* body, int link,
                                         float lower, float upper)
    : lower_bound_(lower), upper_bound_(upper) {
    body_a_ = body;
    link_a_ = link;
}

void MultiBodyJointLimit::create_constraint_rows(
    std::vector<MultiBodySolverConstraint>& rows,
    MultiBodyConstraintSolverData& data,
    float dt) {
    if (!body_a_ || link_a_ < 0) return;

    const auto& model = body_a_->model();
    const auto& joint = model.joints[link_a_];
    const int nv_i = joint.num_qd();
    if (nv_i == 0) return;

    const int ndof = 6 + body_a_->num_dofs();
    const int dof_offset = model.qd_start(link_a_);
    const int q_offset = model.q_start(link_a_);

    float joint_pos = body_a_->q()(q_offset);
    float joint_vel = body_a_->qd()(dof_offset);

    // Row 0: Lower limit
    // Row 1: Upper limit
    for (int row = 0; row < 2; ++row) {
        float penetration;
        float direction;
        if (row == 0) {
            penetration = joint_pos - lower_bound_;
            direction = 1.0f;
        } else {
            penetration = upper_bound_ - joint_pos;
            direction = -1.0f;
        }

        // Only create constraint if limit is violated (penetration < 0)
        if (penetration >= 0.0f) continue;

        int jac_index = data.allocate_jacobian(ndof);
        int vel_index;
        if (body_a_->companion_id < 0) {
            vel_index = data.allocate_delta_velocities(ndof);
            body_a_->companion_id = vel_index;
        } else {
            vel_index = body_a_->companion_id;
        }

        // Jacobian
        float* jac = &data.jacobians[jac_index];
        for (int i = 0; i < ndof; ++i) jac[i] = 0.0f;
        jac[6 + dof_offset] = direction;

        // Unit impulse response
        float* unit_response = &data.delta_velocities_unit_impulse[jac_index];
        body_a_->calc_acceleration_deltas(jac, unit_response);

        float denom = 0.0f;
        for (int i = 0; i < ndof; ++i) {
            denom += jac[i] * unit_response[i];
        }
        if (denom < 1e-10f) continue;

        // RHS: ERP-based position correction + velocity term
        float rel_vel = direction * joint_vel;
        float positional_error = -penetration * erp_ / dt;
        float velocity_error = -rel_vel;

        MultiBodySolverConstraint c;
        c.multi_body_a = body_a_;
        c.link_a = link_a_;
        c.delta_vel_a_index = vel_index;
        c.jac_a_index = jac_index;
        c.jac_diag_ab_inv = 1.0f / denom;
        c.rhs = positional_error + velocity_error;
        c.cfm = 0.0f;
        c.lower_limit = 0.0f;           // One-way: can only push
        c.upper_limit = 1e10f;
        c.applied_impulse = 0.0f;

        rows.push_back(c);
    }
}

}  // namespace novaphy
