#pragma once

#include <vector>

#include "novaphy/multibody/multibody.h"
#include "novaphy/multibody/free_rigid_body.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

/**
 * @brief One scalar constraint row for the projected Gauss–Seidel (PGS) solver.
 *
 * @details Holds offsets into shared Jacobian and delta-velocity buffers, rigid or
 * articulated body pointers, precomputed rigid-body geometric data, impulse
 * limits, and the row right-hand side for a single constraint equation.
 */
struct MultiBodySolverConstraint {
    MultiBody* multi_body_a = nullptr;
    int link_a = -1;
    int delta_vel_a_index = -1; /**< Offset into global delta_velocities array. */
    int jac_a_index = -1;       /**< Offset into global Jacobians array. */

    MultiBody* multi_body_b = nullptr;
    int link_b = -1;
    int delta_vel_b_index = -1;
    int jac_b_index = -1;

    /** @brief Optional free-body endpoints when the constraint does not use multibody_a/b. */
    FreeRigidBody* free_rigid_body_a = nullptr;
    FreeRigidBody* free_rigid_body_b = nullptr;
    // Pre-computed rigid body Jacobian vectors
    Vec3f contact_normal_1 = Vec3f::Zero();      /**< Linear Jacobian dir for body A (= -n or +n). */
    Vec3f contact_normal_2 = Vec3f::Zero();      /**< Linear Jacobian dir for body B. */
    Vec3f relpos1_cross_normal = Vec3f::Zero();  /**< r_a x n (angular Jacobian). */
    Vec3f relpos2_cross_normal = Vec3f::Zero();  /**< r_b x n. */
    Vec3f angular_component_a = Vec3f::Zero();   /**< I_a_inv * relpos1_cross_normal. */
    Vec3f angular_component_b = Vec3f::Zero();   /**< I_b_inv * relpos2_cross_normal. */

    float applied_impulse = 0.0f;
    float jac_diag_ab_inv = 0.0f;  /**< 1 / (J * M^{-1} * J^T + CFM). */
    float rhs = 0.0f;              /**< Right-hand side target velocity. */
    float cfm = 0.0f;              /**< Constraint force mixing (regularization). */
    float lower_limit = 0.0f;
    float upper_limit = 0.0f;
    float rhs_penetration = 0.0f;  /**< Split impulse penetration term. */

    int friction_index = -1;       /**< Index of corresponding normal row (for friction). */
    float friction = 0.0f;         /**< Friction coefficient. */
};

/**
 * @brief Shared data arrays for constraint solver.
 */
struct MultiBodyConstraintSolverData {
    std::vector<float> jacobians;
    std::vector<float> delta_velocities;
    std::vector<float> delta_velocities_unit_impulse;

    int allocate_jacobian(int ndof) {
        int idx = static_cast<int>(jacobians.size());
        jacobians.resize(idx + ndof, 0.0f);
        delta_velocities_unit_impulse.resize(idx + ndof, 0.0f);
        return idx;
    }

    int allocate_delta_velocities(int ndof) {
        int idx = static_cast<int>(delta_velocities.size());
        delta_velocities.resize(idx + ndof, 0.0f);
        return idx;
    }

    void clear() {
        jacobians.clear();
        delta_velocities.clear();
        delta_velocities_unit_impulse.clear();
    }
};

/**
 * @brief Abstract base for user-defined multibody constraints (motors, limits, etc.).
 *
 * @details Implementations expand into one or more MultiBodySolverConstraint rows
 * for the PGS solver each substep.
 */
class MultiBodyConstraint {
public:
    virtual ~MultiBodyConstraint() = default;

    /** @brief Number of scalar constraint rows. */
    virtual int num_rows() const = 0;

    /** @brief Generate solver constraint rows. */
    virtual void create_constraint_rows(
        std::vector<MultiBodySolverConstraint>& rows,
        MultiBodyConstraintSolverData& data,
        float dt) = 0;

    MultiBody* body_a() const { return body_a_; }
    int link_a() const { return link_a_; }
    MultiBody* body_b() const { return body_b_; }
    int link_b() const { return link_b_; }

protected:
    MultiBody* body_a_ = nullptr;
    int link_a_ = -1;
    MultiBody* body_b_ = nullptr;
    int link_b_ = -1;
};

/**
 * @brief Joint motor: drives one scalar joint DOF toward position and/or velocity targets.
 *
 * @details Builds one equality row with ERP position correction, optional PD-style
 * velocity terms, and impulse clamping per substep.
 */
class MultiBodyJointMotor : public MultiBodyConstraint {
public:
    /**
     * @param body The multibody system.
     * @param link Link index whose joint is driven.
     * @param desired_velocity Target joint velocity (rad/s or m/s).
     * @param max_impulse Maximum impulse per step.
     */
    MultiBodyJointMotor(MultiBody* body, int link,
                        float desired_velocity = 0.0f,
                        float max_impulse = 1e10f);

    void set_position_target(float target, float kp) {
        desired_position_ = target; kp_ = kp;
    }
    void set_velocity_target(float target, float kd) {
        desired_velocity_ = target; kd_ = kd;
    }
    void set_erp(float erp) { erp_ = erp; }
    void set_max_impulse(float m) { max_impulse_ = m; }

    int num_rows() const override { return 1; }
    void create_constraint_rows(
        std::vector<MultiBodySolverConstraint>& rows,
        MultiBodyConstraintSolverData& data,
        float dt) override;

private:
    float desired_position_ = 0.0f;
    float desired_velocity_ = 0.0f;
    float kp_ = 0.0f;
    float kd_ = 1.0f;
    float erp_ = 0.2f;
    float max_impulse_ = 1e10f;
    float rhs_clamp_ = 1e10f;
};

/**
 * @brief Joint position limits on one scalar DOF.
 *
 * @details Emits up to two one-sided inequality rows (lower and upper bounds)
 * with Baumgarte-style penetration correction via ERP.
 */
class MultiBodyJointLimit : public MultiBodyConstraint {
public:
    /**
     * @param body The multibody system.
     * @param link Link index whose joint is limited.
     * @param lower Lower position bound.
     * @param upper Upper position bound.
     */
    MultiBodyJointLimit(MultiBody* body, int link, float lower, float upper);

    int num_rows() const override { return 2; }
    void create_constraint_rows(
        std::vector<MultiBodySolverConstraint>& rows,
        MultiBodyConstraintSolverData& data,
        float dt) override;

private:
    float lower_bound_;
    float upper_bound_;
    float erp_ = 0.2f;
};

}  // namespace novaphy
