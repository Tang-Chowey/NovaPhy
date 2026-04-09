#pragma once

#include <Eigen/Dense>
#include <vector>

#include "novaphy/core/articulation.h"
#include "novaphy/dynamics/featherstone.h"
#include "novaphy/math/math_types.h"
#include "novaphy/math/spatial.h"

namespace novaphy {

/**
 * @brief Stateful multibody system backed by Featherstone forward dynamics.
 *
 * @details Holds mutable generalized state (positions, velocities, forces)
 * and delegates forward dynamics to featherstone::forward_dynamics() (CRBA +
 * Cholesky). The FK result and factored mass matrix are cached after each
 * dynamics call for use by the constraint solver.
 *
 * NovaPhy's Articulation treats each link as having a Joint connecting it to
 * its parent. For floating-base robots, the first joint should be JointType::Free.
 *
 * Spatial convention: [angular; linear] (Featherstone standard).
 */
class MultiBody {
public:
    /**
     * @brief Construct MultiBody from articulation model.
     * @param model Articulation topology and inertia (must outlive this object).
     * @param linear_damping Linear velocity damping coefficient.
     * @param angular_damping Angular velocity damping coefficient.
     */
    explicit MultiBody(const Articulation& model,
                       float linear_damping = 0.0f,
                       float angular_damping = 0.0f);

    // --- Accessors ---
    const Articulation& model() const { return model_; }
    int num_links() const { return model_.num_links(); }
    int num_dofs() const { return model_.total_qd(); }
    int num_q() const { return model_.total_q(); }

    // --- Generalized state ---
    VecXf& q() { return q_; }
    const VecXf& q() const { return q_; }
    VecXf& qd() { return qd_; }
    const VecXf& qd() const { return qd_; }
    VecXf& tau() { return tau_; }
    const VecXf& tau() const { return tau_; }

    // --- Per-link external forces (world frame) ---
    void add_link_force(int link, const Vec3f& f);
    void add_link_torque(int link, const Vec3f& t);
    void clear_forces();

    // --- Forward Dynamics ---
    /**
     * @brief Run forward dynamics via CRBA + Cholesky solve.
     *
     * @details Delegates to featherstone::forward_dynamics(). Caches FK result
     * and factored mass matrix for subsequent constraint solver calls.
     *
     * @param dt Time step (s). If 0, only computes accelerations without velocity update.
     * @param is_constraint_pass If true, uses constraint forces instead of external forces.
     */
    void compute_accelerations_aba(float dt, bool is_constraint_pass = false,
                                    const Vec3f& gravity = Vec3f::Zero());

    // --- Constraint Solver Support ---
    /**
     * @brief Compute acceleration deltas from a test force via mass matrix solve.
     *
     * @details Solves H * delta_a = J using the Cholesky factorization cached
     * from the last compute_accelerations_aba() call.
     *
     * @param force_in Test force vector (6 + num_dofs), base [angular; linear] + joints.
     * @param output Acceleration delta vector (6 + num_dofs).
     */
    void calc_acceleration_deltas(const float* force_in, float* output) const;

    /**
     * @brief Compute contact Jacobian for a constraint point on a link.
     *
     * @details Uses cached FK result from last compute_accelerations_aba() call.
     *
     * @param link Link index where contact occurs.
     * @param contact_point World-space contact point (m).
     * @param normal_ang Angular component of constraint normal.
     * @param normal_lin Linear component of constraint normal.
     * @param jac Output Jacobian array (6 + num_dofs).
     */
    void fill_constraint_jacobian(int link, const Vec3f& contact_point,
                                  const Vec3f& normal_ang, const Vec3f& normal_lin,
                                  float* jac) const;


    // --- Position Integration ---
    /**
     * @brief Integrate positions using current velocities.
     *
     * @details Semi-implicit Euler for linear DOFs. Quaternion exponential map
     * for rotational DOFs (Free and Ball joints).
     *
     * @param dt Time step (s).
     */
    void step_positions(float dt);

    /**
     * @brief Apply per-step velocity damping (matches ArticulatedSolver).
     * @details Multiplies all generalized velocities by 0.999 each step.
     *          Must be called after step_positions() to match ArticulatedSolver order.
     */
    void apply_velocity_damping() { qd_ *= 0.999f; }

    // --- Forward Kinematics ---
    /**
     * @brief Update cached world transforms for all links.
     */
    void forward_kinematics();

    /** @brief Get cached world transforms (valid after forward_kinematics()). */
    std::vector<Transform>& cached_world_transforms() { return world_transforms_; }
    const std::vector<Transform>& cached_world_transforms() const { return world_transforms_; }

    /** @brief Companion ID for constraint solver indexing. */
    int companion_id = -1;

private:
    const Articulation& model_;
    float linear_damping_ = 0.0f;
    float angular_damping_ = 0.0f;

    // Generalized state
    VecXf q_;    // positions
    VecXf qd_;   // velocities
    VecXf tau_;  // joint torques

    // Per-link external forces (world frame)
    std::vector<Vec3f> link_forces_;
    std::vector<Vec3f> link_torques_;
    // Constraint forces (world frame)
    std::vector<Vec3f> constraint_forces_;
    std::vector<Vec3f> constraint_torques_;

    // FK cache (populated by compute_accelerations_aba and forward_kinematics)
    featherstone::ForwardKinematicsResult fk_cache_;

    // Mass matrix cache (populated by compute_accelerations_aba)
    MatXf mass_matrix_cache_;
    Eigen::LLT<MatXf> mass_llt_;

    // FK world transforms (alias of fk_cache_.world_transforms for external access)
    std::vector<Transform> world_transforms_;

    // Output buffer (6 + num_dofs)
    VecXf output_;

    // Pre-computed per-link DOF offsets (cached for performance)
    std::vector<int> dof_offsets_;
    std::vector<int> cfg_offsets_;
};

}  // namespace novaphy
