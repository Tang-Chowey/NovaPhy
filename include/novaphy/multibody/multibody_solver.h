#pragma once

#include <memory>
#include <vector>

#include "novaphy/collision/broadphase.h"
#include "novaphy/core/contact.h"
#include "novaphy/core/control.h"
#include "novaphy/core/model.h"
#include "novaphy/multibody/multibody.h"
#include "novaphy/multibody/multibody_constraint.h"
#include "novaphy/multibody/free_rigid_body.h"
#include "novaphy/sim/state.h"

namespace novaphy {

/**
 * @brief Tuning parameters for the multibody PGS constraint solver.
 *
 * @note Warm-starting is not currently implemented by MultiBodySolver.
 * `warmstarting_factor` and `use_warmstarting` are retained for API
 * compatibility and are ignored by the current solver implementation.
 */
struct MultiBodySolverSettings {
    int num_iterations = 10;             /**< PGS iteration count. */
    float erp = 0.2f;                   /**< Error reduction parameter (contacts). */
    float cfm = 0.0f;                   /**< Constraint force mixing (regularization). */
    float linear_slop = 0.005f;         /**< Penetration tolerance before correction (m). */
    float warmstarting_factor = 0.85f;  /**< Unused: warm-starting is not currently implemented. */
    float restitution_threshold = 0.2f; /**< Minimum relative velocity for restitution. */
    bool use_warmstarting = false;      /**< Unused: warm-starting is not currently implemented. */
};

/**
 * @brief Featherstone ABA + PGS constraint solver for articulated multibodies.
 *
 * @details Follows the XPBDSolver pattern: a solver component owned by World
 * that reads/writes SimState and Control. Internally manages MultiBody
 * instances created from Model.articulations and handles collision detection
 * and constraint solving for the articulated pipeline.
 *
 * Pipeline per step:
 *   sync-in -> clear forces -> FK -> ABA -> broadphase -> narrowphase ->
 *   PGS solve -> position integration -> split-impulse ERP -> FK -> sync-out
 */
class MultiBodySolver {
public:
    explicit MultiBodySolver(MultiBodySolverSettings settings = {});

    /**
     * @brief Initialize solver from model definition and initial state.
     *
     * Creates MultiBody instances from Model.articulations, extracts link
     * colliders and static shapes from Model.shapes, and creates joint limit
     * constraints from Joint.limit_enabled.
     */
    void init(const Model& model, const SimState& state);

    /**
     * @brief Advance all articulated bodies by one time step.
     *
     * Reads q/qd from SimState, applies Control inputs, runs the full
     * Featherstone pipeline, and writes updated q/qd back to SimState.
     */
    void step(const Model& model, SimState& state, const Control& control,
              const Vec3f& gravity, float dt);

    const std::vector<ContactPoint>& contacts() const { return contacts_; }

    MultiBodySolverSettings& settings() { return constraint_settings_; }
    const MultiBodySolverSettings& settings() const {
        return constraint_settings_;
    }

    const std::vector<std::unique_ptr<MultiBody>>& bodies() const { return bodies_; }
    const std::vector<std::unique_ptr<FreeRigidBody>>& free_rigid_bodies() const {
        return free_rigid_bodies_;
    }

private:
    struct LinkCollider {
        int body_index;
        int link_index;
        CollisionShape shape;
    };

    struct RBCollider {
        int rb_index;
        CollisionShape shape;
    };

    void sync_state_in(const SimState& state);
    void sync_state_out(SimState& state) const;
    void sync_rb_state_in(const SimState& state);
    void sync_rb_state_out(SimState& state) const;
    void apply_control(const Control& control);
    void run_collision_detection();
    void run_split_impulse_correction();

    // --- PGS (contact + friction + user constraints), formerly MultiBodyConstraintSolver ---
    void solve_constraints(std::vector<MultiBody*>& multibodies,
                           std::vector<FreeRigidBody*>& free_rigid_bodies,
                           std::vector<ContactPoint>& contacts,
                           std::vector<MultiBodyConstraint*>& constraints,
                           float dt);
    void pgs_setup(std::vector<MultiBody*>& multibodies,
                   std::vector<FreeRigidBody*>& free_rigid_bodies,
                   std::vector<ContactPoint>& contacts,
                   std::vector<MultiBodyConstraint*>& constraints,
                   float dt);
    void pgs_setup_rigid_contact(
        FreeRigidBody* rb_a, MultiBody* mb_a, int link_a,
        FreeRigidBody* rb_b, MultiBody* mb_b, int link_b,
        const Vec3f& normal, const Vec3f& point,
        float penetration, float friction, float restitution,
        float dt);
    void pgs_add_friction_constraint(
        FreeRigidBody* rb_a, MultiBody* mb_a, int link_a,
        FreeRigidBody* rb_b, MultiBody* mb_b, int link_b,
        const Vec3f& point, const Vec3f& friction_dir,
        float dt, int normal_index);
    void pgs_solve_iterations();
    float pgs_resolve_row(MultiBodySolverConstraint& c);
    void pgs_finalize(std::vector<MultiBody*>& multibodies,
                    std::vector<FreeRigidBody*>& free_rigid_bodies);

    std::vector<std::unique_ptr<MultiBody>> bodies_;
    std::vector<std::unique_ptr<FreeRigidBody>> free_rigid_bodies_;
    std::vector<std::unique_ptr<MultiBodyConstraint>> persistent_constraints_;
    std::vector<CollisionShape> static_shapes_;
    std::vector<LinkCollider> link_colliders_;
    std::vector<RBCollider> rigid_body_colliders_;

    MultiBodySolverSettings constraint_settings_;
    MultiBodyConstraintSolverData constraint_data_;
    std::vector<MultiBodySolverConstraint> normal_constraints_;
    std::vector<MultiBodySolverConstraint> friction_constraints_;
    std::vector<MultiBodySolverConstraint> non_contact_constraints_;

    SweepAndPrune broadphase_;
    std::vector<ContactPoint> contacts_;
    Vec3f gravity_ = Vec3f::Zero();
};

}  // namespace novaphy
