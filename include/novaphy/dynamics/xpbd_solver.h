#pragma once

#include <span>

#include "novaphy/collision/broadphase.h"
#include "novaphy/core/articulation.h"
#include "novaphy/core/contact.h"
#include "novaphy/core/control.h"
#include "novaphy/core/model.h"
#include "novaphy/math/math_types.h"
#include "novaphy/math/spatial.h"

namespace novaphy {

/**
 * @brief Runtime tuning parameters for the articulated XPBD solver.
 */
struct XPBDSolverSettings {
    int substeps = 10;               /**< Number of simulation substeps per external step. */
    int iterations = 4;              /**< Position-projection iterations per substep. */
    float velocity_damping = 0.999f; /**< Mild post-step damping for generalized velocities. */
    float contact_relaxation = 0.5f; /**< Fraction of contact penetration corrected per projection. */
    float friction_damping = 0.1f;   /**< Tangential damping factor applied when contacts are active. */
};

/**
 * @brief Lightweight runtime statistics from the most recent XPBD step.
 */
struct XPBDStepStats {
    int substeps = 0;               /**< Number of substeps executed during the last call. */
    int iterations = 0;             /**< Number of configured iterations per substep. */
    int projected_constraints = 0;  /**< Number of position projections applied. */
    int contact_count = 0;          /**< Number of contacts generated in the last step. */
};



/**
 * @brief XPBD-style articulated solver.
 */
class XPBDSolver {
public:
    explicit XPBDSolver(XPBDSolverSettings settings = {});

    void step(const Articulation& model,
              VecXf& q,
              VecXf& qd,
              const VecXf& tau,
              const Vec3f& gravity,
              float dt,
              const Control& control = Control(),
              std::span<const SpatialVector> f_ext = {});

    void step_with_contacts(const Articulation& model,
                            const Model& collision_model,
                            const std::vector<CollisionShape>& static_shapes,
                            VecXf& q,
                            VecXf& qd,
                            const VecXf& tau,
                            const Vec3f& gravity,
                            float dt,
                            const Control& control = Control(),
                            std::span<const SpatialVector> f_ext = {},
                            std::vector<ContactPoint>* contacts = nullptr);

    XPBDSolverSettings& settings() { return settings_; }
    const XPBDSolverSettings& settings() const { return settings_; }
    const XPBDStepStats& last_stats() const { return last_stats_; }

private:
    void integrate_positions(const Articulation& model,
                             VecXf& q,
                             const VecXf& qd,
                             float dt) const;
    void normalize_quaternions(const Articulation& model, VecXf& q) const;
    int project_joint_limits(const Articulation& model,
                             VecXf& q,
                             VecXf& qd) const;
    int project_joint_drives(const Articulation& model,
                             VecXf& q,
                             VecXf& qd,
                             const Control& control,
                             float dt) const;
    int project_contacts(const Articulation& model,
                         const Model& collision_model,
                         const std::vector<CollisionShape>& static_shapes,
                         VecXf& q,
                         VecXf& qd,
                         float dt,
                         std::vector<ContactPoint>* contacts);
    int apply_contact_correction(const Articulation& model,
                                 const std::vector<Transform>& link_transforms,
                                 const ContactPoint& contact,
                                 int link_index,
                                 float sign,
                                 VecXf& q,
                                 VecXf& qd,
                                 float dt) const;

    XPBDSolverSettings settings_;
    XPBDStepStats last_stats_;
    SweepAndPrune broadphase_;
};

}  // namespace novaphy
