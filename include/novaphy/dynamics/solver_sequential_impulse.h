#pragma once

#include <vector>

#include "novaphy/collision/broadphase.h"
#include "novaphy/collision/narrowphase.h"
#include "novaphy/dynamics/free_body_solver.h"
#include "novaphy/dynamics/integrator.h"
#include "novaphy/dynamics/solver_base.h"

namespace novaphy {

/**
 * @brief Sequential Impulse (PGS) solver for free rigid bodies.
 *
 * @details Wraps the existing FreeBodySolver with broadphase + narrowphase
 * collision detection into a self-contained SolverBase implementation.
 * Suitable for scenes with only free (non-articulated) rigid bodies.
 *
 * Pipeline per step:
 *   1. Integrate velocities (gravity + external forces)
 *   2. Build AABBs → broadphase (SAP) → narrowphase
 *   3. Solve contacts via PGS with accumulated-impulse clamping
 *   4. Integrate positions
 */
class SolverSequentialImpulse : public SolverBase {
public:
    explicit SolverSequentialImpulse(SolverSettings settings = {})
        : solver_(settings) {}

    void step(const Model& model,
              SimState& state,
              const Control& control,
              float dt,
              const Vec3f& gravity) override;

    std::string name() const override { return "SolverSequentialImpulse"; }

    const std::vector<ContactPoint>& contacts() const override { return contacts_; }

    /** @brief Mutable access to PGS solver settings. */
    SolverSettings& settings() { return solver_.settings(); }
    const SolverSettings& settings() const { return solver_.settings(); }

private:
    FreeBodySolver solver_;
    SweepAndPrune broadphase_;
    std::vector<ContactPoint> contacts_;
};

}  // namespace novaphy
