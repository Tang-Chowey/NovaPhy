#pragma once

#include <span>
#include <string>
#include <vector>

#include "novaphy/core/body.h"
#include "novaphy/core/contact.h"
#include "novaphy/core/control.h"
#include "novaphy/core/device.h"
#include "novaphy/core/model.h"
#include "novaphy/math/math_types.h"
#include "novaphy/sim/performance_monitor.h"
#include "novaphy/sim/state.h"

namespace novaphy {

/**
 * @brief Abstract base class for all physics solvers.
 *
 * @details Defines the interface that every solver (rigid-body, articulated,
 * fluid, …) must implement. The `World` holds a solver via
 * `std::unique_ptr<SolverBase>` and delegates its `step()` call here.
 *
 * Subclass examples:
 *   - `SolverSequentialImpulse` – PGS-based free-body contact solver
 *   - `SolverXPBD`             – position-based articulated solver
 *   - `SolverFeatherstone`     – Featherstone forward-dynamics solver
 *
 * The solver does NOT own the model or the state; those are passed in by
 * the caller. This allows the same solver to be reused with multiple
 * state snapshots (e.g. for RL parallel rollout).
 */
class SolverBase {
public:
    virtual ~SolverBase() = default;

    /**
     * @brief Advance the simulation by one time step.
     *
     * @param [in]     model   Immutable model definition (topology, shapes).
     * @param [in,out] state   Mutable simulation state to update in-place.
     * @param [in]     control External control inputs for this step.
     * @param [in]     dt      Time step in seconds.
     * @param [in]     gravity Gravity vector (m/s^2).
     */
    virtual void step(const Model& model,
                      SimState& state,
                      const Control& control,
                      float dt,
                      const Vec3f& gravity) = 0;

    /**
     * @brief Human-readable solver name (for logging / Python __repr__).
     *
     * @return Name string.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Query the device this solver targets.
     *
     * @return Device descriptor.
     */
    virtual Device device() const { return Device::cpu(); }

    /**
     * @brief Read-only access to contacts generated during the last step.
     *
     * @return Contacts vector (may be empty if solver does not generate contacts).
     */
    virtual const std::vector<ContactPoint>& contacts() const {
        static const std::vector<ContactPoint> empty;
        return empty;
    }

    /**
     * @brief Attach a performance monitor for phase/metric reporting.
     *
     * @param [in] monitor Pointer to an external performance monitor (non-owning).
     */
    void set_performance_monitor(PerformanceMonitor* monitor) { monitor_ = monitor; }

protected:
    PerformanceMonitor* monitor_ = nullptr;  /**< Non-owning pointer to performance monitor. */
};

}  // namespace novaphy
