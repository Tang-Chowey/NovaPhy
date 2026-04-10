#pragma once

#include "novaphy/core/model.h"
#include "novaphy/sim/state.h"
#include "novaphy/vbd/vbd_config.h"
#include "novaphy/novaphy.h"

#include <limits>
#include <memory>

namespace novaphy {

/** Forward declare: implementation is in cpu/ or cuda/; base for backend isolation. */
class VbdSolver;

/**
 * @brief VBD/AVBD-based simulation world.
 *
 * The public interface mirrors `World` / `IPCWorld` as closely as possible:
 *
 * - Uses `Model` as an immutable scene description;
 * - Maintains a mutable `SimState` buffer internally;
 * - Exposes a simple `step()` method that advances time and updates `state()`.
 *
 * CPU and CUDA backends use separate Impl subclasses so the CUDA step path
 * never touches CPU solver code.
 */
class NOVAPHY_API VBDWorld {
public:
    /**
     * @brief Construct a VBD world from a NovaPhy model.
     *
     * @param model  Immutable model containing bodies, shapes and initial transforms.
     * @param config VBD/AVBD configuration parameters.
     */
    explicit VBDWorld(const Model& model, const VBDConfig& config = VBDConfig{});
    ~VBDWorld();

    // Move-only
    VBDWorld(VBDWorld&&) noexcept;
    VBDWorld& operator=(VBDWorld&&) noexcept;
    VBDWorld(const VBDWorld&) = delete;
    VBDWorld& operator=(const VBDWorld&) = delete;

    /**
     * @brief Advance the VBD simulation by one timestep.
     *
     * Time integration, constraint solving, and (in the future) GPU
     * acceleration are all implemented inside the hidden implementation
     * object. This public method remains a simple `step()` call.
     */
    void step();

    // demo3d-style AVBD constraints/forces
    void clear_forces();
    void add_ignore_collision(int body_a, int body_b);
    int add_joint(int body_a, int body_b,
                  const Vec3f& rA, const Vec3f& rB,
                  float stiffnessLin = std::numeric_limits<float>::infinity(),
                  float stiffnessAng = 0.0f,
                  float fracture = std::numeric_limits<float>::infinity());
    int add_spring(int body_a, int body_b,
                   const Vec3f& rA, const Vec3f& rB,
                   float stiffness, float rest = -1.0f);

    /// Access current simulation state (positions, velocities).
    SimState& state();
    const SimState& state() const;

    /// Access the model used to build this world.
    const Model& model() const;

    /// Access the VBD configuration.
    const VBDConfig& config() const;

    /** Backend-specific implementation; step_one() is in cpu/ or cuda/. Public so create_vbd_world_impl can return it. */
    struct Impl {
        virtual void step_one() = 0;
        virtual void clear_forces() = 0;
        virtual void add_ignore_collision(int body_a, int body_b) = 0;
        virtual int add_joint(int body_a, int body_b, const Vec3f& rA, const Vec3f& rB,
                              float stiffnessLin, float stiffnessAng, float fracture) = 0;
        virtual int add_spring(int body_a, int body_b, const Vec3f& rA, const Vec3f& rB,
                               float stiffness, float rest) = 0;
        virtual SimState& state() = 0;
        virtual const SimState& state() const = 0;
        virtual const Model& model() const = 0;
        virtual const VBDConfig& config() const = 0;
        virtual ~Impl() = default;
    };

private:
    std::unique_ptr<Impl> impl_;
};

/** Creates CPU or CUDA Impl based on config; CUDA impl only when NOVAPHY_VBD_CUDA and config.backend==CUDA. */
std::unique_ptr<VBDWorld::Impl> create_vbd_world_impl(const Model& model, const VBDConfig& config);

#if defined(NOVAPHY_VBD_CUDA)
/** CUDA backend impl factory (defined in cuda/vbd_world_cuda.cu). */
std::unique_ptr<VBDWorld::Impl> create_cuda_impl(const Model& model, const VBDConfig& config);
#endif

}  // namespace novaphy

