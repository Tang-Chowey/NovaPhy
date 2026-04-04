#pragma once

#include <span>
#include <vector>

#include "novaphy/math/math_types.h"

namespace novaphy {

/**
 * @brief Physical parameters for one fluid species (multi-material scenes).
 */
struct FluidMaterial {
    float rest_density = 1000.0f; /**< Rest density rho_0 (kg/m^3). */
};

/**
 * @brief SOA storage for fluid particle state.
 *
 * @details Mirrors SimState pattern. Stores position, velocity, and
 * per-particle quantities needed by SPH/PBF solvers.
 */
struct ParticleState {
    std::vector<Vec3f> positions;         /**< Particle positions in world frame (m). */
    std::vector<Vec3f> velocities;        /**< Particle velocities in world frame (m/s). */
    std::vector<float> densities;         /**< Per-particle density (kg/m^3). */
    std::vector<float> lambdas;           /**< PBF constraint multipliers. */
    std::vector<Vec3f> predicted_positions; /**< Predicted positions for PBF solver. */
    std::vector<Vec3f> delta_positions;   /**< PBF position corrections. */
    std::vector<float> particle_masses;   /**< Per-particle mass (kg); empty => uniform mass from solver. */
    std::vector<float> rest_densities;    /**< Per-particle rho_0 (kg/m^3); empty => PBFSettings.rest_density. */

    /**
     * @brief Initialize particle arrays with given positions.
     *
     * @param[in] initial_positions Starting positions for all particles.
     * @param[in] initial_velocity Initial velocity for all particles.
     */
    void init(std::span<const Vec3f> initial_positions,
              const Vec3f& initial_velocity = Vec3f::Zero());

    /**
     * @brief Initialize with per-particle velocity and material data (multi-block / multi-material).
     */
    void init(std::span<const Vec3f> initial_positions,
              std::span<const Vec3f> initial_velocities,
              std::span<const float> particle_masses_in,
              std::span<const float> rest_densities_in);

    /**
     * @brief Get number of particles.
     *
     * @return Particle count.
     */
    int num_particles() const { return static_cast<int>(positions.size()); }

    /**
     * @brief Clear all particle data.
     */
    void clear();
};

/**
 * @brief Configuration for a fluid block to be emitted by ModelBuilder.
 *
 * @details Defines a rectangular region to fill with particles at
 * a given spacing.
 */
struct FluidBlockDef {
    Vec3f lower = Vec3f::Zero();      /**< Lower corner of the fluid block (m). */
    Vec3f upper = Vec3f::Zero();      /**< Upper corner of the fluid block (m). */
    float particle_spacing = 0.02f;   /**< Inter-particle spacing (m). */
    float rest_density = 1000.0f;     /**< Rest density when no material table (kg/m^3). */
    int material_index = 0;           /**< Index into Model::fluid_materials (if non-empty). */
    Vec3f initial_velocity = Vec3f::Zero(); /**< Initial velocity of all particles (m/s). */
};

/**
 * @brief Generate particle positions filling a rectangular block.
 *
 * @param[in] def Block definition (bounds, spacing).
 * @return Vector of generated particle positions.
 */
std::vector<Vec3f> generate_fluid_block(const FluidBlockDef& def);

}  // namespace novaphy
