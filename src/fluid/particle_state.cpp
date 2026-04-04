/**
 * @file particle_state.cpp
 * @brief Fluid particle state storage and block generation.
 */
#include "novaphy/fluid/particle_state.h"

#include <cmath>

namespace novaphy {

void ParticleState::init(std::span<const Vec3f> initial_positions,
                         const Vec3f& initial_velocity) {
    const int n = static_cast<int>(initial_positions.size());
    positions.assign(initial_positions.begin(), initial_positions.end());
    velocities.assign(n, initial_velocity);
    densities.assign(n, 0.0f);
    lambdas.assign(n, 0.0f);
    predicted_positions = positions;
    delta_positions.assign(n, Vec3f::Zero());
    particle_masses.clear();
    rest_densities.clear();
}

void ParticleState::init(std::span<const Vec3f> initial_positions,
                         std::span<const Vec3f> initial_velocities,
                         std::span<const float> particle_masses_in,
                         std::span<const float> rest_densities_in) {
    const int n = static_cast<int>(initial_positions.size());
    if (static_cast<int>(initial_velocities.size()) != n ||
        static_cast<int>(particle_masses_in.size()) != n ||
        static_cast<int>(rest_densities_in.size()) != n) {
        clear();
        return;
    }
    positions.assign(initial_positions.begin(), initial_positions.end());
    velocities.assign(initial_velocities.begin(), initial_velocities.end());
    densities.assign(n, 0.0f);
    lambdas.assign(n, 0.0f);
    predicted_positions = positions;
    delta_positions.assign(n, Vec3f::Zero());
    particle_masses.assign(particle_masses_in.begin(), particle_masses_in.end());
    rest_densities.assign(rest_densities_in.begin(), rest_densities_in.end());
}

void ParticleState::clear() {
    positions.clear();
    velocities.clear();
    densities.clear();
    lambdas.clear();
    predicted_positions.clear();
    delta_positions.clear();
    particle_masses.clear();
    rest_densities.clear();
}

std::vector<Vec3f> generate_fluid_block(const FluidBlockDef& def) {
    std::vector<Vec3f> positions;

    float spacing = def.particle_spacing;
    if (spacing <= 0.0f) return positions;

    // Count particles per axis
    int nx = std::max(1, static_cast<int>(std::floor((def.upper.x() - def.lower.x()) / spacing)) + 1);
    int ny = std::max(1, static_cast<int>(std::floor((def.upper.y() - def.lower.y()) / spacing)) + 1);
    int nz = std::max(1, static_cast<int>(std::floor((def.upper.z() - def.lower.z()) / spacing)) + 1);

    positions.reserve(nx * ny * nz);

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                Vec3f pos(
                    def.lower.x() + ix * spacing,
                    def.lower.y() + iy * spacing,
                    def.lower.z() + iz * spacing
                );
                positions.push_back(pos);
            }
        }
    }

    return positions;
}

}  // namespace novaphy
