/**
 * @file pbf_solver.cpp
 * @brief Position Based Fluids solver (Macklin & Muller, SIGGRAPH 2013).
 */
#include "novaphy/fluid/pbf_solver.h"

#include <algorithm>
#include <cmath>

#include "novaphy/sim/performance_monitor.h"
#include "novaphy/fluid/sph_kernel.h"

namespace novaphy {

namespace {

inline float p_mass(const ParticleState& s, int idx, float uniform_mass) {
    return (idx < static_cast<int>(s.particle_masses.size())) ? s.particle_masses[idx]
                                                               : uniform_mass;
}

inline float p_rho0(const ParticleState& s, int idx, float uniform_rho0) {
    return (idx < static_cast<int>(s.rest_densities.size())) ? s.rest_densities[idx]
                                                             : uniform_rho0;
}

}  // namespace

PBFSolver::PBFSolver(const PBFSettings& settings)
    : settings_(settings), grid_(settings.kernel_radius) {}

void PBFSolver::step(ParticleState& state, float dt, const Vec3f& gravity,
                     float particle_mass) {
    int n = state.num_particles();
    if (n == 0) return;

    PerformanceMonitor* monitor = detail::current_performance_monitor();
    float h = settings_.kernel_radius;
    grid_.set_cell_size(h);

    // 1. Apply external forces and predict positions
    {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.predict");
        for (int i = 0; i < n; ++i) {
            state.velocities[i] += gravity * dt;
            state.predicted_positions[i] = state.positions[i] + state.velocities[i] * dt;
        }
    }

    // 2. Build spatial hash grid from predicted positions
    {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.grid_build");
        grid_.build(state.predicted_positions);
    }

    // 3. Find neighbors for all particles
    {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.neighbor_query");
        neighbors_.resize(n);
        std::vector<int> scratch;
        for (int i = 0; i < n; ++i) {
            grid_.query_neighbors(state.predicted_positions[i], h, scratch);
            neighbors_[i] = scratch;
        }
    }

    // 4. Iterative constraint solving
    for (int iter = 0; iter < settings_.solver_iterations; ++iter) {
        {
            detail::PerformancePhaseScope phase_scope(monitor,
                                                      "fluid.pbf.iteration.compute_density");
            compute_density(state, particle_mass);
        }
        {
            detail::PerformancePhaseScope phase_scope(monitor,
                                                      "fluid.pbf.iteration.compute_lambda");
            compute_lambda(state, particle_mass);
        }
        {
            detail::PerformancePhaseScope phase_scope(monitor,
                                                      "fluid.pbf.iteration.compute_delta_position");
            compute_delta_position(state, particle_mass);
        }

        {
            detail::PerformancePhaseScope phase_scope(monitor,
                                                      "fluid.pbf.iteration.apply_delta");
            for (int i = 0; i < n; ++i) {
                state.predicted_positions[i] += state.delta_positions[i];
            }
        }
    }

    // 5. Update velocities from position change (with max speed clamp)
    float inv_dt = 1.0f / dt;
    float max_speed = h / dt;  // CFL-inspired limit: one kernel radius per step
    float max_speed_sq = max_speed * max_speed;
    {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.update_velocities");
        for (int i = 0; i < n; ++i) {
            state.velocities[i] = (state.predicted_positions[i] - state.positions[i]) * inv_dt;
            float speed_sq = state.velocities[i].squaredNorm();
            if (speed_sq > max_speed_sq) {
                state.velocities[i] *= max_speed / std::sqrt(speed_sq);
            }
        }
    }

    // 6. Apply XSPH viscosity
    if (settings_.xsph_viscosity > 0.0f) {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.xsph_viscosity");
        apply_xsph_viscosity(state, particle_mass);
    }

    // 6b. Vorticity confinement (optional)
    if (settings_.vorticity_epsilon > 1e-8f) {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.vorticity");
        apply_vorticity_confinement(state, dt, particle_mass);
    }

    // 7. Update positions (with optional domain clamping)
    {
        detail::PerformancePhaseScope phase_scope(monitor, "fluid.pbf.commit_positions");
        if (settings_.use_domain_bounds) {
            float pr = settings_.particle_radius;
            Vec3f lo = settings_.domain_lower + Vec3f(pr, pr, pr);
            Vec3f hi = settings_.domain_upper - Vec3f(pr, pr, pr);
            for (int i = 0; i < n; ++i) {
                Vec3f& p = state.predicted_positions[i];
                for (int d = 0; d < 3; ++d) {
                    if (p[d] < lo[d]) {
                        p[d] = lo[d];
                        state.velocities[i][d] = 0.0f;
                    } else if (p[d] > hi[d]) {
                        p[d] = hi[d];
                        state.velocities[i][d] = 0.0f;
                    }
                }
                state.positions[i] = p;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                state.positions[i] = state.predicted_positions[i];
            }
        }
    }
}

void PBFSolver::compute_density(ParticleState& state, float particle_mass) {
    int n = state.num_particles();
    float h = settings_.kernel_radius;

    for (int i = 0; i < n; ++i) {
        float density = 0.0f;
        for (int j : neighbors_[i]) {
            Vec3f r = state.predicted_positions[i] - state.predicted_positions[j];
            float r_sq = r.squaredNorm();
            density += p_mass(state, j, particle_mass) * SPHKernels::poly6(r_sq, h);
        }
        state.densities[i] = density;
    }
}

void PBFSolver::compute_lambda(ParticleState& state, float particle_mass) {
    int n = state.num_particles();
    float h = settings_.kernel_radius;
    float rho0_global = settings_.rest_density;
    float eps = settings_.epsilon;

    for (int i = 0; i < n; ++i) {
        float rho0_i = p_rho0(state, i, rho0_global);
        float inv_rho0_i = 1.0f / rho0_i;
        // Density constraint: C_i = rho_i / rho_0_i - 1
        float C_i = state.densities[i] * inv_rho0_i - 1.0f;

        // Only enforce incompressibility (no tensile/attraction from negative C)
        if (C_i < 0.0f) {
            state.lambdas[i] = 0.0f;
            continue;
        }

        // Compute gradient of C_i wrt each neighbor
        // grad_pk C_i = (1/rho0_i) * m_j * grad W(pi - pk, h)  for k != i
        float sum_grad_sq = 0.0f;
        Vec3f grad_i = Vec3f::Zero();

        for (int j : neighbors_[i]) {
            if (j == i) continue;
            Vec3f r = state.predicted_positions[i] - state.predicted_positions[j];
            Vec3f spiky_g = SPHKernels::spiky_grad(r, h);
            float mj = p_mass(state, j, particle_mass);
            Vec3f grad_j = (mj * inv_rho0_i) * spiky_g;
            sum_grad_sq += grad_j.squaredNorm();
            grad_i += grad_j;  // accumulate grad_C wrt particle i
        }
        sum_grad_sq += grad_i.squaredNorm();

        // Lambda with CFM relaxation
        state.lambdas[i] = -C_i / (sum_grad_sq + eps);
    }
}

void PBFSolver::compute_delta_position(ParticleState& state, float particle_mass) {
    int n = state.num_particles();
    float h = settings_.kernel_radius;
    float rho0_global = settings_.rest_density;

    // Tensile instability correction reference value
    float delta_q = settings_.corr_delta_q * h;
    float w_delta_q = SPHKernels::poly6(delta_q * delta_q, h);

    for (int i = 0; i < n; ++i) {
        float rho0_i = p_rho0(state, i, rho0_global);
        float inv_rho0_i = 1.0f / rho0_i;
        Vec3f delta_p = Vec3f::Zero();

        for (int j : neighbors_[i]) {
            if (j == i) continue;
            Vec3f r = state.predicted_positions[i] - state.predicted_positions[j];
            float r_sq = r.squaredNorm();

            float lambda_sum = state.lambdas[i] + state.lambdas[j];

            // Tensile instability correction (s_corr) — only active when
            // constraint is engaged (at least one lambda non-zero)
            float s_corr = 0.0f;
            if (lambda_sum < -1e-10f && w_delta_q > 1e-10f) {
                float ratio = SPHKernels::poly6(r_sq, h) / w_delta_q;
                // Clamp ratio to [0, 1] to prevent explosive corrections
                // when particles are much closer than delta_q
                ratio = std::min(ratio, 1.0f);
                float ratio_pow = ratio * ratio;  // ratio^2
                if (settings_.corr_n == 4) {
                    ratio_pow = ratio_pow * ratio_pow;  // ratio^4
                }
                s_corr = -settings_.corr_k * ratio_pow;
            }

            float mj = p_mass(state, j, particle_mass);
            delta_p += (lambda_sum + s_corr) * mj * SPHKernels::spiky_grad(r, h);
        }

        state.delta_positions[i] = delta_p * inv_rho0_i;
    }
}

void PBFSolver::apply_xsph_viscosity(ParticleState& state, float particle_mass) {
    int n = state.num_particles();
    float h = settings_.kernel_radius;
    float c = settings_.xsph_viscosity;
    float rho0_global = settings_.rest_density;

    // Compute velocity corrections (use delta_positions as temp buffer)
    for (int i = 0; i < n; ++i) {
        Vec3f v_corr = Vec3f::Zero();
        for (int j : neighbors_[i]) {
            if (j == i) continue;
            Vec3f v_ij = state.velocities[j] - state.velocities[i];
            Vec3f r = state.predicted_positions[i] - state.predicted_positions[j];
            float r_sq = r.squaredNorm();
            float w = SPHKernels::poly6(r_sq, h);
            float rho0_j = p_rho0(state, j, rho0_global);
            float rho_j = (state.densities[j] > 1e-6f) ? state.densities[j] : rho0_j;
            float mj = p_mass(state, j, particle_mass);
            v_corr += (mj / rho_j) * v_ij * w;
        }
        state.delta_positions[i] = v_corr;  // temp storage
    }

    // Apply corrections
    for (int i = 0; i < n; ++i) {
        state.velocities[i] += c * state.delta_positions[i];
    }
}

void PBFSolver::apply_vorticity_confinement(ParticleState& state, float dt,
                                            float uniform_mass) {
    int n = state.num_particles();
    float h = settings_.kernel_radius;
    float eps = settings_.vorticity_epsilon;

    std::vector<Vec3f> omega(static_cast<size_t>(n), Vec3f::Zero());
    for (int i = 0; i < n; ++i) {
        for (int j : neighbors_[i]) {
            if (j == i) continue;
            Vec3f r = state.predicted_positions[i] - state.predicted_positions[j];
            Vec3f grad_w = SPHKernels::spiky_grad(r, h);
            Vec3f v_ij = state.velocities[j] - state.velocities[i];
            float rho_j = std::max(state.densities[j], 1e-6f);
            float mj = p_mass(state, j, uniform_mass);
            omega[static_cast<size_t>(i)] += (mj / rho_j) * v_ij.cross(grad_w);
        }
    }

    std::vector<float> omega_len(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        omega_len[static_cast<size_t>(i)] = omega[static_cast<size_t>(i)].norm();
    }

    std::vector<Vec3f> eta(static_cast<size_t>(n), Vec3f::Zero());
    for (int i = 0; i < n; ++i) {
        for (int j : neighbors_[i]) {
            if (j == i) continue;
            Vec3f r = state.predicted_positions[i] - state.predicted_positions[j];
            float dist = r.norm();
            if (dist < 1e-8f) continue;
            float dw = omega_len[static_cast<size_t>(j)] - omega_len[static_cast<size_t>(i)];
            float r_sq = r.squaredNorm();
            float w = SPHKernels::poly6(r_sq, h);
            eta[static_cast<size_t>(i)] += (dw * w / dist) * r;
        }
    }

    for (int i = 0; i < n; ++i) {
        float om = omega_len[static_cast<size_t>(i)];
        if (om < 1e-8f) continue;
        Vec3f omega_i = omega[static_cast<size_t>(i)];
        Vec3f eta_i = eta[static_cast<size_t>(i)];
        Vec3f f = eta_i.cross(omega_i / om);
        state.velocities[i] += eps * f * dt;
    }
}

}  // namespace novaphy
