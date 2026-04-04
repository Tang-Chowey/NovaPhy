/**
 * @file world.cpp
 * @brief High-level simulation world with pluggable solver.
 */
#include "novaphy/sim/world.h"

#include <cmath>
#include <numbers>

#include "novaphy/fluid/sph_kernel.h"
#include "novaphy/dynamics/featherstone.h"

namespace novaphy {

namespace {

float effective_fluid_rest_density(const Model& model, const FluidBlockDef& block) {
    if (!model.fluid_materials.empty()) {
        const int mi = block.material_index;
        if (mi >= 0 && mi < static_cast<int>(model.fluid_materials.size())) {
            return model.fluid_materials[mi].rest_density;
        }
    }
    return block.rest_density;
}

}  // namespace

World::World(const Model& model, 
             SolverSettings solver_settings,
             XPBDSolverSettings xpbd_settings,
             PBFSettings pbf_settings,
             float fluid_boundary_extent)
    : model_(model),
      solver_(std::make_unique<SolverSequentialImpulse>(solver_settings)),
      xpbd_solver_(xpbd_settings),
      pbf_solver_(pbf_settings),
      boundary_grid_(pbf_settings.kernel_radius),
      gravity_(model.gravity) {
    
    // Initialize simulation state
    state_ = SimState::from_model(model_, Device::cpu());
    solver_->set_performance_monitor(&performance_monitor_);

    // Generate fluid particles from all blocks (multi-block / multi-material)
    std::vector<Vec3f> all_positions;
    std::vector<Vec3f> all_velocities;
    std::vector<float> all_masses;
    std::vector<float> all_rest_rho;

    for (const auto& block : model.fluid_blocks) {
        auto block_positions = generate_fluid_block(block);
        const float rho = effective_fluid_rest_density(model, block);
        const float spacing = block.particle_spacing;
        const float m = rho * spacing * spacing * spacing;
        for (const auto& p : block_positions) {
            all_positions.push_back(p);
            all_velocities.push_back(block.initial_velocity);
            all_masses.push_back(m);
            all_rest_rho.push_back(rho);
        }
        particle_spacing_ = spacing;
    }

    if (!all_positions.empty()) {
        state_.fluid_state.init(all_positions, all_velocities, all_masses, all_rest_rho);
        particle_mass_ = all_masses.empty() ? pbf_settings.particle_mass(particle_spacing_)
                                            : all_masses[0];
    }

    // Generate boundary particles from all rigid body and articulation shapes
    boundary_particles_ = sample_model_boundaries(model, particle_spacing_, fluid_boundary_extent);

    // Compute initial boundary volumes (Akinci)
    if (!boundary_particles_.empty()) {
        std::vector<std::vector<Transform>> art_transforms(model.articulations.size());
        for (size_t a = 0; a < model.articulations.size(); ++a) {
            art_transforms[a] = featherstone::forward_kinematics(model.articulations[a], state_.q[a]).world_transforms;
        }
        auto bpos = boundary_world_positions(boundary_particles_, state_.transforms, art_transforms);
        compute_boundary_volumes(boundary_particles_, bpos, pbf_settings.kernel_radius);
    }
}

void World::step(float dt) {
    step(state_, control_, dt);
    
    // Clear user-applied controls after step
    control_.joint_forces.setZero();
    control_.articulation_joint_forces.clear();
    control_.body_forces.assign(model_.num_bodies(), Vec3f::Zero());
    control_.body_torques.assign(model_.num_bodies(), Vec3f::Zero());
}

void World::step(SimState& state, const Control& control, float dt) {
    performance_monitor_.begin_frame();
    detail::ScopedPerformanceCaptureContext capture_context(&performance_monitor_);

    {
        detail::PerformancePhaseScope total_scope(&performance_monitor_, "world.total");

        int n_fluid = state.fluid_state.num_particles();
        int n_boundary = static_cast<int>(boundary_particles_.size());
        std::vector<Vec3f> boundary_pos;

        // 1 & 2. Fluid Step (boundary update + PBF solve)
        if (n_fluid > 0) {
            detail::PerformancePhaseScope fluid_total_scope(&performance_monitor_, "fluid.total");

            if (n_boundary > 0) {
                detail::PerformancePhaseScope phase_scope(&performance_monitor_, "fluid.boundary.positions");
                std::vector<std::vector<Transform>> art_transforms(model_.articulations.size());
                for (size_t a = 0; a < model_.articulations.size(); ++a) {
                    art_transforms[a] = featherstone::forward_kinematics(model_.articulations[a], state.q[a]).world_transforms;
                }
                boundary_pos = boundary_world_positions(boundary_particles_, state.transforms, art_transforms);
            }

            {
                detail::PerformancePhaseScope phase_scope(&performance_monitor_, "fluid.pbf.total");
                pbf_solver_.step(state.fluid_state, dt, gravity(), particle_mass_);

                if (n_boundary > 0) {
                    apply_fluid_boundary_density(state, boundary_pos);
                    apply_fluid_coupling_forces(state, boundary_pos, dt);
                }
            }
        }

        // 3. Rigid Body Step
        if (model_.num_bodies() > 0) {
            detail::PerformancePhaseScope phase_scope(&performance_monitor_, "rigid.step");
            solver_->step(model_, state, control, dt, gravity());
        }

        // 4. Articulation Step
        if (!model_.articulations.empty()) {
            detail::PerformancePhaseScope phase_scope(&performance_monitor_, "articulation.step");
            int total_articulation_qd = 0;
            for (const auto& articulation : model_.articulations) {
                total_articulation_qd += articulation.total_qd();
            }

            const bool has_flat_joint_forces =
                control.joint_forces.size() == total_articulation_qd;
            int flat_joint_force_offset = 0;

            for (size_t a = 0; a < model_.articulations.size(); ++a) {
                const auto& art = model_.articulations[a];
                
                VecXf tau = VecXf::Zero(art.total_qd());
                if (a < control.articulation_joint_forces.size() &&
                    control.articulation_joint_forces[a].size() == art.total_qd()) {
                    tau = control.articulation_joint_forces[a];
                } else if (has_flat_joint_forces) {
                    tau = control.joint_forces.segment(flat_joint_force_offset, art.total_qd());
                } else if (model_.articulations.size() == 1 &&
                           control.joint_forces.size() == art.total_qd()) {
                    tau = control.joint_forces;
                }

                std::vector<CollisionShape> static_shapes; 
                for (const auto& sb : model_.shapes) {
                    if (sb.body_index < 0 && sb.articulation_index < 0) {
                        static_shapes.push_back(sb);
                    }
                }

                xpbd_solver_.step_with_contacts(art, model_, static_shapes, 
                                                state.q[a], state.qd[a], tau, gravity(), dt, 
                                                control, state.articulation_forces[a]);
                flat_joint_force_offset += art.total_qd();
            }
        }

        // Record metrics
        performance_monitor_.record_metric("fluid_particles", static_cast<double>(n_fluid));
        performance_monitor_.record_metric("boundary_particles", static_cast<double>(n_boundary));
        performance_monitor_.record_metric("pbf_solver_iterations",
            static_cast<double>(pbf_solver_.settings().solver_iterations));
        performance_monitor_.record_metric("kernel_radius",
            static_cast<double>(pbf_solver_.settings().kernel_radius));

        // 5. Clear accumulated forces
        state.clear_forces();
    }

    performance_monitor_.end_frame();
}

void World::apply_fluid_boundary_density(SimState& state, const std::vector<Vec3f>& boundary_world_pos) {
    int n_fluid = state.fluid_state.num_particles();
    float h = pbf_solver_.settings().kernel_radius;
    boundary_grid_.set_cell_size(h);
    boundary_grid_.build(boundary_world_pos);

    float rho0 = pbf_solver_.settings().rest_density;
    std::vector<int> scratch;
    for (int i = 0; i < n_fluid; ++i) {
        float boundary_density = 0.0f;
        boundary_grid_.query_neighbors(state.fluid_state.positions[i], h, scratch);
        for (int j : scratch) {
            Vec3f r = state.fluid_state.positions[i] - boundary_world_pos[j];
            float r_sq = r.squaredNorm();
            boundary_density += rho0 * boundary_particles_[j].volume * SPHKernels::poly6(r_sq, h);
        }
        state.fluid_state.densities[i] += boundary_density;
    }
}

void World::apply_fluid_coupling_forces(SimState& state, const std::vector<Vec3f>& boundary_world_pos, float dt) {
    int n_fluid = state.fluid_state.num_particles();
    int n_boundary = static_cast<int>(boundary_particles_.size());
    float h = pbf_solver_.settings().kernel_radius;
    float rho0 = pbf_solver_.settings().rest_density;

    SpatialHashGrid fluid_grid(h);
    fluid_grid.build(state.fluid_state.positions);

    std::vector<int> scratch;
    for (int b = 0; b < n_boundary; ++b) {
        int body_idx = boundary_particles_[b].body_index;
        int art_idx = boundary_particles_[b].articulation_index;
        int link_idx = boundary_particles_[b].link_index;

        if (body_idx < 0 && art_idx < 0) continue; // Skip static planes

        if (body_idx >= 0 && model_.bodies[body_idx].is_static()) continue;

        Vec3f force = Vec3f::Zero();
        float psi_b = boundary_particles_[b].volume;

        fluid_grid.query_neighbors(boundary_world_pos[b], h, scratch);
        for (int f : scratch) {
            Vec3f r = boundary_world_pos[b] - state.fluid_state.positions[f];
            float r_sq = r.squaredNorm();
            if (r_sq >= h * h) continue;

            float rho_f = std::max(state.fluid_state.densities[f], 1.0f);
            const float rho0_f =
                (f < static_cast<int>(state.fluid_state.rest_densities.size()))
                    ? state.fluid_state.rest_densities[f]
                    : rho0;
            const float m_f =
                (f < static_cast<int>(state.fluid_state.particle_masses.size()))
                    ? state.fluid_state.particle_masses[f]
                    : particle_mass_;
            float pressure = std::max(0.0f, rho_f - rho0_f);
            float pressure_coeff = m_f * pressure / (rho_f * rho_f);

            Vec3f grad = SPHKernels::spiky_grad(r, h);
            force += pressure_coeff * rho0_f * psi_b * grad;
        }

        if (force.squaredNorm() > 1e-20f) {
            if (body_idx >= 0) {
                state.apply_force(body_idx, force);
                Vec3f body_com = state.transforms[body_idx].position;
                Vec3f r_arm = boundary_world_pos[b] - body_com;
                Vec3f torque = r_arm.cross(force);
                state.apply_torque(body_idx, torque);
            } else if (art_idx >= 0 && link_idx >= 0) {
                // World-frame force at boundary point; equivalent spatial wrench at link origin uses r × F.
                Transform link_tf = featherstone::forward_kinematics(model_.articulations[art_idx], state.q[art_idx]).world_transforms[link_idx];
                
                Vec3f r_arm = boundary_world_pos[b] - link_tf.position;
                Vec3f torque = r_arm.cross(force);

                SpatialVector f_spatial;
                f_spatial << torque, force;
                state.apply_articulation_force(art_idx, link_idx, f_spatial);
            }
        }
    }
}

void World::set_gravity(const Vec3f& g) {
    gravity_ = g;
}

SolverSettings& World::solver_settings() {
    auto* si_solver = dynamic_cast<SolverSequentialImpulse*>(solver_.get());
    if (si_solver) return si_solver->settings();
    static SolverSettings fallback;
    return fallback;
}

void World::apply_force(int body_index, const Vec3f& force) {
    auto* si_solver = dynamic_cast<SolverSequentialImpulse*>(solver_.get());
    if (si_solver) {
        if (si_solver->settings().sleep_enabled && body_index >= 0 &&
            body_index < static_cast<int>(state_.sleeping.size())) {
            if (state_.is_sleeping(body_index)) {
                state_.wake_body(body_index);
            }
        }
    }
    state_.apply_force(body_index, force);
}

void World::apply_torque(int body_index, const Vec3f& torque) {
    auto* si_solver = dynamic_cast<SolverSequentialImpulse*>(solver_.get());
    if (si_solver) {
        if (si_solver->settings().sleep_enabled && body_index >= 0 &&
            body_index < static_cast<int>(state_.sleeping.size())) {
            if (state_.is_sleeping(body_index)) {
                state_.wake_body(body_index);
            }
        }
    }
    state_.apply_torque(body_index, torque);
}

}  // namespace novaphy
