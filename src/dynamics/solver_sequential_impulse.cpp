/**
 * @file solver_sequential_impulse.cpp
 * @brief PGS-based free-body solver implementing SolverBase.
 */
#include "novaphy/dynamics/solver_sequential_impulse.h"

namespace novaphy {

void SolverSequentialImpulse::step(const Model& model,
                                   SimState& state,
                                   const Control& control,
                                   float dt,
                                   const Vec3f& gravity) {
    const int n = model.num_bodies();
    const auto& settings = solver_.settings();
    const bool sleep_enabled = settings.sleep_enabled;

    // Apply external body forces/torques from Control
    if (!control.body_forces.empty()) {
        for (int i = 0; i < std::min(n, static_cast<int>(control.body_forces.size())); ++i) {
            state.forces[i] += control.body_forces[i];
        }
    }
    if (!control.body_torques.empty()) {
        for (int i = 0; i < std::min(n, static_cast<int>(control.body_torques.size())); ++i) {
            state.torques[i] += control.body_torques[i];
        }
    }

    int dynamic_body_count = 0;

    // 1. Integrate velocities
    {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.integrate_velocity");
        for (int i = 0; i < n; ++i) {
            if (model.bodies[i].is_static()) continue;
            if (sleep_enabled && state.is_sleeping(i)) continue;
            dynamic_body_count += 1;
            SymplecticEuler::integrate_velocity(
                state.linear_velocities[i],
                state.angular_velocities[i],
                state.forces[i],
                state.torques[i],
                model.bodies[i].inv_mass(),
                model.bodies[i].inv_inertia(),
                gravity, dt);
            if (sleep_enabled) {
                const auto& body = model.bodies[i];
                const Vec3f& lv = state.linear_velocities[i];
                const Vec3f& av = state.angular_velocities[i];
                float energy = body.mass * lv.squaredNorm() + av.dot(body.inertia * av);
                state.update_energy(i, energy, settings.sleep_ema_alpha);
            }
        }
    }

    // 2. Build AABBs
    const int num_shapes = model.num_shapes();
    std::vector<AABB> shape_aabbs(num_shapes);
    std::vector<bool> shape_static(num_shapes);

    {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.broadphase.build_aabbs");
        for (int i = 0; i < num_shapes; ++i) {
            const auto& shape = model.shapes[i];
            if (shape.body_index >= 0) {
                shape_aabbs[i] = shape.compute_aabb(state.transforms[shape.body_index]);
                bool body_static = model.bodies[shape.body_index].is_static();
                if (sleep_enabled && !body_static) {
                    body_static = state.is_sleeping(shape.body_index);
                }
                shape_static[i] = body_static;
            } else {
                shape_aabbs[i] = shape.compute_aabb(Transform::identity());
                shape_static[i] = true;
            }
        }
    }

    // 3. Broadphase
    {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.broadphase.sap");
        broadphase_.update(shape_aabbs, shape_static);
    }

    // 4. Narrowphase
    contacts_.clear();
    {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.narrowphase.total");
        const auto& pairs = broadphase_.get_pairs();
        for (const auto& pair : pairs) {
            if (model.is_collision_pair_filtered(pair.body_a, pair.body_b)) continue;
            const auto& sa = model.shapes[pair.body_a];
            const auto& sb = model.shapes[pair.body_b];
            Transform ta = (sa.body_index >= 0) ? state.transforms[sa.body_index] : Transform::identity();
            Transform tb = (sb.body_index >= 0) ? state.transforms[sb.body_index] : Transform::identity();
            std::vector<ContactPoint> new_contacts;
            if (collide_shapes(sa, ta, sb, tb, new_contacts)) {
                for (auto& cp : new_contacts) {
                    cp.friction = combine_friction(sa.friction, sb.friction);
                    cp.restitution = combine_restitution(sa.restitution, sb.restitution);
                    contacts_.push_back(cp);
                }
            }
        }
    }

    // 5. Islands and sleep
    if (sleep_enabled) {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.build_islands");
        state.build_islands(contacts_);
    }

    // 6. Solve contacts
    {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.solver.total");
        solver_.solve(contacts_, model.bodies, state.transforms,
                      state.linear_velocities, state.angular_velocities,
                      state.sleeping, dt);
    }

    // 7. Wake propagation
    if (sleep_enabled) {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.propagate_wakes");
        for (const auto& contact : contacts_) {
            bool a_sleeping = (contact.body_a >= 0) ? state.is_sleeping(contact.body_a) : false;
            bool b_sleeping = (contact.body_b >= 0) ? state.is_sleeping(contact.body_b) : false;
            if (a_sleeping && !b_sleeping && contact.body_a >= 0) {
                state.propagate_wake_through_island(contact.body_a);
            } else if (!a_sleeping && b_sleeping && contact.body_b >= 0) {
                state.propagate_wake_through_island(contact.body_b);
            }
        }
    }

    // 8. Sleep evaluation
    if (sleep_enabled) {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.evaluate_sleep");
        state.evaluate_sleep(dt, settings.sleep_energy_threshold, settings.sleep_time_required);
    }

    // 9. Integrate positions
    {
        detail::PerformancePhaseScope phase_scope(monitor_, "world.integrate_position");
        for (int i = 0; i < n; ++i) {
            if (model.bodies[i].is_static()) continue;
            if (sleep_enabled && state.is_sleeping(i)) continue;
            SymplecticEuler::integrate_position(
                state.transforms[i],
                state.linear_velocities[i],
                state.angular_velocities[i], dt);
        }
    }

    // 10. Clear forces is now handled by World::step after all sub-solvers run

    // Record metrics
    if (monitor_) {
        const int candidate_pair_count = static_cast<int>(broadphase_.get_pairs().size());
        monitor_->record_metric("bodies", static_cast<double>(model.num_bodies()));
        monitor_->record_metric("dynamic_bodies", static_cast<double>(dynamic_body_count));
        monitor_->record_metric("shapes", static_cast<double>(model.num_shapes()));
        monitor_->record_metric("candidate_pairs", static_cast<double>(candidate_pair_count));
        monitor_->record_metric("contacts", static_cast<double>(contacts_.size()));
        monitor_->record_metric("solver_iterations", static_cast<double>(settings.velocity_iterations));
    }
}

}  // namespace novaphy
