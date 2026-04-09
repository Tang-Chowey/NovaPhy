/**
 * @file multibody_solver.cpp
 * @brief Featherstone ABA + PGS solver component for World.
 *
 * Reads/writes SimState instead of owning state. PGS contact/friction/limit
 * solving is implemented as private methods on this class.
 */
#include "novaphy/multibody/multibody_solver.h"

#include "novaphy/collision/narrowphase.h"
#include "novaphy/core/contact.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace novaphy {

namespace {

void plane_space(const Vec3f& n, Vec3f& t1, Vec3f& t2) {
    if (std::abs(n.z()) > 0.7071068f) {
        float a = n.y() * n.y() + n.z() * n.z();
        float k = 1.0f / std::sqrt(a);
        t1 = Vec3f(0.0f, -n.z() * k, n.y() * k);
        t2 = Vec3f(a * k, -n.x() * t1.z(), n.x() * t1.y());
    } else {
        float a = n.x() * n.x() + n.y() * n.y();
        float k = 1.0f / std::sqrt(a);
        t1 = Vec3f(-n.y() * k, n.x() * k, 0.0f);
        t2 = Vec3f(-n.z() * t1.y(), n.z() * t1.x(), a * k);
    }
}

}  // namespace

MultiBodySolver::MultiBodySolver(MultiBodySolverSettings settings)
    : constraint_settings_(settings) {}

void MultiBodySolver::init(const Model& model, const SimState& state) {
    bodies_.clear();
    free_rigid_bodies_.clear();
    persistent_constraints_.clear();
    static_shapes_.clear();
    link_colliders_.clear();
    rigid_body_colliders_.clear();

    // Create MultiBody instances from articulations
    for (size_t a = 0; a < model.articulations.size(); ++a) {
        const auto& art = model.articulations[a];
        auto body = std::make_unique<MultiBody>(art, art.linear_damping, art.angular_damping);
        if (a < state.q.size() && state.q[a].size() > 0)
            body->q() = state.q[a];
        if (a < state.qd.size() && state.qd[a].size() > 0)
            body->qd() = state.qd[a];
        bodies_.push_back(std::move(body));
    }

    // Create FreeRigidBody instances from free bodies
    for (int b = 0; b < model.num_bodies(); ++b) {
        const auto& rb_data = model.bodies[b];
        Vec3f inertia_diag(rb_data.inertia(0, 0),
                           rb_data.inertia(1, 1),
                           rb_data.inertia(2, 2));
        Transform start_tf = (b < static_cast<int>(model.initial_transforms.size()))
                                  ? model.initial_transforms[b]
                                  : Transform::identity();
        auto rb = std::make_unique<FreeRigidBody>(
            rb_data.mass, inertia_diag, start_tf,
            rb_data.linear_damping, rb_data.angular_damping);
        free_rigid_bodies_.push_back(std::move(rb));
    }

    // Classify shapes
    for (const auto& shape : model.shapes) {
        if (shape.articulation_index >= 0 && shape.link_index >= 0) {
            link_colliders_.push_back({shape.articulation_index, shape.link_index, shape});
        } else if (shape.body_index >= 0) {
            rigid_body_colliders_.push_back({shape.body_index, shape});
        } else {
            static_shapes_.push_back(shape);
        }
    }

    // Create joint limit constraints from joint definitions
    for (size_t a = 0; a < model.articulations.size(); ++a) {
        const auto& art = model.articulations[a];
        for (int link = 0; link < art.num_links(); ++link) {
            const auto& joint = art.joints[link];
            if (joint.limit_enabled &&
                (joint.type == JointType::Revolute || joint.type == JointType::Slide)) {
                persistent_constraints_.push_back(
                    std::make_unique<MultiBodyJointLimit>(
                        bodies_[a].get(), link,
                        joint.lower_limit, joint.upper_limit));
            }
        }
    }
}

void MultiBodySolver::sync_state_in(const SimState& state) {
    for (size_t a = 0; a < bodies_.size(); ++a) {
        if (a < state.q.size() && state.q[a].size() > 0)
            bodies_[a]->q() = state.q[a];
        if (a < state.qd.size() && state.qd[a].size() > 0)
            bodies_[a]->qd() = state.qd[a];
    }
}

void MultiBodySolver::sync_state_out(SimState& state) const {
    for (size_t a = 0; a < bodies_.size(); ++a) {
        if (a < state.q.size())
            state.q[a] = bodies_[a]->q();
        if (a < state.qd.size())
            state.qd[a] = bodies_[a]->qd();
    }
}

void MultiBodySolver::sync_rb_state_in(const SimState& state) {
    for (size_t b = 0; b < free_rigid_bodies_.size(); ++b) {
        if (b < state.transforms.size())
            free_rigid_bodies_[b]->set_world_transform(state.transforms[b]);
        if (b < state.linear_velocities.size())
            free_rigid_bodies_[b]->set_linear_velocity(state.linear_velocities[b]);
        if (b < state.angular_velocities.size())
            free_rigid_bodies_[b]->set_angular_velocity(state.angular_velocities[b]);
    }
}

void MultiBodySolver::sync_rb_state_out(SimState& state) const {
    for (size_t b = 0; b < free_rigid_bodies_.size(); ++b) {
        if (b < state.transforms.size())
            state.transforms[b] = free_rigid_bodies_[b]->world_transform();
        if (b < state.linear_velocities.size())
            state.linear_velocities[b] = free_rigid_bodies_[b]->linear_velocity();
        if (b < state.angular_velocities.size())
            state.angular_velocities[b] = free_rigid_bodies_[b]->angular_velocity();
    }
}

void MultiBodySolver::apply_control(const Control& control) {
    int total_qd = 0;
    for (const auto& body : bodies_) total_qd += body->num_dofs();

    const bool has_flat = control.joint_forces.size() == total_qd;
    int flat_offset = 0;

    for (size_t a = 0; a < bodies_.size(); ++a) {
        auto& body = bodies_[a];
        const int ndof = body->num_dofs();

        VecXf tau = VecXf::Zero(ndof);
        if (a < control.articulation_joint_forces.size() &&
            control.articulation_joint_forces[a].size() == ndof) {
            tau = control.articulation_joint_forces[a];
        } else if (has_flat) {
            tau = control.joint_forces.segment(flat_offset, ndof);
        } else if (bodies_.size() == 1 && control.joint_forces.size() == ndof) {
            tau = control.joint_forces;
        }
        body->tau() = tau;

        flat_offset += ndof;
    }

    // Map JointDrive -> transient motor constraints are not needed here because
    // tau is applied directly through the ABA pass. Joint drives should be
    // converted to torques by the caller (same pattern as XPBDSolver).
}

void MultiBodySolver::run_collision_detection() {
    contacts_.clear();

    const int S = static_cast<int>(static_shapes_.size());
    const int L = static_cast<int>(link_colliders_.size());
    const int R = static_cast<int>(rigid_body_colliders_.size());
    const int N = S + L + R;

    if (N == 0) return;

    std::vector<AABB> shape_aabbs(N);
    std::vector<bool> shape_static(N);

    for (int i = 0; i < S; ++i) {
        shape_aabbs[i]  = static_shapes_[i].compute_aabb(Transform::identity());
        shape_static[i] = true;
    }
    for (int i = 0; i < L; ++i) {
        const auto& lc = link_colliders_[i];
        const Transform& tf =
            bodies_[lc.body_index]->cached_world_transforms()[lc.link_index];
        shape_aabbs[S + i]  = lc.shape.compute_aabb(tf);
        shape_static[S + i] = false;
    }
    for (int i = 0; i < R; ++i) {
        const auto& rbc = rigid_body_colliders_[i];
        const Transform& tf = free_rigid_bodies_[rbc.rb_index]->world_transform();
        shape_aabbs[S + L + i]  = rbc.shape.compute_aabb(tf);
        shape_static[S + L + i] = (free_rigid_bodies_[rbc.rb_index]->inv_mass() <= 0.0f);
    }

    broadphase_.update(shape_aabbs, shape_static);

    for (const auto& pair : broadphase_.get_pairs()) {
        const int ia = pair.body_a;
        const int ib = pair.body_b;

        // cat: 0=static, 1=link, 2=rb
        const int cat_a = (ia < S) ? 0 : (ia < S + L) ? 1 : 2;
        const int cat_b = (ib < S) ? 0 : (ib < S + L) ? 1 : 2;

        const CollisionShape* shape_a = nullptr;
        const CollisionShape* shape_b = nullptr;
        Transform tf_a, tf_b;
        int encoded_a = -1, encoded_b = -1;

        if (cat_a == 0) {
            shape_a   = &static_shapes_[ia];
            tf_a      = Transform::identity();
            encoded_a = -1;
        } else if (cat_a == 1) {
            const auto& lc = link_colliders_[ia - S];
            shape_a   = &lc.shape;
            tf_a      = bodies_[lc.body_index]->cached_world_transforms()[lc.link_index];
            encoded_a = (lc.body_index << 16) | lc.link_index;
        } else {
            const auto& rbc = rigid_body_colliders_[ia - S - L];
            shape_a   = &rbc.shape;
            tf_a      = free_rigid_bodies_[rbc.rb_index]->world_transform();
            encoded_a = 0x80000000 | rbc.rb_index;
        }

        if (cat_b == 0) {
            shape_b   = &static_shapes_[ib];
            tf_b      = Transform::identity();
            encoded_b = -1;
        } else if (cat_b == 1) {
            const auto& lc = link_colliders_[ib - S];
            shape_b   = &lc.shape;
            tf_b      = bodies_[lc.body_index]->cached_world_transforms()[lc.link_index];
            encoded_b = (lc.body_index << 16) | lc.link_index;
        } else {
            const auto& rbc = rigid_body_colliders_[ib - S - L];
            shape_b   = &rbc.shape;
            tf_b      = free_rigid_bodies_[rbc.rb_index]->world_transform();
            encoded_b = 0x80000000 | rbc.rb_index;
        }

        // Filter: adjacent links on same body
        if (cat_a == 1 && cat_b == 1) {
            const auto& lc_a = link_colliders_[ia - S];
            const auto& lc_b = link_colliders_[ib - S];
            if (lc_a.body_index == lc_b.body_index &&
                std::abs(lc_a.link_index - lc_b.link_index) <= 2) continue;
        }

        // Filter: same rigid body
        if (cat_a == 2 && cat_b == 2) {
            const auto& rbc_a = rigid_body_colliders_[ia - S - L];
            const auto& rbc_b = rigid_body_colliders_[ib - S - L];
            if (rbc_a.rb_index == rbc_b.rb_index) continue;
        }

        std::vector<ContactPoint> new_contacts;
        int final_encoded_a = encoded_a;
        int final_encoded_b = encoded_b;
        bool hit = false;

        // Swap link-vs-rb to preserve convention: rb as first arg
        if (cat_a == 1 && cat_b == 2) {
            hit = collide_shapes(*shape_b, tf_b, *shape_a, tf_a, new_contacts);
            final_encoded_a = encoded_b;
            final_encoded_b = encoded_a;
        } else {
            hit = collide_shapes(*shape_a, tf_a, *shape_b, tf_b, new_contacts);
        }

        if (hit) {
            for (auto& cp : new_contacts) {
                cp.friction = combine_friction(shape_a->friction, shape_b->friction);
                cp.restitution = combine_restitution(shape_a->restitution, shape_b->restitution);
                cp.body_a = final_encoded_a;
                cp.body_b = final_encoded_b;
                contacts_.push_back(cp);
            }
        }
    }
}

void MultiBodySolver::run_split_impulse_correction() {
    const float erp  = constraint_settings_.erp;
    const float slop = constraint_settings_.linear_slop;

    auto body_key = [](int encoded) -> unsigned {
        if (encoded == -1 || (encoded & 0x80000000)) return static_cast<unsigned>(encoded);
        return static_cast<unsigned>(encoded >> 16);
    };
    std::unordered_map<long long, const ContactPoint*> deepest;
    deepest.reserve(contacts_.size());
    for (const auto& cp : contacts_) {
        if (cp.penetration <= 0.0f) continue;
        float depth = cp.penetration - slop;
        if (depth <= 0.0f) continue;
        long long key = (static_cast<long long>(body_key(cp.body_a)) << 32)
                       | body_key(cp.body_b);
        auto it = deepest.find(key);
        if (it == deepest.end() || depth > (it->second->penetration - slop))
            deepest[key] = &cp;
    }

    auto apply_mb = [&](int encoded, float sign, const ContactPoint& cp,
                         float corr) {
        if (encoded == -1 || (encoded & 0x80000000)) return;
        int body_idx = encoded >> 16;
        int link_idx = encoded & 0xFFFF;
        if (body_idx < 0 || body_idx >= static_cast<int>(bodies_.size())) return;
        auto& body = bodies_[body_idx];
        const auto& joints = body->model().joints;
        if (joints.empty() || joints[0].type != JointType::Free) return;

        const int ndof = 6 + body->num_dofs();
        std::vector<float> jac(ndof, 0.0f);
        std::vector<float> delta_a(ndof, 0.0f);

        const Vec3f n = cp.normal * sign;
        body->fill_constraint_jacobian(link_idx, cp.position, Vec3f::Zero(), n, jac.data());
        body->calc_acceleration_deltas(jac.data(), delta_a.data());

        float m_eff = 0.0f;
        for (int i = 0; i < ndof; ++i) m_eff += jac[i] * delta_a[i];
        if (m_eff < 1e-6f) return;

        const float lambda = corr / m_eff;
        body->q()(0) += delta_a[6 + 3] * lambda;
        body->q()(1) += delta_a[6 + 4] * lambda;
        body->q()(2) += delta_a[6 + 5] * lambda;
    };

    auto apply_rb = [&](int encoded, float sign, const ContactPoint& cp,
                         float corr) {
        if (!(encoded & 0x80000000)) return;
        int rb_idx = encoded & 0x7FFFFFFF;
        if (rb_idx < 0 || rb_idx >= static_cast<int>(free_rigid_bodies_.size())) return;
        auto& rb = free_rigid_bodies_[rb_idx];
        if (rb->inv_mass() <= 0.0f) return;
        rb->translate(sign * cp.normal * corr);
    };

    for (const auto& [key, cp] : deepest) {
        float depth      = cp->penetration - slop;
        float correction = depth * erp;
        apply_mb(cp->body_a, -1.0f, *cp, correction);
        apply_mb(cp->body_b, +1.0f, *cp, correction);
        apply_rb(cp->body_a, -1.0f, *cp, correction);
        apply_rb(cp->body_b, +1.0f, *cp, correction);
    }
}

void MultiBodySolver::solve_constraints(std::vector<MultiBody*>& multibodies,
                                        std::vector<FreeRigidBody*>& free_rigid_bodies,
                                        std::vector<ContactPoint>& contacts,
                                        std::vector<MultiBodyConstraint*>& constraints,
                                        float dt) {
    pgs_setup(multibodies, free_rigid_bodies, contacts, constraints, dt);
    pgs_solve_iterations();
    pgs_finalize(multibodies, free_rigid_bodies);
}

void MultiBodySolver::pgs_setup(std::vector<MultiBody*>& multibodies,
                                std::vector<FreeRigidBody*>& free_rigid_bodies,
                                std::vector<ContactPoint>& contacts,
                                std::vector<MultiBodyConstraint*>& constraints,
                                float dt) {
    constraint_data_.clear();
    normal_constraints_.clear();
    friction_constraints_.clear();
    non_contact_constraints_.clear();

    for (auto* b : multibodies) {
        b->companion_id = -1;
    }
    for (auto* rb : free_rigid_bodies) {
        rb->delta_linear_velocity() = Vec3f::Zero();
        rb->delta_angular_velocity() = Vec3f::Zero();
    }

    for (auto& cp : contacts) {
        if (cp.penetration <= 0.0f) continue;

        auto decode = [&](int encoded, FreeRigidBody*& out_rb,
                          MultiBody*& out_mb, int& out_link) {
            out_rb = nullptr;
            out_mb = nullptr;
            out_link = -1;
            if (encoded == -1) return;
            if (encoded & 0x80000000) {
                int rb_idx = encoded & 0x7FFFFFFF;
                if (rb_idx >= 0 && rb_idx < static_cast<int>(free_rigid_bodies.size()))
                    out_rb = free_rigid_bodies[rb_idx];
            } else {
                int body_idx = encoded >> 16;
                int link_idx = encoded & 0xFFFF;
                if (body_idx >= 0 && body_idx < static_cast<int>(multibodies.size())) {
                    out_mb = multibodies[body_idx];
                    out_link = link_idx;
                }
            }
        };

        FreeRigidBody* rb_a = nullptr;
        MultiBody* mb_a = nullptr;
        int link_a = -1;
        FreeRigidBody* rb_b = nullptr;
        MultiBody* mb_b = nullptr;
        int link_b = -1;
        decode(cp.body_a, rb_a, mb_a, link_a);
        decode(cp.body_b, rb_b, mb_b, link_b);

        pgs_setup_rigid_contact(rb_a, mb_a, link_a, rb_b, mb_b, link_b,
                                cp.normal, cp.position, cp.penetration,
                                cp.friction, cp.restitution, dt);
    }

    for (auto* c : constraints) {
        c->create_constraint_rows(non_contact_constraints_, constraint_data_, dt);
    }
}

void MultiBodySolver::pgs_setup_rigid_contact(
    FreeRigidBody* rb_a, MultiBody* mb_a, int link_a,
    FreeRigidBody* rb_b, MultiBody* mb_b, int link_b,
    const Vec3f& normal, const Vec3f& point,
    float penetration, float friction_coeff, float restitution,
    float dt) {

    bool a_is_static = (rb_a == nullptr && mb_a == nullptr);
    bool b_is_static = (rb_b == nullptr && mb_b == nullptr);
    if (a_is_static && b_is_static) return;

    float denom = 0.0f;
    float rel_vel = 0.0f;

    MultiBodySolverConstraint c;
    c.multi_body_a = mb_a;
    c.link_a = link_a;
    c.free_rigid_body_a = rb_a;
    c.multi_body_b = mb_b;
    c.link_b = link_b;
    c.free_rigid_body_b = rb_b;

    if (mb_a) {
        int ndof_a = 6 + mb_a->num_dofs();
        c.jac_a_index = constraint_data_.allocate_jacobian(ndof_a);
        if (mb_a->companion_id < 0) {
            c.delta_vel_a_index = constraint_data_.allocate_delta_velocities(ndof_a);
            mb_a->companion_id = c.delta_vel_a_index;
        } else {
            c.delta_vel_a_index = mb_a->companion_id;
        }
        float* jac = &constraint_data_.jacobians[c.jac_a_index];
        mb_a->fill_constraint_jacobian(link_a, point, Vec3f::Zero(), -normal, jac);
        float* unit_resp = &constraint_data_.delta_velocities_unit_impulse[c.jac_a_index];
        mb_a->calc_acceleration_deltas(jac, unit_resp);
        for (int i = 0; i < ndof_a; ++i) denom += jac[i] * unit_resp[i];
        for (int i = 0; i < mb_a->num_dofs(); ++i)
            rel_vel += constraint_data_.jacobians[c.jac_a_index + 6 + i] * mb_a->qd()(i);
    } else if (rb_a) {
        Vec3f r_a = point - rb_a->center_of_mass_position();
        c.contact_normal_1 = -normal;
        c.relpos1_cross_normal = -(r_a.cross(normal));
        c.angular_component_a = rb_a->inv_inertia_tensor_world() * c.relpos1_cross_normal;
        denom += rb_a->inv_mass() + c.relpos1_cross_normal.dot(c.angular_component_a);
        rel_vel += c.contact_normal_1.dot(rb_a->linear_velocity())
                 + c.relpos1_cross_normal.dot(rb_a->angular_velocity());
    }

    if (mb_b) {
        int ndof_b = 6 + mb_b->num_dofs();
        c.jac_b_index = constraint_data_.allocate_jacobian(ndof_b);
        if (mb_b->companion_id < 0) {
            c.delta_vel_b_index = constraint_data_.allocate_delta_velocities(ndof_b);
            mb_b->companion_id = c.delta_vel_b_index;
        } else {
            c.delta_vel_b_index = mb_b->companion_id;
        }
        float* jac = &constraint_data_.jacobians[c.jac_b_index];
        mb_b->fill_constraint_jacobian(link_b, point, Vec3f::Zero(), normal, jac);
        float* unit_resp = &constraint_data_.delta_velocities_unit_impulse[c.jac_b_index];
        mb_b->calc_acceleration_deltas(jac, unit_resp);
        for (int i = 0; i < ndof_b; ++i) denom += jac[i] * unit_resp[i];
        for (int i = 0; i < mb_b->num_dofs(); ++i)
            rel_vel += constraint_data_.jacobians[c.jac_b_index + 6 + i] * mb_b->qd()(i);
    } else if (rb_b) {
        Vec3f r_b = point - rb_b->center_of_mass_position();
        c.contact_normal_2 = normal;
        c.relpos2_cross_normal = r_b.cross(normal);
        c.angular_component_b = rb_b->inv_inertia_tensor_world() * c.relpos2_cross_normal;
        denom += rb_b->inv_mass() + c.relpos2_cross_normal.dot(c.angular_component_b);
        rel_vel += c.contact_normal_2.dot(rb_b->linear_velocity())
                 + c.relpos2_cross_normal.dot(rb_b->angular_velocity());
    }

    denom += constraint_settings_.cfm / dt;
    if (denom < 1e-10f) return;

    float depth = penetration - constraint_settings_.linear_slop;
    float penetration_impulse = depth > 0.0f ? depth * constraint_settings_.erp / dt : 0.0f;
    float restitution_impulse = (-rel_vel > constraint_settings_.restitution_threshold)
                                    ? -restitution * rel_vel
                                    : 0.0f;

    c.jac_diag_ab_inv = 1.0f / denom;
    c.rhs = restitution_impulse - rel_vel;
    c.rhs_penetration = penetration_impulse;
    c.cfm = constraint_settings_.cfm / dt;
    c.lower_limit = 0.0f;
    c.upper_limit = 1e10f;
    c.friction = friction_coeff;
    c.applied_impulse = 0.0f;

    int normal_index = static_cast<int>(normal_constraints_.size());
    normal_constraints_.push_back(c);

    Vec3f t1, t2;
    plane_space(normal, t1, t2);
    pgs_add_friction_constraint(rb_a, mb_a, link_a, rb_b, mb_b, link_b,
                                point, t1, dt, normal_index);
    pgs_add_friction_constraint(rb_a, mb_a, link_a, rb_b, mb_b, link_b,
                                point, t2, dt, normal_index);
}

void MultiBodySolver::pgs_add_friction_constraint(
    FreeRigidBody* rb_a, MultiBody* mb_a, int link_a,
    FreeRigidBody* rb_b, MultiBody* mb_b, int link_b,
    const Vec3f& point, const Vec3f& friction_dir,
    float /*dt*/, int normal_index) {

    float denom = 0.0f;
    float rel_vel = 0.0f;

    MultiBodySolverConstraint c;
    c.multi_body_a = mb_a;
    c.link_a = link_a;
    c.free_rigid_body_a = rb_a;
    c.multi_body_b = mb_b;
    c.link_b = link_b;
    c.free_rigid_body_b = rb_b;

    if (mb_a) {
        int ndof_a = 6 + mb_a->num_dofs();
        c.jac_a_index = constraint_data_.allocate_jacobian(ndof_a);
        c.delta_vel_a_index = mb_a->companion_id;
        float* jac = &constraint_data_.jacobians[c.jac_a_index];
        mb_a->fill_constraint_jacobian(link_a, point, Vec3f::Zero(), -friction_dir, jac);
        float* unit_resp = &constraint_data_.delta_velocities_unit_impulse[c.jac_a_index];
        mb_a->calc_acceleration_deltas(jac, unit_resp);
        for (int i = 0; i < ndof_a; ++i) denom += jac[i] * unit_resp[i];
        for (int i = 0; i < mb_a->num_dofs(); ++i)
            rel_vel += constraint_data_.jacobians[c.jac_a_index + 6 + i] * mb_a->qd()(i);
    } else if (rb_a) {
        Vec3f r_a = point - rb_a->center_of_mass_position();
        c.contact_normal_1 = -friction_dir;
        c.relpos1_cross_normal = -(r_a.cross(friction_dir));
        c.angular_component_a = rb_a->inv_inertia_tensor_world() * c.relpos1_cross_normal;
        denom += rb_a->inv_mass() + c.relpos1_cross_normal.dot(c.angular_component_a);
        rel_vel += c.contact_normal_1.dot(rb_a->linear_velocity())
                 + c.relpos1_cross_normal.dot(rb_a->angular_velocity());
    }

    if (mb_b) {
        int ndof_b = 6 + mb_b->num_dofs();
        c.jac_b_index = constraint_data_.allocate_jacobian(ndof_b);
        c.delta_vel_b_index = mb_b->companion_id;
        float* jac = &constraint_data_.jacobians[c.jac_b_index];
        mb_b->fill_constraint_jacobian(link_b, point, Vec3f::Zero(), friction_dir, jac);
        float* unit_resp = &constraint_data_.delta_velocities_unit_impulse[c.jac_b_index];
        mb_b->calc_acceleration_deltas(jac, unit_resp);
        for (int i = 0; i < ndof_b; ++i) denom += jac[i] * unit_resp[i];
        for (int i = 0; i < mb_b->num_dofs(); ++i)
            rel_vel += constraint_data_.jacobians[c.jac_b_index + 6 + i] * mb_b->qd()(i);
    } else if (rb_b) {
        Vec3f r_b = point - rb_b->center_of_mass_position();
        c.contact_normal_2 = friction_dir;
        c.relpos2_cross_normal = r_b.cross(friction_dir);
        c.angular_component_b = rb_b->inv_inertia_tensor_world() * c.relpos2_cross_normal;
        denom += rb_b->inv_mass() + c.relpos2_cross_normal.dot(c.angular_component_b);
        rel_vel += c.contact_normal_2.dot(rb_b->linear_velocity())
                 + c.relpos2_cross_normal.dot(rb_b->angular_velocity());
    }

    if (denom < 1e-10f) return;

    c.jac_diag_ab_inv = 1.0f / denom;
    c.rhs = -rel_vel;
    c.cfm = 0.0f;
    c.lower_limit = 0.0f;
    c.upper_limit = 0.0f;
    c.friction_index = normal_index;
    c.applied_impulse = 0.0f;

    friction_constraints_.push_back(c);
}

void MultiBodySolver::pgs_solve_iterations() {
    for (int iter = 0; iter < constraint_settings_.num_iterations; ++iter) {
        for (auto& c : non_contact_constraints_) {
            pgs_resolve_row(c);
        }

        for (auto& c : normal_constraints_) {
            pgs_resolve_row(c);
        }

        for (auto& c : friction_constraints_) {
            if (c.friction_index >= 0 &&
                c.friction_index < static_cast<int>(normal_constraints_.size())) {
                float normal_impulse = normal_constraints_[c.friction_index].applied_impulse;
                float friction = normal_constraints_[c.friction_index].friction;
                float limit = friction * std::abs(normal_impulse);
                c.lower_limit = -limit;
                c.upper_limit = limit;
            }
            pgs_resolve_row(c);
        }
    }
}

float MultiBodySolver::pgs_resolve_row(MultiBodySolverConstraint& c) {

    float delta_vel_a_dot_n = 0.0f;
    float delta_vel_b_dot_n = 0.0f;

    if (c.multi_body_a && c.jac_a_index >= 0 && c.delta_vel_a_index >= 0) {
        int ndof = 6 + c.multi_body_a->num_dofs();
        const float* jac = &constraint_data_.jacobians[c.jac_a_index];
        const float* dv = &constraint_data_.delta_velocities[c.delta_vel_a_index];
        for (int i = 0; i < ndof; ++i) {
            delta_vel_a_dot_n += jac[i] * dv[i];
        }
    }
    if (c.free_rigid_body_a) {
        delta_vel_a_dot_n += c.contact_normal_1.dot(c.free_rigid_body_a->delta_linear_velocity())
                           + c.relpos1_cross_normal.dot(c.free_rigid_body_a->delta_angular_velocity());
    }

    if (c.multi_body_b && c.jac_b_index >= 0 && c.delta_vel_b_index >= 0) {
        int ndof = 6 + c.multi_body_b->num_dofs();
        const float* jac = &constraint_data_.jacobians[c.jac_b_index];
        const float* dv = &constraint_data_.delta_velocities[c.delta_vel_b_index];
        for (int i = 0; i < ndof; ++i) {
            delta_vel_b_dot_n += jac[i] * dv[i];
        }
    }
    if (c.free_rigid_body_b) {
        delta_vel_b_dot_n += c.contact_normal_2.dot(c.free_rigid_body_b->delta_linear_velocity())
                           + c.relpos2_cross_normal.dot(c.free_rigid_body_b->delta_angular_velocity());
    }

    float delta_impulse = (c.rhs - c.applied_impulse * c.cfm
        - delta_vel_a_dot_n - delta_vel_b_dot_n) * c.jac_diag_ab_inv;

    float sum = c.applied_impulse + delta_impulse;
    if (sum < c.lower_limit) {
        delta_impulse = c.lower_limit - c.applied_impulse;
        c.applied_impulse = c.lower_limit;
    } else if (sum > c.upper_limit) {
        delta_impulse = c.upper_limit - c.applied_impulse;
        c.applied_impulse = c.upper_limit;
    } else {
        c.applied_impulse = sum;
    }

    if (c.multi_body_a && c.jac_a_index >= 0 && c.delta_vel_a_index >= 0) {
        int ndof = 6 + c.multi_body_a->num_dofs();
        const float* unit_resp = &constraint_data_.delta_velocities_unit_impulse[c.jac_a_index];
        float* dv = &constraint_data_.delta_velocities[c.delta_vel_a_index];
        for (int i = 0; i < ndof; ++i) {
            dv[i] += unit_resp[i] * delta_impulse;
        }
    }
    if (c.free_rigid_body_a && c.free_rigid_body_a->inv_mass() > 0.0f) {
        c.free_rigid_body_a->delta_linear_velocity() +=
            c.contact_normal_1 * c.free_rigid_body_a->inv_mass() * delta_impulse;
        c.free_rigid_body_a->delta_angular_velocity() +=
            c.angular_component_a * delta_impulse;
    }

    if (c.multi_body_b && c.jac_b_index >= 0 && c.delta_vel_b_index >= 0) {
        int ndof = 6 + c.multi_body_b->num_dofs();
        const float* unit_resp = &constraint_data_.delta_velocities_unit_impulse[c.jac_b_index];
        float* dv = &constraint_data_.delta_velocities[c.delta_vel_b_index];
        for (int i = 0; i < ndof; ++i) {
            dv[i] += unit_resp[i] * delta_impulse;
        }
    }
    if (c.free_rigid_body_b && c.free_rigid_body_b->inv_mass() > 0.0f) {
        c.free_rigid_body_b->delta_linear_velocity() +=
            c.contact_normal_2 * c.free_rigid_body_b->inv_mass() * delta_impulse;
        c.free_rigid_body_b->delta_angular_velocity() +=
            c.angular_component_b * delta_impulse;
    }

    return delta_impulse;
}

void MultiBodySolver::pgs_finalize(std::vector<MultiBody*>& multibodies,
                                     std::vector<FreeRigidBody*>& free_rigid_bodies) {
    for (auto* body : multibodies) {
        if (body->companion_id >= 0) {
            int ndof = 6 + body->num_dofs();
            const float* dv = &constraint_data_.delta_velocities[body->companion_id];
            for (int i = 0; i < body->num_dofs(); ++i) {
                body->qd()(i) += dv[6 + i];
            }
        }
        body->companion_id = -1;
    }

    for (auto* rb : free_rigid_bodies) {
        if (rb->inv_mass() > 0.0f) {
            rb->set_linear_velocity(rb->linear_velocity() + rb->delta_linear_velocity());
            rb->set_angular_velocity(rb->angular_velocity() + rb->delta_angular_velocity());
        }
        rb->delta_linear_velocity() = Vec3f::Zero();
        rb->delta_angular_velocity() = Vec3f::Zero();
    }
}

void MultiBodySolver::step(const Model& model, SimState& state,
                           const Control& control, const Vec3f& gravity,
                           float dt) {
    if (dt <= 0.0f) return;

    gravity_ = gravity;

    sync_state_in(state);
    sync_rb_state_in(state);

    apply_control(control);

    for (std::size_t a = 0; a < bodies_.size(); ++a) {
        auto& body = bodies_[a];
        body->clear_forces();

        if (a < state.articulation_forces.size()) {
            const auto& link_wrenches = state.articulation_forces[a];
            for (std::size_t link = 0; link < link_wrenches.size(); ++link) {
                const auto& w = link_wrenches[link];
                body->add_link_torque(static_cast<int>(link), w.head<3>());
                body->add_link_force(static_cast<int>(link), w.tail<3>());
            }
        }
    }

    for (std::size_t i = 0; i < free_rigid_bodies_.size(); ++i) {
        auto& rb = free_rigid_bodies_[i];
        rb->clear_forces();

        if (i < control.body_forces.size())
            rb->apply_force(control.body_forces[i]);
        if (i < control.body_torques.size())
            rb->apply_torque(control.body_torques[i]);
        if (i < state.forces.size())
            rb->apply_force(state.forces[i]);
        if (i < state.torques.size())
            rb->apply_torque(state.torques[i]);

        rb->set_gravity(gravity_);
        rb->apply_gravity();
        rb->integrate_velocities(dt);
        rb->apply_damping(dt);
    }

    for (auto& body : bodies_) {
        body->compute_accelerations_aba(dt, false, gravity_);
    }

    run_collision_detection();

    std::vector<MultiBody*> body_ptrs;
    body_ptrs.reserve(bodies_.size());
    for (auto& b : bodies_) body_ptrs.push_back(b.get());

    std::vector<FreeRigidBody*> rb_ptrs;
    rb_ptrs.reserve(free_rigid_bodies_.size());
    for (auto& rb : free_rigid_bodies_) rb_ptrs.push_back(rb.get());

    std::vector<MultiBodyConstraint*> constraint_ptrs;
    constraint_ptrs.reserve(persistent_constraints_.size());
    for (auto& c : persistent_constraints_) constraint_ptrs.push_back(c.get());

    solve_constraints(body_ptrs, rb_ptrs, contacts_, constraint_ptrs, dt);

    for (auto& body : bodies_) {
        body->step_positions(dt);
        body->apply_velocity_damping();
    }

    for (auto& rb : free_rigid_bodies_) {
        if (rb->inv_mass() <= 0.0f) continue;
        Transform predicted;
        rb->predict_integrated_transform(dt, predicted);
        rb->proceed_to_transform(predicted);
    }

    run_split_impulse_correction();

    for (auto& body : bodies_) {
        body->forward_kinematics();
    }

    sync_state_out(state);
    sync_rb_state_out(state);
}

}  // namespace novaphy
