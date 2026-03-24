#include "novaphy/dynamics/xpbd_solver.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "novaphy/collision/narrowphase.h"
#include "novaphy/dynamics/featherstone.h"

namespace novaphy {

XPBDSolver::XPBDSolver(XPBDSolverSettings settings) : settings_(settings) {}

void XPBDSolver::step(const Articulation& model,
                      VecXf& q,
                      VecXf& qd,
                      const VecXf& tau,
                      const Vec3f& gravity,
                      float dt,
                      const XPBDControl& control) {
    Model empty_model;
    std::vector<CollisionShape> no_static_shapes;
    step_with_contacts(model, empty_model, no_static_shapes, q, qd, tau, gravity, dt, control, nullptr);
}

void XPBDSolver::step_with_contacts(const Articulation& model,
                                    const Model& collision_model,
                                    const std::vector<CollisionShape>& static_shapes,
                                    VecXf& q,
                                    VecXf& qd,
                                    const VecXf& tau,
                                    const Vec3f& gravity,
                                    float dt,
                                    const XPBDControl& control,
                                    std::vector<ContactPoint>* contacts) {
    const int substeps = std::max(1, settings_.substeps);
    const float dt_substep = dt / static_cast<float>(substeps);

    last_stats_ = XPBDStepStats{};
    last_stats_.substeps = substeps;
    last_stats_.iterations = settings_.iterations;

    std::vector<ContactPoint> local_contacts;
    std::vector<ContactPoint> reported_contacts;
    int max_contact_count = 0;
    for (int substep = 0; substep < substeps; ++substep) {
        const VecXf qdd = featherstone::forward_dynamics(model, q, qd, tau, gravity);
        qd += qdd * dt_substep;
        integrate_positions(model, q, qd, dt_substep);
        normalize_quaternions(model, q);

        for (int iter = 0; iter < settings_.iterations; ++iter) {
            last_stats_.projected_constraints += project_joint_limits(model, q, qd);
            last_stats_.projected_constraints += project_joint_drives(model, q, qd, control, dt_substep);
            last_stats_.projected_constraints += project_contacts(
                model, collision_model, static_shapes, q, qd, dt_substep, &local_contacts);
            if (!local_contacts.empty()) {
                reported_contacts = local_contacts;
                max_contact_count = std::max(max_contact_count, static_cast<int>(local_contacts.size()));
            }
            normalize_quaternions(model, q);
        }

        qd *= settings_.velocity_damping;
    }

    last_stats_.contact_count = max_contact_count;
    if (contacts != nullptr) {
        *contacts = reported_contacts;
    }
}

void XPBDSolver::integrate_positions(const Articulation& model,
                                     VecXf& q,
                                     const VecXf& qd,
                                     float dt) const {
    for (int i = 0; i < model.num_links(); ++i) {
        const Joint& joint = model.joints[i];
        const int qi = model.q_start(i);
        const int qdi = model.qd_start(i);

        switch (joint.type) {
            case JointType::Revolute:
                q(qi) += qd(qdi) * dt;
                break;

            case JointType::Fixed:
                break;

            case JointType::Slide:
                q(qi) += qd(qdi) * dt;
                break;

            case JointType::Free: {
                q(qi + 0) += qd(qdi + 3) * dt;
                q(qi + 1) += qd(qdi + 4) * dt;
                q(qi + 2) += qd(qdi + 5) * dt;

                const float wx = qd(qdi + 0);
                const float wy = qd(qdi + 1);
                const float wz = qd(qdi + 2);

                const float qx = q(qi + 3);
                const float qy = q(qi + 4);
                const float qz = q(qi + 5);
                const float qw = q(qi + 6);

                const float dqx = 0.5f * (wx * qw + wz * qy - wy * qz);
                const float dqy = 0.5f * (wy * qw - wz * qx + wx * qz);
                const float dqz = 0.5f * (wz * qw + wy * qx - wx * qy);
                const float dqw = 0.5f * (-wx * qx - wy * qy - wz * qz);

                q(qi + 3) += dqx * dt;
                q(qi + 4) += dqy * dt;
                q(qi + 5) += dqz * dt;
                q(qi + 6) += dqw * dt;
                break;
            }

            case JointType::Ball: {
                const float wx = qd(qdi + 0);
                const float wy = qd(qdi + 1);
                const float wz = qd(qdi + 2);

                const float qx = q(qi + 0);
                const float qy = q(qi + 1);
                const float qz = q(qi + 2);
                const float qw = q(qi + 3);

                const float dqx = 0.5f * (wx * qw + wz * qy - wy * qz);
                const float dqy = 0.5f * (wy * qw - wz * qx + wx * qz);
                const float dqz = 0.5f * (wz * qw + wy * qx - wx * qy);
                const float dqw = 0.5f * (-wx * qx - wy * qy - wz * qz);

                q(qi + 0) += dqx * dt;
                q(qi + 1) += dqy * dt;
                q(qi + 2) += dqz * dt;
                q(qi + 3) += dqw * dt;
                break;
            }
        }
    }
}

void XPBDSolver::normalize_quaternions(const Articulation& model, VecXf& q) const {
    for (int i = 0; i < model.num_links(); ++i) {
        const Joint& joint = model.joints[i];
        const int qi = model.q_start(i);

        if (joint.type == JointType::Free) {
            const float norm = std::sqrt(
                q(qi + 3) * q(qi + 3) + q(qi + 4) * q(qi + 4) +
                q(qi + 5) * q(qi + 5) + q(qi + 6) * q(qi + 6));
            if (norm > 1.0e-6f) {
                q(qi + 3) /= norm;
                q(qi + 4) /= norm;
                q(qi + 5) /= norm;
                q(qi + 6) /= norm;
            }
        } else if (joint.type == JointType::Ball) {
            const float norm = std::sqrt(
                q(qi + 0) * q(qi + 0) + q(qi + 1) * q(qi + 1) +
                q(qi + 2) * q(qi + 2) + q(qi + 3) * q(qi + 3));
            if (norm > 1.0e-6f) {
                q(qi + 0) /= norm;
                q(qi + 1) /= norm;
                q(qi + 2) /= norm;
                q(qi + 3) /= norm;
            }
        }
    }
}

int XPBDSolver::project_joint_limits(const Articulation& model,
                                     VecXf& q,
                                     VecXf& qd) const {
    int projected = 0;
    for (int i = 0; i < model.num_links(); ++i) {
        const Joint& joint = model.joints[i];
        if (!joint.limit_enabled) continue;
        if (joint.type != JointType::Revolute && joint.type != JointType::Slide) continue;

        const int qi = model.q_start(i);
        const int qdi = model.qd_start(i);
        if (q(qi) < joint.lower_limit) {
            q(qi) = joint.lower_limit;
            if (qd(qdi) < 0.0f) qd(qdi) = 0.0f;
            ++projected;
        } else if (q(qi) > joint.upper_limit) {
            q(qi) = joint.upper_limit;
            if (qd(qdi) > 0.0f) qd(qdi) = 0.0f;
            ++projected;
        }
    }
    return projected;
}

int XPBDSolver::project_joint_drives(const Articulation& model,
                                     VecXf& q,
                                     VecXf& qd,
                                     const XPBDControl& control,
                                     float dt) const {
    int projected = 0;
    const size_t drive_count = control.joint_drives.size();
    for (int i = 0; i < model.num_links(); ++i) {
        if (static_cast<size_t>(i) >= drive_count) break;
        const Joint& joint = model.joints[i];
        if (joint.type != JointType::Revolute && joint.type != JointType::Slide) continue;

        const XPBDJointDrive& drive = control.joint_drives[static_cast<size_t>(i)];
        if (drive.mode != JointDriveMode::TargetPosition) continue;

        const int qi = model.q_start(i);
        const int qdi = model.qd_start(i);
        const float error = drive.target_position - q(qi);
        const float alpha = std::clamp(1.0f - std::exp(-drive.stiffness * dt), 0.0f, 1.0f);
        q(qi) += alpha * error;
        const float damping_scale = std::clamp(1.0f - drive.damping * dt, 0.0f, 1.0f);
        qd(qdi) = qd(qdi) * damping_scale + (alpha * error) / std::max(dt, 1.0e-6f);
        ++projected;

        if (joint.limit_enabled) {
            if (q(qi) < joint.lower_limit) {
                q(qi) = joint.lower_limit;
                if (qd(qdi) < 0.0f) qd(qdi) = 0.0f;
            } else if (q(qi) > joint.upper_limit) {
                q(qi) = joint.upper_limit;
                if (qd(qdi) > 0.0f) qd(qdi) = 0.0f;
            }
        }
    }
    return projected;
}

int XPBDSolver::project_contacts(const Articulation& model,
                                 const Model& collision_model,
                                 const std::vector<CollisionShape>& static_shapes,
                                 VecXf& q,
                                 VecXf& qd,
                                 float dt,
                                 std::vector<ContactPoint>* contacts) {
    const size_t dynamic_shape_count = collision_model.shapes.size();
    const size_t total_shape_count = dynamic_shape_count + static_shapes.size();
    if (total_shape_count == 0) {
        if (contacts != nullptr) contacts->clear();
        return 0;
    }

    const std::vector<Transform> link_transforms = featherstone::forward_kinematics(model, q).world_transforms;
    std::vector<AABB> shape_aabbs(total_shape_count);
    std::vector<bool> static_mask(total_shape_count, true);

    auto dynamic_shape_transform = [&](const CollisionShape& shape) -> Transform {
        if (shape.body_index >= 0 && shape.body_index < static_cast<int>(link_transforms.size())) {
            return link_transforms[static_cast<size_t>(shape.body_index)];
        }
        return Transform::identity();
    };

    for (size_t i = 0; i < total_shape_count; ++i) {
        const bool is_dynamic_shape = i < dynamic_shape_count;
        const CollisionShape& shape = is_dynamic_shape ? collision_model.shapes[i] : static_shapes[i - dynamic_shape_count];
        const Transform body_transform = is_dynamic_shape ? dynamic_shape_transform(shape) : Transform::identity();
        shape_aabbs[i] = shape.compute_aabb(body_transform);
        if (shape.body_index >= 0 && shape.body_index < static_cast<int>(collision_model.bodies.size())) {
            static_mask[i] = collision_model.bodies[static_cast<size_t>(shape.body_index)].is_static();
        } else {
            static_mask[i] = true;
        }
    }

    broadphase_.update(shape_aabbs, static_mask);

    std::vector<ContactPoint> local_contacts;
    int projected = 0;
    for (const BroadPhasePair& pair : broadphase_.get_pairs()) {
        const int shape_a_index = pair.body_a;
        const int shape_b_index = pair.body_b;
        const bool a_is_dynamic_shape = shape_a_index >= 0 &&
            shape_a_index < static_cast<int>(dynamic_shape_count);
        const bool b_is_dynamic_shape = shape_b_index >= 0 &&
            shape_b_index < static_cast<int>(dynamic_shape_count);

        if (a_is_dynamic_shape && b_is_dynamic_shape &&
            collision_model.is_collision_pair_filtered(shape_a_index, shape_b_index)) {
            continue;
        }

        const CollisionShape& shape_a = a_is_dynamic_shape
            ? collision_model.shapes[static_cast<size_t>(shape_a_index)]
            : static_shapes[static_cast<size_t>(shape_a_index) - dynamic_shape_count];
        const CollisionShape& shape_b = b_is_dynamic_shape
            ? collision_model.shapes[static_cast<size_t>(shape_b_index)]
            : static_shapes[static_cast<size_t>(shape_b_index) - dynamic_shape_count];

        const Transform transform_a = a_is_dynamic_shape ? dynamic_shape_transform(shape_a) : Transform::identity();
        const Transform transform_b = b_is_dynamic_shape ? dynamic_shape_transform(shape_b) : Transform::identity();

        std::vector<ContactPoint> pair_contacts;
        if (!collide_shapes(shape_a, transform_a, shape_b, transform_b, pair_contacts)) {
            continue;
        }

        for (ContactPoint& contact : pair_contacts) {
            contact.shape_a = shape_a_index;
            contact.shape_b = shape_b_index;
            contact.friction = combine_friction(shape_a.friction, shape_b.friction);
            contact.restitution = combine_restitution(shape_a.restitution, shape_b.restitution);
            local_contacts.push_back(contact);

            const int dynamic_bodies = (contact.body_a >= 0 ? 1 : 0) + (contact.body_b >= 0 ? 1 : 0);
            const float split = dynamic_bodies > 1 ? 0.5f : 1.0f;
            ContactPoint scaled_contact = contact;
            scaled_contact.penetration *= split;
            if (contact.body_a >= 0) {
                projected += apply_contact_correction(
                    model, link_transforms, scaled_contact, contact.body_a, -1.0f, q, qd, dt);
            }
            if (contact.body_b >= 0) {
                projected += apply_contact_correction(
                    model, link_transforms, scaled_contact, contact.body_b, 1.0f, q, qd, dt);
            }
        }
    }

    if (contacts != nullptr) {
        *contacts = local_contacts;
    }
    return projected;
}

int XPBDSolver::apply_contact_correction(const Articulation& model,
                                         const std::vector<Transform>& link_transforms,
                                         const ContactPoint& contact,
                                         int link_index,
                                         float sign,
                                         VecXf& q,
                                         VecXf& qd,
                                         float dt) const {
    if (link_index < 0 || link_index >= model.num_links()) return 0;

    const Vec3f normal = sign * contact.normal;
    float denominator = 0.0f;
    bool has_root_translation = model.num_links() > 0 && model.joints[0].type == JointType::Free;
    struct JacobianEntry {
        int link = -1;
        float value = 0.0f;
    };
    std::vector<JacobianEntry> jacobians;

    if (has_root_translation) {
        denominator += normal.squaredNorm();
    }

    int cursor = link_index;
    while (cursor >= 0) {
        const Joint& joint = model.joints[cursor];
        Transform joint_world = joint.parent_to_joint;
        if (joint.parent >= 0) {
            joint_world = link_transforms[static_cast<size_t>(joint.parent)] * joint.parent_to_joint;
        }

        if (joint.type == JointType::Revolute) {
            const Vec3f axis_world = joint_world.transform_vector(joint.axis.normalized());
            const float jac = normal.dot(axis_world.cross(contact.position - joint_world.position));
            if (std::abs(jac) > 1.0e-6f) {
                jacobians.push_back({cursor, jac});
                denominator += jac * jac;
            }
        } else if (joint.type == JointType::Slide) {
            const Vec3f axis_world = joint_world.transform_vector(joint.axis.normalized());
            const float jac = normal.dot(axis_world);
            if (std::abs(jac) > 1.0e-6f) {
                jacobians.push_back({cursor, jac});
                denominator += jac * jac;
            }
        }

        cursor = joint.parent;
    }

    if (denominator <= 1.0e-6f) {
        return 0;
    }

    const float correction = settings_.contact_relaxation * contact.penetration / denominator;
    if (has_root_translation) {
        const int qi = model.q_start(0);
        const int qdi = model.qd_start(0);
        q(qi + 0) += correction * normal.x();
        q(qi + 1) += correction * normal.y();
        q(qi + 2) += correction * normal.z();

        // Zero out approaching root velocity along contact normal without
        // injecting artificial separation velocity.
        Vec3f linear_velocity(qd(qdi + 3), qd(qdi + 4), qd(qdi + 5));
        const float normal_velocity = linear_velocity.dot(normal);
        if (normal_velocity < 0.0f) {
            linear_velocity -= normal * normal_velocity;
        }
        const Vec3f tangential_velocity = linear_velocity - normal * linear_velocity.dot(normal);
        linear_velocity -= tangential_velocity * std::clamp(contact.friction * settings_.friction_damping,
                                                            0.0f, 1.0f);
        qd(qdi + 3) = linear_velocity.x();
        qd(qdi + 4) = linear_velocity.y();
        qd(qdi + 5) = linear_velocity.z();
    }

    const float joint_damping_scale = std::clamp(
        1.0f - contact.friction * settings_.friction_damping * dt, 0.0f, 1.0f);
    for (const JacobianEntry& entry : jacobians) {
        const int qi = model.q_start(entry.link);
        const int qdi = model.qd_start(entry.link);
        q(qi) += correction * entry.value;
        qd(qdi) = (qd(qdi) + (correction * entry.value) / std::max(dt, 1.0e-6f)) * joint_damping_scale;
    }

    return 1;
}

}  // namespace novaphy
