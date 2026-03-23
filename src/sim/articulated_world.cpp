#include "novaphy/sim/articulated_world.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace novaphy {

ArticulatedWorld::ArticulatedWorld(const SceneBuildResult& scene,
                                   XPBDSolverSettings solver_settings)
    : articulation_(scene.articulation),
      collision_model_(scene.model),
      metadata_(scene.metadata),
      solver_(solver_settings),
      q_(scene.initial_q.size() == scene.articulation.total_q()
             ? scene.initial_q
             : VecXf::Zero(scene.articulation.total_q())),
      qd_(scene.initial_qd.size() == scene.articulation.total_qd()
              ? scene.initial_qd
              : VecXf::Zero(scene.articulation.total_qd())) {
    rebuild_joint_name_map();
    ensure_drive_capacity();
}

void ArticulatedWorld::step(float dt) {
    if (dt <= 0.0f) {
        throw std::invalid_argument("ArticulatedWorld::step requires dt > 0.");
    }

    VecXf tau = VecXf::Zero(articulation_.total_qd());
    solver_.step_with_contacts(articulation_,
                               collision_model_,
                               static_shapes_,
                               q_,
                               qd_,
                               tau,
                               gravity_,
                               dt,
                               control_,
                               &contacts_);
}

void ArticulatedWorld::set_q(const VecXf& q) {
    if (q.size() != articulation_.total_q()) {
        throw std::invalid_argument("ArticulatedWorld::set_q received a vector with the wrong dimension.");
    }
    q_ = q;
}

void ArticulatedWorld::set_qd(const VecXf& qd) {
    if (qd.size() != articulation_.total_qd()) {
        throw std::invalid_argument("ArticulatedWorld::set_qd received a vector with the wrong dimension.");
    }
    qd_ = qd;
}

std::unordered_map<std::string, float> ArticulatedWorld::joint_positions() const {
    std::unordered_map<std::string, float> positions;
    positions.reserve(joint_name_to_q_index_.size());
    for (const auto& entry : joint_name_to_q_index_) {
        positions[entry.first] = q_(entry.second);
    }
    return positions;
}

void ArticulatedWorld::set_joint_positions(const std::unordered_map<std::string, float>& positions) {
    for (const auto& entry : positions) {
        auto q_it = joint_name_to_q_index_.find(entry.first);
        if (q_it == joint_name_to_q_index_.end()) {
            throw std::invalid_argument("Unknown joint name: " + entry.first);
        }
        q_(q_it->second) = entry.second;

        auto qd_it = joint_name_to_qd_index_.find(entry.first);
        if (qd_it != joint_name_to_qd_index_.end()) {
            qd_(qd_it->second) = 0.0f;
        }
    }
}

void ArticulatedWorld::set_default_drive_gains(float stiffness, float damping) {
    default_drive_stiffness_ = std::max(0.0f, stiffness);
    default_drive_damping_ = std::max(0.0f, damping);

    ensure_drive_capacity();
    for (XPBDJointDrive& drive : control_.joint_drives) {
        if (drive.mode == JointDriveMode::Off) {
            continue;
        }
        drive.stiffness = default_drive_stiffness_;
        drive.damping = default_drive_damping_;
    }
}

void ArticulatedWorld::clear_target_positions() {
    ensure_drive_capacity();
    for (XPBDJointDrive& drive : control_.joint_drives) {
        drive.mode = JointDriveMode::Off;
    }
}

void ArticulatedWorld::set_target_positions(const std::unordered_map<std::string, float>& targets) {
    ensure_drive_capacity();
    for (const auto& entry : targets) {
        auto joint_it = joint_name_to_link_index_.find(entry.first);
        if (joint_it == joint_name_to_link_index_.end()) {
            throw std::invalid_argument("Unknown joint name: " + entry.first);
        }

        XPBDJointDrive& drive = control_.joint_drives[static_cast<size_t>(joint_it->second)];
        drive.mode = JointDriveMode::TargetPosition;
        drive.target_position = entry.second;
        if (drive.stiffness <= 0.0f) {
            drive.stiffness = default_drive_stiffness_;
        }
        if (drive.damping <= 0.0f) {
            drive.damping = default_drive_damping_;
        }
    }
}

void ArticulatedWorld::add_static_shape(const CollisionShape& shape) {
    CollisionShape static_shape = shape;
    static_shape.body_index = -1;
    static_shapes_.push_back(static_shape);
}

void ArticulatedWorld::add_ground_plane(const Vec3f& normal,
                                        float offset,
                                        float friction,
                                        float restitution) {
    add_static_shape(CollisionShape::make_plane(normal, offset, friction, restitution));
}

void ArticulatedWorld::ensure_drive_capacity() {
    control_.joint_drives.resize(static_cast<size_t>(articulation_.num_links()));
}

void ArticulatedWorld::rebuild_joint_name_map() {
    joint_names_.clear();
    joint_name_to_link_index_.clear();
    joint_name_to_q_index_.clear();
    joint_name_to_qd_index_.clear();

    std::unordered_set<std::string> seen_names;
    for (const SceneJointMetadata& joint_meta : metadata_.joints) {
        if (joint_meta.articulation_index < 0 || joint_meta.num_qd <= 0 || joint_meta.joint_name.empty()) {
            continue;
        }
        if (!seen_names.insert(joint_meta.joint_name).second) {
            continue;
        }
        joint_names_.push_back(joint_meta.joint_name);
        joint_name_to_link_index_[joint_meta.joint_name] = joint_meta.articulation_index;
        if (joint_meta.num_q == 1 && joint_meta.q_start >= 0) {
            joint_name_to_q_index_[joint_meta.joint_name] = joint_meta.q_start;
        }
        if (joint_meta.num_qd >= 1 && joint_meta.qd_start >= 0) {
            joint_name_to_qd_index_[joint_meta.joint_name] = joint_meta.qd_start;
        }
    }

    if (!joint_names_.empty()) {
        return;
    }

    for (int link_index = 0; link_index < articulation_.num_links(); ++link_index) {
        const Joint& joint = articulation_.joints[link_index];
        if (joint.num_qd() <= 0) {
            continue;
        }

        std::string fallback_name;
        for (const SceneJointMetadata& joint_meta : metadata_.joints) {
            if (joint_meta.articulation_index == link_index && !joint_meta.joint_name.empty()) {
                fallback_name = joint_meta.joint_name;
                break;
            }
        }
        if (fallback_name.empty()) {
            fallback_name = "joint_" + std::to_string(link_index);
        }

        joint_names_.push_back(fallback_name);
        joint_name_to_link_index_[fallback_name] = link_index;
        if (joint.num_q() == 1) {
            joint_name_to_q_index_[fallback_name] = articulation_.q_start(link_index);
        }
        if (joint.num_qd() >= 1) {
            joint_name_to_qd_index_[fallback_name] = articulation_.qd_start(link_index);
        }
    }
}

}  // namespace novaphy
