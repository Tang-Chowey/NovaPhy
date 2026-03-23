#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "novaphy/core/contact.h"
#include "novaphy/core/model.h"
#include "novaphy/dynamics/xpbd_solver.h"
#include "novaphy/io/scene_types.h"

namespace novaphy {

/**
 * @brief High-level articulated simulation container backed by XPBD.
 *
 * @details Wraps an articulated model, its collision geometry, joint-name
 * mapping, static environment shapes, and the reduced-coordinate XPBD solver so
 * Python callers do not need to manage raw q/qd arrays directly.
 */
class ArticulatedWorld {
public:
    explicit ArticulatedWorld(const SceneBuildResult& scene,
                              XPBDSolverSettings solver_settings = {});

    void step(float dt);

    void set_gravity(const Vec3f& gravity) { gravity_ = gravity; }
    const Vec3f& gravity() const { return gravity_; }

    const VecXf& q() const { return q_; }
    void set_q(const VecXf& q);

    const VecXf& qd() const { return qd_; }
    void set_qd(const VecXf& qd);

    const std::vector<std::string>& joint_names() const { return joint_names_; }
    std::unordered_map<std::string, float> joint_positions() const;
    const std::vector<ContactPoint>& contacts() const { return contacts_; }
    const SceneBuildMetadata& metadata() const { return metadata_; }

    void set_joint_positions(const std::unordered_map<std::string, float>& positions);
    void set_default_drive_gains(float stiffness, float damping);
    void clear_target_positions();
    void set_target_positions(const std::unordered_map<std::string, float>& targets);

    void add_static_shape(const CollisionShape& shape);
    void add_ground_plane(const Vec3f& normal,
                          float offset,
                          float friction = 0.8f,
                          float restitution = 0.0f);

    XPBDSolver& solver() { return solver_; }
    const XPBDSolver& solver() const { return solver_; }

private:
    void ensure_drive_capacity();
    void rebuild_joint_name_map();

    Articulation articulation_;
    Model collision_model_;
    SceneBuildMetadata metadata_;
    XPBDSolver solver_;
    XPBDControl control_;
    VecXf q_;
    VecXf qd_;
    Vec3f gravity_ = Vec3f(0.0f, 0.0f, -9.81f);
    float default_drive_stiffness_ = 50.0f;
    float default_drive_damping_ = 5.0f;
    std::vector<std::string> joint_names_;
    std::unordered_map<std::string, int> joint_name_to_link_index_;
    std::unordered_map<std::string, int> joint_name_to_q_index_;
    std::unordered_map<std::string, int> joint_name_to_qd_index_;
    std::vector<CollisionShape> static_shapes_;
    std::vector<ContactPoint> contacts_;
};

}  // namespace novaphy
