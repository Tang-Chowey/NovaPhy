#pragma once

#include <string>
#include <utility>
#include <vector>

#include "novaphy/core/articulation.h"
#include "novaphy/core/joint.h"
#include "novaphy/core/model.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

enum class UrdfGeometryType {
    Box,
    Sphere,
    Cylinder,
    Mesh,
    Unknown
};

struct UrdfGeometry {
    UrdfGeometryType type = UrdfGeometryType::Unknown;
    Vec3f size = Vec3f::Zero();
    float radius = 0.0f;
    float length = 0.0f;
    std::string mesh_filename;
    Vec3f mesh_scale = Vec3f::Ones();
};

struct UrdfVisual {
    Transform origin = Transform::identity();
    UrdfGeometry geometry;
    std::string material_name;
};

struct UrdfCollision {
    Transform origin = Transform::identity();
    UrdfGeometry geometry;
    float friction = 0.5f;
    float restitution = 0.0f;
};

struct UrdfInertial {
    Transform origin = Transform::identity();
    float mass = 0.0f;
    Mat3f inertia = Mat3f::Zero();
};

struct UrdfLink {
    std::string name;
    UrdfInertial inertial;
    std::vector<UrdfVisual> visuals;
    std::vector<UrdfCollision> collisions;
};

struct UrdfJoint {
    std::string name;
    std::string type;
    std::string parent_link;
    std::string child_link;
    Transform origin = Transform::identity();
    Vec3f axis = Vec3f(0.0f, 0.0f, 1.0f);
    float lower_limit = 0.0f;
    float upper_limit = 0.0f;
    float effort_limit = 0.0f;
    float velocity_limit = 0.0f;
    float damping = 0.0f;
    float friction = 0.0f;
};

struct UrdfModelData {
    std::string name;
    std::vector<UrdfLink> links;
    std::vector<UrdfJoint> joints;
};

struct UrdfImportOptions {
    bool floating_base = true;
    bool enable_self_collisions = true;
    bool collapse_fixed_joints = false;
    bool use_visual_collision_fallback = false;
    bool ignore_inertial_definitions = false;
    Transform root_transform = Transform::identity();
};

struct SceneLinkMetadata {
    std::string link_name;
    std::string parent_link_name;
    std::string joint_name;
    std::string body_link_name;
    int urdf_link_index = -1;
    int body_index = -1;
    int articulation_index = -1;
    bool collapsed = false;
};

struct SceneJointMetadata {
    std::string joint_name;
    std::string parent_link_name;
    std::string child_link_name;
    std::string joint_type;
    int urdf_joint_index = -1;
    int articulation_index = -1;
    int q_start = -1;
    int qd_start = -1;
    int num_q = 0;
    int num_qd = 0;
    float lower_limit = 0.0f;
    float upper_limit = 0.0f;
    float effort_limit = 0.0f;
    float velocity_limit = 0.0f;
    float damping = 0.0f;
    float friction = 0.0f;
    bool collapsed = false;
};

struct SceneShapeMetadata {
    int shape_index = -1;
    int body_index = -1;
    int articulation_index = -1;
    int urdf_link_index = -1;
    std::string link_name;
    std::string body_link_name;
    std::string geometry_type;
    std::string source;
};

struct SceneBuildMetadata {
    UrdfImportOptions import_options;
    std::string root_link_name;
    int root_body_index = -1;
    int root_articulation_index = -1;
    std::vector<std::string> link_names;
    std::vector<std::string> joint_names;
    std::vector<std::string> topological_link_order;
    std::vector<std::string> articulation_link_names;
    std::vector<std::string> dof_joint_names;
    std::vector<int> dof_qd_indices;
    std::vector<SceneLinkMetadata> links;
    std::vector<SceneJointMetadata> joints;
    std::vector<SceneShapeMetadata> shapes;
    std::vector<std::pair<std::string, std::string>> filtered_link_pairs;
};

struct UsdAnimationTrack {
    std::string property_name;
    std::vector<std::pair<float, Vec3f>> vec3_samples;
    std::vector<std::pair<float, Vec4f>> vec4_samples;
};

struct UsdPrim {
    std::string path;
    std::string name;
    std::string type_name;
    std::string parent_path;
    Transform local_transform = Transform::identity();
    float mass = 0.0f;
    float density = 0.0f;
    std::string material_binding;
    Vec3f box_half_extents = Vec3f::Zero();
    float sphere_radius = 0.0f;
    std::vector<UsdAnimationTrack> tracks;
};

struct UsdStageData {
    std::string default_prim;
    std::string up_axis = "Y";
    float meters_per_unit = 1.0f;
    std::vector<UsdPrim> prims;
};

struct SceneBuildResult {
    Model model;
    Articulation articulation;
    SceneBuildMetadata metadata;
    VecXf initial_q = VecXf::Zero(0);
    VecXf initial_qd = VecXf::Zero(0);
    std::vector<std::string> warnings;
};

struct RecordedKeyframe {
    float time = 0.0f;
    int body_index = -1;
    Vec3f position = Vec3f::Zero();
    Vec4f rotation = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    Vec3f linear_velocity = Vec3f::Zero();
    Vec3f angular_velocity = Vec3f::Zero();
};

struct RecordedCollisionEvent {
    float time = 0.0f;
    int body_a = -1;
    int body_b = -1;
    Vec3f position = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    float penetration = 0.0f;
};

struct RecordedConstraintReaction {
    float time = 0.0f;
    std::string joint_name;
    VecXf wrench = VecXf::Zero(6);
};

}  // namespace novaphy
