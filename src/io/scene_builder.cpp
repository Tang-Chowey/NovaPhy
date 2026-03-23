#include "novaphy/io/scene_builder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "novaphy/core/model_builder.h"

namespace novaphy {
namespace {

JointType urdf_joint_type(const std::string& t) {
    if (t == "revolute" || t == "continuous") return JointType::Revolute;
    if (t == "prismatic") return JointType::Slide;
    if (t == "floating") return JointType::Free;
    if (t == "spherical") return JointType::Ball;
    return JointType::Fixed;
}

std::string geometry_type_name(UrdfGeometryType type) {
    switch (type) {
        case UrdfGeometryType::Box: return "box";
        case UrdfGeometryType::Sphere: return "sphere";
        case UrdfGeometryType::Cylinder: return "cylinder";
        case UrdfGeometryType::Mesh: return "mesh";
        case UrdfGeometryType::Unknown: return "unknown";
    }
    return "unknown";
}

Mat3f rotated_inertia(const Mat3f& inertia, const Quatf& rotation) {
    const Mat3f R = rotation.normalized().toRotationMatrix();
    return R * inertia * R.transpose();
}

Mat3f parallel_axis_inertia(float mass, const Vec3f& offset) {
    return mass * ((offset.squaredNorm() * Mat3f::Identity()) - (offset * offset.transpose()));
}

RigidBody transform_rigid_body(const RigidBody& body, const Transform& transform) {
    if (body.mass <= 0.0f) return RigidBody::make_static();

    RigidBody transformed = body;
    transformed.com = transform.transform_point(body.com);
    transformed.inertia = rotated_inertia(body.inertia, transform.rotation);
    return transformed;
}

RigidBody combine_rigid_bodies(const std::vector<RigidBody>& bodies) {
    float total_mass = 0.0f;
    Vec3f weighted_com = Vec3f::Zero();
    for (const RigidBody& body : bodies) {
        if (body.mass <= 0.0f) continue;
        total_mass += body.mass;
        weighted_com += body.mass * body.com;
    }

    if (total_mass <= 0.0f) {
        return RigidBody::make_static();
    }

    RigidBody combined;
    combined.mass = total_mass;
    combined.com = weighted_com / total_mass;
    combined.inertia = Mat3f::Zero();
    for (const RigidBody& body : bodies) {
        if (body.mass <= 0.0f) continue;
        const Vec3f offset = body.com - combined.com;
        combined.inertia += body.inertia + parallel_axis_inertia(body.mass, offset);
    }
    return combined;
}

RigidBody urdf_link_to_body(const UrdfLink& link) {
    if (link.inertial.mass <= 0.0f) {
        return RigidBody::make_static();
    }

    RigidBody body;
    body.mass = link.inertial.mass;
    body.com = link.inertial.origin.position;
    body.inertia = rotated_inertia(link.inertial.inertia, link.inertial.origin.rotation);
    return body;
}

RigidBody rigid_body_from_geometry(const UrdfGeometry& geometry,
                                   float mass,
                                   std::vector<std::string>& warnings) {
    if (mass <= 0.0f) {
        return RigidBody::make_static();
    }

    if (geometry.type == UrdfGeometryType::Box) {
        return RigidBody::from_box(mass, geometry.size * 0.5f);
    }
    if (geometry.type == UrdfGeometryType::Sphere) {
        return RigidBody::from_sphere(mass, geometry.radius);
    }
    if (geometry.type == UrdfGeometryType::Cylinder) {
        return RigidBody::from_cylinder(mass, geometry.radius, geometry.length);
    }
    if (geometry.type == UrdfGeometryType::Mesh) {
        warnings.push_back("Mesh inertia approximated as box from mesh scale.");
        return RigidBody::from_box(mass, geometry.mesh_scale.cwiseAbs() * 0.5f);
    }

    warnings.push_back("Unsupported URDF geometry used for inertia; falling back to static body.");
    return RigidBody::make_static();
}

RigidBody geometry_based_link_body(const UrdfLink& link,
                                   std::vector<std::string>& warnings) {
    if (link.inertial.mass <= 0.0f) {
        return RigidBody::make_static();
    }

    if (link.collisions.empty()) {
        warnings.push_back("ignore_inertial_definitions requested but link has no collision geometry; using URDF inertial instead.");
        return urdf_link_to_body(link);
    }

    const float part_mass = link.inertial.mass / static_cast<float>(link.collisions.size());
    std::vector<RigidBody> parts;
    parts.reserve(link.collisions.size());
    for (const UrdfCollision& collision : link.collisions) {
        RigidBody part = rigid_body_from_geometry(collision.geometry, part_mass, warnings);
        if (part.mass <= 0.0f) continue;
        parts.push_back(transform_rigid_body(part, collision.origin));
    }

    if (parts.empty()) {
        warnings.push_back("Geometry-based inertia generation produced no valid shapes; using URDF inertial instead.");
        return urdf_link_to_body(link);
    }

    return combine_rigid_bodies(parts);
}

int add_collision_shape_from_urdf(const UrdfCollision& collision,
                                  int body_idx,
                                  const Transform& body_to_link,
                                  ModelBuilder& builder,
                                  std::vector<std::string>& warnings) {
    const Transform local_transform = body_to_link * collision.origin;

    if (collision.geometry.type == UrdfGeometryType::Box) {
        const Vec3f half = collision.geometry.size * 0.5f;
        return builder.add_shape(CollisionShape::make_box(
            half, body_idx, local_transform, collision.friction, collision.restitution));
    }
    if (collision.geometry.type == UrdfGeometryType::Sphere) {
        return builder.add_shape(CollisionShape::make_sphere(
            collision.geometry.radius, body_idx, local_transform,
            collision.friction, collision.restitution));
    }
    if (collision.geometry.type == UrdfGeometryType::Cylinder) {
        const float r = collision.geometry.radius;
        const float half_l = collision.geometry.length * 0.5f;
        const int shape_idx = builder.add_shape(CollisionShape::make_box(
            Vec3f(r, r, half_l), body_idx, local_transform,
            collision.friction, collision.restitution));
        warnings.push_back("Cylinder collision approximated as box.");
        return shape_idx;
    }
    if (collision.geometry.type == UrdfGeometryType::Mesh) {
        const Vec3f extent = collision.geometry.mesh_scale.cwiseAbs() * 0.5f;
        const int shape_idx = builder.add_shape(CollisionShape::make_box(
            extent, body_idx, local_transform, collision.friction, collision.restitution));
        warnings.push_back("Mesh collision approximated as box.");
        return shape_idx;
    }
    warnings.push_back("Unsupported URDF collision geometry skipped.");
    return -1;
}

}  // namespace

std::vector<const UrdfLink*> topologically_order_links(
    const UrdfModelData& urdf_model,
    std::vector<std::string>& warnings) {
    std::unordered_map<std::string, const UrdfLink*> link_by_name;
    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<std::string>> children;
    std::vector<std::string> declaration_order;
    declaration_order.reserve(urdf_model.links.size());

    for (const UrdfLink& link : urdf_model.links) {
        if (link_by_name.find(link.name) != link_by_name.end()) {
            warnings.push_back("Duplicate URDF link name encountered; using first declaration.");
            continue;
        }
        link_by_name[link.name] = &link;
        indegree[link.name] = 0;
        declaration_order.push_back(link.name);
    }

    for (const UrdfJoint& joint : urdf_model.joints) {
        auto child_it = link_by_name.find(joint.child_link);
        if (child_it == link_by_name.end()) {
            warnings.push_back("URDF joint child link not found in links; joint ignored for ordering.");
            continue;
        }
        if (joint.parent_link.empty()) {
            continue;
        }
        auto parent_it = link_by_name.find(joint.parent_link);
        if (parent_it == link_by_name.end()) {
            warnings.push_back("URDF joint parent link not found in links; joint treated as root attachment.");
            continue;
        }
        children[joint.parent_link].push_back(joint.child_link);
        ++indegree[joint.child_link];
    }

    std::queue<std::string> ready;
    for (const std::string& name : declaration_order) {
        if (indegree[name] == 0) ready.push(name);
    }

    std::vector<const UrdfLink*> ordered_links;
    ordered_links.reserve(declaration_order.size());
    while (!ready.empty()) {
        std::string parent = ready.front();
        ready.pop();
        ordered_links.push_back(link_by_name[parent]);
        auto child_it = children.find(parent);
        if (child_it == children.end()) continue;
        for (const std::string& child : child_it->second) {
            int& degree = indegree[child];
            --degree;
            if (degree == 0) ready.push(child);
        }
    }

    if (ordered_links.size() != declaration_order.size()) {
        warnings.push_back("URDF link graph is cyclic or disconnected; appending unresolved links in declaration order.");
        std::unordered_map<std::string, bool> emitted;
        emitted.reserve(ordered_links.size());
        for (const UrdfLink* link : ordered_links) emitted[link->name] = true;
        for (const std::string& name : declaration_order) {
            if (emitted.find(name) == emitted.end()) {
                ordered_links.push_back(link_by_name[name]);
            }
        }
    }

    return ordered_links;
}

SceneBuildResult SceneBuilderEngine::build_from_urdf(const UrdfModelData& urdf_model,
                                                     const UrdfImportOptions& options) const {
    SceneBuildResult result;
    result.metadata.import_options = options;
    ModelBuilder builder;

    std::unordered_map<std::string, const UrdfLink*> link_by_name;
    std::unordered_map<std::string, int> urdf_link_index_by_name;
    for (size_t i = 0; i < urdf_model.links.size(); ++i) {
        const UrdfLink& link = urdf_model.links[i];
        link_by_name[link.name] = &link;
        urdf_link_index_by_name[link.name] = static_cast<int>(i);
        result.metadata.link_names.push_back(link.name);
    }
    for (const UrdfJoint& joint : urdf_model.joints) {
        result.metadata.joint_names.push_back(joint.name);
    }

    const std::vector<const UrdfLink*> ordered_links =
        topologically_order_links(urdf_model, result.warnings);
    for (const UrdfLink* link : ordered_links) {
        result.metadata.topological_link_order.push_back(link->name);
    }

    std::unordered_map<std::string, const UrdfJoint*> child_joint;
    for (const UrdfJoint& joint : urdf_model.joints) {
        child_joint[joint.child_link] = &joint;
    }

    std::unordered_map<std::string, Transform> world_transform_cache;
    std::function<Transform(const std::string&)> world_transform_of =
        [&](const std::string& link_name) -> Transform {
            auto cache_it = world_transform_cache.find(link_name);
            if (cache_it != world_transform_cache.end()) {
                return cache_it->second;
            }

            Transform world = Transform::identity();
            auto joint_it = child_joint.find(link_name);
            if (joint_it != child_joint.end() && !joint_it->second->parent_link.empty()) {
                auto parent_it = link_by_name.find(joint_it->second->parent_link);
                if (parent_it != link_by_name.end()) {
                    world = world_transform_of(joint_it->second->parent_link) * joint_it->second->origin;
                }
            }
            world_transform_cache[link_name] = world;
            return world;
        };

    std::unordered_map<std::string, std::string> body_link_name_by_link;
    std::function<std::string(const std::string&)> resolve_body_link =
        [&](const std::string& link_name) -> std::string {
            auto cache_it = body_link_name_by_link.find(link_name);
            if (cache_it != body_link_name_by_link.end()) {
                return cache_it->second;
            }

            if (!options.collapse_fixed_joints) {
                body_link_name_by_link[link_name] = link_name;
                return link_name;
            }

            std::string current = link_name;
            std::unordered_set<std::string> visited;
            while (true) {
                auto joint_it = child_joint.find(current);
                if (joint_it == child_joint.end()) break;
                const UrdfJoint* joint = joint_it->second;
                if (joint->type != "fixed") break;
                if (joint->parent_link.empty()) break;
                if (!visited.insert(current).second) break;
                current = joint->parent_link;
            }

            body_link_name_by_link[link_name] = current;
            return current;
        };

    std::string root_link_name;
    for (const UrdfLink* link : ordered_links) {
        if (child_joint.find(link->name) == child_joint.end()) {
            root_link_name = link->name;
            break;
        }
    }
    result.metadata.root_link_name = root_link_name;

    std::unordered_map<std::string, RigidBody> link_body_by_name;
    for (const UrdfLink& link : urdf_model.links) {
        if (options.ignore_inertial_definitions) {
            link_body_by_name[link.name] = geometry_based_link_body(link, result.warnings);
        } else {
            link_body_by_name[link.name] = urdf_link_to_body(link);
        }
    }

    std::unordered_map<std::string, std::vector<std::string>> cluster_members;
    for (const UrdfLink& link : urdf_model.links) {
        cluster_members[resolve_body_link(link.name)].push_back(link.name);
    }

    std::vector<std::string> articulation_link_names;
    articulation_link_names.reserve(ordered_links.size());
    std::unordered_set<std::string> emitted_body_links;
    for (const UrdfLink* link : ordered_links) {
        const std::string body_link_name = resolve_body_link(link->name);
        if (body_link_name != link->name) continue;
        if (!emitted_body_links.insert(body_link_name).second) continue;
        articulation_link_names.push_back(body_link_name);
    }
    result.metadata.articulation_link_names = articulation_link_names;

    std::unordered_map<std::string, int> body_index_by_body_link;
    std::unordered_map<std::string, RigidBody> combined_body_by_body_link;
    for (const std::string& body_link_name : articulation_link_names) {
        std::vector<RigidBody> cluster_bodies;
        const Transform world_body = world_transform_of(body_link_name);
        for (const std::string& member_link_name : cluster_members[body_link_name]) {
            const Transform body_to_link = world_body.inverse() * world_transform_of(member_link_name);
            cluster_bodies.push_back(transform_rigid_body(link_body_by_name[member_link_name], body_to_link));
        }
        RigidBody combined_body = combine_rigid_bodies(cluster_bodies);
        const Transform initial_transform = options.root_transform * world_body;
        const int body_index = builder.add_body(combined_body, initial_transform);
        body_index_by_body_link[body_link_name] = body_index;
        combined_body_by_body_link[body_link_name] = combined_body;
    }

    for (const UrdfLink* link : ordered_links) {
        const std::string body_link_name = resolve_body_link(link->name);
        const auto body_it = body_index_by_body_link.find(body_link_name);
        if (body_it == body_index_by_body_link.end()) continue;

        const Transform body_to_link =
            world_transform_of(body_link_name).inverse() * world_transform_of(link->name);
        for (const UrdfCollision& collision : link->collisions) {
            const int shape_idx = add_collision_shape_from_urdf(
                collision, body_it->second, body_to_link, builder, result.warnings);
            if (shape_idx < 0) continue;

            SceneShapeMetadata shape_meta;
            shape_meta.shape_index = shape_idx;
            shape_meta.body_index = body_it->second;
            shape_meta.articulation_index = body_it->second;
            shape_meta.urdf_link_index = urdf_link_index_by_name[link->name];
            shape_meta.link_name = link->name;
            shape_meta.body_link_name = body_link_name;
            shape_meta.geometry_type = geometry_type_name(collision.geometry.type);
            shape_meta.source = "collision";
            result.metadata.shapes.push_back(shape_meta);
        }

        if (link->collisions.empty() && !link->visuals.empty()) {
            if (options.use_visual_collision_fallback) {
                const UrdfVisual& visual = link->visuals.front();
                UrdfCollision proxy;
                proxy.origin = visual.origin;
                proxy.geometry = visual.geometry;
                const int shape_idx = add_collision_shape_from_urdf(
                    proxy, body_it->second, body_to_link, builder, result.warnings);
                if (shape_idx >= 0) {
                    SceneShapeMetadata shape_meta;
                    shape_meta.shape_index = shape_idx;
                    shape_meta.body_index = body_it->second;
                    shape_meta.articulation_index = body_it->second;
                    shape_meta.urdf_link_index = urdf_link_index_by_name[link->name];
                    shape_meta.link_name = link->name;
                    shape_meta.body_link_name = body_link_name;
                    shape_meta.geometry_type = geometry_type_name(proxy.geometry.type);
                    shape_meta.source = "visual_fallback";
                    result.metadata.shapes.push_back(shape_meta);
                }
                result.warnings.push_back("Visual geometry used as fallback collision.");
            } else {
                result.warnings.push_back("Visual geometry present but fallback collision is disabled.");
            }
        }
    }

    result.model = builder.build();

    result.articulation.joints.reserve(articulation_link_names.size());
    result.articulation.bodies.reserve(articulation_link_names.size());
    for (const std::string& body_link_name : articulation_link_names) {
        Joint joint;
        auto incoming_joint_it = child_joint.find(body_link_name);
        if (incoming_joint_it != child_joint.end()) {
            const UrdfJoint& urdf_joint = *incoming_joint_it->second;
            joint.type = urdf_joint_type(urdf_joint.type);
            if (urdf_joint.axis.norm() > 1.0e-6f) {
                joint.axis = urdf_joint.axis.normalized();
            }
            const bool has_position_limits =
                (std::abs(urdf_joint.lower_limit) > 1.0e-6f) ||
                (std::abs(urdf_joint.upper_limit) > 1.0e-6f) ||
                (urdf_joint.lower_limit < urdf_joint.upper_limit);
            if ((joint.type == JointType::Revolute || joint.type == JointType::Slide) &&
                has_position_limits) {
                joint.limit_enabled = true;
                joint.lower_limit = urdf_joint.lower_limit;
                joint.upper_limit = urdf_joint.upper_limit;
            }
            const std::string parent_body_link = resolve_body_link(urdf_joint.parent_link);
            auto parent_body_it = body_index_by_body_link.find(parent_body_link);
            joint.parent = parent_body_it != body_index_by_body_link.end() ? parent_body_it->second : -1;
            const Transform parent_body_to_parent_link =
                world_transform_of(parent_body_link).inverse() * world_transform_of(urdf_joint.parent_link);
            joint.parent_to_joint = parent_body_to_parent_link * urdf_joint.origin;
        } else {
            joint.type = options.floating_base ? JointType::Free : JointType::Fixed;
            joint.parent = -1;
            joint.parent_to_joint = Transform::identity();
        }
        result.articulation.joints.push_back(joint);
        result.articulation.bodies.push_back(combined_body_by_body_link[body_link_name]);
    }
    result.articulation.build_spatial_inertias();

    if (!root_link_name.empty()) {
        const std::string root_body_link = resolve_body_link(root_link_name);
        auto root_body_it = body_index_by_body_link.find(root_body_link);
        if (root_body_it != body_index_by_body_link.end()) {
            result.metadata.root_body_index = root_body_it->second;
            result.metadata.root_articulation_index = root_body_it->second;
        }
    }

    result.metadata.links.reserve(urdf_model.links.size());
    for (size_t i = 0; i < urdf_model.links.size(); ++i) {
        const UrdfLink& link = urdf_model.links[i];
        SceneLinkMetadata meta;
        meta.link_name = link.name;
        meta.urdf_link_index = static_cast<int>(i);
        meta.body_link_name = resolve_body_link(link.name);
        auto body_it = body_index_by_body_link.find(meta.body_link_name);
        if (body_it != body_index_by_body_link.end()) {
            meta.body_index = body_it->second;
            meta.articulation_index = body_it->second;
        }
        auto joint_it = child_joint.find(link.name);
        if (joint_it != child_joint.end()) {
            meta.parent_link_name = joint_it->second->parent_link;
            meta.joint_name = joint_it->second->name;
        }
        meta.collapsed = options.collapse_fixed_joints && meta.body_link_name != link.name;
        result.metadata.links.push_back(meta);
    }

    result.metadata.joints.reserve(urdf_model.joints.size());
    for (size_t i = 0; i < urdf_model.joints.size(); ++i) {
        const UrdfJoint& joint = urdf_model.joints[i];
        SceneJointMetadata meta;
        meta.joint_name = joint.name;
        meta.parent_link_name = joint.parent_link;
        meta.child_link_name = joint.child_link;
        meta.joint_type = joint.type;
        meta.urdf_joint_index = static_cast<int>(i);
        meta.lower_limit = joint.lower_limit;
        meta.upper_limit = joint.upper_limit;
        meta.effort_limit = joint.effort_limit;
        meta.velocity_limit = joint.velocity_limit;
        meta.damping = joint.damping;
        meta.friction = joint.friction;
        meta.collapsed = options.collapse_fixed_joints && joint.type == "fixed";

        if (!meta.collapsed) {
            const std::string body_link_name = resolve_body_link(joint.child_link);
            auto art_it = body_index_by_body_link.find(body_link_name);
            if (art_it != body_index_by_body_link.end()) {
                meta.articulation_index = art_it->second;
                meta.q_start = result.articulation.q_start(meta.articulation_index);
                meta.qd_start = result.articulation.qd_start(meta.articulation_index);
                meta.num_q = result.articulation.joints[meta.articulation_index].num_q();
                meta.num_qd = result.articulation.joints[meta.articulation_index].num_qd();
            }
        }

        result.metadata.joints.push_back(meta);
    }

    result.initial_q = VecXf::Zero(result.articulation.total_q());
    result.initial_qd = VecXf::Zero(result.articulation.total_qd());
    for (int art_idx = 0; art_idx < result.articulation.num_links(); ++art_idx) {
        const Joint& joint = result.articulation.joints[art_idx];
        const int qi = result.articulation.q_start(art_idx);
        const std::string& body_link_name = articulation_link_names[art_idx];

        if (joint.parent < 0 && joint.type == JointType::Free) {
            const Transform root_pose = options.root_transform * world_transform_of(body_link_name);
            result.initial_q(qi + 0) = root_pose.position.x();
            result.initial_q(qi + 1) = root_pose.position.y();
            result.initial_q(qi + 2) = root_pose.position.z();
            result.initial_q(qi + 3) = root_pose.rotation.x();
            result.initial_q(qi + 4) = root_pose.rotation.y();
            result.initial_q(qi + 5) = root_pose.rotation.z();
            result.initial_q(qi + 6) = root_pose.rotation.w();
        } else if (joint.type == JointType::Ball) {
            result.initial_q(qi + 0) = 0.0f;
            result.initial_q(qi + 1) = 0.0f;
            result.initial_q(qi + 2) = 0.0f;
            result.initial_q(qi + 3) = 1.0f;
        }
    }

    result.metadata.dof_joint_names.reserve(result.articulation.total_qd());
    result.metadata.dof_qd_indices.reserve(result.articulation.total_qd());
    for (int art_idx = 0; art_idx < result.articulation.num_links(); ++art_idx) {
        const int num_qd = result.articulation.joints[art_idx].num_qd();
        std::string dof_joint_name = "__root__";
        auto joint_it = child_joint.find(articulation_link_names[art_idx]);
        if (joint_it != child_joint.end()) {
            dof_joint_name = joint_it->second->name;
        }
        for (int local_dof = 0; local_dof < num_qd; ++local_dof) {
            result.metadata.dof_qd_indices.push_back(
                static_cast<int>(result.metadata.dof_qd_indices.size()));
            result.metadata.dof_joint_names.push_back(dof_joint_name);
        }
    }

    std::unordered_map<std::string, std::string> parent_body_link_by_body_link;
    for (size_t art_idx = 0; art_idx < articulation_link_names.size(); ++art_idx) {
        const int parent_index = result.articulation.joints[art_idx].parent;
        if (parent_index >= 0) {
            parent_body_link_by_body_link[articulation_link_names[art_idx]] =
                articulation_link_names[static_cast<size_t>(parent_index)];
        }
    }

    std::unordered_set<unsigned long long> filtered_shape_pair_keys;
    std::unordered_set<std::string> filtered_link_pair_keys;
    auto add_filtered_pair =
        [&](int shape_a, int shape_b, const std::string& link_a, const std::string& link_b) {
            if (shape_a == shape_b) return;
            const int a = std::min(shape_a, shape_b);
            const int b = std::max(shape_a, shape_b);
            const unsigned long long shape_key =
                (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32) |
                static_cast<unsigned long long>(static_cast<unsigned int>(b));
            if (filtered_shape_pair_keys.insert(shape_key).second) {
                result.model.collision_filter_pairs.push_back({a, b});
            }

            const std::pair<std::string, std::string> link_pair =
                (link_a < link_b) ? std::make_pair(link_a, link_b) : std::make_pair(link_b, link_a);
            const std::string link_key = link_pair.first + "\n" + link_pair.second;
            if (filtered_link_pair_keys.insert(link_key).second) {
                result.metadata.filtered_link_pairs.push_back(link_pair);
            }
        };

    for (size_t i = 0; i < result.metadata.shapes.size(); ++i) {
        const SceneShapeMetadata& shape_a = result.metadata.shapes[i];
        for (size_t j = i + 1; j < result.metadata.shapes.size(); ++j) {
            const SceneShapeMetadata& shape_b = result.metadata.shapes[j];
            bool filter_pair = false;
            if (shape_a.body_index == shape_b.body_index) {
                filter_pair = true;
            } else if (!options.enable_self_collisions) {
                filter_pair = true;
            } else {
                auto parent_a_it = parent_body_link_by_body_link.find(shape_a.body_link_name);
                auto parent_b_it = parent_body_link_by_body_link.find(shape_b.body_link_name);
                const bool a_is_parent = parent_b_it != parent_body_link_by_body_link.end() &&
                                         parent_b_it->second == shape_a.body_link_name;
                const bool b_is_parent = parent_a_it != parent_body_link_by_body_link.end() &&
                                         parent_a_it->second == shape_b.body_link_name;
                filter_pair = a_is_parent || b_is_parent;
            }

            if (filter_pair) {
                add_filtered_pair(shape_a.shape_index, shape_b.shape_index,
                                  shape_a.link_name, shape_b.link_name);
            }
        }
    }

    return result;
}

SceneBuildResult SceneBuilderEngine::build_from_openusd(const UsdStageData& stage) const {
    SceneBuildResult result;
    ModelBuilder builder;
    std::unordered_map<std::string, int> prim_to_body;

    for (const UsdPrim& prim : stage.prims) {
        if (prim.mass <= 0.0f && prim.box_half_extents.isZero(0.0f) && prim.sphere_radius <= 0.0f) {
            continue;
        }
        RigidBody body;
        if (prim.mass > 0.0f) {
            body.mass = prim.mass;
            const Vec3f half = prim.box_half_extents.isZero(0.0f)
                                  ? Vec3f(0.25f, 0.25f, 0.25f)
                                  : prim.box_half_extents;
            body = RigidBody::from_box(prim.mass, half);
        } else {
            body = RigidBody::make_static();
        }
        const int body_idx = builder.add_body(body, prim.local_transform);
        prim_to_body[prim.path] = body_idx;

        if (!prim.box_half_extents.isZero(0.0f)) {
            builder.add_shape(CollisionShape::make_box(prim.box_half_extents, body_idx));
        } else if (prim.sphere_radius > 0.0f) {
            builder.add_shape(CollisionShape::make_sphere(prim.sphere_radius, body_idx));
        } else {
            builder.add_shape(CollisionShape::make_box(Vec3f(0.25f, 0.25f, 0.25f), body_idx));
            result.warnings.push_back("USD prim missing collider size, default box inserted.");
        }
    }

    result.model = builder.build();

    for (const UsdPrim& prim : stage.prims) {
        if (prim.type_name.find("Joint") == std::string::npos) continue;
        Joint j;
        if (prim.type_name == "PhysicsRevoluteJoint") j.type = JointType::Revolute;
        else if (prim.type_name == "PhysicsPrismaticJoint") j.type = JointType::Slide;
        else if (prim.type_name == "PhysicsSphericalJoint") j.type = JointType::Ball;
        else j.type = JointType::Fixed;
        j.parent = -1;
        j.parent_to_joint = prim.local_transform;
        result.articulation.joints.push_back(j);
        result.articulation.bodies.push_back(RigidBody::make_static());
    }

    if (result.articulation.joints.empty()) {
        result.articulation.joints.resize(result.model.num_bodies(), Joint{JointType::Free});
        result.articulation.bodies = result.model.bodies;
    }
    if (result.articulation.bodies.size() != result.articulation.joints.size()) {
        result.articulation.bodies.resize(result.articulation.joints.size(), RigidBody::make_static());
    }
    result.articulation.build_spatial_inertias();
    return result;
}

}  // namespace novaphy
