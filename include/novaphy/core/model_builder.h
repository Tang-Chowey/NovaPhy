#pragma once

#include <string>
#include <vector>

#include "novaphy/core/body.h"
#include "novaphy/core/shape.h"
#include "novaphy/core/site.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

/** @brief Forward declaration of immutable model container. */
struct Model;
struct CollisionFilterPair;

/**
 * @brief Builder utility for constructing immutable simulation models.
 *
 * @details Provides both low-level (add_body + add_shape) and high-level
 * Newton-style convenience methods (add_shape_box, add_shape_sphere, etc.)
 * for constructing scenes. High-level methods create both the body and shape
 * in a single call.
 */
class ModelBuilder {
public:
    /** @brief Construct an empty model builder. */
    ModelBuilder() = default;

    // ---- Low-level API (backward compatible) ----

    /**
     * @brief Add a rigid body with initial world transform.
     *
     * @param [in] body Body inertial properties.
     * @param [in] transform Initial world transform.
     * @return Index of inserted body.
     */
    int add_body(const RigidBody& body, const Transform& transform = Transform::identity());

    /**
     * @brief Add a collision shape.
     *
     * @param [in] shape Collision shape descriptor.
     * @return Index of inserted shape.
     */
    int add_shape(const CollisionShape& shape);

    /**
     * @brief Add infinite horizontal ground plane.
     *
     * @param [in] y Plane offset along +Y world axis in meters.
     * @param [in] friction Friction coefficient.
     * @param [in] restitution Restitution coefficient.
     * @return Index of inserted plane shape.
     */
    int add_ground_plane(float y = 0.0f, float friction = 0.5f, float restitution = 0.0f);

    // ---- Newton-style convenience API ----

    /**
     * @brief Add a box body+shape in one call.
     *
     * @param [in] half_extents Box half extents in meters.
     * @param [in] transform Initial world transform.
     * @param [in] density Mass density in kg/m^3 (auto-computes mass & inertia).
     * @param [in] friction Friction coefficient.
     * @param [in] restitution Restitution coefficient.
     * @param [in] is_static If true, body is static (mass = 0).
     * @return Index of inserted body.
     */
    int add_shape_box(const Vec3f& half_extents,
                      const Transform& transform = Transform::identity(),
                      float density = 1000.0f,
                      float friction = 0.5f,
                      float restitution = 0.3f,
                      bool is_static = false);

    /**
     * @brief Add a sphere body+shape in one call.
     *
     * @param [in] radius Sphere radius in meters.
     * @param [in] transform Initial world transform.
     * @param [in] density Mass density in kg/m^3.
     * @param [in] friction Friction coefficient.
     * @param [in] restitution Restitution coefficient.
     * @param [in] is_static If true, body is static (mass = 0).
     * @return Index of inserted body.
     */
    int add_shape_sphere(float radius,
                         const Transform& transform = Transform::identity(),
                         float density = 1000.0f,
                         float friction = 0.5f,
                         float restitution = 0.3f,
                         bool is_static = false);

    /**
     * @brief Add a cylinder body+shape in one call.
     *
     * @param [in] radius Cylinder radius in meters.
     * @param [in] half_length Half-length along local Z axis in meters.
     * @param [in] transform Initial world transform.
     * @param [in] density Mass density in kg/m^3.
     * @param [in] friction Friction coefficient.
     * @param [in] restitution Restitution coefficient.
     * @param [in] is_static If true, body is static (mass = 0).
     * @return Index of inserted body.
     */
    int add_shape_cylinder(float radius, float half_length,
                           const Transform& transform = Transform::identity(),
                           float density = 1000.0f,
                           float friction = 0.5f,
                           float restitution = 0.3f,
                           bool is_static = false);

    /**
     * @brief Add an additional collision shape attached to an existing body.
     *
     * @param [in] body_index Existing body to attach this shape to.
     * @param [in] shape Shape descriptor (body_index will be overwritten).
     * @return Index of inserted shape.
     */
    int add_shape_to_body(int body_index, const CollisionShape& shape);

    // ---- Collision filtering ----

    /**
     * @brief Disable collision between two shapes.
     *
     * @param [in] shape_a First shape index.
     * @param [in] shape_b Second shape index.
     */
    void add_collision_filter(int shape_a, int shape_b);

    int add_site(const Site& site);
    int add_site(int body_index,
                 const Transform& local_transform = Transform::identity(),
                 const std::string& label = "");
    int add_site_on_link(int articulation_index, int link_index,
                         const Transform& local_transform = Transform::identity(),
                         const std::string& label = "");
    int num_sites() const { return static_cast<int>(sites_.size()); }

    // ---- Scene settings ----

    /**
     * @brief Set the gravity vector for the built model.
     *
     * @param [in] gravity Gravity in world coordinates (m/s^2).
     */
    void set_gravity(const Vec3f& gravity) { gravity_ = gravity; }

    // ---- Build ----

    /**
     * @brief Build immutable `Model` from accumulated entities.
     *
     * @return Immutable scene model.
     */
    Model build() const;

    // ---- Queries ----

    int num_bodies() const { return static_cast<int>(bodies_.size()); }
    int num_shapes() const { return static_cast<int>(shapes_.size()); }

private:
    std::vector<RigidBody> bodies_;
    std::vector<Transform> initial_transforms_;
    std::vector<CollisionShape> shapes_;
    std::vector<CollisionFilterPair> collision_filter_pairs_;
    std::vector<Site> sites_;
    Vec3f gravity_ = Vec3f(0.0f, -9.81f, 0.0f);
};

}  // namespace novaphy
