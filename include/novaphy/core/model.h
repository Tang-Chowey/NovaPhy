#pragma once

#include <algorithm>
#include <vector>

#include "novaphy/core/articulation.h"
#include "novaphy/core/body.h"
#include "novaphy/core/shape.h"
#include "novaphy/core/site.h"
#include "novaphy/fluid/particle_state.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

/**
 * @brief Disabled collision pair in shape-index space.
 *
 * @details Shape indices are stored in canonical ascending order so the pair can
 * be compared directly or looked up with a simple linear scan.
 */
struct CollisionFilterPair {
    int shape_a = -1;  /**< First shape index in canonical ascending order. */
    int shape_b = -1;  /**< Second shape index in canonical ascending order. */
};

/**
 * @brief Immutable free-body scene model.
 *
 * @details Describes rigid bodies, their initial world transforms, collision
 * shapes, and optional disabled collision pairs. Instances are typically
 * created with `ModelBuilder` or the URDF/USD scene builders.
 */
struct Model {
    std::vector<RigidBody> bodies;                /**< Rigid-body inertial properties. */
    std::vector<Transform> initial_transforms;    /**< Initial body transforms in world coordinates. */
    std::vector<CollisionShape> shapes;           /**< Collision shapes attached to bodies/world. */
    std::vector<Articulation> articulations;      /**< Tree-structured articulated robots/mechanisms. */
    std::vector<FluidBlockDef> fluid_blocks;      /**< Definitions for generating fluid particles. */
    std::vector<FluidMaterial> fluid_materials;   /**< Optional multi-material table (indexed by block.material_index). */
    std::vector<CollisionFilterPair> collision_filter_pairs;  /**< Disabled shape-pair list. */
    std::vector<Site> sites;  /**< Named reference frames attached to bodies/links. */
    Vec3f gravity = Vec3f(0.0f, -9.81f, 0.0f);   /**< World gravity vector (m/s^2). */

    /**
     * @brief Get number of bodies in the model.
     *
     * @return Body count.
     */
    int num_bodies() const { return static_cast<int>(bodies.size()); }

    /**
     * @brief Get number of collision shapes in the model.
     *
     * @return Shape count.
     */
    int num_shapes() const { return static_cast<int>(shapes.size()); }

    int num_sites() const { return static_cast<int>(sites.size()); }

    /**
     * @brief Check whether a shape pair is disabled by the model filter.
     *
     * @param [in] shape_a First shape index.
     * @param [in] shape_b Second shape index.
     * @return True if the pair should be skipped before narrowphase.
     */
    bool is_collision_pair_filtered(int shape_a, int shape_b) const {
        if (shape_a == shape_b) return true;
        const int a = std::min(shape_a, shape_b);
        const int b = std::max(shape_a, shape_b);
        for (const CollisionFilterPair& pair : collision_filter_pairs) {
            if (pair.shape_a == a && pair.shape_b == b) {
                return true;
            }
        }
        return false;
    }
};

}  // namespace novaphy
