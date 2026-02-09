#pragma once

#include <vector>

#include "novaphy/core/contact.h"
#include "novaphy/core/shape.h"

namespace novaphy {
namespace narrowphase {

/// Sphere vs Sphere collision
bool collide_sphere_sphere(const CollisionShape& a, const Transform& ta,
                           const CollisionShape& b, const Transform& tb,
                           std::vector<ContactPoint>& contacts);

/// Sphere vs Plane collision
bool collide_sphere_plane(const CollisionShape& sphere, const Transform& ts,
                          const CollisionShape& plane,
                          std::vector<ContactPoint>& contacts);

/// Box vs Sphere collision
bool collide_box_sphere(const CollisionShape& box, const Transform& tb,
                        const CollisionShape& sphere, const Transform& ts,
                        std::vector<ContactPoint>& contacts);

/// Box vs Plane collision
bool collide_box_plane(const CollisionShape& box, const Transform& tb,
                       const CollisionShape& plane,
                       std::vector<ContactPoint>& contacts);

/// Box vs Box collision (SAT-based)
bool collide_box_box(const CollisionShape& a, const Transform& ta,
                     const CollisionShape& b, const Transform& tb,
                     std::vector<ContactPoint>& contacts);

}  // namespace narrowphase

/// Collision dispatcher: calls the appropriate narrowphase function
/// based on shape types.
bool collide_shapes(const CollisionShape& a, const Transform& ta,
                    const CollisionShape& b, const Transform& tb,
                    std::vector<ContactPoint>& contacts);

}  // namespace novaphy
