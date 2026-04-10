#pragma once

#include "novaphy/core/shape.h"
#include "novaphy/math/math_types.h"

#include "novaphy/novaphy.h"

#include <vector>

namespace novaphy {

/**
 * @brief Vertices and tetrahedra for a tetrahedral mesh.
 *
 * Used as intermediate representation when converting NovaPhy analytic
 * shapes to libuipc SimplicialComplex geometry.
 * Coordinates are in double precision for libuipc compatibility.
 */
struct TetMeshData {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector4i> tetrahedra;
};

/**
 * @brief Convert a box (given by half-extents) to a tetrahedral mesh.
 *
 * Produces 8 corner vertices and 5 tetrahedra filling the box volume.
 *
 * @param half_extents Box half-extents in meters.
 * @return Tet mesh representation.
 */
NOVAPHY_API TetMeshData box_to_tetmesh(const Vec3f& half_extents);

/**
 * @brief Convert a sphere to a tetrahedral mesh via icosphere subdivision.
 *
 * Creates an icosphere surface mesh and tetrahedralizes from the center.
 *
 * @param radius Sphere radius in meters.
 * @param subdivisions Number of icosphere subdivisions (0 = icosahedron).
 * @return Tet mesh representation.
 */
NOVAPHY_API TetMeshData sphere_to_tetmesh(float radius, int subdivisions = 1);

/**
 * @brief Convert a cylinder to a tetrahedral mesh.
 *
 * Creates a prism-like mesh with top/bottom disk fans and lateral tets.
 *
 * @param radius    Cylinder radius in meters.
 * @param half_length Half-length along the local Z axis.
 * @param n_segments Number of radial segments around the circumference.
 * @return Tet mesh representation.
 */
TetMeshData cylinder_to_tetmesh(float radius, float half_length, int n_segments = 16);

}  // namespace novaphy
