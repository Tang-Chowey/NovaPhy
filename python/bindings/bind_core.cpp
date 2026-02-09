#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "novaphy/core/aabb.h"
#include "novaphy/core/body.h"
#include "novaphy/core/contact.h"
#include "novaphy/core/shape.h"

namespace py = pybind11;
using namespace novaphy;

void bind_core(py::module_& m) {
    // --- ShapeType enum ---
    py::enum_<ShapeType>(m, "ShapeType")
        .value("Box", ShapeType::Box)
        .value("Sphere", ShapeType::Sphere)
        .value("Plane", ShapeType::Plane);

    // --- RigidBody ---
    py::class_<RigidBody>(m, "RigidBody")
        .def(py::init<>())
        .def_readwrite("mass", &RigidBody::mass)
        .def_readwrite("inertia", &RigidBody::inertia)
        .def_readwrite("com", &RigidBody::com)
        .def("inv_mass", &RigidBody::inv_mass)
        .def("inv_inertia", &RigidBody::inv_inertia)
        .def("is_static", &RigidBody::is_static)
        .def_static("from_box", &RigidBody::from_box,
                     py::arg("mass"), py::arg("half_extents"))
        .def_static("from_sphere", &RigidBody::from_sphere,
                     py::arg("mass"), py::arg("radius"))
        .def_static("make_static", &RigidBody::make_static);

    // --- AABB ---
    py::class_<AABB>(m, "AABB")
        .def(py::init<>())
        .def(py::init<const Vec3f&, const Vec3f&>(), py::arg("min"), py::arg("max"))
        .def_readwrite("min", &AABB::min)
        .def_readwrite("max", &AABB::max)
        .def("overlaps", &AABB::overlaps, py::arg("other"))
        .def("center", &AABB::center)
        .def("half_extents", &AABB::half_extents)
        .def("surface_area", &AABB::surface_area)
        .def("is_valid", &AABB::is_valid)
        .def_static("from_sphere", &AABB::from_sphere,
                     py::arg("center"), py::arg("radius"));

    // --- CollisionShape ---
    py::class_<CollisionShape>(m, "CollisionShape")
        .def(py::init<>())
        .def_readwrite("type", &CollisionShape::type)
        .def_readwrite("local_transform", &CollisionShape::local_transform)
        .def_readwrite("friction", &CollisionShape::friction)
        .def_readwrite("restitution", &CollisionShape::restitution)
        .def_readwrite("body_index", &CollisionShape::body_index)
        .def_property("box_half_extents",
            [](const CollisionShape& s) { return s.box.half_extents; },
            [](CollisionShape& s, const Vec3f& v) { s.box.half_extents = v; })
        .def_property("sphere_radius",
            [](const CollisionShape& s) { return s.sphere.radius; },
            [](CollisionShape& s, float r) { s.sphere.radius = r; })
        .def_property("plane_normal",
            [](const CollisionShape& s) { return s.plane.normal; },
            [](CollisionShape& s, const Vec3f& n) { s.plane.normal = n; })
        .def_property("plane_offset",
            [](const CollisionShape& s) { return s.plane.offset; },
            [](CollisionShape& s, float d) { s.plane.offset = d; })
        .def_static("make_box", &CollisionShape::make_box,
                     py::arg("half_extents"), py::arg("body_idx"),
                     py::arg("local") = Transform::identity(),
                     py::arg("friction") = 0.5f, py::arg("restitution") = 0.3f)
        .def_static("make_sphere", &CollisionShape::make_sphere,
                     py::arg("radius"), py::arg("body_idx"),
                     py::arg("local") = Transform::identity(),
                     py::arg("friction") = 0.5f, py::arg("restitution") = 0.3f)
        .def_static("make_plane", &CollisionShape::make_plane,
                     py::arg("normal"), py::arg("offset"),
                     py::arg("friction") = 0.5f, py::arg("restitution") = 0.0f)
        .def("compute_aabb", &CollisionShape::compute_aabb, py::arg("body_transform"));

    // --- ContactPoint ---
    py::class_<ContactPoint>(m, "ContactPoint")
        .def(py::init<>())
        .def_readwrite("position", &ContactPoint::position)
        .def_readwrite("normal", &ContactPoint::normal)
        .def_readwrite("penetration", &ContactPoint::penetration)
        .def_readwrite("body_a", &ContactPoint::body_a)
        .def_readwrite("body_b", &ContactPoint::body_b)
        .def_readwrite("friction", &ContactPoint::friction)
        .def_readwrite("restitution", &ContactPoint::restitution);
}
