#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "novaphy/collision/broadphase.h"
#include "novaphy/collision/narrowphase.h"

namespace py = pybind11;
using namespace novaphy;

void bind_collision(py::module_& m) {
    // --- BroadPhasePair ---
    py::class_<BroadPhasePair>(m, "BroadPhasePair")
        .def(py::init<>())
        .def_readwrite("body_a", &BroadPhasePair::body_a)
        .def_readwrite("body_b", &BroadPhasePair::body_b);

    // --- SweepAndPrune ---
    py::class_<SweepAndPrune>(m, "SweepAndPrune")
        .def(py::init<>())
        .def("update", &SweepAndPrune::update,
             py::arg("body_aabbs"), py::arg("static_mask"))
        .def("get_pairs", &SweepAndPrune::get_pairs);

    // --- Collision dispatcher ---
    // Return (bool, list[ContactPoint]) tuple instead of passing contacts by reference
    m.def("collide_shapes",
          [](const CollisionShape& a, const Transform& ta,
             const CollisionShape& b, const Transform& tb) {
              std::vector<ContactPoint> contacts;
              bool result = collide_shapes(a, ta, b, tb, contacts);
              return std::make_pair(result, contacts);
          },
          py::arg("shape_a"), py::arg("transform_a"),
          py::arg("shape_b"), py::arg("transform_b"),
          "Test collision between two shapes. Returns (hit, contacts).");
}
