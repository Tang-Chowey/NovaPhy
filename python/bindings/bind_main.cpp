#include <pybind11/pybind11.h>

#include "novaphy/novaphy.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "NovaPhy: A 3D physics engine for embodied intelligence";

    m.def("version", &novaphy::version, "Returns the NovaPhy version string");
}
