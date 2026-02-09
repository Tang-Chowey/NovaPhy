# NovaPhy - Claude Code Rules

## Project Overview

NovaPhy is a C++17/Python 3D physics engine for embodied intelligence (robotics, RL, sim-to-real).
Working directory: `E:\NovaPhy`

## Architecture

- **ModelBuilder** builds scenes -> **Model** (immutable) -> **World** runs simulation
- Two solver pipelines:
  - **Free bodies**: Broadphase(SAP) -> Narrowphase -> Sequential Impulse (PGS)
  - **Articulated bodies**: FK -> RNEA -> CRBA -> Cholesky -> Semi-implicit Euler
- Python visualization via Polyscope

## Code Conventions

### C++
- **float32 only** - never use `double`. Use Eigen `*f` types: `Vec3f`, `Mat3f`, `Quatf`, `VecXf`, `MatXf`
- `using Scalar = float;` defined in `novaphy_types.h`
- Modern C++17: RAII, `std::unique_ptr` for ownership, `std::vector` for collections
- Header files in `include/novaphy/`, source files in `src/`
- Header-only for simple utilities (AABB, math typedefs)
- `.h` for headers, `.cpp` for sources
- `#pragma once` for include guards
- Namespace: `novaphy` for all code
- Spatial algebra convention: **[angular; linear]** (Featherstone convention)

### Python
- Package: `novaphy` (imports from `novaphy._core` C++ extension)
- Visualization: `novaphy.viz` module using Polyscope
- Demos: `demos/` directory, each script is standalone

### Naming
- C++ classes/structs: `PascalCase` (e.g., `RigidBody`, `SweepAndPrune`)
- C++ functions/methods: `snake_case` (e.g., `forward_kinematics`, `compute_aabb`)
- C++ member variables: `snake_case` with trailing underscore for private (e.g., `pairs_`)
- Python: standard PEP 8
- Files: `snake_case.h`, `snake_case.cpp`

## Build System

- **CMake** root at `CMakeLists.txt`
- **vcpkg** for C++ deps (eigen3, pybind11, gtest) - manifest in `vcpkg.json`
- **scikit-build-core** for pip install - config in `pyproject.toml`
- **Conda** for Python env - config in `environment.yml`

### Build Commands
```bash
# Development install
conda activate novaphy
pip install -e ".[dev]"

# C++ tests (standalone build)
cmake --preset default
cmake --build build
cd build && ctest --output-on-failure

# Python tests
pytest tests/python/ -v

# Run a demo
python demos/demo_stack.py
```

## File Organization

| Directory | Purpose |
|-----------|---------|
| `include/novaphy/math/` | Math types, spatial algebra |
| `include/novaphy/core/` | Body, Shape, Joint, Model, AABB, Contact |
| `include/novaphy/collision/` | Broadphase, Narrowphase |
| `include/novaphy/dynamics/` | Integrator, Solvers, Featherstone |
| `include/novaphy/sim/` | World, State, Solver interface |
| `src/` | C++ implementations (mirrors include/) |
| `python/novaphy/` | Python package |
| `python/bindings/` | pybind11 binding files |
| `tests/cpp/` | Google Test C++ tests |
| `tests/python/` | pytest Python tests |
| `demos/` | 10 demo scripts + shared utils |

## Testing Rules

- Every new C++ feature needs a Google Test in `tests/cpp/`
- Every new Python-exposed feature needs a pytest in `tests/python/`
- Test names: `test_<module>.cpp` / `test_<module>.py`
- Use analytical comparisons for physics (free fall, pendulum period)
- Tolerance: 1% for 1000 steps at dt=1/240

## Key Design Patterns

- `ModelBuilder` (mutable) -> `Model` (immutable) -> `World` (simulation)
- Collision: broadphase filters, narrowphase generates contacts, solver resolves
- Featherstone: FK -> bias forces (RNEA) -> mass matrix (CRBA) -> Cholesky solve -> integrate
- Contact solver: accumulated impulse clamping, warm starting, Baumgarte stabilization
