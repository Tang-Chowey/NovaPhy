# NovaPhy

A 3D physics engine for embodied intelligence applications.

## Features

- **Rigid Body Dynamics** with contact/collision (Sequential Impulse solver)
- **Articulated Body Dynamics** (Featherstone CRBA in reduced coordinates)
- **Collision Detection**: Sweep-and-Prune broadphase + Box/Sphere/Plane narrowphase
- **Joint Types**: Revolute (hinge), Free, Fixed
- **Python API** via pybind11 with Polyscope visualization
- **pip-installable** C++17 core via scikit-build-core

## Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- [vcpkg](https://vcpkg.io/) with `VCPKG_ROOT` environment variable set
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate novaphy

# Install vcpkg dependencies (automatic via manifest)
# Ensure VCPKG_ROOT is set to your vcpkg installation

# Install NovaPhy in development mode
pip install -e ".[dev]"
```

### Run a Demo

```bash
python demos/demo_stack.py
```

### Verify Installation

```python
import novaphy
print(novaphy.version())  # 0.1.0
```

## Architecture

```
User (Python) -> ModelBuilder -> Model -> World -> step(dt)
                                           |-> Free bodies:  SAP -> Narrowphase -> Sequential Impulse
                                           |-> Articulated:  FK -> RNEA -> CRBA -> Cholesky -> Integrate
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core | C++17 (float32) |
| Math | Eigen3 |
| Bindings | pybind11 |
| Build | CMake + scikit-build-core |
| C++ Deps | vcpkg |
| Visualization | Polyscope |

## License

MIT
