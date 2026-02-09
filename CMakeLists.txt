cmake_minimum_required(VERSION 3.15...4.0)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Eigen3 via vcpkg
find_package(Eigen3 CONFIG REQUIRED)

# Find pybind11 from pip-installed package (not vcpkg)
find_package(pybind11 CONFIG REQUIRED)

# Core C++ library
add_subdirectory(src)

# Python bindings
add_subdirectory(python/bindings)

# Tests (only when building standalone, not via pip install)
if(NOT DEFINED SKBUILD)
    find_package(GTest CONFIG QUIET)
    if(GTest_FOUND)
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()
