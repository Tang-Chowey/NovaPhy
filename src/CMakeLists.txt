# NovaPhy core static library

add_library(novaphy_core STATIC
    math/spatial.cpp
    math/math_utils.cpp
    core/body.cpp
    core/shape.cpp
    collision/broadphase.cpp
    collision/narrowphase.cpp
)

target_include_directories(novaphy_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(novaphy_core PUBLIC
    Eigen3::Eigen
)

target_compile_definitions(novaphy_core PUBLIC
    EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
)
