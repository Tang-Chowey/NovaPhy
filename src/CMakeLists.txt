# NovaPhy core static library

configure_file(
    ${PROJECT_SOURCE_DIR}/include/novaphy/novaphy_config.h.in
    ${PROJECT_BINARY_DIR}/include/novaphy/novaphy_config.h
)

add_library(novaphy_core STATIC
    math/spatial.cpp
    math/math_utils.cpp
    core/body.cpp
    core/shape.cpp
    core/model_builder.cpp
    core/joint.cpp
    collision/broadphase.cpp
    collision/narrowphase.cpp
    collision/narrowphase_sat.cpp
    dynamics/integrator.cpp
    dynamics/free_body_solver.cpp
    dynamics/featherstone.cpp
    dynamics/articulated_solver.cpp
    dynamics/xpbd_solver.cpp
    multibody/multibody.cpp
    multibody/multibody_constraint.cpp
    multibody/multibody_solver.cpp
    multibody/free_rigid_body.cpp
    dynamics/solver_sequential_impulse.cpp
    sim/performance_monitor.cpp
    sim/state.cpp
    sim/world.cpp
    fluid/particle_state.cpp
    fluid/neighbor_search.cpp
    fluid/pbf_solver.cpp
    fluid/boundary.cpp
    io/urdf_parser.cpp
    io/openusd_importer.cpp
    io/scene_builder.cpp
    io/simulation_exporter.cpp
    io/feature_completeness.cpp
)

target_include_directories(novaphy_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(novaphy_core PUBLIC
    Eigen3::Eigen
    tinyxml2::tinyxml2
)

target_compile_definitions(novaphy_core 
    PUBLIC
        EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
    PRIVATE
        NOVAPHY_BUILDING
)

novaphy_header_set(novaphy_core
    BASE_DIR ${PROJECT_SOURCE_DIR}/include
    PATTERNS
        ${PROJECT_SOURCE_DIR}/include/novaphy/collision/*.h
        ${PROJECT_SOURCE_DIR}/include/novaphy/core/*.h
        ${PROJECT_SOURCE_DIR}/include/novaphy/dynamics/*.h
        ${PROJECT_SOURCE_DIR}/include/novaphy/fluid/*.h
        ${PROJECT_SOURCE_DIR}/include/novaphy/math/*.h
        ${PROJECT_SOURCE_DIR}/include/novaphy/sim/*.h
    FILES 
        ${PROJECT_SOURCE_DIR}/include/novaphy/novaphy.h
        ${PROJECT_SOURCE_DIR}/include/novaphy/novaphy_types.h
)
novaphy_header_set(novaphy_core
    BASE_DIR ${PROJECT_BINARY_DIR}/include
    FILES ${PROJECT_BINARY_DIR}/include/novaphy/novaphy_config.h
)
novaphy_export_library(novaphy_core COMPONENT core XNAME core DEPENDS Eigen3)
