# Python bindings via pybind11

set(BIND_SOURCES
    bind_main.cpp
    bind_math.cpp
    bind_core.cpp
    bind_collision.cpp
    bind_performance.cpp
    bind_sim.cpp
    bind_dynamics.cpp
    bind_fluid.cpp
    bind_io.cpp
)

if(NOVAPHY_WITH_IPC)
    list(APPEND BIND_SOURCES bind_ipc.cpp)
endif()

# VBD/AVBD bindings (always built; implementation currently)
list(APPEND BIND_SOURCES bind_vbd.cpp)

pybind11_add_module(_core ${BIND_SOURCES})

target_link_libraries(_core PRIVATE
    novaphy_core
)

if(NOVAPHY_WITH_IPC)
    target_link_libraries(_core PRIVATE novaphy_ipc)
    # C++20 needed for bind_ipc.cpp which includes libuipc headers
    set_source_files_properties(bind_ipc.cpp PROPERTIES
        COMPILE_FLAGS "$<$<CXX_COMPILER_ID:MSVC>:/std:c++20>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-std=c++20>"
    )

endif()

target_link_libraries(_core PRIVATE novaphy_vbd)

novaphy_export_pybind(_core)

if (SKBUILD AND NOVAPHY_BUNDLE_DEPENDENCIES)
    novaphy_bundle_dependencies(_core)
    # The dependency of libuipc is hidden by CMake's target system, so we need to explicitly bundle it here
    novaphy_bundle_files(GLOB
        "${CMAKE_BINARY_DIR}/${NOVAPHY_CONFIG_TYPE}/bin/*.dll" 
        "${CMAKE_BINARY_DIR}/${NOVAPHY_CONFIG_TYPE}/bin/*.so" 
        "${CMAKE_BINARY_DIR}/${NOVAPHY_CONFIG_TYPE}/bin/*.dylib"
    )
endif()
