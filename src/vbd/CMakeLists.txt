# CPU pipeline: always built. Layout: src/vbd/cpu/; GPU in src/vbd/vbd_cuda/ (under vbd, migration-ready).
# release_cuda_buffers() stub is in vbd_solver.cpp when NOVAPHY_VBD_CUDA is unset.
add_library(novaphy_vbd SHARED
    cpu/vbd_world.cpp
    cpu/vbd_solver.cpp
)

# GPU pipeline: only when -DNOVAPHY_WITH_VBD_CUDA=ON + CUDA compiler. Sources under vbd/vbd_cuda/.
if(NOVAPHY_WITH_VBD_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(novaphy_vbd PRIVATE
        vbd_cuda/vbd_solver_cuda.cu
        vbd_cuda/vbd_world_cuda.cpp
    )
    target_compile_definitions(novaphy_vbd PRIVATE NOVAPHY_VBD_CUDA)
    set_target_properties(novaphy_vbd PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )
    message(STATUS "NovaPhy: VBD CUDA backend ENABLED")
else()
    message(STATUS "NovaPhy: VBD CUDA backend DISABLED (set -DNOVAPHY_WITH_VBD_CUDA=ON and provide CUDA to enable)")
endif()

target_include_directories(novaphy_vbd PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(novaphy_vbd PUBLIC
    novaphy_core
)

# The best practice is that every module has their own compile definitions, instead of relying on the "global" ones. 
# This makes it more clear and maintainable.
# 
# But it's works for now since we only have few module that needs it, 
# and we want to avoid unnecessary compile definitions for other modules. 
# We can refactor this later when we have more modules.
target_compile_definitions(novaphy_vbd PRIVATE
    PRIVATE
        NOVAPHY_BUILDING
)

novaphy_header_set(novaphy_vbd
    BASE_DIR ${PROJECT_SOURCE_DIR}/include
    PATTERNS
        ${PROJECT_SOURCE_DIR}/include/novaphy/vbd/*.h
)

novaphy_export_library(novaphy_vbd COMPONENT vbd XNAME vbd)