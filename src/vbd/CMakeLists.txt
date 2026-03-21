# CPU pipeline: always built. Layout: src/vbd/cpu/; GPU in src/vbd/vbd_cuda/ (under vbd, migration-ready).
# release_cuda_buffers() stub is in vbd_solver.cpp when NOVAPHY_VBD_CUDA is unset.
if(MSVC)
    add_library(novaphy_vbd STATIC
        cpu/vbd_world.cpp
        cpu/vbd_solver.cpp
    )
else()
    add_library(novaphy_vbd SHARED
        cpu/vbd_world.cpp
        cpu/vbd_solver.cpp
    )
endif()

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
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(novaphy_vbd PUBLIC
    novaphy_core
)

