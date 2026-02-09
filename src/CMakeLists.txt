# NovaPhy core static library

add_library(novaphy_core STATIC
    stub.cpp
)

target_include_directories(novaphy_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(novaphy_core PUBLIC
    Eigen3::Eigen
)

# Enforce float32: no accidental double usage through Eigen defaults
target_compile_definitions(novaphy_core PUBLIC
    EIGEN_DEFAULT_DENSE_INDEX_TYPE=int
)
