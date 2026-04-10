# The sample C++ Project use NovaPhy by find_package

cmake_minimum_required(VERSION 3.15...4.0)

project(NovaPhyFindPackageTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(novaphy CONFIG REQUIRED)

# Export the novaphy::core target for use in this project

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE novaphy::core)