# The sample C++ project uses NovaPhy via add_subdirectory

cmake_minimum_required(VERSION 3.15...4.0)

project(NovaPhyAddSubdirectoryTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Before build the project, copy NovaPhy source code to ./NovaPhy directory
add_subdirectory(NovaPhy)

# Export the novaphy::core target for use in this project

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE novaphy::core)