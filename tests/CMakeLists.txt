# C++ tests using Google Test

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# Will add test files as they are created in later stages
# Example pattern:
# file(GLOB TEST_SOURCES cpp/test_*.cpp)
# foreach(test_source ${TEST_SOURCES})
#     get_filename_component(test_name ${test_source} NAME_WE)
#     add_executable(${test_name} ${test_source})
#     target_link_libraries(${test_name} PRIVATE
#         novaphy_core
#         GTest::gtest_main
#     )
#     gtest_discover_tests(${test_name})
# endforeach()
