# The module detects the environment and applying some patch.

set(NOVAPHY_OVERLAY_PORTS "")

if (CMAKE_HOST_WIN32)
    # Windows-specific overlay ports 
    # (more specifically, for MSVC, but MSVC can't be detected before project command)
    list(APPEND NOVAPHY_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/port/eigen3")
endif()

set(_novaphy_vcpkg_overlay_ports "${VCPKG_OVERLAY_PORTS}")
list(APPEND _novaphy_vcpkg_overlay_ports ${NOVAPHY_OVERLAY_PORTS})
list(REMOVE_DUPLICATES _novaphy_vcpkg_overlay_ports)
set(VCPKG_OVERLAY_PORTS "${_novaphy_vcpkg_overlay_ports}" CACHE STRING "Overlay ports for vcpkg" FORCE)
message(STATUS "NovaPhy: VCPKG_OVERLAY_PORTS set to ${VCPKG_OVERLAY_PORTS}")