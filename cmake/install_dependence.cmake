# Resolve and install runtime dependencies for the given target
# at install time, excluding common system libraries.

include_guard()

set(WINDOWS_IGNORE_REGEX
    "api-ms-.*" 
    "ext-ms-.*" 
    "kernel32\\.dll" 
    "user32\\.dll" 
    "gdi32\\.dll"
    "advapi32\\.dll"
    "shell32\\.dll"
    "msvcrt\\.dll"
    "ucrtbase\\.dll"
    "ntdll\\.dll" 
    "imm32\\.dll" 
    "ws2_32\\.dll" 
    "comctl32\\.dll"
    "ole32\\.dll" 
    "shlwapi\\.dll" 
    "msvcp_win\\.dll"
    "msvcp.*\\.dll"
    "vcruntime.*\\.dll"
)

set(LINUX_IGNORE_REGEX
    "^libc\\.so.*" 
    "^libpthread\\.so.*" 
    "^librt\\.so.*" 
    "^libdl\\.so.*"
    "^libm\\.so.*"
    "^libgcc_s\\.so.*" 
    "^libX11\\.so.*" 
    "^libglib-.*"
    "^libstdc\\+\\+\\.so.*"
    "^ld-linux.*"
    "^libasound\\.so.*"
)

function(novaphy_install_dependencies target_type target_path dst package_dir)
    message(STATUS "Installing dependencies for ${target_path}")
    message(DEBUG "  Target type: ${target_type}")
    message(DEBUG "  Target path: ${target_path}")
    message(DEBUG "  Destination: ${dst}")
    message(DEBUG "  Package library directory: ${package_dir}")
    if (target_type STREQUAL "EXECUTABLE")
        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR resolved_deps
            UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
            EXECUTABLES ${target_path}
            PRE_EXCLUDE_REGEXES
                ${WINDOWS_IGNORE_REGEX}
                ${LINUX_IGNORE_REGEX}
        )
    elseif (target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY")
        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR resolved_deps
            UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
            LIBRARIES ${target_path}
            PRE_EXCLUDE_REGEXES
                ${WINDOWS_IGNORE_REGEX}
                ${LINUX_IGNORE_REGEX}
        )
    else()
        message(FATAL_ERROR "Target type '${target_type}' is not supported for dependency installation")
    endif()
    
    list(LENGTH resolved_deps resolved_count)
    list(LENGTH unresolved_deps unresolved_count)
    message(DEBUG "  Resolved dependencies: ${resolved_count}")
    message(DEBUG "  Unresolved dependencies: ${unresolved_count}")
    
    if (unresolved_deps)
        message(WARNING "Unresolved dependencies for ${target_path}: ${unresolved_deps}")
    endif()
    
    if (resolved_deps)
        message(DEBUG "  Installing ${resolved_count} dependencies:")
    else()
        message(DEBUG "  No dependencies to install")
    endif()
    
    foreach(dep IN LISTS resolved_deps)
        message(DEBUG "    Installing: ${dep}")
        get_filename_component(dep_name ${dep} NAME)
        set(dep_full_path ${dst}/${dep_name})
        set(dep_ex_full_path ${package_dir}/${dep_name})

        if (EXISTS "${dep_ex_full_path}" AND NOT "${dep}" IS_NEWER_THAN "${dep_ex_full_path}")
            continue()
        endif()

        if (NOT EXISTS "${dep_full_path}" OR NOT "${dep_full_path}" IS_NEWER_THAN "${dep}")
            file(INSTALL DESTINATION "${dst}" FILES "${dep}")
        endif()
    endforeach()
endfunction()