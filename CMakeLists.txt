cmake_minimum_required(VERSION 3.20)

# Add custom modules to the search path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(wrf-plugin
    VERSION     0.2.0
    LANGUAGES   C CXX
)

# Set global language standard to C++20.
set(CMAKE_CXX_STANDARD 20)

add_compile_options(
    # NOTE: C++20 warning adjustments for this project.
    -Wall
    -Wextra
    -Wno-ambiguous-reversed-operator
    # NOTE: GDB < 10.x can't handle DWARFv5 correctly.
    $<$<CXX_COMPILER_ID:Clang>:-fdebug-default-version=4>
)

add_library(plugin_rrtmg SHARED
    src/plugin.cpp
    src/Constants.cpp
)

target_include_directories(plugin_rrtmg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(runner
    src/runner.cpp
)

target_link_libraries(runner plugin_rrtmg -lnetcdf)
