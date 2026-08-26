cmake_minimum_required(VERSION 3.21)
project(xf_qd_runner LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(XF_QD_RUNNER_BUILD_TESTS
       "Build xf_qd_runner's own tests"
       ${PROJECT_IS_TOP_LEVEL})

# ---- dependencies -------------------------------------------------------------
find_package(OpenMP REQUIRED)

# xfac_quad manages itself (and transitively QTgrid-quad, numeric-type-quad, QD).
# Its tests, and those of every layer below it, stay off unless the matching
# <LAYER>_BUILD_TESTS option is explicitly set.
add_subdirectory(extern/xfac_quad)

# ---- own paths ----------------------------------------------------------------
set(XF_QD_RUNNER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SPDLOG_INCLUDE_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/extern/spdlog/include)

if(NOT EXISTS ${SPDLOG_INCLUDE_DIR}/spdlog/spdlog.h)
    message(FATAL_ERROR
        "spdlog not found at ${SPDLOG_INCLUDE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ---- interface target: carries our headers + transitive deps ------------------
add_library(xf_qd_runner_deps INTERFACE)

target_include_directories(xf_qd_runner_deps INTERFACE
    ${XF_QD_RUNNER_INCLUDE_DIR})

# Header-only spdlog (with its bundled fmt). SYSTEM so its own warnings don't
# surface under -Wall -Wextra -Wshadow.
target_include_directories(xf_qd_runner_deps SYSTEM INTERFACE
    ${SPDLOG_INCLUDE_DIR})

target_compile_features(xf_qd_runner_deps INTERFACE cxx_std_20)

target_link_libraries(xf_qd_runner_deps INTERFACE
    xfac_quad_deps
    OpenMP::OpenMP_CXX)

# ---- test / exec targets ------------------------------------------------------
if(XF_QD_RUNNER_BUILD_TESTS)
    enable_testing()

    # One entry per test/test_<name>.cpp -> target xf_qd_runner_test_<name>
    set(_xf_qd_runner_tests
        runner
        determinism
        param)

    foreach(_name IN LISTS _xf_qd_runner_tests)
        add_executable(xf_qd_runner_test_${_name} test/test_${_name}.cpp)
        target_link_libraries(xf_qd_runner_test_${_name} PRIVATE xf_qd_runner_deps)
        target_compile_options(xf_qd_runner_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow)
        target_compile_options(xf_qd_runner_test_${_name} PRIVATE -UNDEBUG)
        target_include_directories(xf_qd_runner_test_${_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/test)
        add_test(NAME xf_qd_runner_test_${_name}
                 COMMAND xf_qd_runner_test_${_name})
    endforeach()
endif()