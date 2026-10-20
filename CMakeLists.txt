cmake_minimum_required(VERSION 3.16)

project(EdgeNPU-RT VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EDGE_NPU_BUILD_TESTS "Build tests" ON)
option(EDGE_NPU_BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(runtime)
add_subdirectory(simulator)

if(EDGE_NPU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(EDGE_NPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()