add_library(tinynpu_runtime src/device.cpp src/gemm.cpp)

target_include_directories(tinynpu_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)