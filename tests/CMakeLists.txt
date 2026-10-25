add_executable(runtime_smoke runtime_smoke.cpp)

target_link_libraries(runtime_smoke
    PRIVATE
        tinynpu_runtime
        tinynpu_simulator
)

add_test(
    NAME runtime_smoke
    COMMAND runtime_smoke
)

add_executable(gemm_test
    gemm_test.cpp
)

target_link_libraries(gemm_test
    PRIVATE
        tinynpu_runtime
)

add_executable(command_test
    command_test.cpp
)

target_link_libraries(command_test
    PRIVATE
        tinynpu_runtime
)