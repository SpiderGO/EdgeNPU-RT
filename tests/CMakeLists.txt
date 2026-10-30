# 测试可执行文件
add_executable(runtime_smoke runtime_smoke.cpp)
target_link_libraries(runtime_smoke
    PRIVATE
        tinynpu_runtime
        tinynpu_simulator
)

add_executable(gemm_test gemm_test.cpp)
target_link_libraries(gemm_test
    PRIVATE
        tinynpu_runtime
)

add_executable(command_test command_test.cpp)
target_link_libraries(command_test
    PRIVATE
        tinynpu_runtime
)

add_executable(backend_test backend_test.cpp)
target_link_libraries(backend_test
    PRIVATE
        tinynpu_runtime
)

add_executable(fpga_backend_test fpga_backend_test.cpp)
target_link_libraries(fpga_backend_test
    PRIVATE
        tinynpu_runtime
)

# 注册测试
add_test(NAME runtime_smoke COMMAND runtime_smoke)
add_test(NAME command_test  COMMAND command_test)
add_test(NAME gemm_test     COMMAND gemm_test)
add_test(NAME backend_test  COMMAND backend_test)
add_test(NAME fpga_backend_test  COMMAND fpga_backend_test)