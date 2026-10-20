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