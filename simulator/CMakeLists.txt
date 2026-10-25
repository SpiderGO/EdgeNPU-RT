add_library(tinynpu_simulator STATIC
    src/cpu_simulator.cpp
)

target_include_directories(tinynpu_simulator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tinynpu_simulator
    PUBLIC
        tinynpu_runtime
)

add_executable(simulator_demo
    src/simulator.cpp
)

target_link_libraries(simulator_demo
    PRIVATE
        tinynpu_simulator
)