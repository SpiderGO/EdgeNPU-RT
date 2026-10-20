add_library(tinynpu_simulator src/simulator.cpp)

target_include_directories(tinynpu_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)