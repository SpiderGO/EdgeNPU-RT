#include "tinynpu/device.h"
#include "tinynpu/command.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    tinynpu::Device device("cpu-simulator");

    auto tensor = device.allocate_tensor(
        {2, 3},
        tinynpu::DataType::FP32
    );

    assert(tensor.element_count() == 6);
    assert(tensor.nbytes() == 6 * sizeof(float));
    assert(device.backend_name() == "cpu-simulator");
    assert(tensor.buffer_handle() == tinynpu::kInvalidBufferHandle);
    assert(!tensor.has_handle());

    auto a = device.allocate_tensor(
        {2, 3},
        tinynpu::DataType::FP32
    );

    auto b = device.allocate_tensor(
        {3, 4},
        tinynpu::DataType::FP32
    );

    auto c = device.allocate_tensor(
        {2, 4},
        tinynpu::DataType::FP32
    );

    const auto gemm_command =
        tinynpu::GemmCommand::from_tensors(a, b, c);

    assert(gemm_command.m == 2);
    assert(gemm_command.n == 4);
    assert(gemm_command.k == 3);

    assert(gemm_command.a_handle == a.buffer_handle());
    assert(gemm_command.b_handle == b.buffer_handle());
    assert(gemm_command.c_handle == c.buffer_handle());
    // 验证 Tensor 与 Buffer 大小不一致时会被拒绝
    bool size_mismatch_rejected = false;

    try {
        tinynpu::Tensor invalid_tensor(
            {2, 3},
            tinynpu::DataType::FP32,
            tinynpu::Buffer(sizeof(float))
        );
    } catch (const std::invalid_argument&) {
        size_mismatch_rejected = true;
    }

    assert(size_mismatch_rejected);

    // 验证 Tensor 大小计算溢出时会被拒绝
    bool overflow_rejected = false;

    try {
        device.allocate_tensor(
            {
                std::numeric_limits<std::size_t>::max(),
                2
            },
            tinynpu::DataType::FP32
        );
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }

    assert(overflow_rejected);

    std::cout << "Runtime smoke test passed\n";
    std::cout << "Backend: "
              << device.backend_name()
              << '\n';
    std::cout << "Tensor elements: "
              << tensor.element_count()
              << '\n';
    std::cout << "Tensor bytes: "
              << tensor.nbytes()
              << '\n';

    return 0;
}