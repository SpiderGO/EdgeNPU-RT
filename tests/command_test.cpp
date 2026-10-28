#include "tinynpu/command.h"
#include "tinynpu/device.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    tinynpu::Device device;

    auto a = device.allocate_tensor(
        {2, 3},
        tinynpu::DataType::FP32
    );

    auto b = device.allocate_tensor(
        {3, 2},
        tinynpu::DataType::FP32
    );

    auto c = device.allocate_tensor(
        {2, 2},
        tinynpu::DataType::FP32
    );

    const float a_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const float b_values[] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    auto* a_data = static_cast<float*>(a.data());
    auto* b_data = static_cast<float*>(b.data());

    for (int i = 0; i < 6; ++i) {
        a_data[i] = a_values[i];
        b_data[i] = b_values[i];
    }

    tinynpu::Event event;
    tinynpu::CommandQueue queue;

    // 使用新的 Tensor 接口提交 GEMM
    assert(queue.submit_gemm(a, b, c, &event));

    assert(queue.status() == tinynpu::CommandStatus::Submitted);
    assert(!event.is_complete());

    assert(queue.wait());

    assert(queue.status() == tinynpu::CommandStatus::Completed);
    assert(event.is_complete());

    const float expected[] = {
        58.0f,  64.0f,
        139.0f, 154.0f
    };

    const auto* c_data = static_cast<const float*>(c.data());

    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(c_data[i] - expected[i]) < 1e-5f);
    }

    std::cout << "tensor command_test passed\n";
    return 0;
}