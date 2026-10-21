#include "tinynpu/device.h"

#include <iostream>
#include <vector>

int main() {
    tinynpu::Device device("cpu-simulator");

    auto tensor = device.allocate_tensor(
        {2, 3},
        tinynpu::DataType::FP32
    );

    if (tensor.element_count() != 6) {
        std::cerr << "Unexpected element count" << std::endl;
        return 1;
    }

    if (tensor.nbytes() != 6 * sizeof(float)) {
        std::cerr << "Unexpected tensor size" << std::endl;
        return 1;
    }

    if (device.backend_name() != "cpu-simulator") {
        std::cerr << "Unexpected backend name" << std::endl;
        return 1;
    }

    std::cout << "Runtime smoke test passed" << std::endl;
    std::cout << "Backend: " << device.backend_name() << std::endl;
    std::cout << "Tensor elements: " << tensor.element_count() << std::endl;
    std::cout << "Tensor bytes: " << tensor.nbytes() << std::endl;

    return 0;
}