#include "tinynpu/fpga_backend.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    // const std::string missing_device =
    //     "/dev/tinynpu_test_device_not_exist";

    // tinynpu::FpgaGemmExecutor executor(
    //     missing_device
    // );

    // assert(executor.device_path() == missing_device);
    // assert(!executor.is_available());

    // std::cout << "fpga_backend_test passed\n";
    // std::cout << "Device path: "
    //           << executor.device_path()
    //           << '\n';

    tinynpu::FpgaGemmExecutor executor("/dev/null");

    tinynpu::Device device("cpu-simulator");

    auto tensor = device.allocate_tensor(
        {2, 2},
        tinynpu::DataType::FP32
    );

    assert(!tensor.has_handle());

    const bool registered =
        executor.register_tensor(tensor);

    assert(!registered);
    assert(!tensor.has_handle());

    return 0;
}