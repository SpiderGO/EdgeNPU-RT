#include "tinynpu/command.h"
#include "tinynpu/cpu_backend.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    const float a[] = {
        2.0f
    };

    const float b[] = {
        3.0f
    };

    float c[] = {
        0.0f
    };

    tinynpu::GemmCommand command;
    command.a = a;
    command.b = b;
    command.c = c;
    command.m = 1;
    command.n = 1;
    command.k = 1;

    tinynpu::CpuGemmExecutor executor;

    assert(executor.execute(command));
    assert(std::fabs(c[0] - 6.0f) < 1e-5f);

    std::cout << "backend_test passed\n";
    return 0;
}