#include "tinynpu/cpu_simulator.h"

#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t m = 256;
    constexpr std::size_t n = 256;
    constexpr std::size_t k = 256;

    std::vector<float> a(m * k, 1.0f);
    std::vector<float> b(k * n, 2.0f);
    std::vector<float> c(m * n, 0.0f);

    tinynpu::CpuSimulator simulator;

    const auto stats = simulator.benchmark_gemm(
        a.data(),
        b.data(),
        c.data(),
        m,
        n,
        k,
        3,
        10
    );

    if (!stats.success) {
        std::cerr << "GEMM benchmark failed\n";
        return 1;
    }

    std::cout << "CPU Simulator GEMM benchmark\n";
    std::cout << "Matrix shape: "
              << m << "x" << k << " * "
              << k << "x" << n << "\n";
    std::cout << "Average latency: "
              << stats.average_ms << " ms\n";
    std::cout << "Throughput: "
              << stats.gflops << " GFLOPS\n";

    // 1*2 + 1*2 + ... 共 256 项
    // 每个输出元素应为 512
    std::cout << "Output[0]: " << c[0] << "\n";

    return 0;
}