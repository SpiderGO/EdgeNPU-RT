#include "tinynpu/cpu_simulator.h"

#include "tinynpu/gemm.h"

#include <chrono>

namespace tinynpu {

GemmStats CpuSimulator::benchmark_gemm(const float* a,
                                       const float* b,
                                       float* c,
                                       std::size_t m,
                                       std::size_t n,
                                       std::size_t k,
                                       std::size_t warmup,
                                       std::size_t iterations) const {
    GemmStats stats;

    if (a == nullptr || b == nullptr || c == nullptr ||
        iterations == 0) {
        return stats;
    }

    for (std::size_t i = 0; i < warmup; ++i) {
        if (!gemm_f32(a, b, c, m, n, k)) {
            return stats;
        }
    }

    const auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < iterations; ++i) {
        if (!gemm_f32(a, b, c, m, n, k)) {
            return stats;
        }
    }

    const auto end = std::chrono::steady_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    stats.success = true;
    stats.average_ms = total_ms / static_cast<double>(iterations);

    const double operations =
        2.0 * static_cast<double>(m) *
        static_cast<double>(n) *
        static_cast<double>(k);

    stats.gflops =
        operations / (stats.average_ms * 1.0e6);

    return stats;
}

}  // namespace tinynpu