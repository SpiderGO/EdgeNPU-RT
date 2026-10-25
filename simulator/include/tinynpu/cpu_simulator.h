#pragma once

#include <cstddef>

namespace tinynpu {

struct GemmStats {
    bool success{false};
    double average_ms{0.0};
    double gflops{0.0};
};

class CpuSimulator {
public:
    GemmStats benchmark_gemm(const float* a,
                             const float* b,
                             float* c,
                             std::size_t m,
                             std::size_t n,
                             std::size_t k,
                             std::size_t warmup = 3,
                             std::size_t iterations = 10) const;
};

}  // namespace tinynpu