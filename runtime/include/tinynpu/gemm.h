#pragma once

#include <cstddef>

namespace tinynpu {

bool gemm_f32(const float* a,
              const float* b,
              float* c,
              std::size_t m,
              std::size_t n,
              std::size_t k);

}  // namespace tinynpu