#include "tinynpu/gemm.h"

namespace tinynpu {

bool gemm_f32(const float* a,
              const float* b,
              float* c,
              std::size_t m,
              std::size_t n,
              std::size_t k) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        return false;
    }

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for(int p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    return true;
}

}  // namespace tinynpu