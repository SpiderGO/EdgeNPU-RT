#include "tinynpu/gemm.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    const float a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const float b[] = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    float c[4] = {};

    const bool success = tinynpu::gemm_f32(
        a, b, c,
        2,  // M
        2,  // N
        3   // K
    );

    assert(success);

    // 结果应为：
    // [58,  64]
    // [139, 154]
    const float expected[] = {
        58.0f,  64.0f,
        139.0f, 154.0f
    };

    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(c[i] - expected[i]) < 1e-5f);
    }

    std::cout << "gemm_test passed\n";
    return 0;
}