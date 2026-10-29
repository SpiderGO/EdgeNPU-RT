#include "tinynpu/cpu_backend.h"
#include "tinynpu/gemm.h"
#include "tinynpu/command.h"

namespace tinynpu {

bool CpuGemmExecutor::execute(const GemmCommand& command) {
    return gemm_f32(
        command.a,
        command.b,
        command.c,
        command.m,
        command.n,
        command.k
    );
};

}