#pragma once

#include "tinynpu/backend.h"

namespace tinynpu {

class CpuGemmExecutor final : public GemmExecutor {
public:
    bool execute(const GemmCommand& command) override;
};

}//namespace tinynpu