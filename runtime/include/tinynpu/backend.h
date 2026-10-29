#pragma once

namespace tinynpu {

struct GemmCommand;

class GemmExecutor {
public:
    virtual ~GemmExecutor() = default;

    virtual bool execute(const GemmCommand& command) = 0;
};
} //namespace tinynpu