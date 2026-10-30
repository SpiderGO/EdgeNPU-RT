#pragma once

#include <string>

#include "tinynpu/backend.h"

namespace tinynpu {

class FpgaGemmExecutor final : public GemmExecutor {
public:
    explicit FpgaGemmExecutor(
        std::string device_path = "/dev/tinynpu0"
    );

    bool execute(const GemmCommand& command) override;

    bool is_available() const noexcept;

    const std::string& device_path() const noexcept;

private:
    std::string device_path_;
};

}  // namespace tinynpu