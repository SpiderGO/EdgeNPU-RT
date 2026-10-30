#pragma once

#include <cstddef>
#include <memory>
#include "tinynpu/backend.h"
#include "tinynpu/device.h"
namespace tinynpu {

enum class CommandStatus {
    Idle,
    Submitted,
    Completed,
    Failed
};

class Event {
public:
    void reset() noexcept;
    void signal() noexcept;
    bool is_complete() const noexcept;

private:
    bool completed_{false};
};

struct GemmCommand {
    const float* a{nullptr};
    const float* b{nullptr};
    float* c{nullptr};

    std::uint32_t a_handle{0};
    std::uint32_t b_handle{0};
    std::uint32_t c_handle{0};

    static GemmCommand from_tensors(const Tensor& a, const Tensor& b, Tensor& c);

    std::size_t m{0};
    std::size_t n{0};
    std::size_t k{0};

    Event* completion_event{nullptr};
};

class CommandQueue {
public:
    CommandQueue();

    explicit CommandQueue(std::unique_ptr<GemmExecutor> executor);

    bool submit(const GemmCommand& command);
    bool submit_gemm(const Tensor& a,
                     const Tensor& b,
                     Tensor& c,
                     Event* completion_event = nullptr);
    bool wait();

    CommandStatus status() const noexcept;

private:
    GemmCommand command_{};
    CommandStatus status_{CommandStatus::Idle};
    std::unique_ptr<GemmExecutor> executor_;
};

}  // namespace tinynpu