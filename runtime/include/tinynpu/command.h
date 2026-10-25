#pragma once

#include <cstddef>

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

    std::size_t m{0};
    std::size_t n{0};
    std::size_t k{0};

    Event* completion_event{nullptr};
};

class CommandQueue {
public:
    bool submit(const GemmCommand& command);
    bool wait();

    CommandStatus status() const noexcept;

private:
    GemmCommand command_{};
    CommandStatus status_{CommandStatus::Idle};
};

}  // namespace tinynpu