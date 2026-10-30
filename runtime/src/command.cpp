#include "tinynpu/command.h"

#include "tinynpu/cpu_backend.h"

#include <stdexcept>
#include <memory>
#include <utility>
namespace tinynpu {

tinynpu::GemmCommand tinynpu::GemmCommand::from_tensors(
    const Tensor& a,
    const Tensor& b,
    Tensor& c
) {
    if (a.shape().size() != 2 ||
        b.shape().size() != 2 ||
        c.shape().size() != 2) {
        throw std::invalid_argument(
            "GEMM tensors must be two-dimensional"
        );
    }

    const std::size_t m = a.shape()[0];
    const std::size_t k = a.shape()[1];
    const std::size_t n = b.shape()[1];

    if (b.shape()[0] != k) {
        throw std::invalid_argument(
            "GEMM dimensions are incompatible"
        );
    }

    if (c.shape()[0] != m || c.shape()[1] != n) {
        throw std::invalid_argument(
            "GEMM output shape is incorrect"
        );
    }

    GemmCommand command{};

    command.m = static_cast<std::uint32_t>(m);
    command.n = static_cast<std::uint32_t>(n);
    command.k = static_cast<std::uint32_t>(k);

    command.a_handle = a.buffer_handle();
    command.b_handle = b.buffer_handle();
    command.c_handle = c.buffer_handle();

    command.a = static_cast<const float*>(a.data());
    command.b = static_cast<const float*>(b.data());
    command.c = static_cast<float*>(c.data());

    return command;
}

void Event::reset() noexcept {
    completed_ = false;
}

void Event::signal() noexcept {
    completed_ = true;
}

bool Event::is_complete() const noexcept {
    return completed_;
}

CommandQueue::CommandQueue()
    : CommandQueue(std::make_unique<CpuGemmExecutor>()) {
}

CommandQueue::CommandQueue(std::unique_ptr<GemmExecutor> executor)
    : executor_(std::move(executor)) {
}

bool CommandQueue::submit(const GemmCommand& command) {
    if (command.a == nullptr ||
        command.b == nullptr ||
        command.c == nullptr ||
        command.m == 0 ||
        command.n == 0 ||
        command.k == 0) {
        status_ = CommandStatus::Failed;
        return false;
    }

    if (status_ == CommandStatus::Submitted) {
        return false;
    }

    command_ = command;

    if (command_.completion_event != nullptr) {
        command_.completion_event->reset();
    }

    status_ = CommandStatus::Submitted;
    return true;
}

bool CommandQueue::submit_gemm(
    const Tensor& a,
    const Tensor& b,
    Tensor& c,
    Event* completion_event
) {
    if (a.dtype() != DataType::FP32 ||
        b.dtype() != DataType::FP32 ||
        c.dtype() != DataType::FP32) {
        status_ = CommandStatus::Failed;
        return false;
    }

    try {
        GemmCommand command =
            GemmCommand::from_tensors(a, b, c);

        command.completion_event = completion_event;

        return submit(command);
    } catch (const std::invalid_argument&) {
        status_ = CommandStatus::Failed;
        return false;
    }
}

bool CommandQueue::wait() {
    if (status_ != CommandStatus::Submitted) {
        return false;
    }

    if(executor_ == nullptr) {
        status_ = CommandStatus::Failed;
        return false;
    }
    const bool success = executor_->execute(command_);

    if (!success) {
        status_ = CommandStatus::Failed;
        return false;
    }

    if (command_.completion_event != nullptr) {
        command_.completion_event->signal();
    }

    status_ = CommandStatus::Completed;
    return true;
}

CommandStatus CommandQueue::status() const noexcept {
    return status_;
}

}  // namespace tinynpu