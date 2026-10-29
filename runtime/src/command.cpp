#include "tinynpu/command.h"

#include "tinynpu/cpu_backend.h"

#include <memory>
#include <utility>
namespace tinynpu {

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
    // 当前 Tensor 版 GEMM 只支持 FP32
    if (a.dtype() != DataType::FP32 ||
        b.dtype() != DataType::FP32 ||
        c.dtype() != DataType::FP32) {
        status_ = CommandStatus::Failed;
        return false;
    }

    // 当前 GEMM 只支持二维 Tensor
    if (a.shape().size() != 2 ||
        b.shape().size() != 2 ||
        c.shape().size() != 2) {
        status_ = CommandStatus::Failed;
        return false;
    }

    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    const auto& c_shape = c.shape();

    const std::size_t m = a_shape[0];
    const std::size_t k = a_shape[1];

    const std::size_t b_k = b_shape[0];
    const std::size_t n = b_shape[1];

    // 矩阵乘法要求：
    //
    // A: [M, K]
    // B: [K, N]
    // C: [M, N]
    if (k != b_k ||
        c_shape[0] != m ||
        c_shape[1] != n ||
        m == 0 ||
        n == 0 ||
        k == 0) {
        status_ = CommandStatus::Failed;
        return false;
    }

    GemmCommand command;
    command.a = static_cast<const float*>(a.data());
    command.b = static_cast<const float*>(b.data());
    command.c = static_cast<float*>(c.data());
    command.m = m;
    command.n = n;
    command.k = k;
    command.completion_event = completion_event;

    // Tensor 接口最终复用现有底层 float* 接口
    return submit(command);
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