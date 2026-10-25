#include "tinynpu/command.h"

#include "tinynpu/gemm.h"

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

bool CommandQueue::wait() {
    if (status_ != CommandStatus::Submitted) {
        return false;
    }

    const bool success = gemm_f32(
        command_.a,
        command_.b,
        command_.c,
        command_.m,
        command_.n,
        command_.k
    );

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