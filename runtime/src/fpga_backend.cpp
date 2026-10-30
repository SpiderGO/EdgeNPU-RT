#include "tinynpu/fpga_backend.h"

#include "tinynpu/command.h"

#include "tinynpu_ioctl.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <utility>

namespace tinynpu {

FpgaGemmExecutor::FpgaGemmExecutor(std::string device_path) : device_path_(std::move(device_path)) {
}

bool FpgaGemmExecutor::is_available() const noexcept {
    return ::access(
        device_path_.c_str(),
        F_OK
    ) == 0;
}

const std::string& FpgaGemmExecutor::device_path() const noexcept {
    return device_path_;
}

bool FpgaGemmExecutor::execute(const GemmCommand& command) {
    if (!is_available()) {
        return false;
    }

    if (command.a_handle == 0 ||
        command.b_handle == 0 ||
        command.c_handle == 0 ||
        command.m == 0 ||
        command.n == 0 ||
        command.k == 0) {
        return false;
    }

    const int fd = ::open(
        device_path_.c_str(),
        O_RDWR | O_CLOEXEC
    );

    if (fd < 0) {
        return false;
    }

    tinynpu_gemm_submit request{};

    request.a_handle = command.a_handle;
    request.b_handle = command.b_handle;
    request.c_handle = command.c_handle;

    request.m = static_cast<std::uint32_t>(command.m);
    request.n = static_cast<std::uint32_t>(command.n);
    request.k = static_cast<std::uint32_t>(command.k);

    request.dtype = 0;  // FP32
    request.flags = 0;
    request.sequence = 0;

    const int submit_result = ::ioctl(
        fd,
        TINYNPU_IOCTL_SUBMIT_GEMM,
        &request
    );

    if (submit_result < 0) {
        ::close(fd);
        return false;
    }

    tinynpu_wait_request wait_request{};

    wait_request.sequence = request.sequence;

    const int wait_result = ::ioctl(
        fd,
        TINYNPU_IOCTL_WAIT,
        &wait_request
    );

    ::close(fd);

    if (wait_result < 0) {
        return false;
    }

    return wait_request.status == 0;
}

}  // namespace tinynpu