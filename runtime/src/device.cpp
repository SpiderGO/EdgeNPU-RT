#include "tinynpu/device.h"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tinynpu {

namespace {

std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs,
    const char* message
) {
    if (rhs != 0 &&
        lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error(message);
    }

    return lhs * rhs;
}

std::size_t checked_element_count(
    const std::vector<std::size_t>& shape
) {
    if (shape.empty()) {
        return 1;
    }

    std::size_t elements = 1;

    for (std::size_t dim : shape) {
        elements = checked_multiply(
            elements,
            dim,
            "Tensor element count overflow"
        );
    }

    return elements;
}

std::size_t checked_tensor_bytes(
    const std::vector<std::size_t>& shape,
    DataType dtype
) {
    return checked_multiply(
        checked_element_count(shape),
        data_type_size(dtype),
        "Tensor byte size overflow"
    );
}

}  // namespace

std::size_t data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return sizeof(float);

        case DataType::INT8:
            return sizeof(std::int8_t);

        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

Buffer::Buffer(std::size_t bytes, BufferHandle handle) : storage_(bytes, 0), handle_(handle) {
}

void* Buffer::data() {
    return storage_.data();
}

const void* Buffer::data() const {
    return storage_.data();
}

std::size_t Buffer::size() const {
    return storage_.size();
}

bool Buffer::empty() const {
    return storage_.empty();
}

BufferHandle Buffer::handle() const noexcept {
    return handle_;
}

bool Buffer::has_handle() const noexcept {
    return handle_ != kInvalidBufferHandle;
}

Tensor::Tensor(
    std::vector<std::size_t> shape,
    DataType dtype,
    Buffer buffer
)
    : shape_(std::move(shape)),
      dtype_(dtype),
      buffer_(std::move(buffer)) {
    const std::size_t expected_bytes =
        checked_tensor_bytes(shape_, dtype_);

    if (buffer_.size() != expected_bytes) {
        throw std::invalid_argument(
            "Tensor buffer size does not match shape and data type"
        );
    }
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

DataType Tensor::dtype() const {
    return dtype_;
}

std::size_t Tensor::element_count() const {
    return checked_element_count(shape_);
}

std::size_t Tensor::nbytes() const {
    return element_count() * data_type_size(dtype_);
}

BufferHandle Tensor::buffer_handle() const noexcept {
    return buffer_.handle();
}

bool Tensor::has_handle() const noexcept {
    return buffer_.has_handle();
}

void* Tensor::data() {
    return buffer_.data();
}

const void* Tensor::data() const {
    return buffer_.data();
}

Device::Device(std::string backend_name)
    : backend_name_(std::move(backend_name)) {
}

Buffer Device::allocate(std::size_t bytes) const {
    return Buffer(bytes);
}

Tensor Device::allocate_tensor(
    const std::vector<std::size_t>& shape,
    DataType dtype
) const {
    const std::size_t bytes =
        checked_tensor_bytes(shape, dtype);

    return Tensor(
        shape,
        dtype,
        Buffer(bytes)
    );
}

const std::string& Device::backend_name() const {
    return backend_name_;
}

}  // namespace tinynpu