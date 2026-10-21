#include "tinynpu/device.h"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace tinynpu {

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

Buffer::Buffer(std::size_t bytes)
    : storage_(bytes, 0) {
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

Tensor::Tensor(std::vector<std::size_t> shape,
               DataType dtype,
               Buffer buffer)
    : shape_(std::move(shape)),
      dtype_(dtype),
      buffer_(std::move(buffer)) {
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

DataType Tensor::dtype() const {
    return dtype_;
}

std::size_t Tensor::element_count() const {
    if (shape_.empty()) {
        return 1;
    }

    return std::accumulate(
        shape_.begin(),
        shape_.end(),
        std::size_t{1},
        [](std::size_t result, std::size_t dim) {
            return result * dim;
        }
    );
}

std::size_t Tensor::nbytes() const {
    return element_count() * data_type_size(dtype_);
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
        [&]() {
            if (shape.empty()) {
                return data_type_size(dtype);
            }

            std::size_t elements = 1;
            for (std::size_t dim : shape) {
                elements *= dim;
            }

            return elements * data_type_size(dtype);
        }();

    return Tensor(shape, dtype, Buffer(bytes));
}

const std::string& Device::backend_name() const {
    return backend_name_;
}

}  // namespace tinynpu