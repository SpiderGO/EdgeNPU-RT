#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinynpu {

enum class DataType {
    FP32,
    INT8
};

std::size_t data_type_size(DataType dtype);

class Buffer {
public:
    explicit Buffer(std::size_t bytes);

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    ~Buffer() = default;

    void* data();
    const void* data() const;

    std::size_t size() const;
    bool empty() const;

private:
    std::vector<std::uint8_t> storage_;
};

class Tensor {
public:
    Tensor(std::vector<std::size_t> shape, DataType dtype, Buffer buffer);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    ~Tensor() = default;

    const std::vector<std::size_t>& shape() const;
    DataType dtype() const;

    std::size_t element_count() const;
    std::size_t nbytes() const;

    void* data();
    const void* data() const;

private:
    std::vector<std::size_t> shape_;
    DataType dtype_;
    Buffer buffer_;
};

class Device {
public:
    explicit Device(std::string backend_name = "cpu-simulator");

    Buffer allocate(std::size_t bytes) const;

    Tensor allocate_tensor(
        const std::vector<std::size_t>& shape,
        DataType dtype
    ) const;

    const std::string& backend_name() const;

private:
    std::string backend_name_;
};

}  // namespace tinynpu