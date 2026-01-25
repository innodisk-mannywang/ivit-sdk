/**
 * @file tensor.hpp
 * @brief Tensor class definition
 */

#ifndef IVIT_CORE_TENSOR_HPP
#define IVIT_CORE_TENSOR_HPP

#include "ivit/core/common.hpp"
#include <vector>
#include <memory>
#include <cstring>

namespace ivit {

/**
 * @brief Tensor information
 */
struct TensorInfo {
    std::string name;          ///< Tensor name
    Shape shape;               ///< Tensor shape
    DataType dtype;            ///< Data type
    Layout layout;             ///< Tensor layout

    /**
     * @brief Get total number of elements
     */
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    /**
     * @brief Get size in bytes
     */
    size_t byte_size() const {
        size_t elem_size = 4; // default float32
        switch (dtype) {
            case DataType::Float16: elem_size = 2; break;
            case DataType::Int8:
            case DataType::UInt8: elem_size = 1; break;
            case DataType::Int64: elem_size = 8; break;
            default: elem_size = 4; break;
        }
        return numel() * elem_size;
    }
};

/**
 * @brief Lightweight tensor class
 *
 * Simple tensor container for data exchange.
 * For heavy computation, use backend-specific tensors.
 */
class Tensor {
public:
    Tensor() = default;

    /**
     * @brief Construct tensor with shape and data type
     */
    Tensor(const Shape& shape, DataType dtype = DataType::Float32)
        : info_{.name = "", .shape = shape, .dtype = dtype, .layout = Layout::NCHW}
    {
        data_.resize(info_.byte_size());
    }

    /**
     * @brief Construct from raw data
     */
    template<typename T>
    Tensor(const std::vector<T>& data, const Shape& shape)
        : info_{.name = "", .shape = shape, .dtype = DataType::Float32, .layout = Layout::NCHW}
    {
        // Infer dtype from T
        if constexpr (std::is_same_v<T, float>) {
            info_.dtype = DataType::Float32;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            info_.dtype = DataType::Int32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            info_.dtype = DataType::Int64;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            info_.dtype = DataType::UInt8;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            info_.dtype = DataType::Int8;
        }

        data_.resize(data.size() * sizeof(T));
        std::memcpy(data_.data(), data.data(), data_.size());
    }

    // Accessors
    const TensorInfo& info() const { return info_; }
    const Shape& shape() const { return info_.shape; }
    DataType dtype() const { return info_.dtype; }
    Layout layout() const { return info_.layout; }
    int64_t numel() const { return info_.numel(); }
    size_t byte_size() const { return data_.size(); }

    // Data access
    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }

    template<typename T>
    T* data_ptr() { return reinterpret_cast<T*>(data_.data()); }

    template<typename T>
    const T* data_ptr() const { return reinterpret_cast<const T*>(data_.data()); }

    // Setters
    void set_name(const std::string& name) { info_.name = name; }
    void set_layout(Layout layout) { info_.layout = layout; }

    /**
     * @brief Reshape tensor (must have same number of elements)
     */
    void reshape(const Shape& new_shape) {
        int64_t old_numel = info_.numel();
        info_.shape = new_shape;
        if (info_.numel() != old_numel) {
            throw IVITError("Reshape: element count mismatch");
        }
    }

    /**
     * @brief Clone tensor
     */
    Tensor clone() const {
        Tensor t;
        t.info_ = info_;
        t.data_ = data_;
        return t;
    }

private:
    TensorInfo info_;
    std::vector<uint8_t> data_;
};

} // namespace ivit

#endif // IVIT_CORE_TENSOR_HPP
