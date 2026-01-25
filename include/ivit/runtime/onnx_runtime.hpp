/**
 * @file onnx_runtime.hpp
 * @brief ONNX Runtime backend
 */

#ifndef IVIT_RUNTIME_ONNX_RUNTIME_HPP
#define IVIT_RUNTIME_ONNX_RUNTIME_HPP

#include "ivit/runtime/runtime.hpp"

#ifdef IVIT_HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace ivit {

/**
 * @brief ONNX Runtime session wrapper
 */
struct ONNXSession {
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;
    bool use_cuda = false;
    int device_id = 0;
};

/**
 * @brief ONNX Runtime backend implementation
 */
class ONNXRuntimeBackend : public IRuntime {
public:
    ONNXRuntimeBackend();
    ~ONNXRuntimeBackend() override;

    // IRuntime interface
    BackendType type() const override { return BackendType::ONNXRuntime; }
    std::string name() const override { return "ONNXRuntime"; }
    bool is_available() const override;
    std::vector<std::string> supported_formats() const override;
    std::vector<DeviceInfo> get_devices() const override;

    void* load_model(
        const std::string& path,
        const RuntimeConfig& config
    ) override;

    void unload_model(void* handle) override;

    std::vector<TensorInfo> get_input_info(void* handle) const override;
    std::vector<TensorInfo> get_output_info(void* handle) const override;

    std::map<std::string, Tensor> infer(
        void* handle,
        const std::map<std::string, Tensor>& inputs
    ) override;

    void convert_model(
        const std::string& src_path,
        const std::string& dst_path,
        const RuntimeConfig& config
    ) override;

    // ONNX Runtime-specific methods

    /**
     * @brief Check if CUDA execution provider is available
     */
    bool cuda_available() const;

    /**
     * @brief Get ONNX Runtime version
     */
    std::string get_version() const;

    /**
     * @brief Set number of threads
     */
    void set_num_threads(int num_threads);

private:
    void initialize();

    DataType ort_to_dtype(ONNXTensorElementDataType ort_type) const;
    ONNXTensorElementDataType dtype_to_ort(DataType dtype) const;

    TensorInfo extract_tensor_info(
        Ort::Session& session,
        size_t index,
        bool is_input
    );

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::mutex mutex_;
    bool initialized_ = false;
    int num_threads_ = 0;  // 0 = auto
};

/**
 * @brief Get global ONNX Runtime instance
 */
ONNXRuntimeBackend& get_onnx_runtime();

} // namespace ivit

#endif // IVIT_HAS_ONNXRUNTIME

#endif // IVIT_RUNTIME_ONNX_RUNTIME_HPP
