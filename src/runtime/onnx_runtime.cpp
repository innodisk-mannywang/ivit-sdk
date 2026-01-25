/**
 * @file onnx_runtime.cpp
 * @brief ONNX Runtime backend implementation
 */

#include "ivit/runtime/onnx_runtime.hpp"

#ifdef IVIT_HAS_ONNXRUNTIME

#include <algorithm>
#include <fstream>
#include <thread>

namespace ivit {

// ============================================================================
// ONNXRuntimeBackend implementation
// ============================================================================

ONNXRuntimeBackend::ONNXRuntimeBackend() {
    initialize();
}

ONNXRuntimeBackend::~ONNXRuntimeBackend() = default;

void ONNXRuntimeBackend::initialize() {
    if (initialized_) return;

    env_ = std::make_unique<Ort::Env>(
        ORT_LOGGING_LEVEL_WARNING,
        "ivit-onnxruntime"
    );

    session_options_ = std::make_unique<Ort::SessionOptions>();

    // Set default thread count
    if (num_threads_ > 0) {
        session_options_->SetIntraOpNumThreads(num_threads_);
    } else {
        // Use hardware concurrency
        int threads = std::thread::hardware_concurrency();
        if (threads > 0) {
            session_options_->SetIntraOpNumThreads(threads);
        }
    }

    // Enable graph optimization
    session_options_->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    initialized_ = true;
}

bool ONNXRuntimeBackend::is_available() const {
    return true;  // ONNX Runtime is always available as fallback
}

std::vector<std::string> ONNXRuntimeBackend::supported_formats() const {
    return {".onnx"};
}

std::vector<DeviceInfo> ONNXRuntimeBackend::get_devices() const {
    std::vector<DeviceInfo> devices;

    // CPU is always available
    DeviceInfo cpu;
    cpu.id = "cpu";
    cpu.name = "CPU";
    cpu.backend = "onnxruntime";
    cpu.type = "cpu";
    cpu.is_available = true;
    cpu.supported_precisions = {Precision::FP32};
    devices.push_back(cpu);

    // Check CUDA availability
    if (cuda_available()) {
        DeviceInfo cuda;
        cuda.id = "cuda:0";
        cuda.name = "CUDA GPU";
        cuda.backend = "onnxruntime";
        cuda.type = "gpu";
        cuda.is_available = true;
        cuda.supported_precisions = {Precision::FP32, Precision::FP16};
        devices.push_back(cuda);
    }

    return devices;
}

void* ONNXRuntimeBackend::load_model(
    const std::string& path,
    const RuntimeConfig& config
) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto wrapper = new ONNXSession();

    // Create session options
    Ort::SessionOptions options;

    // Set thread count
    if (config.num_threads > 0) {
        options.SetIntraOpNumThreads(config.num_threads);
    }

    // Enable graph optimization
    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    // Add execution providers based on device
    std::string device = config.device;
    std::transform(device.begin(), device.end(), device.begin(), ::tolower);

    bool use_cuda = false;
    int device_id = 0;

    if (device.find("cuda") != std::string::npos ||
        device.find("gpu") != std::string::npos) {
        // Try to add CUDA execution provider
        if (cuda_available()) {
            if (device.find(":") != std::string::npos) {
                device_id = std::stoi(device.substr(device.find(":") + 1));
            }

            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            options.AppendExecutionProvider_CUDA(cuda_options);
            use_cuda = true;
        }
    }

    // Create session
    try {
        wrapper->session = std::make_unique<Ort::Session>(
            *env_,
            path.c_str(),
            options
        );
    } catch (const Ort::Exception& e) {
        delete wrapper;
        throw ModelLoadError(e.what());
    }

    wrapper->use_cuda = use_cuda;
    wrapper->device_id = device_id;

    // Get input information
    size_t num_inputs = wrapper->session->GetInputCount();
    wrapper->input_names.reserve(num_inputs);
    wrapper->input_infos.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
        auto info = extract_tensor_info(*wrapper->session, i, true);

        // Get input name
        Ort::AllocatedStringPtr name_ptr = wrapper->session->GetInputNameAllocated(
            i, wrapper->allocator
        );
        std::string name = name_ptr.get();

        info.name = name;
        wrapper->input_names.push_back(name);
        wrapper->input_infos.push_back(info);
    }

    // Get output information
    size_t num_outputs = wrapper->session->GetOutputCount();
    wrapper->output_names.reserve(num_outputs);
    wrapper->output_infos.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; i++) {
        auto info = extract_tensor_info(*wrapper->session, i, false);

        // Get output name
        Ort::AllocatedStringPtr name_ptr = wrapper->session->GetOutputNameAllocated(
            i, wrapper->allocator
        );
        std::string name = name_ptr.get();

        info.name = name;
        wrapper->output_names.push_back(name);
        wrapper->output_infos.push_back(info);
    }

    // Prepare C-string arrays for inference
    for (const auto& name : wrapper->input_names) {
        wrapper->input_names_cstr.push_back(name.c_str());
    }
    for (const auto& name : wrapper->output_names) {
        wrapper->output_names_cstr.push_back(name.c_str());
    }

    return wrapper;
}

void ONNXRuntimeBackend::unload_model(void* handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    ONNXSession* session = static_cast<ONNXSession*>(handle);
    delete session;
}

std::vector<TensorInfo> ONNXRuntimeBackend::get_input_info(void* handle) const {
    ONNXSession* session = static_cast<ONNXSession*>(handle);
    return session->input_infos;
}

std::vector<TensorInfo> ONNXRuntimeBackend::get_output_info(void* handle) const {
    ONNXSession* session = static_cast<ONNXSession*>(handle);
    return session->output_infos;
}

std::map<std::string, Tensor> ONNXRuntimeBackend::infer(
    void* handle,
    const std::map<std::string, Tensor>& inputs
) {
    ONNXSession* session = static_cast<ONNXSession*>(handle);

    // Create input tensors
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(session->input_infos.size());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    for (const auto& info : session->input_infos) {
        auto it = inputs.find(info.name);
        if (it == inputs.end()) {
            throw InferenceError("Missing input: " + info.name);
        }

        const Tensor& input = it->second;

        // Convert shape to int64_t
        std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

        // Create ORT tensor
        ONNXTensorElementDataType ort_type = dtype_to_ort(input.dtype());

        Ort::Value tensor = Ort::Value::CreateTensor(
            memory_info,
            const_cast<void*>(input.data()),
            input.byte_size(),
            shape.data(),
            shape.size(),
            ort_type
        );

        input_tensors.push_back(std::move(tensor));
    }

    // Run inference
    std::vector<Ort::Value> output_tensors;

    try {
        output_tensors = session->session->Run(
            Ort::RunOptions{nullptr},
            session->input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            session->output_names_cstr.data(),
            session->output_names_cstr.size()
        );
    } catch (const Ort::Exception& e) {
        throw InferenceError(e.what());
    }

    // Extract output tensors
    std::map<std::string, Tensor> outputs;

    for (size_t i = 0; i < output_tensors.size(); i++) {
        const auto& info = session->output_infos[i];
        auto& ort_tensor = output_tensors[i];

        // Get tensor info
        auto type_info = ort_tensor.GetTensorTypeAndShapeInfo();
        auto ort_shape = type_info.GetShape();
        auto ort_type = type_info.GetElementType();

        // Convert to iVIT tensor
        Shape shape(ort_shape.begin(), ort_shape.end());
        DataType dtype = ort_to_dtype(ort_type);
        Tensor output(shape, dtype);
        output.set_name(info.name);

        // Copy data
        void* data_ptr = ort_tensor.GetTensorMutableData<void>();
        std::memcpy(output.data(), data_ptr, output.byte_size());

        outputs[info.name] = std::move(output);
    }

    return outputs;
}

void ONNXRuntimeBackend::convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const RuntimeConfig& config
) {
    // ONNX Runtime doesn't need conversion
    // Just copy the file if source and destination are different
    if (src_path != dst_path) {
        std::ifstream src(src_path, std::ios::binary);
        std::ofstream dst(dst_path, std::ios::binary);
        dst << src.rdbuf();
    }
}

bool ONNXRuntimeBackend::cuda_available() const {
    // Check if CUDA provider is available
    std::vector<std::string> providers = Ort::GetAvailableProviders();
    for (const auto& provider : providers) {
        if (provider == "CUDAExecutionProvider") {
            return true;
        }
    }
    return false;
}

std::string ONNXRuntimeBackend::get_version() const {
    return std::to_string(ORT_API_VERSION);
}

void ONNXRuntimeBackend::set_num_threads(int num_threads) {
    num_threads_ = num_threads;
}

TensorInfo ONNXRuntimeBackend::extract_tensor_info(
    Ort::Session& session,
    size_t index,
    bool is_input
) {
    TensorInfo info;

    Ort::TypeInfo type_info = is_input ?
        session.GetInputTypeInfo(index) :
        session.GetOutputTypeInfo(index);

    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // Get shape
    auto ort_shape = tensor_info.GetShape();
    for (auto dim : ort_shape) {
        // Handle dynamic dimensions (marked as -1)
        info.shape.push_back(dim > 0 ? dim : 1);
    }

    // Get data type
    info.dtype = ort_to_dtype(tensor_info.GetElementType());

    // Determine layout
    if (info.shape.size() == 4) {
        info.layout = Layout::NCHW;  // Assume NCHW for 4D tensors
    } else if (info.shape.size() == 2) {
        info.layout = Layout::NC;
    } else {
        info.layout = Layout::Unknown;
    }

    return info;
}

DataType ONNXRuntimeBackend::ort_to_dtype(ONNXTensorElementDataType ort_type) const {
    switch (ort_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DataType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DataType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return DataType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DataType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DataType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return DataType::Bool;
        default: return DataType::Unknown;
    }
}

ONNXTensorElementDataType ONNXRuntimeBackend::dtype_to_ort(DataType dtype) const {
    switch (dtype) {
        case DataType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DataType::Float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case DataType::Int8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case DataType::UInt8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case DataType::Int32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DataType::Int64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DataType::Bool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
}

ONNXRuntimeBackend& get_onnx_runtime() {
    static ONNXRuntimeBackend instance;
    return instance;
}

} // namespace ivit

#endif // IVIT_HAS_ONNXRUNTIME
