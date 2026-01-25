/**
 * @file tensorrt_runtime.cpp
 * @brief TensorRT backend runtime implementation
 */

#include "ivit/runtime/tensorrt_runtime.hpp"

#ifdef IVIT_HAS_TENSORRT

#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace ivit {

// ============================================================================
// TensorRTLogger implementation
// ============================================================================

void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    // Filter out info messages unless verbose mode is on
    if (severity == Severity::kINFO && !verbose_) {
        return;
    }

    const char* severity_str = "";
    switch (severity) {
        case Severity::kINTERNAL_ERROR: severity_str = "[INTERNAL_ERROR]"; break;
        case Severity::kERROR: severity_str = "[ERROR]"; break;
        case Severity::kWARNING: severity_str = "[WARNING]"; break;
        case Severity::kINFO: severity_str = "[INFO]"; break;
        case Severity::kVERBOSE: severity_str = "[VERBOSE]"; break;
    }

    if (severity <= Severity::kWARNING) {
        std::cerr << "TensorRT " << severity_str << ": " << msg << std::endl;
    } else if (verbose_) {
        std::cout << "TensorRT " << severity_str << ": " << msg << std::endl;
    }
}

// ============================================================================
// TensorRTEngine implementation
// ============================================================================

TensorRTEngine::~TensorRTEngine() {
    // Free device buffers
    for (void* buf : device_buffers) {
        if (buf) {
            cudaFree(buf);
        }
    }

    // Destroy CUDA stream
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

// ============================================================================
// TensorRTRuntime implementation
// ============================================================================

TensorRTRuntime::TensorRTRuntime() {
    initialize();
}

TensorRTRuntime::~TensorRTRuntime() = default;

void TensorRTRuntime::initialize() {
    if (initialized_) return;

    runtime_.reset(
        nvinfer1::createInferRuntime(logger_),
        [](nvinfer1::IRuntime* r) { delete r; }
    );

    if (!runtime_) {
        throw IVITError("Failed to create TensorRT runtime");
    }

    initialized_ = true;
}

bool TensorRTRuntime::is_available() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

std::vector<std::string> TensorRTRuntime::supported_formats() const {
    return {".onnx", ".engine", ".trt", ".plan"};
}

std::vector<DeviceInfo> TensorRTRuntime::get_devices() const {
    std::vector<DeviceInfo> devices;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return devices;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        DeviceInfo info;
        info.id = "cuda:" + std::to_string(i);
        info.name = props.name;
        info.backend = "tensorrt";
        info.type = "gpu";
        info.memory_total = props.totalGlobalMem;
        info.is_available = true;

        // Check supported precisions based on compute capability
        info.supported_precisions = {Precision::FP32};

        // FP16 support (Compute Capability >= 5.3 for full support)
        if (props.major > 5 || (props.major == 5 && props.minor >= 3)) {
            info.supported_precisions.push_back(Precision::FP16);
        }

        // INT8 support (Compute Capability >= 6.1)
        if (props.major > 6 || (props.major == 6 && props.minor >= 1)) {
            info.supported_precisions.push_back(Precision::INT8);
        }

        devices.push_back(info);
    }

    return devices;
}

std::string TensorRTRuntime::get_cache_path(
    const std::string& model_path,
    const RuntimeConfig& config
) const {
    namespace fs = std::filesystem;

    // Generate cache filename based on model path, precision, and GPU
    std::string model_name = fs::path(model_path).stem().string();
    std::string precision_str = "fp32";
    if (config.precision == Precision::FP16) precision_str = "fp16";
    else if (config.precision == Precision::INT8) precision_str = "int8";

    // Get GPU info for cache key
    cudaDeviceProp props;
    int device_id = 0;
    if (config.device.find("cuda:") == 0) {
        device_id = std::stoi(config.device.substr(5));
    }
    cudaGetDeviceProperties(&props, device_id);

    // Use compute capability as part of cache key
    std::string gpu_key = std::to_string(props.major) + std::to_string(props.minor);

    // Determine cache directory
    std::string cache_dir = config.cache_dir;
    if (cache_dir.empty()) {
        // Default: same directory as model
        cache_dir = fs::path(model_path).parent_path().string();
        if (cache_dir.empty()) cache_dir = ".";
    }

    std::string cache_name = model_name + "_trt_" + precision_str + "_sm" + gpu_key + ".engine";
    return (fs::path(cache_dir) / cache_name).string();
}

void* TensorRTRuntime::load_model(
    const std::string& path,
    const RuntimeConfig& config
) {
    std::lock_guard<std::mutex> lock(mutex_);
    namespace fs = std::filesystem;

    // Parse device ID
    int device_id = 0;
    if (config.device.find("cuda:") == 0) {
        device_id = std::stoi(config.device.substr(5));
    }

    // Set CUDA device
    cudaSetDevice(device_id);

    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    std::shared_ptr<nvinfer1::ICudaEngine> engine;

    if (ext == ".onnx") {
        // Check for cached engine
        if (config.use_cache) {
            std::string cache_path = get_cache_path(path, config);

            if (fs::exists(cache_path)) {
                // Check if cache is newer than source
                auto src_time = fs::last_write_time(path);
                auto cache_time = fs::last_write_time(cache_path);

                if (cache_time > src_time) {
                    std::cout << "[TensorRT] Loading cached engine: " << cache_path << std::endl;
                    engine = load_engine(cache_path);
                }
            }
        }

        // Build engine if no cache
        if (!engine) {
            std::cout << "[TensorRT] Building engine from ONNX (this may take a while)..." << std::endl;
            engine = build_engine_from_onnx(path, config);

            // Save to cache
            if (config.use_cache && engine) {
                std::string cache_path = get_cache_path(path, config);
                std::cout << "[TensorRT] Saving engine to cache: " << cache_path << std::endl;
                save_engine(engine.get(), cache_path);
            }
        }
    } else if (ext == ".engine" || ext == ".trt" || ext == ".plan") {
        // Load serialized engine
        engine = load_engine(path);
    } else {
        throw UnsupportedFormatError(ext);
    }

    if (!engine) {
        throw ModelLoadError("Failed to create TensorRT engine");
    }

    return create_engine_wrapper(engine, device_id);
}

void TensorRTRuntime::unload_model(void* handle) {
    std::lock_guard<std::mutex> lock(mutex_);

    TensorRTEngine* engine = static_cast<TensorRTEngine*>(handle);
    delete engine;
}

std::vector<TensorInfo> TensorRTRuntime::get_input_info(void* handle) const {
    TensorRTEngine* engine = static_cast<TensorRTEngine*>(handle);
    return engine->input_infos;
}

std::vector<TensorInfo> TensorRTRuntime::get_output_info(void* handle) const {
    TensorRTEngine* engine = static_cast<TensorRTEngine*>(handle);
    return engine->output_infos;
}

std::map<std::string, Tensor> TensorRTRuntime::infer(
    void* handle,
    const std::map<std::string, Tensor>& inputs
) {
    TensorRTEngine* engine = static_cast<TensorRTEngine*>(handle);

    // Set CUDA device
    cudaSetDevice(engine->device_id);

    // Copy input data to device
    for (size_t i = 0; i < engine->input_infos.size(); i++) {
        const auto& info = engine->input_infos[i];
        auto it = inputs.find(info.name);

        if (it == inputs.end()) {
            throw InferenceError("Missing input: " + info.name);
        }

        const Tensor& input = it->second;
        cudaMemcpyAsync(
            engine->device_buffers[i],
            input.data(),
            input.byte_size(),
            cudaMemcpyHostToDevice,
            engine->stream
        );
    }

    // Run inference (TensorRT 10 API uses enqueueV3)
    bool success = engine->context->enqueueV3(engine->stream);

    if (!success) {
        throw InferenceError("TensorRT inference failed");
    }

    // Copy output data to host
    std::map<std::string, Tensor> outputs;
    size_t input_count = engine->input_infos.size();

    for (size_t i = 0; i < engine->output_infos.size(); i++) {
        const auto& info = engine->output_infos[i];
        Tensor output(info.shape, info.dtype);
        output.set_name(info.name);

        cudaMemcpyAsync(
            output.data(),
            engine->device_buffers[input_count + i],
            output.byte_size(),
            cudaMemcpyDeviceToHost,
            engine->stream
        );

        outputs[info.name] = std::move(output);
    }

    // Synchronize
    cudaStreamSynchronize(engine->stream);

    return outputs;
}

void TensorRTRuntime::convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const RuntimeConfig& config
) {
    namespace fs = std::filesystem;

    std::string src_ext = fs::path(src_path).extension().string();
    std::transform(src_ext.begin(), src_ext.end(), src_ext.begin(), ::tolower);

    if (src_ext != ".onnx") {
        throw IVITError("TensorRT conversion only supports ONNX input");
    }

    auto engine = build_engine_from_onnx(src_path, config);
    save_engine(engine.get(), dst_path);
}

std::shared_ptr<nvinfer1::ICudaEngine> TensorRTRuntime::build_engine_from_onnx(
    const std::string& onnx_path,
    const RuntimeConfig& config
) {
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_)
    );
    if (!builder) {
        throw IVITError("Failed to create TensorRT builder");
    }

    // Create network
    const auto explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    );
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch)
    );
    if (!network) {
        throw IVITError("Failed to create TensorRT network");
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_)
    );
    if (!parser) {
        throw IVITError("Failed to create ONNX parser");
    }

    // Parse ONNX model
    if (!parser->parseFromFile(
            onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
        )) {
        std::string errors;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            errors += parser->getError(i)->desc();
            errors += "\n";
        }
        throw ModelLoadError("ONNX parsing failed: " + errors);
    }

    // Create builder config
    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    if (!builder_config) {
        throw IVITError("Failed to create builder config");
    }

    // Set precision
    if (config.precision == Precision::FP16) {
        if (builder->platformHasFastFp16()) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    } else if (config.precision == Precision::INT8) {
        if (builder->platformHasFastInt8()) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // Note: INT8 calibration would be needed for full INT8 inference
        }
    }

    // Set memory pool limit (use 1GB as default)
    builder_config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        1ULL << 30  // 1GB
    );

    // Build engine
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builder_config)
    );
    if (!serialized_engine) {
        throw IVITError("Failed to build TensorRT engine");
    }

    // Deserialize engine
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(
            serialized_engine->data(),
            serialized_engine->size()
        ),
        [](nvinfer1::ICudaEngine* e) { delete e; }
    );

    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine> TensorRTRuntime::load_engine(
    const std::string& engine_path
) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw ModelLoadError("Cannot open engine file: " + engine_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw ModelLoadError("Cannot read engine file: " + engine_path);
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(buffer.data(), size),
        [](nvinfer1::ICudaEngine* e) { delete e; }
    );

    if (!engine) {
        throw ModelLoadError("Failed to deserialize TensorRT engine");
    }

    return engine;
}

void TensorRTRuntime::save_engine(
    nvinfer1::ICudaEngine* engine,
    const std::string& path
) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine->serialize()
    );

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw IVITError("Cannot open file for writing: " + path);
    }

    file.write(
        static_cast<const char*>(serialized->data()),
        serialized->size()
    );
}

void TensorRTRuntime::set_device(int device_id) {
    default_device_ = device_id;
}

std::string TensorRTRuntime::get_version() const {
    int version = NV_TENSORRT_VERSION;
    int major = version / 1000;
    int minor = (version % 1000) / 100;
    int patch = version % 100;
    return std::to_string(major) + "." +
           std::to_string(minor) + "." +
           std::to_string(patch);
}

TensorRTEngine* TensorRTRuntime::create_engine_wrapper(
    std::shared_ptr<nvinfer1::ICudaEngine> engine,
    int device_id
) {
    auto wrapper = new TensorRTEngine();
    wrapper->engine = engine;
    wrapper->device_id = device_id;

    // Create execution context
    wrapper->context.reset(
        engine->createExecutionContext(),
        [](nvinfer1::IExecutionContext* c) { delete c; }
    );

    if (!wrapper->context) {
        delete wrapper;
        throw IVITError("Failed to create execution context");
    }

    // Create CUDA stream
    cudaStreamCreate(&wrapper->stream);

    // Get IO tensor info (TensorRT 10 API)
    int num_io_tensors = engine->getNbIOTensors();
    wrapper->bindings.resize(num_io_tensors);
    wrapper->device_buffers.resize(num_io_tensors);

    for (int i = 0; i < num_io_tensors; i++) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType nv_dtype = engine->getTensorDataType(name);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(name);

        TensorInfo info;
        info.name = name;
        info.dtype = nv_to_dtype(nv_dtype);

        // Convert dims to shape
        for (int d = 0; d < dims.nbDims; d++) {
            info.shape.push_back(dims.d[d]);
        }

        // Determine layout
        if (dims.nbDims == 4) {
            info.layout = Layout::NCHW;
        } else if (dims.nbDims == 2) {
            info.layout = Layout::NC;
        } else {
            info.layout = Layout::Unknown;
        }

        // Allocate device buffer
        size_t buffer_size = info.byte_size();
        cudaMalloc(&wrapper->device_buffers[i], buffer_size);
        wrapper->bindings[i] = wrapper->device_buffers[i];

        // Set tensor address in context
        wrapper->context->setTensorAddress(name, wrapper->device_buffers[i]);

        // Store info
        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            wrapper->input_infos.push_back(info);
        } else {
            wrapper->output_infos.push_back(info);
        }
    }

    return wrapper;
}

DataType TensorRTRuntime::nv_to_dtype(nvinfer1::DataType nv_type) const {
    switch (nv_type) {
        case nvinfer1::DataType::kFLOAT: return DataType::Float32;
        case nvinfer1::DataType::kHALF: return DataType::Float16;
        case nvinfer1::DataType::kINT8: return DataType::Int8;
        case nvinfer1::DataType::kINT32: return DataType::Int32;
        case nvinfer1::DataType::kBOOL: return DataType::Bool;
        default: return DataType::Unknown;
    }
}

nvinfer1::DataType TensorRTRuntime::dtype_to_nv(DataType dtype) const {
    switch (dtype) {
        case DataType::Float32: return nvinfer1::DataType::kFLOAT;
        case DataType::Float16: return nvinfer1::DataType::kHALF;
        case DataType::Int8: return nvinfer1::DataType::kINT8;
        case DataType::Int32: return nvinfer1::DataType::kINT32;
        case DataType::Bool: return nvinfer1::DataType::kBOOL;
        default: return nvinfer1::DataType::kFLOAT;
    }
}

TensorRTRuntime& get_tensorrt_runtime() {
    static TensorRTRuntime instance;
    return instance;
}

} // namespace ivit

#endif // IVIT_HAS_TENSORRT
