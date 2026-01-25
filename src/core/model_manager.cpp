/**
 * @file model_manager.cpp
 * @brief Model manager implementation
 */

#include "ivit/core/model.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace ivit {

// ============================================================================
// Model base implementation
// ============================================================================

Results Model::predict(
    const std::string& image_path,
    const InferConfig& config
) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw InferenceError("Failed to load image: " + image_path);
    }
    return predict(image, config);
}

// ============================================================================
// ModelManager implementation
// ============================================================================

ModelManager& ModelManager::instance() {
    static ModelManager instance;
    return instance;
}

std::shared_ptr<Model> ModelManager::load(
    const std::string& path,
    const LoadConfig& config
) {
    // Check cache
    std::string cache_key = path + "_" + config.device + "_" + config.backend;
    if (config.use_cache && cache_.count(cache_key)) {
        return cache_[cache_key];
    }

    // Determine backend
    BackendType backend_type;
    if (config.backend == "auto" || config.backend.empty()) {
        backend_type = BackendType::Auto;
    } else if (config.backend == "openvino") {
        backend_type = BackendType::OpenVINO;
    } else if (config.backend == "tensorrt") {
        backend_type = BackendType::TensorRT;
    } else if (config.backend == "snpe") {
        backend_type = BackendType::SNPE;
    } else {
        backend_type = BackendType::ONNXRuntime;
    }

    // Get runtime
    auto& factory = RuntimeFactory::instance();
    std::shared_ptr<IRuntime> runtime;

    if (backend_type == BackendType::Auto) {
        runtime = factory.get_best_runtime(path, config.device);
    } else {
        runtime = factory.get_runtime(backend_type);
    }

    if (!runtime || !runtime->is_available()) {
        throw ModelLoadError("No available runtime for backend");
    }

    // Check file exists
    if (!fs::exists(path)) {
        throw ModelLoadError("Model file not found: " + path);
    }

    // Load model through runtime
    RuntimeConfig rt_config;
    rt_config.device = config.device;
    rt_config.cache_dir = config.cache_dir.empty() ? cache_dir_ : config.cache_dir;

    if (config.precision == "fp16") {
        rt_config.precision = Precision::FP16;
    } else if (config.precision == "int8") {
        rt_config.precision = Precision::INT8;
    }

    // Create model wrapper
    // Note: This would create a concrete Model subclass based on the runtime
    // For now, this is a placeholder
    // auto model = std::make_shared<RuntimeModel>(runtime, path, rt_config);

    // Cache and return
    // if (config.use_cache) {
    //     cache_[cache_key] = model;
    // }

    // return model;

    // Placeholder - actual implementation would return the created model
    throw ModelLoadError("Model loading not yet implemented");
}

void ModelManager::convert(
    const std::string& src_path,
    const std::string& dst_path,
    BackendType target_backend,
    Precision precision
) {
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_runtime(target_backend);

    if (!runtime) {
        throw ModelError("Target backend not available");
    }

    RuntimeConfig config;
    config.precision = precision;

    runtime->convert_model(src_path, dst_path, config);
}

void ModelManager::clear_cache() {
    cache_.clear();
}

void ModelManager::set_cache_dir(const std::string& path) {
    cache_dir_ = path;

    // Create directory if it doesn't exist
    if (!path.empty() && !fs::exists(path)) {
        fs::create_directories(path);
    }
}

} // namespace ivit
