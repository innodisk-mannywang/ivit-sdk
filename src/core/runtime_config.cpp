/**
 * @file runtime_config.cpp
 * @brief Runtime configuration implementations
 */

#include "ivit/core/runtime_config.hpp"

namespace ivit {

std::map<std::string, std::string> OpenVINOConfig::to_ov_config() const {
    std::map<std::string, std::string> config;

    config["PERFORMANCE_HINT"] = performance_mode;

    if (num_streams > 0) {
        config["NUM_STREAMS"] = std::to_string(num_streams);
    }

    if (inference_precision != "FP32") {
        config["INFERENCE_PRECISION_HINT"] = inference_precision;
    }

    if (enable_cpu_pinning) {
        config["ENABLE_CPU_PINNING"] = "YES";
    }

    if (num_threads > 0) {
        config["INFERENCE_NUM_THREADS"] = std::to_string(num_threads);
    }

    if (!cache_dir.empty()) {
        config["CACHE_DIR"] = cache_dir;
    }

    for (const auto& [key, value] : device_properties) {
        config[key] = value;
    }

    return config;
}

} // namespace ivit
