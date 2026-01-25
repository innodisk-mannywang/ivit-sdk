/**
 * @file device_manager.cpp
 * @brief Device manager implementation
 */

#include "ivit/core/device.hpp"
#include "ivit/runtime/runtime.hpp"
#include <algorithm>
#include <mutex>
#include <filesystem>
#include <cstdlib>
#include <iostream>

#ifdef IVIT_HAS_OPENVINO
#include <openvino/openvino.hpp>
#endif

#ifdef IVIT_HAS_TENSORRT
#include <cuda_runtime.h>
#endif

namespace ivit {

// ============================================================================
// DeviceManager implementation
// ============================================================================

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    discover_devices();
}

std::vector<DeviceInfo> DeviceManager::list_devices() {
    if (!initialized_) {
        discover_devices();
    }
    return devices_;
}

DeviceInfo DeviceManager::get_device(const std::string& device_id) {
    auto devices = list_devices();

    for (const auto& dev : devices) {
        if (dev.id == device_id) {
            return dev;
        }
    }

    throw DeviceNotFoundError(device_id);
}

DeviceInfo DeviceManager::get_best_device(
    const std::string& task,
    const std::string& priority
) {
    auto devices = list_devices();

    if (devices.empty()) {
        throw DeviceError("No devices available");
    }

    // Priority order based on selection criteria
    std::vector<std::string> type_order;
    if (priority == "performance") {
        type_order = {"gpu", "npu", "vpu", "cpu"};
    } else if (priority == "efficiency") {
        type_order = {"npu", "vpu", "gpu", "cpu"};
    } else { // memory
        type_order = {"gpu", "cpu", "npu", "vpu"};
    }

    // Sort devices by priority
    std::sort(devices.begin(), devices.end(),
        [&type_order](const DeviceInfo& a, const DeviceInfo& b) {
            auto it_a = std::find(type_order.begin(), type_order.end(), a.type);
            auto it_b = std::find(type_order.begin(), type_order.end(), b.type);

            int idx_a = (it_a != type_order.end()) ?
                        std::distance(type_order.begin(), it_a) :
                        type_order.size();
            int idx_b = (it_b != type_order.end()) ?
                        std::distance(type_order.begin(), it_b) :
                        type_order.size();

            return idx_a < idx_b;
        });

    // Return first available device
    for (const auto& dev : devices) {
        if (dev.is_available) {
            return dev;
        }
    }

    return devices[0];
}

DeviceStatus DeviceManager::get_device_status(const std::string& device_id) {
    DeviceStatus status;
    status.id = device_id;

    // TODO: Implement actual status retrieval
    // This would query the specific backend for device statistics

    return status;
}

bool DeviceManager::supports_precision(
    const std::string& device_id,
    Precision precision
) {
    auto dev = get_device(device_id);

    for (const auto& p : dev.supported_precisions) {
        if (p == precision) {
            return true;
        }
    }

    return false;
}

bool DeviceManager::supports_format(
    const std::string& device_id,
    const std::string& format
) {
    // TODO: Implement format support checking
    return true;
}

void DeviceManager::refresh() {
    devices_.clear();
    initialized_ = false;
    discover_devices();
}

void DeviceManager::discover_devices() {
    devices_.clear();

    discover_openvino_devices();
    discover_tensorrt_devices();
    discover_snpe_devices();

    // Always add CPU fallback
    bool has_cpu = false;
    for (const auto& dev : devices_) {
        if (dev.type == "cpu") {
            has_cpu = true;
            break;
        }
    }

    if (!has_cpu) {
        DeviceInfo cpu;
        cpu.id = "cpu";
        cpu.name = "CPU (Fallback)";
        cpu.backend = "onnxruntime";
        cpu.type = "cpu";
        cpu.is_available = true;
        cpu.supported_precisions = {Precision::FP32};
        devices_.push_back(cpu);
    }

    initialized_ = true;
}

void DeviceManager::discover_openvino_devices() {
#ifdef IVIT_HAS_OPENVINO
    try {
        ov::Core core;
        auto available = core.get_available_devices();

        for (const auto& device_name : available) {
            DeviceInfo info;

            // Get full device name
            try {
                info.name = core.get_property(device_name, ov::device::full_name);
            } catch (...) {
                info.name = device_name;
            }

            // Determine device type and ID
            if (device_name.find("CPU") != std::string::npos) {
                info.id = "cpu";
                info.type = "cpu";
            } else if (device_name.find("GPU") != std::string::npos) {
                info.id = "gpu:0";
                info.type = "gpu";
            } else if (device_name.find("NPU") != std::string::npos) {
                info.id = "npu";
                info.type = "npu";
            } else if (device_name.find("MYRIAD") != std::string::npos ||
                       device_name.find("VPU") != std::string::npos) {
                info.id = "vpu";
                info.type = "vpu";
            } else {
                info.id = device_name;
                info.type = "unknown";
            }

            info.backend = "openvino";
            info.is_available = true;
            info.supported_precisions = {
                Precision::FP32,
                Precision::FP16,
                Precision::INT8
            };

            devices_.push_back(info);
        }
    } catch (...) {
        // OpenVINO not available or error
    }
#endif
}

void DeviceManager::discover_tensorrt_devices() {
#ifdef IVIT_HAS_TENSORRT
    try {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        if (err == cudaSuccess && device_count > 0) {
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
                info.supported_precisions = {
                    Precision::FP32,
                    Precision::FP16,
                    Precision::INT8
                };

                devices_.push_back(info);
            }
        }
    } catch (...) {
        // CUDA/TensorRT not available
    }
#endif
}

void DeviceManager::discover_snpe_devices() {
#ifdef IVIT_HAS_SNPE
    // SNPE device discovery
    // This would use SNPE API to detect available compute units
    // (CPU, GPU, DSP, HTP, etc.)
#endif
}

// ============================================================================
// Helper functions
// ============================================================================

std::pair<BackendType, std::string> parse_device_string(const std::string& device) {
    std::string dev_lower = device;
    std::transform(dev_lower.begin(), dev_lower.end(), dev_lower.begin(), ::tolower);

    if (dev_lower == "auto") {
        return {BackendType::Auto, ""};
    }

    if (dev_lower.rfind("cuda", 0) == 0) {
        // Extract device index if present
        if (dev_lower.length() > 5 && dev_lower[4] == ':') {
            return {BackendType::TensorRT, dev_lower.substr(5)};
        }
        return {BackendType::TensorRT, "0"};
    }

    if (dev_lower == "cpu" || dev_lower.rfind("cpu:", 0) == 0) {
        return {BackendType::OpenVINO, "CPU"};
    }

    if (dev_lower == "gpu" || dev_lower.rfind("gpu:", 0) == 0) {
        return {BackendType::OpenVINO, "GPU"};
    }

    if (dev_lower == "npu") {
        return {BackendType::OpenVINO, "NPU"};
    }

    if (dev_lower == "vpu" || dev_lower == "myriad") {
        return {BackendType::OpenVINO, "MYRIAD"};
    }

    if (dev_lower == "hexagon" || dev_lower == "dsp" || dev_lower == "htp") {
        return {BackendType::SNPE, dev_lower};
    }

    // Default to CPU
    return {BackendType::ONNXRuntime, "CPU"};
}

BackendType get_backend_for_device(const std::string& device) {
    return parse_device_string(device).first;
}

bool cuda_is_available() {
#ifdef IVIT_HAS_TENSORRT
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

int cuda_device_count() {
#ifdef IVIT_HAS_TENSORRT
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

bool openvino_is_available() {
#ifdef IVIT_HAS_OPENVINO
    try {
        ov::Core core;
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool snpe_is_available() {
#ifdef IVIT_HAS_SNPE
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Free functions (declared in ivit.hpp)
// ============================================================================

std::vector<DeviceInfo> list_devices() {
    return DeviceManager::instance().list_devices();
}

DeviceInfo get_best_device(
    const std::string& task,
    const std::string& priority
) {
    return DeviceManager::instance().get_best_device(task, priority);
}

// Global log level setting
static std::string g_log_level = "info";

void set_log_level(const std::string& level) {
    g_log_level = level;
    // TODO: Actually configure logging
}

// Global cache directory
static std::string g_cache_dir = "";

void set_cache_dir(const std::string& path) {
    g_cache_dir = path;
}

std::string get_cache_dir() {
    if (g_cache_dir.empty()) {
        // Default cache directory
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + "/.cache/ivit";
        }
        return "/tmp/ivit_cache";
    }
    return g_cache_dir;
}

void convert_model(
    const std::string& src_path,
    const std::string& dst_path,
    const std::string& device,
    const std::string& precision
) {
    namespace fs = std::filesystem;

    // Determine target backend from device
    BackendType backend = get_backend_for_device(device);

    if (backend == BackendType::Auto) {
        // Determine from output extension
        std::string ext = fs::path(dst_path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".engine" || ext == ".trt" || ext == ".plan") {
            backend = BackendType::TensorRT;
        } else if (ext == ".xml") {
            backend = BackendType::OpenVINO;
        } else {
            throw IVITError("Cannot determine target backend from extension: " + ext);
        }
    }

    // Create runtime config
    RuntimeConfig config;
    config.device = device;

    // Parse precision
    if (precision == "fp16" || precision == "FP16") {
        config.precision = Precision::FP16;
    } else if (precision == "int8" || precision == "INT8") {
        config.precision = Precision::INT8;
    } else {
        config.precision = Precision::FP32;
    }

    // Get runtime and convert
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_runtime(backend);

    if (!runtime) {
        throw IVITError("Backend not available: " + to_string(backend));
    }

    runtime->convert_model(src_path, dst_path, config);
}

void clear_cache(const std::string& cache_dir) {
    namespace fs = std::filesystem;

    std::string dir = cache_dir.empty() ? get_cache_dir() : cache_dir;

    if (fs::exists(dir)) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".engine" || ext == ".trt" || ext == ".blob") {
                    fs::remove(entry.path());
                }
            }
        }
    }
}

} // namespace ivit
