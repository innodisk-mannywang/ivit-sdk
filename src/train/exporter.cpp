/**
 * @file exporter.cpp
 * @brief Model exporter implementation
 */

#include "ivit/train/exporter.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

#ifdef IVIT_HAS_TORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace ivit {
namespace train {

namespace fs = std::filesystem;

// ============================================================================
// Implementation Details
// ============================================================================

struct ModelExporter::Impl {
#ifdef IVIT_HAS_TORCH
    torch::jit::Module jit_model;
    bool has_model = false;
#endif
};

// ============================================================================
// Construction
// ============================================================================

#ifdef IVIT_HAS_TORCH
ModelExporter::ModelExporter(torch::nn::AnyModule model, torch::Device device)
    : device_(device)
    , impl_(std::make_unique<Impl>())
{
    // Note: AnyModule is tricky to trace, we'll handle this case specially
}
#endif

ModelExporter::ModelExporter(const std::string& checkpoint_path)
    : checkpoint_path_(checkpoint_path)
    , impl_(std::make_unique<Impl>())
{
#ifdef IVIT_HAS_TORCH
    if (!fs::exists(checkpoint_path)) {
        throw std::runtime_error("Checkpoint not found: " + checkpoint_path);
    }

    try {
        impl_->jit_model = torch::jit::load(checkpoint_path);
        impl_->has_model = true;
        device_ = torch::kCPU;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load checkpoint: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

ModelExporter::~ModelExporter() = default;

// ============================================================================
// Export Methods
// ============================================================================

std::string ModelExporter::export_model(const std::string& path, const ExportOptions& options) {
    std::string format = options.format;
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);

    if (format == "onnx") {
        return export_onnx(path, options.input_shape, options.opset_version, options.dynamic_batch);
    } else if (format == "torchscript" || format == "pt") {
        return export_torchscript(path, options.input_shape);
    } else if (format == "openvino") {
        std::vector<cv::Mat> calib_data;  // Empty for now
        return export_openvino(path, options.input_shape, options.quantize, calib_data);
    } else if (format == "tensorrt") {
        return export_tensorrt(path, options.input_shape, options.quantize);
    }

    throw std::invalid_argument("Unsupported export format: " + format);
}

std::string ModelExporter::export_onnx(
    const std::string& path,
    const std::vector<int64_t>& input_shape,
    int opset_version,
    bool dynamic_batch
) {
#ifdef IVIT_HAS_TORCH
    // Create parent directory
    fs::path dir = fs::path(path).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    std::cout << "[ModelExporter] Exporting to ONNX: " << path << std::endl;

    // For ONNX export, we first need to save as TorchScript
    // then convert using external tools
    std::string ts_path = path;
    size_t pos = ts_path.find(".onnx");
    if (pos != std::string::npos) {
        ts_path.replace(pos, 5, "_temp.pt");
    } else {
        ts_path += "_temp.pt";
    }

    // Export TorchScript first
    export_torchscript(ts_path, input_shape);

    std::cout << "[ModelExporter] Note: For full ONNX export, use Python:" << std::endl;
    std::cout << "  import torch" << std::endl;
    std::cout << "  model = torch.jit.load('" << ts_path << "')" << std::endl;
    std::cout << "  torch.onnx.export(model, torch.randn(" << input_shape[0];
    for (size_t i = 1; i < input_shape.size(); ++i) {
        std::cout << ", " << input_shape[i];
    }
    std::cout << "), '" << path << "')" << std::endl;

    // Save class labels
    save_labels(path);

    return ts_path;
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

std::string ModelExporter::export_torchscript(
    const std::string& path,
    const std::vector<int64_t>& input_shape
) {
#ifdef IVIT_HAS_TORCH
    // Create parent directory
    fs::path dir = fs::path(path).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    std::cout << "[ModelExporter] Exporting to TorchScript: " << path << std::endl;

    if (impl_->has_model) {
        // We already have a JIT model, just save it
        impl_->jit_model.save(path);
        std::cout << "[ModelExporter] TorchScript export complete: " << path << std::endl;
    } else {
        throw std::runtime_error("No model available to export");
    }

    // Save class labels
    save_labels(path);

    return path;
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

std::string ModelExporter::export_openvino(
    const std::string& path,
    const std::vector<int64_t>& input_shape,
    const std::string& quantize,
    const std::vector<cv::Mat>& calibration_data
) {
#ifdef IVIT_HAS_TORCH
    // First export to ONNX
    std::string onnx_path = fs::path(path).replace_extension(".onnx").string();
    export_onnx(onnx_path, input_shape, 17, true);

    std::cout << "[ModelExporter] Converting to OpenVINO IR: " << path << std::endl;

    std::cout << "[ModelExporter] To convert to OpenVINO, use:" << std::endl;
    std::cout << "  mo --input_model " << onnx_path << " --output_dir "
              << fs::path(path).parent_path().string() << std::endl;

    if (quantize == "fp16") {
        std::cout << "  Add: --compress_to_fp16" << std::endl;
    } else if (quantize == "int8") {
        std::cout << "  For INT8, use OpenVINO NNCF toolkit" << std::endl;
    }

    return onnx_path;
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

std::string ModelExporter::export_tensorrt(
    const std::string& path,
    const std::vector<int64_t>& input_shape,
    const std::string& quantize
) {
#ifdef IVIT_HAS_TORCH
    // First export to ONNX
    std::string onnx_path = fs::path(path).replace_extension(".onnx").string();
    export_onnx(onnx_path, input_shape, 17, true);

    std::cout << "[ModelExporter] Converting to TensorRT: " << path << std::endl;

    std::cout << "[ModelExporter] To convert to TensorRT, use:" << std::endl;
    std::cout << "  trtexec --onnx=" << onnx_path
              << " --saveEngine=" << fs::path(path).replace_extension(".engine").string();

    if (quantize == "fp16") {
        std::cout << " --fp16";
    } else if (quantize == "int8") {
        std::cout << " --int8";
    }
    std::cout << std::endl;

    return onnx_path;
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

std::string ModelExporter::quantize_onnx_fp16(const std::string& onnx_path) {
    std::string output_path = onnx_path;
    size_t pos = output_path.rfind(".onnx");
    if (pos != std::string::npos) {
        output_path.insert(pos, "_fp16");
    } else {
        output_path += "_fp16.onnx";
    }

    std::cout << "[ModelExporter] FP16 quantization requires onnxconverter-common:" << std::endl;
    std::cout << "  python -c \"from onnxconverter_common import float16; "
              << "import onnx; model = onnx.load('" << onnx_path << "'); "
              << "model_fp16 = float16.convert_float_to_float16(model); "
              << "onnx.save(model_fp16, '" << output_path << "')\"" << std::endl;

    return output_path;
}

void ModelExporter::set_class_names(const std::vector<std::string>& class_names) {
    class_names_ = class_names;
}

void ModelExporter::save_labels(const std::string& model_path) {
    if (class_names_.empty()) return;

    std::string labels_path = fs::path(model_path).replace_extension(".txt").string();
    std::ofstream file(labels_path);
    for (const auto& name : class_names_) {
        file << name << "\n";
    }
    std::cout << "[ModelExporter] Saved labels to " << labels_path << std::endl;
}

// ============================================================================
// Convenience Function
// ============================================================================

std::string export_model(
    const std::string& model_or_checkpoint,
    const std::string& path,
    const ExportOptions& options
) {
    ModelExporter exporter(model_or_checkpoint);
    return exporter.export_model(path, options);
}

} // namespace train
} // namespace ivit
