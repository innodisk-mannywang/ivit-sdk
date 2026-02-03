/**
 * @file exporter.hpp
 * @brief Model exporter for converting trained models to deployment formats
 */

#ifndef IVIT_TRAIN_EXPORTER_HPP
#define IVIT_TRAIN_EXPORTER_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "ivit/train/config.hpp"

#ifdef IVIT_HAS_TORCH
#include <torch/torch.h>
#endif

namespace ivit {
namespace train {

/**
 * @brief Export trained models to various deployment formats
 *
 * Supports:
 * - ONNX (cross-platform)
 * - TorchScript (PyTorch native)
 * - OpenVINO IR (Intel optimization)
 * - TensorRT Engine (NVIDIA optimization)
 *
 * @example
 * ```cpp
 * ModelExporter exporter(model, device);
 *
 * // Export to ONNX
 * exporter.export_onnx("model.onnx", {1, 3, 224, 224});
 *
 * // Export to TorchScript
 * exporter.export_torchscript("model.pt", {1, 3, 224, 224});
 *
 * // Export with quantization
 * ExportOptions options;
 * options.quantize = "fp16";
 * exporter.export("model.onnx", options);
 * ```
 */
class ModelExporter {
public:
#ifdef IVIT_HAS_TORCH
    /**
     * @brief Create exporter with PyTorch model
     *
     * @param model PyTorch model
     * @param device Device the model is on
     */
    ModelExporter(torch::nn::AnyModule model, torch::Device device);
#endif

    /**
     * @brief Create exporter from checkpoint file
     *
     * @param checkpoint_path Path to checkpoint file
     */
    explicit ModelExporter(const std::string& checkpoint_path);

    ~ModelExporter();

    /**
     * @brief Export model to specified format
     *
     * @param path Output path
     * @param options Export options
     * @return Path to exported model
     */
    std::string export_model(
        const std::string& path,
        const ExportOptions& options = ExportOptions{}
    );

    /**
     * @brief Export to ONNX format
     *
     * @param path Output path (.onnx)
     * @param input_shape Input tensor shape (NCHW)
     * @param opset_version ONNX opset version
     * @param dynamic_batch Enable dynamic batch size
     * @return Path to exported model
     */
    std::string export_onnx(
        const std::string& path,
        const std::vector<int64_t>& input_shape = {1, 3, 224, 224},
        int opset_version = 17,
        bool dynamic_batch = true
    );

    /**
     * @brief Export to TorchScript format
     *
     * @param path Output path (.pt)
     * @param input_shape Input tensor shape (NCHW)
     * @return Path to exported model
     */
    std::string export_torchscript(
        const std::string& path,
        const std::vector<int64_t>& input_shape = {1, 3, 224, 224}
    );

    /**
     * @brief Export to OpenVINO IR format
     *
     * First exports to ONNX, then converts using OpenVINO tools.
     *
     * @param path Output path (.xml)
     * @param input_shape Input tensor shape (NCHW)
     * @param quantize Quantization mode ("", "fp16", "int8")
     * @param calibration_data Calibration images for INT8 quantization
     * @return Path to exported model
     */
    std::string export_openvino(
        const std::string& path,
        const std::vector<int64_t>& input_shape = {1, 3, 224, 224},
        const std::string& quantize = "",
        const std::vector<cv::Mat>& calibration_data = {}
    );

    /**
     * @brief Export to TensorRT engine format
     *
     * First exports to ONNX, then converts using TensorRT.
     *
     * @param path Output path (.engine)
     * @param input_shape Input tensor shape (NCHW)
     * @param quantize Quantization mode ("", "fp16", "int8")
     * @return Path to exported model
     */
    std::string export_tensorrt(
        const std::string& path,
        const std::vector<int64_t>& input_shape = {1, 3, 224, 224},
        const std::string& quantize = ""
    );

    /**
     * @brief Apply FP16 quantization to ONNX model
     *
     * @param onnx_path Path to ONNX model
     * @return Path to quantized model
     */
    static std::string quantize_onnx_fp16(const std::string& onnx_path);

    /**
     * @brief Set class names for metadata
     */
    void set_class_names(const std::vector<std::string>& class_names);

    /**
     * @brief Get class names
     */
    const std::vector<std::string>& class_names() const { return class_names_; }

private:
    void save_labels(const std::string& model_path);

#ifdef IVIT_HAS_TORCH
    torch::nn::AnyModule model_;
    torch::Device device_{torch::kCPU};
#endif

    std::string checkpoint_path_;
    std::vector<std::string> class_names_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Convenience function to export a model
 *
 * @param model_or_checkpoint Model or path to checkpoint
 * @param path Output path
 * @param options Export options
 * @return Path to exported model
 */
std::string export_model(
    const std::string& model_or_checkpoint,
    const std::string& path,
    const ExportOptions& options = ExportOptions{}
);

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_EXPORTER_HPP
