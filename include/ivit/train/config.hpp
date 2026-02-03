/**
 * @file config.hpp
 * @brief Configuration structures for training module
 */

#ifndef IVIT_TRAIN_CONFIG_HPP
#define IVIT_TRAIN_CONFIG_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace ivit {
namespace train {

/**
 * @brief Training configuration
 */
struct TrainerConfig {
    int epochs = 10;                        ///< Number of training epochs
    float learning_rate = 0.001f;           ///< Learning rate
    int batch_size = 32;                    ///< Batch size
    std::string optimizer = "adam";         ///< Optimizer type: "adam", "sgd", "adamw"
    std::string device = "cuda:0";          ///< Training device
    bool freeze_backbone = true;            ///< Freeze backbone for transfer learning
    int num_workers = 4;                    ///< Number of data loader workers
    float weight_decay = 0.0f;              ///< Weight decay (L2 regularization)
    float momentum = 0.9f;                  ///< Momentum for SGD
    bool mixed_precision = false;           ///< Enable mixed precision training
    int log_interval = 10;                  ///< Log every N batches
    std::string checkpoint_dir = "";        ///< Directory to save checkpoints
    int seed = 42;                          ///< Random seed for reproducibility
};

/**
 * @brief Model export options
 */
struct ExportOptions {
    std::string format = "onnx";            ///< Export format: "onnx", "torchscript", "openvino", "tensorrt"
    std::string optimize_for = "";          ///< Target hardware: "intel_cpu", "intel_npu", "nvidia_gpu"
    std::string quantize = "";              ///< Quantization mode: "", "fp16", "int8"
    std::vector<int64_t> input_shape = {1, 3, 224, 224};  ///< Input shape (NCHW)
    int opset_version = 17;                 ///< ONNX opset version
    bool dynamic_batch = true;              ///< Enable dynamic batch size
    std::string calibration_data = "";      ///< Path to calibration data for INT8
};

/**
 * @brief Training metrics
 */
struct TrainingMetrics {
    int epoch = 0;                          ///< Current epoch
    float loss = 0.0f;                      ///< Training loss
    float accuracy = 0.0f;                  ///< Training accuracy
    float val_loss = 0.0f;                  ///< Validation loss
    float val_accuracy = 0.0f;              ///< Validation accuracy
    float learning_rate = 0.0f;             ///< Current learning rate
    double elapsed_seconds = 0.0;           ///< Time elapsed for this epoch
};

/**
 * @brief Dataset split configuration
 */
struct SplitConfig {
    float train_ratio = 0.8f;               ///< Training set ratio
    float val_ratio = 0.2f;                 ///< Validation set ratio
    int seed = 42;                          ///< Random seed for splitting
};

/**
 * @brief Augmentation configuration
 */
struct AugmentationConfig {
    int target_size = 224;                  ///< Target image size
    bool random_flip = true;                ///< Enable random horizontal flip
    float flip_probability = 0.5f;          ///< Flip probability
    bool color_jitter = true;               ///< Enable color jittering
    float brightness = 0.2f;                ///< Brightness adjustment range
    float contrast = 0.2f;                  ///< Contrast adjustment range
    float saturation = 0.2f;                ///< Saturation adjustment range
    float hue = 0.1f;                       ///< Hue adjustment range
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};  ///< Normalization mean (ImageNet)
    std::vector<float> std = {0.229f, 0.224f, 0.225f};   ///< Normalization std (ImageNet)
};

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_CONFIG_HPP
