/**
 * @file train.hpp
 * @brief Convenience header for iVIT training module
 *
 * Include this header to get all training-related classes and functions.
 *
 * @example
 * ```cpp
 * #include <ivit/train/train.hpp>
 *
 * using namespace ivit::train;
 *
 * // Create dataset
 * auto dataset = std::make_shared<ImageFolderDataset>("./data");
 *
 * // Create transforms
 * auto transform = get_train_augmentation(224);
 *
 * // Create trainer
 * TrainerConfig config;
 * config.epochs = 20;
 * config.learning_rate = 0.001f;
 *
 * Trainer trainer("resnet50", dataset, config);
 *
 * // Train with callbacks
 * auto history = trainer.fit({
 *     std::make_shared<EarlyStopping>("val_loss", 5),
 *     std::make_shared<ModelCheckpoint>("checkpoints/best.pt"),
 *     std::make_shared<ProgressLogger>(10),
 * });
 *
 * // Export model
 * trainer.export_model("model.onnx");
 * ```
 */

#ifndef IVIT_TRAIN_TRAIN_HPP
#define IVIT_TRAIN_TRAIN_HPP

// Configuration
#include "ivit/train/config.hpp"

// Dataset
#include "ivit/train/dataset.hpp"

// Augmentation
#include "ivit/train/augmentation.hpp"

// Callbacks
#include "ivit/train/callbacks.hpp"

// Trainer
#include "ivit/train/trainer.hpp"

// Exporter
#include "ivit/train/exporter.hpp"

namespace ivit {
namespace train {

/**
 * @brief Check if training module is available (LibTorch compiled in)
 */
inline bool is_training_available() {
#ifdef IVIT_HAS_TORCH
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get training module version
 */
inline std::string training_version() {
    return "1.0.0";
}

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_TRAIN_HPP
