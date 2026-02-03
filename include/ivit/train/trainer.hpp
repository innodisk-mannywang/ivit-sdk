/**
 * @file trainer.hpp
 * @brief Trainer class for transfer learning with LibTorch
 */

#ifndef IVIT_TRAIN_TRAINER_HPP
#define IVIT_TRAIN_TRAINER_HPP

#include <string>
#include <vector>
#include <memory>
#include <map>

#include "ivit/train/config.hpp"
#include "ivit/train/dataset.hpp"
#include "ivit/train/callbacks.hpp"

#ifdef IVIT_HAS_TORCH
#include <torch/torch.h>
#endif

namespace ivit {
namespace train {

/**
 * @brief Trainer for fine-tuning models with transfer learning
 *
 * Supports classification and detection models with a simple API.
 * Uses LibTorch (PyTorch C++ API) for training.
 *
 * @example
 * ```cpp
 * auto dataset = std::make_shared<ImageFolderDataset>("./data");
 *
 * TrainerConfig config;
 * config.epochs = 20;
 * config.learning_rate = 0.001f;
 * config.device = "cuda:0";
 *
 * Trainer trainer("resnet50", dataset, config);
 * auto history = trainer.fit({
 *     std::make_shared<EarlyStopping>("val_loss", 5),
 *     std::make_shared<ModelCheckpoint>("best.pt"),
 * });
 *
 * auto metrics = trainer.evaluate();
 * trainer.export_model("model.onnx");
 * ```
 */
class Trainer {
public:
    /**
     * @brief Create trainer with model name and dataset
     *
     * @param model Model name (e.g., "resnet50", "efficientnet_b0") or path to checkpoint
     * @param dataset Training dataset
     * @param config Training configuration
     * @param val_dataset Validation dataset (optional, uses train_split if not provided)
     */
    Trainer(
        const std::string& model,
        std::shared_ptr<IDataset> dataset,
        const TrainerConfig& config = TrainerConfig{},
        std::shared_ptr<IDataset> val_dataset = nullptr
    );

#ifdef IVIT_HAS_TORCH
    /**
     * @brief Create trainer with existing PyTorch model
     *
     * @param model Pre-initialized torch::nn::Module
     * @param dataset Training dataset
     * @param config Training configuration
     * @param val_dataset Validation dataset (optional)
     */
    Trainer(
        torch::nn::AnyModule model,
        std::shared_ptr<IDataset> dataset,
        const TrainerConfig& config = TrainerConfig{},
        std::shared_ptr<IDataset> val_dataset = nullptr
    );
#endif

    ~Trainer();

    // Prevent copying
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    // Allow moving
    Trainer(Trainer&&) noexcept;
    Trainer& operator=(Trainer&&) noexcept;

    /**
     * @brief Train the model
     *
     * @param callbacks List of training callbacks
     * @return Training history (metrics per epoch)
     */
    std::vector<TrainingMetrics> fit(
        const std::vector<std::shared_ptr<ITrainingCallback>>& callbacks = {}
    );

    /**
     * @brief Evaluate model on a dataset
     *
     * @param dataset Dataset to evaluate on (default: validation set)
     * @return Evaluation metrics
     */
    TrainingMetrics evaluate(std::shared_ptr<IDataset> dataset = nullptr);

    /**
     * @brief Export trained model to deployment format
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
     * @brief Load a checkpoint
     *
     * @param path Path to checkpoint file
     */
    void load_checkpoint(const std::string& path);

    /**
     * @brief Save a checkpoint
     *
     * @param path Path to save checkpoint
     */
    void save_checkpoint(const std::string& path);

    /**
     * @brief Unfreeze backbone layers for full fine-tuning
     *
     * @param num_layers Number of layers to unfreeze from the end (-1 for all)
     */
    void unfreeze_backbone(int num_layers = -1);

    /**
     * @brief Freeze backbone layers
     */
    void freeze_backbone();

    /**
     * @brief Set learning rate
     */
    void set_learning_rate(float lr);

    /**
     * @brief Stop training (can be called from callbacks)
     */
    void stop_training() { stop_training_ = true; }

    /**
     * @brief Check if training should stop
     */
    bool should_stop() const { return stop_training_; }

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get training configuration
     */
    const TrainerConfig& config() const { return config_; }

    /**
     * @brief Get training history
     */
    const std::vector<TrainingMetrics>& history() const { return history_; }

    /**
     * @brief Get current epoch (0-indexed)
     */
    int current_epoch() const { return current_epoch_; }

    /**
     * @brief Get total number of epochs
     */
    int epochs() const { return config_.epochs; }

    /**
     * @brief Get batches per epoch
     */
    int batches_per_epoch() const { return batches_per_epoch_; }

    /**
     * @brief Get current learning rate
     */
    float learning_rate() const;

    /**
     * @brief Get number of classes
     */
    size_t num_classes() const;

    /**
     * @brief Get class names
     */
    std::vector<std::string> class_names() const;

    /**
     * @brief Get device string
     */
    std::string device() const { return config_.device; }

    /**
     * @brief Get model name
     */
    std::string model_name() const { return model_name_; }

    /**
     * @brief Check if using CUDA
     */
    bool is_cuda() const;

    // ========================================================================
    // Static Methods
    // ========================================================================

    /**
     * @brief List available pretrained models
     */
    static std::vector<std::string> available_models();

    /**
     * @brief Check if LibTorch is available
     */
    static bool is_torch_available();

    /**
     * @brief Check if CUDA is available for training
     */
    static bool is_cuda_available();

    /**
     * @brief Get LibTorch version
     */
    static std::string torch_version();

private:
    void init_torch();
    void init_model(const std::string& model);
    void init_optimizer();
    void init_dataloaders();
    void modify_classifier(int num_classes);
    void freeze_backbone_layers();

    // Training loop helpers
    TrainingMetrics train_epoch(int epoch, const std::vector<std::shared_ptr<ITrainingCallback>>& callbacks);
    TrainingMetrics validate_epoch();

    // Model and training state
    std::string model_name_;
    TrainerConfig config_;
    std::shared_ptr<IDataset> dataset_;
    std::shared_ptr<IDataset> val_dataset_;
    std::vector<TrainingMetrics> history_;

    int current_epoch_ = 0;
    int batches_per_epoch_ = 0;
    bool stop_training_ = false;

#ifdef IVIT_HAS_TORCH
    torch::Device device_{torch::kCPU};
    torch::nn::AnyModule model_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    // DataLoader will be created per-epoch to allow shuffling
#endif

    // Implementation details (pimpl for ABI stability)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_TRAINER_HPP
