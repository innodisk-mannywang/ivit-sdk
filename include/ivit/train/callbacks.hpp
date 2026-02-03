/**
 * @file callbacks.hpp
 * @brief Training callbacks for monitoring and controlling the training process
 */

#ifndef IVIT_TRAIN_CALLBACKS_HPP
#define IVIT_TRAIN_CALLBACKS_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <optional>
#include "ivit/train/config.hpp"

namespace ivit {
namespace train {

// Forward declarations
class Trainer;

/**
 * @brief Abstract base class for training callbacks
 *
 * Callbacks can hook into various stages of training:
 * - on_train_start: Called when training begins
 * - on_train_end: Called when training ends
 * - on_epoch_start: Called at the start of each epoch
 * - on_epoch_end: Called at the end of each epoch
 * - on_batch_start: Called before each batch
 * - on_batch_end: Called after each batch
 */
class ITrainingCallback {
public:
    virtual ~ITrainingCallback() = default;

    /**
     * @brief Called when training starts
     */
    virtual void on_train_start(Trainer& trainer) {}

    /**
     * @brief Called when training ends
     */
    virtual void on_train_end(Trainer& trainer) {}

    /**
     * @brief Called at the start of each epoch
     */
    virtual void on_epoch_start(Trainer& trainer, int epoch) {}

    /**
     * @brief Called at the end of each epoch
     *
     * @param trainer Trainer instance
     * @param epoch Current epoch (0-indexed)
     * @param metrics Metrics for this epoch
     */
    virtual void on_epoch_end(
        Trainer& trainer,
        int epoch,
        const TrainingMetrics& metrics
    ) {}

    /**
     * @brief Called before each batch
     */
    virtual void on_batch_start(Trainer& trainer, int batch_idx) {}

    /**
     * @brief Called after each batch
     *
     * @param trainer Trainer instance
     * @param batch_idx Batch index
     * @param loss Batch loss
     */
    virtual void on_batch_end(Trainer& trainer, int batch_idx, float loss) {}

    /**
     * @brief Get string representation
     */
    virtual std::string repr() const = 0;
};

/**
 * @brief Stop training when a monitored metric stops improving
 *
 * @example
 * ```cpp
 * auto early_stop = std::make_shared<EarlyStopping>("val_loss", 5, 0.0f, "min");
 * trainer.fit({early_stop});
 * ```
 */
class EarlyStopping : public ITrainingCallback {
public:
    /**
     * @brief Create early stopping callback
     *
     * @param monitor Metric to monitor (default: "val_loss")
     * @param patience Number of epochs with no improvement to wait
     * @param min_delta Minimum change to qualify as improvement
     * @param mode "min" or "max" (whether lower or higher is better)
     */
    EarlyStopping(
        const std::string& monitor = "val_loss",
        int patience = 5,
        float min_delta = 0.0f,
        const std::string& mode = "min"
    );

    void on_train_start(Trainer& trainer) override;
    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    std::string repr() const override;

    /**
     * @brief Get the epoch at which training was stopped
     */
    int stopped_epoch() const { return stopped_epoch_; }

    /**
     * @brief Check if training should stop
     */
    bool should_stop() const { return should_stop_; }

private:
    bool is_improvement(float current) const;
    float get_metric_value(const TrainingMetrics& metrics) const;

    std::string monitor_;
    int patience_;
    float min_delta_;
    std::string mode_;

    std::optional<float> best_value_;
    int counter_ = 0;
    int stopped_epoch_ = 0;
    bool should_stop_ = false;
};

/**
 * @brief Save model checkpoints during training
 *
 * @example
 * ```cpp
 * auto checkpoint = std::make_shared<ModelCheckpoint>(
 *     "checkpoints/model_{epoch:02d}.pt",
 *     "val_loss",
 *     true,   // save_best_only
 *     "min"
 * );
 * ```
 */
class ModelCheckpoint : public ITrainingCallback {
public:
    /**
     * @brief Create model checkpoint callback
     *
     * @param filepath Path template (can include {epoch}, {val_loss}, etc.)
     * @param monitor Metric to monitor for best model
     * @param save_best_only Only save when monitored metric improves
     * @param mode "min" or "max"
     */
    ModelCheckpoint(
        const std::string& filepath = "checkpoint.pt",
        const std::string& monitor = "val_loss",
        bool save_best_only = true,
        const std::string& mode = "min"
    );

    void on_train_start(Trainer& trainer) override;
    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    std::string repr() const override;

    /**
     * @brief Get path to best saved model
     */
    std::string best_path() const { return best_path_; }

private:
    bool is_improvement(float current) const;
    float get_metric_value(const TrainingMetrics& metrics) const;
    std::string format_filepath(int epoch, const TrainingMetrics& metrics) const;
    void save_checkpoint(Trainer& trainer, const std::string& filepath, int epoch, const TrainingMetrics& metrics);

    std::string filepath_;
    std::string monitor_;
    bool save_best_only_;
    std::string mode_;

    std::optional<float> best_value_;
    std::string best_path_;
};

/**
 * @brief Log training progress to console
 */
class ProgressLogger : public ITrainingCallback {
public:
    /**
     * @brief Create progress logger
     *
     * @param log_frequency Log every N batches
     */
    explicit ProgressLogger(int log_frequency = 10);

    void on_epoch_start(Trainer& trainer, int epoch) override;
    void on_batch_end(Trainer& trainer, int batch_idx, float loss) override;
    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    std::string repr() const override;

private:
    int log_frequency_;
    std::chrono::steady_clock::time_point epoch_start_time_;
    std::vector<float> batch_losses_;
};

/**
 * @brief Learning rate scheduler callback
 *
 * @example
 * ```cpp
 * // Step decay every 10 epochs
 * auto scheduler = std::make_shared<LRScheduler>("step", 10, 0.1f);
 *
 * // Cosine annealing
 * auto scheduler = std::make_shared<LRScheduler>("cosine", 100);
 * ```
 */
class LRScheduler : public ITrainingCallback {
public:
    /**
     * @brief Create LR scheduler
     *
     * @param scheduler_type Type: "step", "cosine", "plateau", "exponential"
     * @param step_size Steps for step/cosine schedulers
     * @param gamma Decay factor
     * @param patience Patience for plateau scheduler
     */
    LRScheduler(
        const std::string& scheduler_type = "step",
        int step_size = 10,
        float gamma = 0.1f,
        int patience = 5
    );

    void on_train_start(Trainer& trainer) override;
    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    std::string repr() const override;

    /**
     * @brief Get current learning rate
     */
    float current_lr() const { return current_lr_; }

private:
    void step(int epoch, float val_loss);
    void update_lr_step(int epoch);
    void update_lr_cosine(int epoch);
    void update_lr_plateau(float val_loss);
    void update_lr_exponential(int epoch);

    std::string scheduler_type_;
    int step_size_;
    float gamma_;
    int patience_;

    float base_lr_ = 0.0f;
    float current_lr_ = 0.0f;
    int total_epochs_ = 0;

    // For plateau scheduler
    std::optional<float> best_val_loss_;
    int plateau_counter_ = 0;
};

/**
 * @brief Log training metrics to TensorBoard
 */
class TensorBoardLogger : public ITrainingCallback {
public:
    /**
     * @brief Create TensorBoard logger
     *
     * @param log_dir Directory for TensorBoard logs
     */
    explicit TensorBoardLogger(const std::string& log_dir = "runs");

    void on_train_start(Trainer& trainer) override;
    void on_batch_end(Trainer& trainer, int batch_idx, float loss) override;
    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    void on_train_end(Trainer& trainer) override;
    std::string repr() const override;

private:
    std::string log_dir_;
    // Note: Actual TensorBoard writer will be handled via LibTorch's torch::tensorboard
    // or a simple file-based implementation for CSV logging
    bool initialized_ = false;
};

/**
 * @brief CSV logger for training metrics
 */
class CSVLogger : public ITrainingCallback {
public:
    /**
     * @brief Create CSV logger
     *
     * @param filepath Path to CSV file
     * @param append Append to existing file
     */
    explicit CSVLogger(const std::string& filepath = "training_log.csv", bool append = false);

    void on_train_start(Trainer& trainer) override;
    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    void on_train_end(Trainer& trainer) override;
    std::string repr() const override;

private:
    std::string filepath_;
    bool append_;
    bool header_written_ = false;
};

/**
 * @brief Lambda callback for custom logic
 */
class LambdaCallback : public ITrainingCallback {
public:
    using EpochEndFn = std::function<void(Trainer&, int, const TrainingMetrics&)>;
    using BatchEndFn = std::function<void(Trainer&, int, float)>;

    LambdaCallback(
        EpochEndFn on_epoch_end_fn = nullptr,
        BatchEndFn on_batch_end_fn = nullptr
    );

    void on_epoch_end(Trainer& trainer, int epoch, const TrainingMetrics& metrics) override;
    void on_batch_end(Trainer& trainer, int batch_idx, float loss) override;
    std::string repr() const override;

private:
    EpochEndFn on_epoch_end_fn_;
    BatchEndFn on_batch_end_fn_;
};

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_CALLBACKS_HPP
