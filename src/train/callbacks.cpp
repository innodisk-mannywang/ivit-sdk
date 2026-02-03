/**
 * @file callbacks.cpp
 * @brief Training callbacks implementation
 */

#include "ivit/train/callbacks.hpp"
#include "ivit/train/trainer.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <filesystem>

namespace ivit {
namespace train {

namespace fs = std::filesystem;

// ============================================================================
// EarlyStopping
// ============================================================================

EarlyStopping::EarlyStopping(
    const std::string& monitor,
    int patience,
    float min_delta,
    const std::string& mode
)
    : monitor_(monitor)
    , patience_(patience)
    , min_delta_(min_delta)
    , mode_(mode)
{}

void EarlyStopping::on_train_start(Trainer& trainer) {
    best_value_.reset();
    counter_ = 0;
    should_stop_ = false;
}

void EarlyStopping::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    float current = get_metric_value(metrics);
    if (std::isnan(current)) {
        std::cerr << "[EarlyStopping] Warning: metric '" << monitor_
                  << "' not found or is NaN" << std::endl;
        return;
    }

    if (!best_value_.has_value()) {
        best_value_ = current;
        return;
    }

    if (is_improvement(current)) {
        best_value_ = current;
        counter_ = 0;
    } else {
        counter_++;
        if (counter_ >= patience_) {
            trainer.stop_training();
            stopped_epoch_ = epoch;
            should_stop_ = true;
            std::cout << "[EarlyStopping] Stopped at epoch " << epoch
                      << " (best " << monitor_ << ": " << best_value_.value() << ")"
                      << std::endl;
        }
    }
}

bool EarlyStopping::is_improvement(float current) const {
    if (mode_ == "min") {
        return current < best_value_.value() - min_delta_;
    }
    return current > best_value_.value() + min_delta_;
}

float EarlyStopping::get_metric_value(const TrainingMetrics& metrics) const {
    if (monitor_ == "loss") return metrics.loss;
    if (monitor_ == "accuracy") return metrics.accuracy;
    if (monitor_ == "val_loss") return metrics.val_loss;
    if (monitor_ == "val_accuracy") return metrics.val_accuracy;
    return std::nanf("");
}

std::string EarlyStopping::repr() const {
    return "EarlyStopping(monitor='" + monitor_ + "', patience=" + std::to_string(patience_) + ")";
}

// ============================================================================
// ModelCheckpoint
// ============================================================================

ModelCheckpoint::ModelCheckpoint(
    const std::string& filepath,
    const std::string& monitor,
    bool save_best_only,
    const std::string& mode
)
    : filepath_(filepath)
    , monitor_(monitor)
    , save_best_only_(save_best_only)
    , mode_(mode)
{}

void ModelCheckpoint::on_train_start(Trainer& trainer) {
    best_value_.reset();
    best_path_.clear();

    // Create directory if needed
    fs::path dir = fs::path(filepath_).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

void ModelCheckpoint::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    float current = get_metric_value(metrics);
    std::string filepath = format_filepath(epoch, metrics);

    if (save_best_only_) {
        if (std::isnan(current)) {
            std::cerr << "[ModelCheckpoint] Warning: metric '" << monitor_
                      << "' not found" << std::endl;
            return;
        }

        if (!best_value_.has_value() || is_improvement(current)) {
            best_value_ = current;
            best_path_ = filepath;
            save_checkpoint(trainer, filepath, epoch, metrics);
            std::cout << "[ModelCheckpoint] Saved best model to " << filepath << std::endl;
        }
    } else {
        save_checkpoint(trainer, filepath, epoch, metrics);
    }
}

bool ModelCheckpoint::is_improvement(float current) const {
    if (mode_ == "min") {
        return current < best_value_.value();
    }
    return current > best_value_.value();
}

float ModelCheckpoint::get_metric_value(const TrainingMetrics& metrics) const {
    if (monitor_ == "loss") return metrics.loss;
    if (monitor_ == "accuracy") return metrics.accuracy;
    if (monitor_ == "val_loss") return metrics.val_loss;
    if (monitor_ == "val_accuracy") return metrics.val_accuracy;
    return std::nanf("");
}

std::string ModelCheckpoint::format_filepath(int epoch, const TrainingMetrics& metrics) const {
    std::string result = filepath_;

    // Replace {epoch} placeholder
    size_t pos = result.find("{epoch");
    if (pos != std::string::npos) {
        size_t end = result.find("}", pos);
        std::string format_spec = result.substr(pos, end - pos + 1);
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << epoch;
        result.replace(pos, format_spec.length(), oss.str());
    }

    // Replace metric placeholders
    auto replace_metric = [&result](const std::string& name, float value) {
        size_t pos = result.find("{" + name);
        if (pos != std::string::npos) {
            size_t end = result.find("}", pos);
            std::string format_spec = result.substr(pos, end - pos + 1);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << value;
            result.replace(pos, format_spec.length(), oss.str());
        }
    };

    replace_metric("loss", metrics.loss);
    replace_metric("accuracy", metrics.accuracy);
    replace_metric("val_loss", metrics.val_loss);
    replace_metric("val_accuracy", metrics.val_accuracy);

    return result;
}

void ModelCheckpoint::save_checkpoint(
    Trainer& trainer,
    const std::string& filepath,
    int epoch,
    const TrainingMetrics& metrics
) {
    trainer.save_checkpoint(filepath);
}

std::string ModelCheckpoint::repr() const {
    return "ModelCheckpoint(filepath='" + filepath_ + "', monitor='" + monitor_ + "')";
}

// ============================================================================
// ProgressLogger
// ============================================================================

ProgressLogger::ProgressLogger(int log_frequency)
    : log_frequency_(log_frequency)
{}

void ProgressLogger::on_epoch_start(Trainer& trainer, int epoch) {
    epoch_start_time_ = std::chrono::steady_clock::now();
    batch_losses_.clear();
    std::cout << "Epoch " << (epoch + 1) << "/" << trainer.epochs() << std::endl;
}

void ProgressLogger::on_batch_end(Trainer& trainer, int batch_idx, float loss) {
    batch_losses_.push_back(loss);

    if ((batch_idx + 1) % log_frequency_ == 0) {
        // Calculate average loss of recent batches
        size_t start = batch_losses_.size() > static_cast<size_t>(log_frequency_)
                       ? batch_losses_.size() - log_frequency_ : 0;
        float avg_loss = std::accumulate(
            batch_losses_.begin() + start, batch_losses_.end(), 0.0f
        ) / (batch_losses_.size() - start);

        std::cout << "  Batch " << (batch_idx + 1) << ": loss=" << std::fixed
                  << std::setprecision(4) << avg_loss << std::endl;
    }
}

void ProgressLogger::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    auto elapsed = std::chrono::steady_clock::now() - epoch_start_time_;
    double elapsed_sec = std::chrono::duration<double>(elapsed).count();

    float avg_loss = batch_losses_.empty() ? 0.0f :
        std::accumulate(batch_losses_.begin(), batch_losses_.end(), 0.0f) / batch_losses_.size();

    std::cout << "Epoch " << (epoch + 1) << " completed in "
              << std::fixed << std::setprecision(1) << elapsed_sec << "s"
              << " - loss=" << std::setprecision(4) << avg_loss
              << ", accuracy=" << std::setprecision(2) << metrics.accuracy << "%";

    if (metrics.val_loss > 0) {
        std::cout << ", val_loss=" << std::setprecision(4) << metrics.val_loss
                  << ", val_accuracy=" << std::setprecision(2) << metrics.val_accuracy << "%";
    }
    std::cout << std::endl;
}

std::string ProgressLogger::repr() const {
    return "ProgressLogger(log_frequency=" + std::to_string(log_frequency_) + ")";
}

// ============================================================================
// LRScheduler
// ============================================================================

LRScheduler::LRScheduler(
    const std::string& scheduler_type,
    int step_size,
    float gamma,
    int patience
)
    : scheduler_type_(scheduler_type)
    , step_size_(step_size)
    , gamma_(gamma)
    , patience_(patience)
{}

void LRScheduler::on_train_start(Trainer& trainer) {
    base_lr_ = trainer.config().learning_rate;
    current_lr_ = base_lr_;
    total_epochs_ = trainer.epochs();
    best_val_loss_.reset();
    plateau_counter_ = 0;
}

void LRScheduler::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    step(epoch, metrics.val_loss);
    trainer.set_learning_rate(current_lr_);
}

void LRScheduler::step(int epoch, float val_loss) {
    if (scheduler_type_ == "step") {
        update_lr_step(epoch);
    } else if (scheduler_type_ == "cosine") {
        update_lr_cosine(epoch);
    } else if (scheduler_type_ == "plateau") {
        update_lr_plateau(val_loss);
    } else if (scheduler_type_ == "exponential") {
        update_lr_exponential(epoch);
    }
}

void LRScheduler::update_lr_step(int epoch) {
    if ((epoch + 1) % step_size_ == 0) {
        current_lr_ *= gamma_;
    }
}

void LRScheduler::update_lr_cosine(int epoch) {
    // Cosine annealing
    current_lr_ = base_lr_ * 0.5f * (1.0f + std::cos(M_PI * epoch / total_epochs_));
}

void LRScheduler::update_lr_plateau(float val_loss) {
    if (std::isnan(val_loss)) return;

    if (!best_val_loss_.has_value() || val_loss < best_val_loss_.value()) {
        best_val_loss_ = val_loss;
        plateau_counter_ = 0;
    } else {
        plateau_counter_++;
        if (plateau_counter_ >= patience_) {
            current_lr_ *= gamma_;
            plateau_counter_ = 0;
            std::cout << "[LRScheduler] Reducing learning rate to " << current_lr_ << std::endl;
        }
    }
}

void LRScheduler::update_lr_exponential(int epoch) {
    current_lr_ = base_lr_ * std::pow(gamma_, epoch);
}

std::string LRScheduler::repr() const {
    return "LRScheduler(type='" + scheduler_type_ + "')";
}

// ============================================================================
// TensorBoardLogger
// ============================================================================

TensorBoardLogger::TensorBoardLogger(const std::string& log_dir)
    : log_dir_(log_dir)
{}

void TensorBoardLogger::on_train_start(Trainer& trainer) {
    // Create log directory
    if (!fs::exists(log_dir_)) {
        fs::create_directories(log_dir_);
    }
    initialized_ = true;
    std::cout << "[TensorBoardLogger] Logging to " << log_dir_ << std::endl;
}

void TensorBoardLogger::on_batch_end(Trainer& trainer, int batch_idx, float loss) {
    // Note: Full TensorBoard integration would require SummaryWriter
    // For now, this is a placeholder that could be extended
}

void TensorBoardLogger::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    // Log metrics to a simple text file that can be parsed
    std::ofstream file(log_dir_ + "/scalars.txt", std::ios::app);
    file << "epoch=" << epoch
         << " loss=" << metrics.loss
         << " accuracy=" << metrics.accuracy
         << " val_loss=" << metrics.val_loss
         << " val_accuracy=" << metrics.val_accuracy
         << " lr=" << metrics.learning_rate
         << std::endl;
}

void TensorBoardLogger::on_train_end(Trainer& trainer) {
    // Close writer if needed
}

std::string TensorBoardLogger::repr() const {
    return "TensorBoardLogger(log_dir='" + log_dir_ + "')";
}

// ============================================================================
// CSVLogger
// ============================================================================

CSVLogger::CSVLogger(const std::string& filepath, bool append)
    : filepath_(filepath), append_(append)
{}

void CSVLogger::on_train_start(Trainer& trainer) {
    // Create directory if needed
    fs::path dir = fs::path(filepath_).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    // Write header if not appending or file doesn't exist
    if (!append_ || !fs::exists(filepath_)) {
        std::ofstream file(filepath_);
        file << "epoch,loss,accuracy,val_loss,val_accuracy,learning_rate,elapsed_seconds"
             << std::endl;
        header_written_ = true;
    }
}

void CSVLogger::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    std::ofstream file(filepath_, std::ios::app);
    file << epoch << ","
         << std::fixed << std::setprecision(6)
         << metrics.loss << ","
         << metrics.accuracy << ","
         << metrics.val_loss << ","
         << metrics.val_accuracy << ","
         << metrics.learning_rate << ","
         << metrics.elapsed_seconds
         << std::endl;
}

void CSVLogger::on_train_end(Trainer& trainer) {
    std::cout << "[CSVLogger] Training log saved to " << filepath_ << std::endl;
}

std::string CSVLogger::repr() const {
    return "CSVLogger(filepath='" + filepath_ + "')";
}

// ============================================================================
// LambdaCallback
// ============================================================================

LambdaCallback::LambdaCallback(
    EpochEndFn on_epoch_end_fn,
    BatchEndFn on_batch_end_fn
)
    : on_epoch_end_fn_(std::move(on_epoch_end_fn))
    , on_batch_end_fn_(std::move(on_batch_end_fn))
{}

void LambdaCallback::on_epoch_end(
    Trainer& trainer,
    int epoch,
    const TrainingMetrics& metrics
) {
    if (on_epoch_end_fn_) {
        on_epoch_end_fn_(trainer, epoch, metrics);
    }
}

void LambdaCallback::on_batch_end(Trainer& trainer, int batch_idx, float loss) {
    if (on_batch_end_fn_) {
        on_batch_end_fn_(trainer, batch_idx, loss);
    }
}

std::string LambdaCallback::repr() const {
    return "LambdaCallback()";
}

} // namespace train
} // namespace ivit
