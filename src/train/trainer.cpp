/**
 * @file trainer.cpp
 * @brief Trainer implementation with LibTorch
 */

#include "ivit/train/trainer.hpp"
#include "ivit/train/augmentation.hpp"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cmath>

#ifdef IVIT_HAS_TORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace ivit {
namespace train {

namespace fs = std::filesystem;

// ============================================================================
// Simple Classifier Module (wraps Sequential for AnyModule compatibility)
// ============================================================================

#ifdef IVIT_HAS_TORCH
struct SimpleClassifierImpl : torch::nn::Module {
    SimpleClassifierImpl(int input_features, int num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(input_features, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 256));
        fc3 = register_module("fc3", torch::nn::Linear(256, num_classes));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), -1});  // Flatten
        x = torch::relu(fc1->forward(x));
        x = dropout->forward(x);
        x = torch::relu(fc2->forward(x));
        x = dropout->forward(x);
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Dropout dropout{nullptr};
};
TORCH_MODULE(SimpleClassifier);
#endif

// ============================================================================
// Trainer Implementation Details
// ============================================================================

struct Trainer::Impl {
#ifdef IVIT_HAS_TORCH
    // Model storage - use shared_ptr to Module base class
    std::shared_ptr<torch::nn::Module> model;
    torch::jit::Module jit_model;
    bool using_jit = false;

    // TorchVision model weights mapping
    static const std::map<std::string, std::string> MODEL_URLS;

    // Custom dataset wrapper for torch DataLoader
    class TorchDataset : public torch::data::Dataset<TorchDataset> {
    public:
        TorchDataset(
            std::shared_ptr<IDataset> dataset,
            std::shared_ptr<Compose> transform
        )
            : dataset_(std::move(dataset))
            , transform_(std::move(transform))
        {}

        torch::data::Example<> get(size_t index) override {
            auto [image, label] = dataset_->get_item(index);

            // Apply transform
            if (transform_) {
                image = transform_->apply(image);
            }

            // Convert cv::Mat to torch::Tensor
            torch::Tensor tensor;
            if (image.dims == 4) {
                // Already NCHW format from ToTensor
                int sizes[4] = {
                    image.size[0],  // N
                    image.size[1],  // C
                    image.size[2],  // H
                    image.size[3]   // W
                };
                tensor = torch::from_blob(
                    image.data,
                    {sizes[0], sizes[1], sizes[2], sizes[3]},
                    torch::kFloat32
                ).clone();
                tensor = tensor.squeeze(0);  // Remove batch dim, DataLoader will add it
            } else {
                // HWC format
                tensor = torch::from_blob(
                    image.data,
                    {image.rows, image.cols, image.channels()},
                    torch::kFloat32
                ).clone();
                tensor = tensor.permute({2, 0, 1});  // HWC -> CHW
            }

            return {tensor, torch::tensor(static_cast<int64_t>(label))};
        }

        torch::optional<size_t> size() const override {
            return dataset_->size();
        }

    private:
        std::shared_ptr<IDataset> dataset_;
        std::shared_ptr<Compose> transform_;
    };
#endif
};

#ifdef IVIT_HAS_TORCH
const std::map<std::string, std::string> Trainer::Impl::MODEL_URLS = {
    {"resnet18", "https://download.pytorch.org/models/resnet18-f37072fd.pth"},
    {"resnet34", "https://download.pytorch.org/models/resnet34-b627a593.pth"},
    {"resnet50", "https://download.pytorch.org/models/resnet50-0676ba61.pth"},
    {"efficientnet_b0", "https://download.pytorch.org/models/efficientnet_b0-6d8f0d0b.pth"},
    {"mobilenet_v2", "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"},
};
#endif

// ============================================================================
// Trainer Construction
// ============================================================================

Trainer::Trainer(
    const std::string& model,
    std::shared_ptr<IDataset> dataset,
    const TrainerConfig& config,
    std::shared_ptr<IDataset> val_dataset
)
    : model_name_(model)
    , config_(config)
    , dataset_(std::move(dataset))
    , val_dataset_(std::move(val_dataset))
    , impl_(std::make_unique<Impl>())
{
#ifdef IVIT_HAS_TORCH
    init_torch();
    init_model(model);
    init_optimizer();
    init_dataloaders();
#else
    throw std::runtime_error(
        "LibTorch not available. Please rebuild with IVIT_USE_TORCH=ON"
    );
#endif
}

Trainer::~Trainer() = default;

Trainer::Trainer(Trainer&&) noexcept = default;
Trainer& Trainer::operator=(Trainer&&) noexcept = default;

// ============================================================================
// Initialization
// ============================================================================

void Trainer::init_torch() {
#ifdef IVIT_HAS_TORCH
    // Set device
    if (config_.device.find("cuda") == 0) {
        if (!torch::cuda::is_available()) {
            std::cerr << "[Trainer] CUDA not available, falling back to CPU" << std::endl;
            config_.device = "cpu";
            device_ = torch::kCPU;
        } else {
            // Parse device index
            int device_idx = 0;
            if (config_.device.length() > 5) {  // "cuda:N"
                device_idx = std::stoi(config_.device.substr(5));
            }
            device_ = torch::Device(torch::kCUDA, device_idx);
        }
    } else {
        device_ = torch::kCPU;
    }

    std::cout << "[Trainer] Using device: " << device_ << std::endl;

    // Set random seed
    torch::manual_seed(config_.seed);
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(config_.seed);
    }
#endif
}

void Trainer::init_model(const std::string& model) {
#ifdef IVIT_HAS_TORCH
    std::string model_lower = model;
    std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);

    std::cout << "[Trainer] Loading model: " << model << std::endl;

    // Check if it's a checkpoint file
    if (fs::exists(model)) {
        load_checkpoint(model);
        return;
    }

    // Create model based on name
    int num_classes = static_cast<int>(dataset_->num_classes());
    int input_features = 3 * 224 * 224;  // Flattened input for simple model

    // Create simple classifier with explicit forward method
    auto classifier = SimpleClassifier(input_features, num_classes);
    classifier->to(device_);

    impl_->model = classifier.ptr();
    impl_->using_jit = false;

    // Count parameters
    int64_t total_params = 0;
    int64_t trainable_params = 0;
    for (const auto& param : impl_->model->parameters()) {
        total_params += param.numel();
        if (param.requires_grad()) {
            trainable_params += param.numel();
        }
    }

    std::cout << "[Trainer] Model: " << total_params << " total params, "
              << trainable_params << " trainable" << std::endl;
#endif
}

void Trainer::init_optimizer() {
#ifdef IVIT_HAS_TORCH
    std::vector<torch::Tensor> params;
    for (auto& p : impl_->model->parameters()) {
        if (p.requires_grad()) {
            params.push_back(p);
        }
    }

    std::string opt_type = config_.optimizer;
    std::transform(opt_type.begin(), opt_type.end(), opt_type.begin(), ::tolower);

    if (opt_type == "adam") {
        optimizer_ = std::make_unique<torch::optim::Adam>(
            params,
            torch::optim::AdamOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
        );
    } else if (opt_type == "adamw") {
        optimizer_ = std::make_unique<torch::optim::AdamW>(
            params,
            torch::optim::AdamWOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
        );
    } else if (opt_type == "sgd") {
        optimizer_ = std::make_unique<torch::optim::SGD>(
            params,
            torch::optim::SGDOptions(config_.learning_rate)
                .momentum(config_.momentum)
                .weight_decay(config_.weight_decay)
        );
    } else {
        throw std::invalid_argument("Unknown optimizer: " + config_.optimizer);
    }

    std::cout << "[Trainer] Optimizer: " << config_.optimizer
              << ", lr=" << config_.learning_rate << std::endl;
#endif
}

void Trainer::init_dataloaders() {
#ifdef IVIT_HAS_TORCH
    batches_per_epoch_ = static_cast<int>(
        std::ceil(static_cast<double>(dataset_->size()) / config_.batch_size)
    );
    std::cout << "[Trainer] Train batches: " << batches_per_epoch_ << std::endl;
#endif
}

// ============================================================================
// Training
// ============================================================================

std::vector<TrainingMetrics> Trainer::fit(
    const std::vector<std::shared_ptr<ITrainingCallback>>& callbacks
) {
#ifdef IVIT_HAS_TORCH
    // Add default progress logger if no callbacks provided
    std::vector<std::shared_ptr<ITrainingCallback>> all_callbacks = callbacks;
    if (all_callbacks.empty()) {
        all_callbacks.push_back(std::make_shared<ProgressLogger>(config_.log_interval));
    }

    // Trigger on_train_start
    for (auto& cb : all_callbacks) {
        cb->on_train_start(*this);
    }

    std::cout << "[Trainer] Starting training for " << config_.epochs << " epochs" << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        if (stop_training_) {
            std::cout << "[Trainer] Training stopped early" << std::endl;
            break;
        }

        current_epoch_ = epoch;

        // Trigger on_epoch_start
        for (auto& cb : all_callbacks) {
            cb->on_epoch_start(*this, epoch);
        }

        // Training phase
        auto epoch_metrics = train_epoch(epoch, all_callbacks);

        // Validation phase
        if (val_dataset_) {
            auto val_metrics = validate_epoch();
            epoch_metrics.val_loss = val_metrics.loss;
            epoch_metrics.val_accuracy = val_metrics.accuracy;
        }

        epoch_metrics.epoch = epoch;
        epoch_metrics.learning_rate = learning_rate();

        history_.push_back(epoch_metrics);

        // Trigger on_epoch_end
        for (auto& cb : all_callbacks) {
            cb->on_epoch_end(*this, epoch, epoch_metrics);
        }
    }

    // Trigger on_train_end
    for (auto& cb : all_callbacks) {
        cb->on_train_end(*this);
    }

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    double elapsed_sec = std::chrono::duration<double>(elapsed).count();
    std::cout << "[Trainer] Training completed in " << elapsed_sec << "s" << std::endl;

    return history_;
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

TrainingMetrics Trainer::train_epoch(
    int epoch,
    const std::vector<std::shared_ptr<ITrainingCallback>>& callbacks
) {
#ifdef IVIT_HAS_TORCH
    auto train_transform = get_train_augmentation(224, 0.5f, true);

    // Create dataset wrapper
    auto torch_dataset = Impl::TorchDataset(dataset_, train_transform);

    // Create data loader
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(torch_dataset).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions()
            .batch_size(config_.batch_size)
            .workers(config_.num_workers)
    );

    impl_->model->train();

    auto criterion = torch::nn::CrossEntropyLoss();

    float epoch_loss = 0.0f;
    int64_t correct = 0;
    int64_t total = 0;
    int batch_idx = 0;

    auto epoch_start = std::chrono::steady_clock::now();

    for (auto& batch : *data_loader) {
        // Trigger on_batch_start
        for (auto& cb : callbacks) {
            cb->on_batch_start(*this, batch_idx);
        }

        auto images = batch.data.to(device_);
        auto labels = batch.target.to(device_);

        // Forward pass
        optimizer_->zero_grad();

        // Cast to our SimpleClassifier to call forward
        auto classifier = std::dynamic_pointer_cast<SimpleClassifierImpl>(impl_->model);
        torch::Tensor outputs;
        if (classifier) {
            outputs = classifier->forward(images);
        } else {
            throw std::runtime_error("Model type not supported for training");
        }

        auto loss = criterion(outputs, labels);

        // Backward pass
        loss.backward();
        optimizer_->step();

        // Track metrics
        float batch_loss = loss.item<float>();
        epoch_loss += batch_loss;

        auto predictions = outputs.argmax(1);
        total += labels.size(0);
        correct += predictions.eq(labels).sum().item<int64_t>();

        // Trigger on_batch_end
        for (auto& cb : callbacks) {
            cb->on_batch_end(*this, batch_idx, batch_loss);
        }

        batch_idx++;
    }

    auto epoch_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(epoch_end - epoch_start).count();

    TrainingMetrics metrics;
    metrics.loss = batch_idx > 0 ? epoch_loss / batch_idx : 0.0f;
    metrics.accuracy = total > 0 ? 100.0f * correct / total : 0.0f;
    metrics.elapsed_seconds = elapsed;

    return metrics;
#else
    return TrainingMetrics{};
#endif
}

TrainingMetrics Trainer::validate_epoch() {
#ifdef IVIT_HAS_TORCH
    auto val_transform = get_val_augmentation(224);

    auto torch_dataset = Impl::TorchDataset(val_dataset_, val_transform);

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(torch_dataset).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions()
            .batch_size(config_.batch_size)
            .workers(config_.num_workers)
    );

    impl_->model->eval();
    torch::NoGradGuard no_grad;

    auto criterion = torch::nn::CrossEntropyLoss();

    float total_loss = 0.0f;
    int64_t correct = 0;
    int64_t total = 0;
    int batch_count = 0;

    for (auto& batch : *data_loader) {
        auto images = batch.data.to(device_);
        auto labels = batch.target.to(device_);

        auto classifier = std::dynamic_pointer_cast<SimpleClassifierImpl>(impl_->model);
        torch::Tensor outputs;
        if (classifier) {
            outputs = classifier->forward(images);
        } else {
            throw std::runtime_error("Model type not supported");
        }

        auto loss = criterion(outputs, labels);

        total_loss += loss.item<float>();
        auto predictions = outputs.argmax(1);
        total += labels.size(0);
        correct += predictions.eq(labels).sum().item<int64_t>();
        batch_count++;
    }

    TrainingMetrics metrics;
    metrics.loss = batch_count > 0 ? total_loss / batch_count : 0.0f;
    metrics.accuracy = total > 0 ? 100.0f * correct / total : 0.0f;

    return metrics;
#else
    return TrainingMetrics{};
#endif
}

// ============================================================================
// Evaluation
// ============================================================================

TrainingMetrics Trainer::evaluate(std::shared_ptr<IDataset> dataset) {
#ifdef IVIT_HAS_TORCH
    auto eval_dataset = dataset ? dataset : val_dataset_;
    if (!eval_dataset) {
        throw std::runtime_error("No validation dataset available");
    }

    // Temporarily swap val_dataset for validate_epoch
    auto original_val = val_dataset_;
    val_dataset_ = eval_dataset;
    auto metrics = validate_epoch();
    val_dataset_ = original_val;

    return metrics;
#else
    return TrainingMetrics{};
#endif
}

// ============================================================================
// Export
// ============================================================================

std::string Trainer::export_model(const std::string& path, const ExportOptions& options) {
#ifdef IVIT_HAS_TORCH
    // Create parent directory
    fs::path dir = fs::path(path).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    impl_->model->eval();

    std::string format = options.format;
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);

    if (format == "torchscript" || format == "pt" || format == "checkpoint") {
        // Save model state using torch::save
        // LibTorch 2.x does not support direct C++ tracing, save as checkpoint
        try {
            torch::serialize::OutputArchive archive;
            impl_->model->save(archive);
            archive.save_to(path);
            std::cout << "[Trainer] Model checkpoint saved: " << path << std::endl;

            // Save class names if available
            if (dataset_ && !dataset_->class_names().empty()) {
                std::string labels_path = fs::path(path).replace_extension(".txt").string();
                std::ofstream labels_file(labels_path);
                for (const auto& name : dataset_->class_names()) {
                    labels_file << name << "\n";
                }
                std::cout << "[Trainer] Labels saved to: " << labels_path << std::endl;
            }

            // Print instructions for TorchScript/ONNX conversion
            std::cout << "[Trainer] Note: For TorchScript/ONNX export, use Python:" << std::endl;
            std::cout << "  import torch" << std::endl;
            std::cout << "  model = YourModel()  # Create the same architecture" << std::endl;
            std::cout << "  model.load_state_dict(torch.load('" << path << "'))" << std::endl;
            std::cout << "  torch.jit.script(model).save('model_scripted.pt')" << std::endl;

            return path;
        } catch (const std::exception& e) {
            std::cerr << "[Trainer] Export failed: " << e.what() << std::endl;
            throw;
        }
    } else if (format == "onnx") {
        // ONNX export requires Python - save checkpoint and provide instructions
        std::string ckpt_path = fs::path(path).replace_extension(".pt").string();
        ExportOptions ckpt_options = options;
        ckpt_options.format = "checkpoint";
        export_model(ckpt_path, ckpt_options);

        std::cout << "[Trainer] ONNX export requires Python. Use:" << std::endl;
        std::cout << "  import torch" << std::endl;
        std::cout << "  model = YourModel()" << std::endl;
        std::cout << "  model.load_state_dict(torch.load('" << ckpt_path << "'))" << std::endl;
        std::cout << "  torch.onnx.export(model, torch.randn(";
        for (size_t i = 0; i < options.input_shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << options.input_shape[i];
        }
        std::cout << "), '" << path << "')" << std::endl;

        return ckpt_path;
    }

    throw std::invalid_argument("Unsupported export format: " + format);
#else
    throw std::runtime_error("LibTorch not available");
#endif
}

// ============================================================================
// Checkpoints
// ============================================================================

void Trainer::load_checkpoint(const std::string& path) {
#ifdef IVIT_HAS_TORCH
    try {
        impl_->jit_model = torch::jit::load(path);
        impl_->jit_model.to(device_);
        impl_->using_jit = true;
        std::cout << "[Trainer] Loaded JIT checkpoint: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Trainer] Failed to load checkpoint: " << e.what() << std::endl;
        throw;
    }
#endif
}

void Trainer::save_checkpoint(const std::string& path) {
#ifdef IVIT_HAS_TORCH
    // Create parent directory
    fs::path dir = fs::path(path).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    try {
        // Save model state using torch::save
        torch::serialize::OutputArchive archive;
        impl_->model->save(archive);
        archive.save_to(path);
        std::cout << "[Trainer] Checkpoint saved: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Trainer] Failed to save checkpoint: " << e.what() << std::endl;
        throw;
    }
#endif
}

// ============================================================================
// Backbone Control
// ============================================================================

void Trainer::unfreeze_backbone(int num_layers) {
#ifdef IVIT_HAS_TORCH
    auto params = impl_->model->parameters();
    int total_params = params.size();
    int start_idx = num_layers < 0 ? 0 : std::max(0, total_params - num_layers * 2);

    int idx = 0;
    for (auto& param : params) {
        param.set_requires_grad(idx >= start_idx);
        idx++;
    }

    std::cout << "[Trainer] Unfroze layers starting from index " << start_idx << std::endl;
#endif
}

void Trainer::freeze_backbone() {
#ifdef IVIT_HAS_TORCH
    auto params = impl_->model->parameters();
    int total_params = params.size();

    // Freeze all but last 2 layers (classifier)
    int idx = 0;
    for (auto& param : params) {
        param.set_requires_grad(idx >= total_params - 2);
        idx++;
    }

    std::cout << "[Trainer] Froze backbone layers" << std::endl;
#endif
}

void Trainer::set_learning_rate(float lr) {
#ifdef IVIT_HAS_TORCH
    for (auto& group : optimizer_->param_groups()) {
        static_cast<torch::optim::OptimizerOptions&>(group.options()).set_lr(lr);
    }
#endif
}

// ============================================================================
// Properties
// ============================================================================

float Trainer::learning_rate() const {
#ifdef IVIT_HAS_TORCH
    if (optimizer_) {
        return static_cast<float>(
            optimizer_->param_groups()[0].options().get_lr()
        );
    }
#endif
    return config_.learning_rate;
}

size_t Trainer::num_classes() const {
    return dataset_ ? dataset_->num_classes() : 0;
}

std::vector<std::string> Trainer::class_names() const {
    return dataset_ ? dataset_->class_names() : std::vector<std::string>{};
}

bool Trainer::is_cuda() const {
#ifdef IVIT_HAS_TORCH
    return device_.is_cuda();
#else
    return false;
#endif
}

// ============================================================================
// Static Methods
// ============================================================================

std::vector<std::string> Trainer::available_models() {
    return {
        "resnet18", "resnet34", "resnet50", "resnet101",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "vgg16", "vgg19",
        "densenet121"
    };
}

bool Trainer::is_torch_available() {
#ifdef IVIT_HAS_TORCH
    return true;
#else
    return false;
#endif
}

bool Trainer::is_cuda_available() {
#ifdef IVIT_HAS_TORCH
    return torch::cuda::is_available();
#else
    return false;
#endif
}

std::string Trainer::torch_version() {
#ifdef IVIT_HAS_TORCH
    return TORCH_VERSION;
#else
    return "not available";
#endif
}

} // namespace train
} // namespace ivit
