/**
 * @file classifier.cpp
 * @brief Image classifier implementation
 */

#include "ivit/vision/classifier.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <filesystem>

namespace ivit {
namespace vision {

// ============================================================================
// Default ImageNet labels (first 10 for demo, full list would be loaded)
// ============================================================================
static const std::vector<std::string> IMAGENET_LABELS = {
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich"
    // ... 1000 classes total
};

// ============================================================================
// Classifier implementation
// ============================================================================

Classifier::Classifier(
    const std::string& model_path,
    const std::string& device,
    const LoadConfig& config
) {
    // Create load config with classification task hint
    LoadConfig load_config = config;
    load_config.device = device;
    load_config.task = "classification";

    // Load model
    model_ = ModelManager::instance().load(model_path, load_config);

    // Get input size from model
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            // NCHW format
            input_size_ = cv::Size(
                static_cast<int>(shape[3]),  // width
                static_cast<int>(shape[2])   // height
            );
        } else {
            input_size_ = cv::Size(224, 224);  // Default
        }
    } else {
        input_size_ = cv::Size(224, 224);
    }

    // Set default normalization (ImageNet)
    mean_ = {0.485f, 0.456f, 0.406f};
    std_ = {0.229f, 0.224f, 0.225f};

    // Try to load labels
    labels_ = model_->labels();
    if (labels_.empty()) {
        // Try to load from file
        namespace fs = std::filesystem;
        fs::path model_dir = fs::path(model_path).parent_path();
        fs::path labels_path = model_dir / "labels.txt";

        if (fs::exists(labels_path)) {
            std::ifstream file(labels_path);
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    labels_.push_back(line);
                }
            }
        }

        // Fallback to default labels
        if (labels_.empty()) {
            labels_ = IMAGENET_LABELS;
        }
    }
}

Classifier::Classifier(std::shared_ptr<Model> model)
    : model_(model)
{
    // Get input size from model
    auto input_info = model_->input_info();
    if (!input_info.empty()) {
        const auto& shape = input_info[0].shape;
        if (shape.size() >= 4) {
            input_size_ = cv::Size(
                static_cast<int>(shape[3]),
                static_cast<int>(shape[2])
            );
        } else {
            input_size_ = cv::Size(224, 224);
        }
    } else {
        input_size_ = cv::Size(224, 224);
    }

    mean_ = {0.485f, 0.456f, 0.406f};
    std_ = {0.229f, 0.224f, 0.225f};
    labels_ = model_->labels();
}

Results Classifier::predict(const cv::Mat& image, int top_k) {
    Results results;
    results.image_size = image.size();

    auto start = std::chrono::high_resolution_clock::now();

    // Preprocess
    cv::Mat processed = preprocess(image);

    // Create input tensor
    auto input_info = model_->input_info();
    if (input_info.empty()) {
        throw InferenceError("Model has no inputs");
    }

    const auto& info = input_info[0];
    Tensor input(info.shape, DataType::Float32);
    input.set_name(info.name);

    // Copy preprocessed data to tensor
    std::memcpy(input.data(), processed.data, input.byte_size());

    // Run inference
    std::map<std::string, Tensor> inputs;
    inputs[info.name] = std::move(input);

    auto outputs = model_->infer_raw(inputs);

    // Postprocess
    results = postprocess(outputs, top_k);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    results.inference_time_ms = duration.count() / 1000.0f;
    results.device_used = model_->device();
    results.image_size = image.size();

    return results;
}

Results Classifier::predict(const std::string& image_path, int top_k) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, top_k);
}

std::vector<Results> Classifier::predict_batch(
    const std::vector<cv::Mat>& images,
    int top_k
) {
    std::vector<Results> results_list;
    results_list.reserve(images.size());

    // TODO: Implement true batch inference
    // For now, process images sequentially
    for (const auto& image : images) {
        results_list.push_back(predict(image, top_k));
    }

    return results_list;
}

cv::Mat Classifier::preprocess(const cv::Mat& image) {
    cv::Mat processed;

    // Resize to input size
    cv::resize(image, processed, input_size_);

    // Convert BGR to RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // Convert to float [0, 1]
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);

    // Apply normalization (ImageNet mean/std)
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    // Merge and convert to NCHW format
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Create NCHW tensor data
    int h = normalized.rows;
    int w = normalized.cols;
    int c = normalized.channels();

    cv::Mat nchw(1, c * h * w, CV_32F);
    float* ptr = nchw.ptr<float>();

    // Convert HWC to CHW
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                ptr[ch * h * w + y * w + x] = normalized.at<cv::Vec3f>(y, x)[ch];
            }
        }
    }

    return nchw;
}

Results Classifier::postprocess(
    const std::map<std::string, Tensor>& outputs,
    int top_k
) {
    Results results;

    if (outputs.empty()) {
        return results;
    }

    // Get first output tensor
    const Tensor& output = outputs.begin()->second;
    const float* data = output.data_ptr<float>();
    int64_t num_classes = output.numel();

    // Apply softmax
    std::vector<float> probs(num_classes);
    float max_val = *std::max_element(data, data + num_classes);
    float sum = 0.0f;

    for (int64_t i = 0; i < num_classes; i++) {
        probs[i] = std::exp(data[i] - max_val);
        sum += probs[i];
    }

    for (int64_t i = 0; i < num_classes; i++) {
        probs[i] /= sum;
    }

    // Find top-k predictions
    std::vector<std::pair<float, int>> scores;
    scores.reserve(num_classes);

    for (int64_t i = 0; i < num_classes; i++) {
        scores.emplace_back(probs[i], static_cast<int>(i));
    }

    // Partial sort to get top-k
    int k = std::min(top_k, static_cast<int>(num_classes));
    std::partial_sort(
        scores.begin(),
        scores.begin() + k,
        scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Create classification results
    for (int i = 0; i < k; i++) {
        auto [score, class_id] = scores[i];

        ClassificationResult cls;
        cls.class_id = class_id;
        cls.score = score;

        if (class_id < static_cast<int>(labels_.size())) {
            cls.label = labels_[class_id];
        } else {
            cls.label = "class_" + std::to_string(class_id);
        }

        results.classifications.push_back(cls);
    }

    return results;
}

} // namespace vision
} // namespace ivit
