/**
 * @file model.cpp
 * @brief Model class implementation
 */

#include "ivit/core/model.hpp"
#include "ivit/runtime/runtime.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace ivit {

// ============================================================================
// InferenceModel - Concrete model implementation using runtime backends
// ============================================================================

class InferenceModel : public Model {
public:
    InferenceModel(
        std::shared_ptr<IRuntime> runtime,
        void* handle,
        const std::string& name,
        TaskType task,
        const std::string& device
    )
        : runtime_(runtime)
        , handle_(handle)
        , name_(name)
        , task_(task)
        , device_(device)
    {
        input_info_ = runtime_->get_input_info(handle_);
        output_info_ = runtime_->get_output_info(handle_);
    }

    ~InferenceModel() override {
        if (runtime_ && handle_) {
            runtime_->unload_model(handle_);
        }
    }

    // Properties
    std::string name() const override { return name_; }
    TaskType task() const override { return task_; }
    std::string device() const override { return device_; }
    BackendType backend() const override { return runtime_->type(); }
    std::vector<TensorInfo> input_info() const override { return input_info_; }
    std::vector<TensorInfo> output_info() const override { return output_info_; }

    size_t memory_usage() const override {
        // Estimate based on tensor sizes
        size_t total = 0;
        for (const auto& info : input_info_) {
            total += info.byte_size();
        }
        for (const auto& info : output_info_) {
            total += info.byte_size();
        }
        return total;
    }

    std::vector<std::string> labels() const override { return labels_; }

    void set_labels(const std::vector<std::string>& labels) {
        labels_ = labels;
    }

    // Inference
    Results predict(
        const cv::Mat& image,
        const InferConfig& config
    ) override {
        Results results;
        results.image_size = image.size();
        results.device_used = device_;

        auto start = std::chrono::high_resolution_clock::now();

        // Preprocess image
        auto inputs = preprocess(image);

        // Run inference
        auto outputs = runtime_->infer(handle_, inputs);

        // Postprocess results
        postprocess(outputs, image.size(), config, results);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        results.inference_time_ms = duration.count() / 1000.0f;

        return results;
    }

    std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        const InferConfig& config
    ) override {
        std::vector<Results> results_list;
        results_list.reserve(images.size());

        // For now, process images sequentially
        // TODO: Implement true batch inference
        for (const auto& image : images) {
            results_list.push_back(predict(image, config));
        }

        return results_list;
    }

    std::map<std::string, Tensor> infer_raw(
        const std::map<std::string, Tensor>& inputs
    ) override {
        return runtime_->infer(handle_, inputs);
    }

    void warmup(int iterations) override {
        if (input_info_.empty()) return;

        // Create dummy input
        const auto& info = input_info_[0];
        Tensor dummy(info.shape, info.dtype);
        dummy.set_name(info.name);

        std::map<std::string, Tensor> inputs;
        inputs[info.name] = std::move(dummy);

        for (int i = 0; i < iterations; i++) {
            runtime_->infer(handle_, inputs);
        }
    }

private:
    std::map<std::string, Tensor> preprocess(const cv::Mat& image) {
        if (input_info_.empty()) {
            throw InferenceError("Model has no inputs");
        }

        const auto& info = input_info_[0];
        const auto& shape = info.shape;

        // Assume NCHW layout
        int64_t batch = shape.size() > 0 ? shape[0] : 1;
        int64_t channels = shape.size() > 1 ? shape[1] : 3;
        int64_t height = shape.size() > 2 ? shape[2] : 224;
        int64_t width = shape.size() > 3 ? shape[3] : 224;

        // Resize image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height));

        // Convert to float and normalize
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

        // Convert BGR to RGB if needed
        if (channels == 3 && float_img.channels() == 3) {
            cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);
        }

        // Create tensor
        Tensor input(shape, DataType::Float32);
        input.set_name(info.name);
        input.set_layout(Layout::NCHW);

        // Copy data in NCHW format
        float* ptr = input.data_ptr<float>();
        std::vector<cv::Mat> channels_vec;
        cv::split(float_img, channels_vec);

        for (int c = 0; c < channels; c++) {
            cv::Mat channel = channels_vec[c];
            std::memcpy(
                ptr + c * height * width,
                channel.data,
                height * width * sizeof(float)
            );
        }

        std::map<std::string, Tensor> inputs;
        inputs[info.name] = std::move(input);
        return inputs;
    }

    void postprocess(
        const std::map<std::string, Tensor>& outputs,
        cv::Size image_size,
        const InferConfig& config,
        Results& results
    ) {
        // Route to appropriate postprocessor based on task type
        switch (task_) {
            case TaskType::Classification:
                postprocess_classification(outputs, config, results);
                break;
            case TaskType::Detection:
                postprocess_detection(outputs, image_size, config, results);
                break;
            case TaskType::Segmentation:
                postprocess_segmentation(outputs, image_size, config, results);
                break;
            default:
                // Just store raw outputs
                break;
        }
    }

    void postprocess_classification(
        const std::map<std::string, Tensor>& outputs,
        const InferConfig& config,
        Results& results
    ) {
        if (outputs.empty()) return;

        // Get first output (assuming single output)
        const Tensor& output = outputs.begin()->second;
        const float* data = output.data_ptr<float>();
        int64_t num_classes = output.numel();

        // Find top-k predictions
        std::vector<std::pair<float, int>> scores;
        scores.reserve(num_classes);

        for (int i = 0; i < num_classes; i++) {
            scores.emplace_back(data[i], i);
        }

        // Sort by score descending
        std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // Apply softmax normalization if needed
        float max_score = scores[0].first;
        float sum = 0;
        for (auto& [score, idx] : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (auto& [score, idx] : scores) {
            score /= sum;
        }

        // Store results
        for (size_t i = 0; i < std::min(scores.size(), size_t(10)); i++) {
            auto [score, class_id] = scores[i];

            if (score < config.conf_threshold) break;

            ClassificationResult cls;
            cls.class_id = class_id;
            cls.score = score;
            cls.label = (class_id < labels_.size()) ?
                        labels_[class_id] :
                        "class_" + std::to_string(class_id);

            results.classifications.push_back(cls);
        }
    }

    void postprocess_detection(
        const std::map<std::string, Tensor>& outputs,
        cv::Size image_size,
        const InferConfig& config,
        Results& results
    ) {
        if (outputs.empty()) return;

        // This is a simplified YOLO-style postprocessor
        // Real implementation would need model-specific handling

        const Tensor& output = outputs.begin()->second;
        const float* data = output.data_ptr<float>();
        const auto& shape = output.shape();

        // YOLO output typically: [batch, num_anchors, 5+num_classes] or similar
        // This is a simplified version for demonstration

        int num_predictions = (shape.size() > 1) ? shape[1] : shape[0];
        int prediction_size = (shape.size() > 2) ? shape[2] : 6;  // x,y,w,h,conf,class

        std::vector<Detection> detections;

        for (int i = 0; i < num_predictions; i++) {
            const float* pred = data + i * prediction_size;

            float confidence = pred[4];
            if (confidence < config.conf_threshold) continue;

            // Find best class
            int best_class = 0;
            float best_class_score = 0;
            for (int c = 5; c < prediction_size; c++) {
                if (pred[c] > best_class_score) {
                    best_class_score = pred[c];
                    best_class = c - 5;
                }
            }

            float final_score = confidence * best_class_score;
            if (final_score < config.conf_threshold) continue;

            // Check class filter
            if (!config.classes.empty()) {
                bool found = false;
                for (int cls : config.classes) {
                    if (cls == best_class) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;
            }

            // Convert from center format to corner format
            float cx = pred[0] * image_size.width;
            float cy = pred[1] * image_size.height;
            float w = pred[2] * image_size.width;
            float h = pred[3] * image_size.height;

            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;

            Detection det;
            det.bbox = BBox(x1, y1, x2, y2);
            det.class_id = best_class;
            det.confidence = final_score;
            det.label = (best_class < labels_.size()) ?
                        labels_[best_class] :
                        "class_" + std::to_string(best_class);

            detections.push_back(det);
        }

        // Apply NMS
        results.detections = apply_nms(detections, config.iou_threshold);

        // Limit to max detections
        if (results.detections.size() > config.max_detections) {
            results.detections.resize(config.max_detections);
        }
    }

    void postprocess_segmentation(
        const std::map<std::string, Tensor>& outputs,
        cv::Size image_size,
        const InferConfig& config,
        Results& results
    ) {
        if (outputs.empty()) return;

        const Tensor& output = outputs.begin()->second;
        const auto& shape = output.shape();

        // Assume output shape: [batch, num_classes, height, width]
        int num_classes = (shape.size() > 1) ? shape[1] : 1;
        int height = (shape.size() > 2) ? shape[2] : image_size.height;
        int width = (shape.size() > 3) ? shape[3] : image_size.width;

        // Create segmentation mask (argmax over classes)
        cv::Mat mask(height, width, CV_32S);
        const float* data = output.data_ptr<float>();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int best_class = 0;
                float best_score = -std::numeric_limits<float>::max();

                for (int c = 0; c < num_classes; c++) {
                    float score = data[c * height * width + y * width + x];
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }

                mask.at<int>(y, x) = best_class;
            }
        }

        // Resize to original image size
        if (mask.size() != image_size) {
            cv::resize(mask, mask, image_size, 0, 0, cv::INTER_NEAREST);
        }

        results.segmentation_mask = mask;
    }

    std::vector<Detection> apply_nms(
        std::vector<Detection>& detections,
        float iou_threshold
    ) {
        // Sort by confidence
        std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });

        std::vector<Detection> result;
        std::vector<bool> suppressed(detections.size(), false);

        for (size_t i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;

            result.push_back(detections[i]);

            for (size_t j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;

                // Only suppress same class
                if (detections[i].class_id != detections[j].class_id) continue;

                float iou = detections[i].bbox.iou(detections[j].bbox);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    std::shared_ptr<IRuntime> runtime_;
    void* handle_;
    std::string name_;
    TaskType task_;
    std::string device_;
    std::vector<TensorInfo> input_info_;
    std::vector<TensorInfo> output_info_;
    std::vector<std::string> labels_;
};

// ============================================================================
// Model path-based predict
// ============================================================================

Results Model::predict(
    const std::string& image_path,
    const InferConfig& config
) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw IVITError("Failed to load image: " + image_path);
    }
    return predict(image, config);
}

// ============================================================================
// ModelManager implementation
// ============================================================================

ModelManager& ModelManager::instance() {
    static ModelManager instance;
    return instance;
}

std::shared_ptr<Model> ModelManager::load(
    const std::string& path,
    const LoadConfig& config
) {
    // Check cache
    std::string cache_key = path + "_" + config.device + "_" + config.backend;
    auto it = cache_.find(cache_key);
    if (it != cache_.end() && config.use_cache) {
        return it->second;
    }

    // Get appropriate runtime
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_best_runtime(path, config.device);

    if (!runtime) {
        throw IVITError("No suitable runtime found for model: " + path);
    }

    // Create runtime config
    RuntimeConfig rt_config;
    rt_config.device = config.device;
    rt_config.num_threads = 0;  // Auto

    if (config.precision == "fp16") {
        rt_config.precision = Precision::FP16;
    } else if (config.precision == "int8") {
        rt_config.precision = Precision::INT8;
    } else {
        rt_config.precision = Precision::FP32;
    }

    rt_config.cache_dir = config.cache_dir.empty() ? cache_dir_ : config.cache_dir;

    // Load model
    void* handle = runtime->load_model(path, rt_config);

    // Determine task type
    TaskType task = TaskType::Unknown;
    if (config.task == "classification") {
        task = TaskType::Classification;
    } else if (config.task == "detection") {
        task = TaskType::Detection;
    } else if (config.task == "segmentation") {
        task = TaskType::Segmentation;
    }

    // Create model wrapper
    auto model = std::make_shared<InferenceModel>(
        runtime,
        handle,
        path,
        task,
        config.device
    );

    // Cache model
    if (config.use_cache) {
        cache_[cache_key] = model;
    }

    return model;
}

void ModelManager::convert(
    const std::string& src_path,
    const std::string& dst_path,
    BackendType target_backend,
    Precision precision
) {
    auto& factory = RuntimeFactory::instance();
    auto runtime = factory.get_runtime(target_backend);

    if (!runtime) {
        throw IVITError("Target runtime not available");
    }

    RuntimeConfig config;
    config.precision = precision;

    runtime->convert_model(src_path, dst_path, config);
}

void ModelManager::clear_cache() {
    cache_.clear();
}

void ModelManager::set_cache_dir(const std::string& path) {
    cache_dir_ = path;
}

} // namespace ivit
