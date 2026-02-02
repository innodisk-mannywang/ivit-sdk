/**
 * @file model.hpp
 * @brief Model class definition
 */

#ifndef IVIT_CORE_MODEL_HPP
#define IVIT_CORE_MODEL_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/tensor.hpp"
#include "ivit/core/result.hpp"
#include "ivit/core/callback.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <future>
#include <functional>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>

namespace ivit {

// Forward declarations
class IRuntime;
struct OpenVINOConfig;
struct TensorRTConfig;
struct QNNConfig;
class VideoSource;

/**
 * @brief Model class for inference
 *
 * Represents a loaded model that can perform inference.
 * This is the main interface users interact with for inference.
 */
class Model {
public:
    Model() = default;
    virtual ~Model();

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Allow moving
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    // ========================================================================
    // Properties
    // ========================================================================

    virtual std::string name() const = 0;
    virtual TaskType task() const = 0;
    virtual std::string device() const = 0;
    virtual BackendType backend() const = 0;
    virtual std::vector<TensorInfo> input_info() const = 0;
    virtual std::vector<TensorInfo> output_info() const = 0;
    virtual size_t memory_usage() const = 0;
    virtual std::vector<std::string> labels() const = 0;

    // ========================================================================
    // Inference
    // ========================================================================

    /**
     * @brief Run inference on a single image (BGR format)
     */
    virtual Results predict(
        const cv::Mat& image,
        const InferConfig& config = InferConfig{}
    ) = 0;

    /**
     * @brief Run inference on an image file
     */
    Results predict(
        const std::string& image_path,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Run batch inference
     */
    virtual std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        const InferConfig& config = InferConfig{}
    ) = 0;

    // ========================================================================
    // Advanced Inference
    // ========================================================================

    /**
     * @brief Run raw tensor-in/tensor-out inference
     */
    virtual std::map<std::string, Tensor> infer_raw(
        const std::map<std::string, Tensor>& inputs
    ) = 0;

    /**
     * @brief Warmup the model
     */
    virtual void warmup(int iterations = 3) = 0;

    // ========================================================================
    // Async Inference
    // ========================================================================

    /**
     * @brief Run inference asynchronously
     */
    virtual std::future<Results> predict_async(
        const cv::Mat& image,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Run batch inference asynchronously
     */
    virtual std::future<std::vector<Results>> predict_batch_async(
        const std::vector<cv::Mat>& images,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Run concurrent inference with concurrency limit
     *
     * @param images Vector of input images
     * @param max_concurrent Maximum concurrent inferences
     * @param config Inference configuration
     * @return Vector of results in same order as inputs
     */
    virtual std::vector<Results> predict_concurrent(
        const std::vector<cv::Mat>& images,
        int max_concurrent = 4,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Submit inference with callback (fire-and-forget)
     */
    virtual void submit_inference(
        const cv::Mat& image,
        std::function<void(Results)> callback,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Shutdown async executor and wait for pending tasks
     */
    void shutdown_async();

    // ========================================================================
    // Video Streaming
    // ========================================================================

    /**
     * @brief Stream inference result iterator
     */
    struct StreamResult {
        Results results;
        cv::Mat frame;
        int frame_index = 0;
        double fps = 0.0;
        bool end_of_stream = false;
    };

    /**
     * @brief Stream iterator for video inference
     */
    class StreamIterator {
    public:
        StreamIterator() : model_(nullptr), source_(nullptr) {}
        StreamIterator(Model* model, VideoSource* source,
                       const InferConfig& config);

        StreamResult next();
        bool has_next() const;

    private:
        Model* model_;
        VideoSource* source_;
        InferConfig config_;
        int frame_index_ = 0;
        std::chrono::steady_clock::time_point start_time_;
    };

    /**
     * @brief Create a streaming inference iterator over a video source
     *
     * @param source Video file path, camera index string ("0"), or RTSP URL
     * @param config Inference configuration
     * @return StreamIterator for iterating over results
     */
    StreamIterator stream(
        const std::string& source,
        const InferConfig& config = InferConfig{}
    );

    // ========================================================================
    // Callback System
    // ========================================================================

    /**
     * @brief Register a callback for an event
     *
     * @param event Event name: "pre_process", "post_process", "infer_start",
     *              "infer_end", "batch_start", "batch_end",
     *              "stream_start", "stream_frame", "stream_end"
     * @param callback Function to call
     * @param priority Higher priority = called first
     * @return Callback ID for later removal
     */
    int on(
        const std::string& event,
        std::function<void(const CallbackContext&)> callback,
        int priority = 0
    );

    /**
     * @brief Remove a specific callback by ID
     *
     * @param event Event name
     * @param callback_id ID returned by on()
     * @return true if removed
     */
    bool remove_callback(const std::string& event, int callback_id);

    /**
     * @brief Remove all callbacks for an event
     *
     * @param event Event name
     * @return Number removed
     */
    int remove_all_callbacks(const std::string& event);

    /**
     * @brief Get the callback manager
     */
    CallbackManager& callback_manager() { return callbacks_; }

    // ========================================================================
    // Hardware Configuration
    // ========================================================================

    /**
     * @brief Configure OpenVINO-specific settings
     */
    virtual void configure_openvino(const OpenVINOConfig& config);

    /**
     * @brief Configure TensorRT-specific settings
     */
    virtual void configure_tensorrt(const TensorRTConfig& config);

    /**
     * @brief Configure QNN-specific settings
     */
    virtual void configure_qnn(const QNNConfig& config);

    // ========================================================================
    // TTA (Test-Time Augmentation)
    // ========================================================================

    /**
     * @brief Run inference with test-time augmentation
     *
     * @param image Input image
     * @param augmentations Augmentation names: "hflip", "vflip", "rotate90",
     *                      "rotate180", "rotate270", "scale_up", "scale_down"
     * @param config Inference configuration
     * @return Merged results from all augmentations
     */
    virtual Results predict_tta(
        const cv::Mat& image,
        const std::vector<std::string>& augmentations = {"original", "hflip"},
        const InferConfig& config = InferConfig{}
    );

protected:
    /**
     * @brief Trigger a callback event
     */
    void trigger_callback(
        CallbackEvent event,
        double latency_ms = 0.0,
        int batch_size = 1,
        int frame_index = -1,
        const Results* results = nullptr
    );

    CallbackManager callbacks_;

private:
    // Async executor
    void ensure_executor();

    std::unique_ptr<std::thread> executor_thread_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> executor_running_{false};
    std::unique_ptr<VideoSource> stream_source_;
};

/**
 * @brief Model manager for loading and managing models
 */
class ModelManager {
public:
    static ModelManager& instance();

    std::shared_ptr<Model> load(
        const std::string& path,
        const LoadConfig& config = LoadConfig{}
    );

    void convert(
        const std::string& src_path,
        const std::string& dst_path,
        BackendType target_backend,
        Precision precision = Precision::FP32
    );

    void clear_cache();
    void set_cache_dir(const std::string& path);

private:
    ModelManager() = default;
    ~ModelManager() = default;

    mutable std::mutex mutex_;
    std::string cache_dir_;
    std::map<std::string, std::shared_ptr<Model>> cache_;
};

} // namespace ivit

#endif // IVIT_CORE_MODEL_HPP
