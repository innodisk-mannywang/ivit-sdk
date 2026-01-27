/**
 * @file model.hpp
 * @brief Model class definition
 */

#ifndef IVIT_CORE_MODEL_HPP
#define IVIT_CORE_MODEL_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/tensor.hpp"
#include "ivit/core/result.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <future>
#include <functional>

namespace ivit {

// Forward declaration
class IRuntime;

/**
 * @brief Model class for inference
 *
 * Represents a loaded model that can perform inference.
 * This is the main interface users interact with for inference.
 */
class Model {
public:
    Model() = default;
    virtual ~Model() = default;

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Allow moving
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get model name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get task type
     */
    virtual TaskType task() const = 0;

    /**
     * @brief Get device ID
     */
    virtual std::string device() const = 0;

    /**
     * @brief Get backend type
     */
    virtual BackendType backend() const = 0;

    /**
     * @brief Get input tensor information
     */
    virtual std::vector<TensorInfo> input_info() const = 0;

    /**
     * @brief Get output tensor information
     */
    virtual std::vector<TensorInfo> output_info() const = 0;

    /**
     * @brief Get model memory usage in bytes
     */
    virtual size_t memory_usage() const = 0;

    /**
     * @brief Get class labels (if available)
     */
    virtual std::vector<std::string> labels() const = 0;

    // ========================================================================
    // Inference
    // ========================================================================

    /**
     * @brief Run inference on a single image
     *
     * @param image Input image (BGR format)
     * @param config Inference configuration
     * @return Inference results
     */
    virtual Results predict(
        const cv::Mat& image,
        const InferConfig& config = InferConfig{}
    ) = 0;

    /**
     * @brief Run inference on an image file
     *
     * @param image_path Path to image file
     * @param config Inference configuration
     * @return Inference results
     */
    Results predict(
        const std::string& image_path,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Run batch inference
     *
     * @param images Vector of input images
     * @param config Inference configuration
     * @return Vector of inference results
     */
    virtual std::vector<Results> predict_batch(
        const std::vector<cv::Mat>& images,
        const InferConfig& config = InferConfig{}
    ) = 0;

    // ========================================================================
    // Advanced
    // ========================================================================

    /**
     * @brief Run raw inference (for advanced users)
     *
     * @param inputs Input tensors
     * @return Output tensors
     */
    virtual std::map<std::string, Tensor> infer_raw(
        const std::map<std::string, Tensor>& inputs
    ) = 0;

    /**
     * @brief Warmup the model
     *
     * @param iterations Number of warmup iterations
     */
    virtual void warmup(int iterations = 3) = 0;

    // ========================================================================
    // Async Inference
    // ========================================================================

    /**
     * @brief Run inference asynchronously
     *
     * @param image Input image (BGR format)
     * @param config Inference configuration
     * @return Future containing inference results
     */
    virtual std::future<Results> predict_async(
        const cv::Mat& image,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Run batch inference asynchronously
     *
     * @param images Vector of input images
     * @param config Inference configuration
     * @return Future containing vector of inference results
     */
    virtual std::future<std::vector<Results>> predict_batch_async(
        const std::vector<cv::Mat>& images,
        const InferConfig& config = InferConfig{}
    );

    /**
     * @brief Submit inference with callback
     *
     * Fire-and-forget style inference that invokes callback when complete.
     *
     * @param image Input image
     * @param callback Function called with results
     * @param config Inference configuration
     */
    virtual void submit_inference(
        const cv::Mat& image,
        std::function<void(Results)> callback,
        const InferConfig& config = InferConfig{}
    );
};

/**
 * @brief Model manager for loading and managing models
 */
class ModelManager {
public:
    static ModelManager& instance();

    /**
     * @brief Load a model
     *
     * @param path Model path or Model Zoo name
     * @param config Load configuration
     * @return Loaded model
     */
    std::shared_ptr<Model> load(
        const std::string& path,
        const LoadConfig& config = LoadConfig{}
    );

    /**
     * @brief Convert model format
     *
     * @param src_path Source model path
     * @param dst_path Destination model path
     * @param target_backend Target backend
     * @param precision Target precision
     */
    void convert(
        const std::string& src_path,
        const std::string& dst_path,
        BackendType target_backend,
        Precision precision = Precision::FP32
    );

    /**
     * @brief Clear model cache
     */
    void clear_cache();

    /**
     * @brief Set cache directory
     */
    void set_cache_dir(const std::string& path);

private:
    ModelManager() = default;
    ~ModelManager() = default;

    mutable std::mutex mutex_;           ///< Mutex for thread-safe cache access
    std::string cache_dir_;
    std::map<std::string, std::shared_ptr<Model>> cache_;
};

} // namespace ivit

#endif // IVIT_CORE_MODEL_HPP
