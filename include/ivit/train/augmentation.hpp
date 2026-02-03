/**
 * @file augmentation.hpp
 * @brief Data augmentation transforms for training
 */

#ifndef IVIT_TRAIN_AUGMENTATION_HPP
#define IVIT_TRAIN_AUGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include "ivit/train/dataset.hpp"

namespace ivit {
namespace train {

/**
 * @brief Abstract base class for image transforms
 */
class ITransform {
public:
    virtual ~ITransform() = default;

    /**
     * @brief Apply transform to image
     *
     * @param image Input image (RGB)
     * @return Transformed image
     */
    virtual cv::Mat apply(const cv::Mat& image) const = 0;

    /**
     * @brief Apply transform to image and detection target
     *
     * @param image Input image (RGB)
     * @param target Detection target with boxes
     * @return Tuple of (transformed_image, transformed_target)
     */
    virtual std::tuple<cv::Mat, DetectionTarget> apply(
        const cv::Mat& image,
        const DetectionTarget& target
    ) const {
        // Default: only transform image, keep target unchanged
        return {apply(image), target};
    }

    /**
     * @brief Get string representation
     */
    virtual std::string repr() const = 0;
};

/**
 * @brief Compose multiple transforms together
 *
 * @example
 * ```cpp
 * auto transform = std::make_shared<Compose>(std::vector<std::shared_ptr<ITransform>>{
 *     std::make_shared<Resize>(224),
 *     std::make_shared<RandomHorizontalFlip>(),
 *     std::make_shared<Normalize>(),
 * });
 * cv::Mat result = transform->apply(image);
 * ```
 */
class Compose : public ITransform {
public:
    explicit Compose(std::vector<std::shared_ptr<ITransform>> transforms);

    cv::Mat apply(const cv::Mat& image) const override;
    std::tuple<cv::Mat, DetectionTarget> apply(
        const cv::Mat& image,
        const DetectionTarget& target
    ) const override;
    std::string repr() const override;

    /**
     * @brief Add a transform to the pipeline
     */
    void add(std::shared_ptr<ITransform> transform);

    /**
     * @brief Get number of transforms
     */
    size_t size() const { return transforms_.size(); }

private:
    std::vector<std::shared_ptr<ITransform>> transforms_;
};

/**
 * @brief Resize image to target size
 *
 * @example
 * ```cpp
 * auto resize = std::make_shared<Resize>(224);                    // Square
 * auto resize = std::make_shared<Resize>(480, 640);               // H x W
 * auto resize = std::make_shared<Resize>(480, 640, true, 114);    // Letterbox
 * ```
 */
class Resize : public ITransform {
public:
    /**
     * @brief Resize to square
     */
    explicit Resize(int size, bool keep_ratio = false, int pad_value = 114);

    /**
     * @brief Resize to specific height and width
     */
    Resize(int height, int width, bool keep_ratio = false, int pad_value = 114);

    cv::Mat apply(const cv::Mat& image) const override;
    std::tuple<cv::Mat, DetectionTarget> apply(
        const cv::Mat& image,
        const DetectionTarget& target
    ) const override;
    std::string repr() const override;

private:
    int height_;
    int width_;
    bool keep_ratio_;
    int pad_value_;
};

/**
 * @brief Randomly flip image horizontally
 */
class RandomHorizontalFlip : public ITransform {
public:
    explicit RandomHorizontalFlip(float p = 0.5f);

    cv::Mat apply(const cv::Mat& image) const override;
    std::tuple<cv::Mat, DetectionTarget> apply(
        const cv::Mat& image,
        const DetectionTarget& target
    ) const override;
    std::string repr() const override;

private:
    float p_;
    mutable std::mt19937 rng_;
};

/**
 * @brief Randomly flip image vertically
 */
class RandomVerticalFlip : public ITransform {
public:
    explicit RandomVerticalFlip(float p = 0.5f);

    cv::Mat apply(const cv::Mat& image) const override;
    std::tuple<cv::Mat, DetectionTarget> apply(
        const cv::Mat& image,
        const DetectionTarget& target
    ) const override;
    std::string repr() const override;

private:
    float p_;
    mutable std::mt19937 rng_;
};

/**
 * @brief Randomly rotate image
 */
class RandomRotation : public ITransform {
public:
    /**
     * @brief Create random rotation transform
     *
     * @param degrees Maximum rotation in degrees (+/- degrees)
     * @param p Probability of applying rotation
     */
    explicit RandomRotation(float degrees, float p = 0.5f);

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;

private:
    float degrees_;
    float p_;
    mutable std::mt19937 rng_;
};

/**
 * @brief Randomly adjust brightness, contrast, saturation, and hue
 *
 * @example
 * ```cpp
 * auto jitter = std::make_shared<ColorJitter>(0.2f, 0.2f, 0.2f, 0.1f);
 * ```
 */
class ColorJitter : public ITransform {
public:
    /**
     * @brief Create color jitter transform
     *
     * @param brightness Brightness adjustment range (0-1)
     * @param contrast Contrast adjustment range (0-1)
     * @param saturation Saturation adjustment range (0-1)
     * @param hue Hue adjustment range (0-0.5)
     */
    ColorJitter(
        float brightness = 0.2f,
        float contrast = 0.2f,
        float saturation = 0.2f,
        float hue = 0.1f
    );

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;

private:
    cv::Mat adjust_brightness(const cv::Mat& image, float factor) const;
    cv::Mat adjust_contrast(const cv::Mat& image, float factor) const;
    cv::Mat adjust_saturation(const cv::Mat& image, float factor) const;
    cv::Mat adjust_hue(const cv::Mat& image, float factor) const;

    float brightness_;
    float contrast_;
    float saturation_;
    float hue_;
    mutable std::mt19937 rng_;
};

/**
 * @brief Normalize image with mean and std
 *
 * Converts image to float32 and normalizes with:
 *     output = (input / 255.0 - mean) / std
 *
 * @example
 * ```cpp
 * // ImageNet normalization (default)
 * auto normalize = std::make_shared<Normalize>();
 *
 * // Custom normalization
 * auto normalize = std::make_shared<Normalize>(
 *     {0.5f, 0.5f, 0.5f},
 *     {0.5f, 0.5f, 0.5f}
 * );
 * ```
 */
class Normalize : public ITransform {
public:
    /**
     * @brief Create normalize transform with ImageNet defaults
     */
    Normalize();

    /**
     * @brief Create normalize transform with custom mean/std
     *
     * @param mean Per-channel mean
     * @param std Per-channel standard deviation
     */
    Normalize(
        const std::vector<float>& mean,
        const std::vector<float>& std
    );

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

/**
 * @brief Convert image to NCHW tensor format
 *
 * Converts HWC uint8 image to NCHW float32:
 *     - HWC (H, W, 3) -> CHW (3, H, W) -> NCHW (1, 3, H, W)
 *     - If input is already float32, no scaling is applied
 *     - If input is uint8, values are scaled to [0, 1]
 */
class ToTensor : public ITransform {
public:
    ToTensor() = default;

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;
};

/**
 * @brief Center crop image
 */
class CenterCrop : public ITransform {
public:
    explicit CenterCrop(int size);
    CenterCrop(int height, int width);

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;

private:
    int height_;
    int width_;
};

/**
 * @brief Random crop image
 */
class RandomCrop : public ITransform {
public:
    explicit RandomCrop(int size);
    RandomCrop(int height, int width);

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;

private:
    int height_;
    int width_;
    mutable std::mt19937 rng_;
};

/**
 * @brief Gaussian blur
 */
class GaussianBlur : public ITransform {
public:
    explicit GaussianBlur(int kernel_size = 5, float sigma = 0.0f);

    cv::Mat apply(const cv::Mat& image) const override;
    std::string repr() const override;

private:
    int kernel_size_;
    float sigma_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Get default augmentation pipeline for inference
 *
 * @param size Target image size
 * @return Compose transform
 */
std::shared_ptr<Compose> get_default_augmentation(int size = 224);

/**
 * @brief Get training augmentation pipeline
 *
 * @param size Target image size
 * @param flip_p Horizontal flip probability
 * @param color_jitter Enable color jittering
 * @return Compose transform
 */
std::shared_ptr<Compose> get_train_augmentation(
    int size = 224,
    float flip_p = 0.5f,
    bool color_jitter = true
);

/**
 * @brief Get validation augmentation pipeline (no random transforms)
 *
 * @param size Target image size
 * @return Compose transform
 */
std::shared_ptr<Compose> get_val_augmentation(int size = 224);

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_AUGMENTATION_HPP
