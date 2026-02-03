/**
 * @file augmentation.cpp
 * @brief Data augmentation transforms implementation
 */

#include "ivit/train/augmentation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ivit {
namespace train {

// ============================================================================
// Compose
// ============================================================================

Compose::Compose(std::vector<std::shared_ptr<ITransform>> transforms)
    : transforms_(std::move(transforms))
{}

cv::Mat Compose::apply(const cv::Mat& image) const {
    cv::Mat result = image.clone();
    for (const auto& transform : transforms_) {
        result = transform->apply(result);
    }
    return result;
}

std::tuple<cv::Mat, DetectionTarget> Compose::apply(
    const cv::Mat& image,
    const DetectionTarget& target
) const {
    cv::Mat img = image.clone();
    DetectionTarget tgt = target;

    for (const auto& transform : transforms_) {
        std::tie(img, tgt) = transform->apply(img, tgt);
    }

    return {img, tgt};
}

std::string Compose::repr() const {
    std::string s = "Compose([";
    for (size_t i = 0; i < transforms_.size(); ++i) {
        s += transforms_[i]->repr();
        if (i < transforms_.size() - 1) s += ", ";
    }
    s += "])";
    return s;
}

void Compose::add(std::shared_ptr<ITransform> transform) {
    transforms_.push_back(std::move(transform));
}

// ============================================================================
// Resize
// ============================================================================

Resize::Resize(int size, bool keep_ratio, int pad_value)
    : height_(size), width_(size), keep_ratio_(keep_ratio), pad_value_(pad_value)
{}

Resize::Resize(int height, int width, bool keep_ratio, int pad_value)
    : height_(height), width_(width), keep_ratio_(keep_ratio), pad_value_(pad_value)
{}

cv::Mat Resize::apply(const cv::Mat& image) const {
    cv::Mat output;

    if (keep_ratio_) {
        // Letterbox resize
        int orig_h = image.rows;
        int orig_w = image.cols;
        float scale = std::min(static_cast<float>(width_) / orig_w,
                               static_cast<float>(height_) / orig_h);
        int new_w = static_cast<int>(orig_w * scale);
        int new_h = static_cast<int>(orig_h * scale);
        int pad_w = (width_ - new_w) / 2;
        int pad_h = (height_ - new_h) / 2;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h));

        output = cv::Mat(height_, width_, image.type(),
                         cv::Scalar(pad_value_, pad_value_, pad_value_));
        resized.copyTo(output(cv::Rect(pad_w, pad_h, new_w, new_h)));
    } else {
        cv::resize(image, output, cv::Size(width_, height_));
    }

    return output;
}

std::tuple<cv::Mat, DetectionTarget> Resize::apply(
    const cv::Mat& image,
    const DetectionTarget& target
) const {
    int orig_h = image.rows;
    int orig_w = image.cols;

    cv::Mat output;
    DetectionTarget new_target = target;

    if (keep_ratio_) {
        float scale = std::min(static_cast<float>(width_) / orig_w,
                               static_cast<float>(height_) / orig_h);
        int new_w = static_cast<int>(orig_w * scale);
        int new_h = static_cast<int>(orig_h * scale);
        int pad_w = (width_ - new_w) / 2;
        int pad_h = (height_ - new_h) / 2;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h));

        output = cv::Mat(height_, width_, image.type(),
                         cv::Scalar(pad_value_, pad_value_, pad_value_));
        resized.copyTo(output(cv::Rect(pad_w, pad_h, new_w, new_h)));

        // Update boxes
        for (auto& box : new_target.boxes) {
            box[0] = box[0] * scale + pad_w;  // x1
            box[1] = box[1] * scale + pad_h;  // y1
            box[2] = box[2] * scale + pad_w;  // x2
            box[3] = box[3] * scale + pad_h;  // y2
        }
    } else {
        cv::resize(image, output, cv::Size(width_, height_));

        // Scale boxes
        float scale_x = static_cast<float>(width_) / orig_w;
        float scale_y = static_cast<float>(height_) / orig_h;

        for (auto& box : new_target.boxes) {
            box[0] *= scale_x;  // x1
            box[1] *= scale_y;  // y1
            box[2] *= scale_x;  // x2
            box[3] *= scale_y;  // y2
        }
    }

    return {output, new_target};
}

std::string Resize::repr() const {
    return "Resize(size=(" + std::to_string(height_) + ", " + std::to_string(width_) +
           "), keep_ratio=" + (keep_ratio_ ? "true" : "false") + ")";
}

// ============================================================================
// RandomHorizontalFlip
// ============================================================================

RandomHorizontalFlip::RandomHorizontalFlip(float p)
    : p_(p), rng_(std::random_device{}())
{}

cv::Mat RandomHorizontalFlip::apply(const cv::Mat& image) const {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng_) < p_) {
        cv::Mat flipped;
        cv::flip(image, flipped, 1);  // Horizontal flip
        return flipped;
    }
    return image.clone();
}

std::tuple<cv::Mat, DetectionTarget> RandomHorizontalFlip::apply(
    const cv::Mat& image,
    const DetectionTarget& target
) const {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng_) < p_) {
        cv::Mat flipped;
        cv::flip(image, flipped, 1);

        DetectionTarget new_target = target;
        int w = image.cols;

        for (auto& box : new_target.boxes) {
            float x1 = box[0];
            float x2 = box[2];
            box[0] = w - x2;  // new x1
            box[2] = w - x1;  // new x2
        }

        return {flipped, new_target};
    }

    return {image.clone(), target};
}

std::string RandomHorizontalFlip::repr() const {
    return "RandomHorizontalFlip(p=" + std::to_string(p_) + ")";
}

// ============================================================================
// RandomVerticalFlip
// ============================================================================

RandomVerticalFlip::RandomVerticalFlip(float p)
    : p_(p), rng_(std::random_device{}())
{}

cv::Mat RandomVerticalFlip::apply(const cv::Mat& image) const {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng_) < p_) {
        cv::Mat flipped;
        cv::flip(image, flipped, 0);  // Vertical flip
        return flipped;
    }
    return image.clone();
}

std::tuple<cv::Mat, DetectionTarget> RandomVerticalFlip::apply(
    const cv::Mat& image,
    const DetectionTarget& target
) const {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng_) < p_) {
        cv::Mat flipped;
        cv::flip(image, flipped, 0);

        DetectionTarget new_target = target;
        int h = image.rows;

        for (auto& box : new_target.boxes) {
            float y1 = box[1];
            float y2 = box[3];
            box[1] = h - y2;  // new y1
            box[3] = h - y1;  // new y2
        }

        return {flipped, new_target};
    }

    return {image.clone(), target};
}

std::string RandomVerticalFlip::repr() const {
    return "RandomVerticalFlip(p=" + std::to_string(p_) + ")";
}

// ============================================================================
// RandomRotation
// ============================================================================

RandomRotation::RandomRotation(float degrees, float p)
    : degrees_(degrees), p_(p), rng_(std::random_device{}())
{}

cv::Mat RandomRotation::apply(const cv::Mat& image) const {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    if (prob_dist(rng_) < p_) {
        std::uniform_real_distribution<float> angle_dist(-degrees_, degrees_);
        float angle = angle_dist(rng_);

        int h = image.rows;
        int w = image.cols;
        cv::Point2f center(w / 2.0f, h / 2.0f);

        cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(image, rotated, M, cv::Size(w, h),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
        return rotated;
    }

    return image.clone();
}

std::string RandomRotation::repr() const {
    return "RandomRotation(degrees=" + std::to_string(degrees_) +
           ", p=" + std::to_string(p_) + ")";
}

// ============================================================================
// ColorJitter
// ============================================================================

ColorJitter::ColorJitter(float brightness, float contrast, float saturation, float hue)
    : brightness_(brightness)
    , contrast_(contrast)
    , saturation_(saturation)
    , hue_(hue)
    , rng_(std::random_device{}())
{}

cv::Mat ColorJitter::apply(const cv::Mat& image) const {
    cv::Mat result = image.clone();

    // Random order of adjustments
    std::vector<int> order = {0, 1, 2, 3};
    std::shuffle(order.begin(), order.end(), rng_);

    for (int op : order) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float factor = dist(rng_);

        switch (op) {
            case 0:
                if (brightness_ > 0)
                    result = adjust_brightness(result, brightness_ * factor);
                break;
            case 1:
                if (contrast_ > 0)
                    result = adjust_contrast(result, 1.0f + contrast_ * factor);
                break;
            case 2:
                if (saturation_ > 0)
                    result = adjust_saturation(result, 1.0f + saturation_ * factor);
                break;
            case 3:
                if (hue_ > 0)
                    result = adjust_hue(result, hue_ * factor);
                break;
        }
    }

    return result;
}

cv::Mat ColorJitter::adjust_brightness(const cv::Mat& image, float factor) const {
    cv::Mat result;
    image.convertTo(result, CV_32F);
    result = result + factor * 255.0f;
    // Clip values to [0, 255]
    cv::max(result, 0.0f, result);
    cv::min(result, 255.0f, result);
    result.convertTo(result, CV_8U);
    return result;
}

cv::Mat ColorJitter::adjust_contrast(const cv::Mat& image, float factor) const {
    cv::Mat result;
    image.convertTo(result, CV_32F);
    cv::Scalar mean = cv::mean(result);
    result = (result - mean) * factor + mean;
    // Clip values to [0, 255]
    cv::max(result, 0.0f, result);
    cv::min(result, 255.0f, result);
    result.convertTo(result, CV_8U);
    return result;
}

cv::Mat ColorJitter::adjust_saturation(const cv::Mat& image, float factor) const {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_RGB2HSV);
    hsv.convertTo(hsv, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    channels[1] = channels[1] * factor;
    // Clip saturation to [0, 255]
    cv::max(channels[1], 0.0f, channels[1]);
    cv::min(channels[1], 255.0f, channels[1]);
    cv::merge(channels, hsv);

    hsv.convertTo(hsv, CV_8U);
    cv::Mat result;
    cv::cvtColor(hsv, result, cv::COLOR_HSV2RGB);
    return result;
}

cv::Mat ColorJitter::adjust_hue(const cv::Mat& image, float factor) const {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_RGB2HSV);
    hsv.convertTo(hsv, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    channels[0] = channels[0] + factor * 180.0f;

    // Wrap hue values to [0, 180)
    for (int i = 0; i < channels[0].rows; ++i) {
        float* row = channels[0].ptr<float>(i);
        for (int j = 0; j < channels[0].cols; ++j) {
            while (row[j] < 0) row[j] += 180.0f;
            while (row[j] >= 180.0f) row[j] -= 180.0f;
        }
    }

    cv::merge(channels, hsv);
    hsv.convertTo(hsv, CV_8U);

    cv::Mat result;
    cv::cvtColor(hsv, result, cv::COLOR_HSV2RGB);
    return result;
}

std::string ColorJitter::repr() const {
    return "ColorJitter(brightness=" + std::to_string(brightness_) +
           ", contrast=" + std::to_string(contrast_) +
           ", saturation=" + std::to_string(saturation_) +
           ", hue=" + std::to_string(hue_) + ")";
}

// ============================================================================
// Normalize
// ============================================================================

Normalize::Normalize()
    : mean_({0.485f, 0.456f, 0.406f})
    , std_({0.229f, 0.224f, 0.225f})
{}

Normalize::Normalize(const std::vector<float>& mean, const std::vector<float>& std)
    : mean_(mean), std_(std)
{
    if (mean_.size() != 3 || std_.size() != 3) {
        throw std::invalid_argument("Mean and std must have 3 values");
    }
}

cv::Mat Normalize::apply(const cv::Mat& image) const {
    cv::Mat result;
    image.convertTo(result, CV_32F, 1.0 / 255.0);

    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(result, channels);

    // Normalize each channel
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    cv::merge(channels, result);
    return result;
}

std::string Normalize::repr() const {
    auto vec_to_str = [](const std::vector<float>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            s += std::to_string(v[i]);
            if (i < v.size() - 1) s += ", ";
        }
        s += "]";
        return s;
    };

    return "Normalize(mean=" + vec_to_str(mean_) + ", std=" + vec_to_str(std_) + ")";
}

// ============================================================================
// ToTensor
// ============================================================================

cv::Mat ToTensor::apply(const cv::Mat& image) const {
    cv::Mat result;

    // Ensure float32
    if (image.depth() != CV_32F) {
        image.convertTo(result, CV_32F, 1.0 / 255.0);
    } else {
        result = image.clone();
    }

    // HWC -> CHW
    std::vector<cv::Mat> channels;
    cv::split(result, channels);

    // Create NCHW tensor (1, C, H, W)
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();

    cv::Mat tensor(1, c * h * w, CV_32F);
    float* ptr = tensor.ptr<float>(0);

    for (int ch = 0; ch < c; ++ch) {
        const float* channel_ptr = channels[ch].ptr<float>(0);
        std::copy(channel_ptr, channel_ptr + h * w, ptr + ch * h * w);
    }

    // Reshape to (1, C, H, W) - stored as 4D
    int sizes[] = {1, c, h, w};
    tensor = tensor.reshape(1, 4, sizes);

    return tensor;
}

std::string ToTensor::repr() const {
    return "ToTensor()";
}

// ============================================================================
// CenterCrop
// ============================================================================

CenterCrop::CenterCrop(int size) : height_(size), width_(size) {}

CenterCrop::CenterCrop(int height, int width) : height_(height), width_(width) {}

cv::Mat CenterCrop::apply(const cv::Mat& image) const {
    int h = image.rows;
    int w = image.cols;

    int crop_h = std::min(height_, h);
    int crop_w = std::min(width_, w);

    int y = (h - crop_h) / 2;
    int x = (w - crop_w) / 2;

    return image(cv::Rect(x, y, crop_w, crop_h)).clone();
}

std::string CenterCrop::repr() const {
    return "CenterCrop(size=(" + std::to_string(height_) + ", " + std::to_string(width_) + "))";
}

// ============================================================================
// RandomCrop
// ============================================================================

RandomCrop::RandomCrop(int size) : height_(size), width_(size), rng_(std::random_device{}()) {}

RandomCrop::RandomCrop(int height, int width) : height_(height), width_(width), rng_(std::random_device{}()) {}

cv::Mat RandomCrop::apply(const cv::Mat& image) const {
    int h = image.rows;
    int w = image.cols;

    int crop_h = std::min(height_, h);
    int crop_w = std::min(width_, w);

    std::uniform_int_distribution<int> y_dist(0, h - crop_h);
    std::uniform_int_distribution<int> x_dist(0, w - crop_w);

    int y = y_dist(rng_);
    int x = x_dist(rng_);

    return image(cv::Rect(x, y, crop_w, crop_h)).clone();
}

std::string RandomCrop::repr() const {
    return "RandomCrop(size=(" + std::to_string(height_) + ", " + std::to_string(width_) + "))";
}

// ============================================================================
// GaussianBlur
// ============================================================================

GaussianBlur::GaussianBlur(int kernel_size, float sigma)
    : kernel_size_(kernel_size | 1)  // Ensure odd
    , sigma_(sigma)
{}

cv::Mat GaussianBlur::apply(const cv::Mat& image) const {
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(kernel_size_, kernel_size_), sigma_);
    return result;
}

std::string GaussianBlur::repr() const {
    return "GaussianBlur(kernel_size=" + std::to_string(kernel_size_) +
           ", sigma=" + std::to_string(sigma_) + ")";
}

// ============================================================================
// Convenience Functions
// ============================================================================

std::shared_ptr<Compose> get_default_augmentation(int size) {
    return std::make_shared<Compose>(std::vector<std::shared_ptr<ITransform>>{
        std::make_shared<Resize>(size),
        std::make_shared<Normalize>(),
        std::make_shared<ToTensor>(),
    });
}

std::shared_ptr<Compose> get_train_augmentation(int size, float flip_p, bool color_jitter) {
    std::vector<std::shared_ptr<ITransform>> transforms;

    transforms.push_back(std::make_shared<Resize>(size));

    if (flip_p > 0) {
        transforms.push_back(std::make_shared<RandomHorizontalFlip>(flip_p));
    }

    if (color_jitter) {
        transforms.push_back(std::make_shared<ColorJitter>(0.2f, 0.2f, 0.2f, 0.1f));
    }

    transforms.push_back(std::make_shared<Normalize>());
    transforms.push_back(std::make_shared<ToTensor>());

    return std::make_shared<Compose>(std::move(transforms));
}

std::shared_ptr<Compose> get_val_augmentation(int size) {
    return std::make_shared<Compose>(std::vector<std::shared_ptr<ITransform>>{
        std::make_shared<Resize>(size),
        std::make_shared<Normalize>(),
        std::make_shared<ToTensor>(),
    });
}

} // namespace train
} // namespace ivit
