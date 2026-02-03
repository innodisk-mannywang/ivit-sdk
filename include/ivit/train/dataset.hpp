/**
 * @file dataset.hpp
 * @brief Dataset interfaces and implementations for training
 */

#ifndef IVIT_TRAIN_DATASET_HPP
#define IVIT_TRAIN_DATASET_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <map>
#include <functional>
#include <filesystem>

namespace ivit {
namespace train {

/**
 * @brief Detection target with bounding boxes and labels
 */
struct DetectionTarget {
    std::vector<std::array<float, 4>> boxes;  ///< Boxes as [x1, y1, x2, y2]
    std::vector<int> labels;                   ///< Class labels
    int image_id = 0;                          ///< Image ID (for COCO)
};

/**
 * @brief Abstract base class for datasets
 *
 * All dataset implementations must provide:
 * - size(): Number of samples
 * - get_item(): Get sample by index
 * - num_classes(): Number of classes
 * - class_names(): List of class names
 */
class IDataset {
public:
    virtual ~IDataset() = default;

    /**
     * @brief Get number of samples in the dataset
     */
    virtual size_t size() const = 0;

    /**
     * @brief Get sample by index (classification)
     *
     * @param idx Sample index
     * @return Tuple of (image, label)
     */
    virtual std::tuple<cv::Mat, int> get_item(size_t idx) const = 0;

    /**
     * @brief Get sample by index with detection targets
     *
     * @param idx Sample index
     * @return Tuple of (image, detection_target)
     */
    virtual std::tuple<cv::Mat, DetectionTarget> get_detection_item(size_t idx) const {
        // Default implementation for classification datasets
        auto [image, label] = get_item(idx);
        DetectionTarget target;
        target.labels.push_back(label);
        return {image, target};
    }

    /**
     * @brief Get number of classes
     */
    virtual size_t num_classes() const = 0;

    /**
     * @brief Get class names
     */
    virtual std::vector<std::string> class_names() const = 0;

    /**
     * @brief Check if this is a detection dataset
     */
    virtual bool is_detection_dataset() const { return false; }
};

/**
 * @brief Dataset that loads images from folder structure
 *
 * Expected structure:
 *     root/
 *         class1/
 *             img1.jpg
 *             img2.jpg
 *         class2/
 *             img3.jpg
 *             img4.jpg
 *
 * @example
 * ```cpp
 * auto dataset = std::make_shared<ImageFolderDataset>("./data", nullptr, 0.8, "train");
 * auto [image, label] = dataset->get_item(0);
 * ```
 */
class ImageFolderDataset : public IDataset {
public:
    /**
     * @brief Supported image extensions
     */
    static const std::vector<std::string> SUPPORTED_EXTENSIONS;

    /**
     * @brief Create dataset from folder structure
     *
     * @param root Root directory containing class folders
     * @param transform Transform function to apply (optional)
     * @param train_split Fraction of data for training (0-1)
     * @param split "train", "val", or "all"
     * @param seed Random seed for reproducible splits
     */
    ImageFolderDataset(
        const std::string& root,
        std::function<cv::Mat(const cv::Mat&)> transform = nullptr,
        float train_split = 0.8f,
        const std::string& split = "train",
        int seed = 42
    );

    size_t size() const override;
    std::tuple<cv::Mat, int> get_item(size_t idx) const override;
    size_t num_classes() const override;
    std::vector<std::string> class_names() const override;

    /**
     * @brief Get a calibration subset for quantization
     *
     * @param n_samples Number of samples (default: 100)
     * @return Vector of images
     */
    std::vector<cv::Mat> calibration_set(size_t n_samples = 100) const;

private:
    void load_dataset();
    void apply_split();
    static bool is_supported_extension(const std::string& ext);

    std::filesystem::path root_;
    std::function<cv::Mat(const cv::Mat&)> transform_;
    float train_split_;
    std::string split_;
    int seed_;

    std::vector<std::string> classes_;
    std::vector<std::pair<std::filesystem::path, int>> samples_;
    std::vector<std::pair<std::filesystem::path, int>> all_samples_;
};

/**
 * @brief Dataset for COCO format annotations
 *
 * @example
 * ```cpp
 * auto dataset = std::make_shared<COCODataset>(
 *     "./coco/images",
 *     "./coco/annotations/instances_train2017.json"
 * );
 * auto [image, target] = dataset->get_detection_item(0);
 * ```
 */
class COCODataset : public IDataset {
public:
    /**
     * @brief Create COCO dataset
     *
     * @param root Root directory containing images
     * @param annotation_file Path to COCO JSON annotation file
     * @param transform Transform function
     * @param split Dataset split (informational only)
     */
    COCODataset(
        const std::string& root,
        const std::string& annotation_file,
        std::function<std::tuple<cv::Mat, DetectionTarget>(const cv::Mat&, const DetectionTarget&)> transform = nullptr,
        const std::string& split = "train"
    );

    size_t size() const override;
    std::tuple<cv::Mat, int> get_item(size_t idx) const override;
    std::tuple<cv::Mat, DetectionTarget> get_detection_item(size_t idx) const override;
    size_t num_classes() const override;
    std::vector<std::string> class_names() const override;
    bool is_detection_dataset() const override { return true; }

private:
    void load_annotations();

    std::filesystem::path root_;
    std::filesystem::path annotation_file_;
    std::function<std::tuple<cv::Mat, DetectionTarget>(const cv::Mat&, const DetectionTarget&)> transform_;
    std::string split_;

    std::map<int, std::string> categories_;
    std::vector<std::string> class_names_;
    std::map<int, std::map<std::string, std::string>> images_;  // id -> {file_name, ...}
    std::map<int, std::vector<std::map<std::string, float>>> annotations_;  // image_id -> [{bbox, category_id}, ...]
    std::vector<int> sample_ids_;
};

/**
 * @brief Dataset for YOLO format annotations
 *
 * Expected structure:
 *     root/
 *         images/
 *             train/
 *                 img1.jpg
 *             val/
 *                 img2.jpg
 *         labels/
 *             train/
 *                 img1.txt
 *             val/
 *                 img2.txt
 *
 * Label format (per line): class_id cx cy w h (normalized 0-1)
 *
 * @example
 * ```cpp
 * auto dataset = std::make_shared<YOLODataset>("./yolo_data", "train");
 * ```
 */
class YOLODataset : public IDataset {
public:
    /**
     * @brief Create YOLO dataset
     *
     * @param root Root directory
     * @param split "train" or "val"
     * @param transform Transform function
     * @param class_names Class names (optional, can be loaded from classes.txt)
     */
    YOLODataset(
        const std::string& root,
        const std::string& split = "train",
        std::function<std::tuple<cv::Mat, DetectionTarget>(const cv::Mat&, const DetectionTarget&)> transform = nullptr,
        const std::vector<std::string>& class_names = {}
    );

    size_t size() const override;
    std::tuple<cv::Mat, int> get_item(size_t idx) const override;
    std::tuple<cv::Mat, DetectionTarget> get_detection_item(size_t idx) const override;
    size_t num_classes() const override;
    std::vector<std::string> class_names() const override;
    bool is_detection_dataset() const override { return true; }

private:
    void load_dataset();

    std::filesystem::path root_;
    std::string split_;
    std::function<std::tuple<cv::Mat, DetectionTarget>(const cv::Mat&, const DetectionTarget&)> transform_;
    std::vector<std::string> class_names_;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> samples_;  // (image_path, label_path)
};

/**
 * @brief Split dataset indices into train and validation sets
 *
 * @param dataset Dataset to split
 * @param train_ratio Ratio for training set
 * @param seed Random seed
 * @return Pair of (train_indices, val_indices)
 */
std::pair<std::vector<size_t>, std::vector<size_t>> split_dataset(
    const IDataset& dataset,
    float train_ratio = 0.8f,
    int seed = 42
);

} // namespace train
} // namespace ivit

#endif // IVIT_TRAIN_DATASET_HPP
