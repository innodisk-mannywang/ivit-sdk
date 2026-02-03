/**
 * @file dataset.cpp
 * @brief Dataset implementations for training
 */

#include "ivit/train/dataset.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

// JSON parsing (simple implementation without external dependency)
namespace {

// Simple JSON parser for COCO format
class SimpleJSON {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type = Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<SimpleJSON> arr_val;
    std::map<std::string, SimpleJSON> obj_val;

    static SimpleJSON parse(const std::string& json) {
        size_t pos = 0;
        return parse_value(json, pos);
    }

    const SimpleJSON& operator[](const std::string& key) const {
        static SimpleJSON null_val;
        if (type == Object) {
            auto it = obj_val.find(key);
            if (it != obj_val.end()) return it->second;
        }
        return null_val;
    }

    const SimpleJSON& operator[](size_t idx) const {
        static SimpleJSON null_val;
        if (type == Array && idx < arr_val.size()) {
            return arr_val[idx];
        }
        return null_val;
    }

    size_t size() const {
        if (type == Array) return arr_val.size();
        if (type == Object) return obj_val.size();
        return 0;
    }

    int to_int() const { return static_cast<int>(num_val); }
    double to_double() const { return num_val; }
    const std::string& to_string() const { return str_val; }

private:
    static void skip_whitespace(const std::string& json, size_t& pos) {
        while (pos < json.size() && std::isspace(json[pos])) pos++;
    }

    static SimpleJSON parse_value(const std::string& json, size_t& pos) {
        skip_whitespace(json, pos);
        if (pos >= json.size()) return SimpleJSON();

        char c = json[pos];
        if (c == '{') return parse_object(json, pos);
        if (c == '[') return parse_array(json, pos);
        if (c == '"') return parse_string(json, pos);
        if (c == 't' || c == 'f') return parse_bool(json, pos);
        if (c == 'n') return parse_null(json, pos);
        if (c == '-' || std::isdigit(c)) return parse_number(json, pos);

        return SimpleJSON();
    }

    static SimpleJSON parse_object(const std::string& json, size_t& pos) {
        SimpleJSON obj;
        obj.type = Object;
        pos++; // skip '{'
        skip_whitespace(json, pos);

        if (json[pos] == '}') { pos++; return obj; }

        while (pos < json.size()) {
            skip_whitespace(json, pos);
            auto key = parse_string(json, pos);
            skip_whitespace(json, pos);
            if (json[pos] == ':') pos++;
            skip_whitespace(json, pos);
            obj.obj_val[key.str_val] = parse_value(json, pos);
            skip_whitespace(json, pos);
            if (json[pos] == '}') { pos++; break; }
            if (json[pos] == ',') pos++;
        }
        return obj;
    }

    static SimpleJSON parse_array(const std::string& json, size_t& pos) {
        SimpleJSON arr;
        arr.type = Array;
        pos++; // skip '['
        skip_whitespace(json, pos);

        if (json[pos] == ']') { pos++; return arr; }

        while (pos < json.size()) {
            skip_whitespace(json, pos);
            arr.arr_val.push_back(parse_value(json, pos));
            skip_whitespace(json, pos);
            if (json[pos] == ']') { pos++; break; }
            if (json[pos] == ',') pos++;
        }
        return arr;
    }

    static SimpleJSON parse_string(const std::string& json, size_t& pos) {
        SimpleJSON str;
        str.type = String;
        pos++; // skip opening quote
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                char escaped = json[pos];
                if (escaped == 'n') str.str_val += '\n';
                else if (escaped == 't') str.str_val += '\t';
                else if (escaped == 'r') str.str_val += '\r';
                else str.str_val += escaped;
            } else {
                str.str_val += json[pos];
            }
            pos++;
        }
        if (pos < json.size()) pos++; // skip closing quote
        return str;
    }

    static SimpleJSON parse_number(const std::string& json, size_t& pos) {
        SimpleJSON num;
        num.type = Number;
        size_t start = pos;
        if (json[pos] == '-') pos++;
        while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+' || json[pos] == '-')) {
            if ((json[pos] == 'e' || json[pos] == 'E') && pos > start) pos++;
            else if ((json[pos] == '+' || json[pos] == '-') && pos > start && (json[pos-1] == 'e' || json[pos-1] == 'E')) pos++;
            else if (std::isdigit(json[pos]) || json[pos] == '.') pos++;
            else break;
        }
        num.num_val = std::stod(json.substr(start, pos - start));
        return num;
    }

    static SimpleJSON parse_bool(const std::string& json, size_t& pos) {
        SimpleJSON b;
        b.type = Bool;
        if (json.substr(pos, 4) == "true") {
            b.bool_val = true;
            pos += 4;
        } else if (json.substr(pos, 5) == "false") {
            b.bool_val = false;
            pos += 5;
        }
        return b;
    }

    static SimpleJSON parse_null(const std::string& json, size_t& pos) {
        pos += 4; // skip "null"
        return SimpleJSON();
    }
};

} // anonymous namespace

namespace ivit {
namespace train {

namespace fs = std::filesystem;

// ============================================================================
// ImageFolderDataset
// ============================================================================

const std::vector<std::string> ImageFolderDataset::SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"
};

ImageFolderDataset::ImageFolderDataset(
    const std::string& root,
    std::function<cv::Mat(const cv::Mat&)> transform,
    float train_split,
    const std::string& split,
    int seed
)
    : root_(root)
    , transform_(std::move(transform))
    , train_split_(train_split)
    , split_(split)
    , seed_(seed)
{
    // Convert split to lowercase
    std::transform(split_.begin(), split_.end(), split_.begin(), ::tolower);

    if (!fs::exists(root_)) {
        throw std::runtime_error("Dataset root does not exist: " + root);
    }

    load_dataset();
    apply_split();

    std::cout << "[ImageFolderDataset] Loaded " << samples_.size()
              << " samples from " << root_ << std::endl;
    std::cout << "[ImageFolderDataset] Classes: " << classes_.size() << std::endl;
}

void ImageFolderDataset::load_dataset() {
    // Get class directories
    std::vector<fs::path> class_dirs;
    for (const auto& entry : fs::directory_iterator(root_)) {
        std::string dirname = entry.path().filename().string();
        if (entry.is_directory() && !dirname.empty() && dirname[0] != '.') {
            class_dirs.push_back(entry.path());
        }
    }

    if (class_dirs.empty()) {
        throw std::runtime_error("No class directories found in " + root_.string());
    }

    // Sort for reproducibility
    std::sort(class_dirs.begin(), class_dirs.end());

    // Build class list
    for (const auto& dir : class_dirs) {
        classes_.push_back(dir.filename().string());
    }

    // Load samples
    for (size_t class_idx = 0; class_idx < class_dirs.size(); ++class_idx) {
        for (const auto& entry : fs::directory_iterator(class_dirs[class_idx])) {
            if (entry.is_regular_file() && is_supported_extension(entry.path().extension().string())) {
                all_samples_.emplace_back(entry.path(), static_cast<int>(class_idx));
            }
        }
    }

    if (all_samples_.empty()) {
        throw std::runtime_error("No valid images found in " + root_.string());
    }

    // Shuffle with fixed seed for reproducibility
    std::mt19937 rng(seed_);
    std::shuffle(all_samples_.begin(), all_samples_.end(), rng);
}

void ImageFolderDataset::apply_split() {
    size_t n_total = all_samples_.size();
    size_t n_train = static_cast<size_t>(n_total * train_split_);

    if (split_ == "train") {
        samples_.assign(all_samples_.begin(), all_samples_.begin() + n_train);
    } else if (split_ == "val") {
        samples_.assign(all_samples_.begin() + n_train, all_samples_.end());
    } else {
        samples_ = all_samples_;
    }
}

bool ImageFolderDataset::is_supported_extension(const std::string& ext) {
    std::string lower_ext = ext;
    std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
    return std::find(SUPPORTED_EXTENSIONS.begin(), SUPPORTED_EXTENSIONS.end(), lower_ext)
           != SUPPORTED_EXTENSIONS.end();
}

size_t ImageFolderDataset::size() const {
    return samples_.size();
}

std::tuple<cv::Mat, int> ImageFolderDataset::get_item(size_t idx) const {
    if (idx >= samples_.size()) {
        throw std::out_of_range("Index out of range: " + std::to_string(idx));
    }

    const auto& [img_path, label] = samples_[idx];

    cv::Mat image = cv::imread(img_path.string());
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path.string());
    }

    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (transform_) {
        image = transform_(image);
    }

    return {image, label};
}

size_t ImageFolderDataset::num_classes() const {
    return classes_.size();
}

std::vector<std::string> ImageFolderDataset::class_names() const {
    return classes_;
}

std::vector<cv::Mat> ImageFolderDataset::calibration_set(size_t n_samples) const {
    std::vector<cv::Mat> images;
    n_samples = std::min(n_samples, samples_.size());

    std::vector<size_t> indices(samples_.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i = 0; i < n_samples; ++i) {
        auto [image, _] = get_item(indices[i]);
        images.push_back(image);
    }

    return images;
}

// ============================================================================
// COCODataset
// ============================================================================

COCODataset::COCODataset(
    const std::string& root,
    const std::string& annotation_file,
    std::function<std::tuple<cv::Mat, DetectionTarget>(const cv::Mat&, const DetectionTarget&)> transform,
    const std::string& split
)
    : root_(root)
    , annotation_file_(annotation_file)
    , transform_(std::move(transform))
    , split_(split)
{
    if (!fs::exists(annotation_file_)) {
        throw std::runtime_error("Annotation file not found: " + annotation_file);
    }

    load_annotations();
}

void COCODataset::load_annotations() {
    // Read entire file
    std::ifstream file(annotation_file_);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();

    auto coco = SimpleJSON::parse(json_str);

    // Build category mapping
    const auto& categories = coco["categories"];
    for (size_t i = 0; i < categories.size(); ++i) {
        int id = categories[i]["id"].to_int();
        std::string name = categories[i]["name"].to_string();
        categories_[id] = name;
    }

    // Build class names list
    for (const auto& [id, name] : categories_) {
        class_names_.push_back(name);
    }

    // Build image info mapping
    const auto& images = coco["images"];
    for (size_t i = 0; i < images.size(); ++i) {
        int id = images[i]["id"].to_int();
        images_[id]["file_name"] = images[i]["file_name"].to_string();
    }

    // Group annotations by image
    const auto& anns = coco["annotations"];
    for (size_t i = 0; i < anns.size(); ++i) {
        int img_id = anns[i]["image_id"].to_int();
        const auto& bbox = anns[i]["bbox"];

        std::map<std::string, float> ann;
        ann["x"] = static_cast<float>(bbox[0].to_double());
        ann["y"] = static_cast<float>(bbox[1].to_double());
        ann["w"] = static_cast<float>(bbox[2].to_double());
        ann["h"] = static_cast<float>(bbox[3].to_double());
        ann["category_id"] = static_cast<float>(anns[i]["category_id"].to_int());

        annotations_[img_id].push_back(ann);
    }

    // Create sample list
    for (const auto& [img_id, _] : annotations_) {
        sample_ids_.push_back(img_id);
    }

    std::cout << "[COCODataset] Loaded " << sample_ids_.size()
              << " images with " << anns.size() << " annotations" << std::endl;
}

size_t COCODataset::size() const {
    return sample_ids_.size();
}

std::tuple<cv::Mat, int> COCODataset::get_item(size_t idx) const {
    // For classification-style access, return first label
    auto [image, target] = get_detection_item(idx);
    int label = target.labels.empty() ? 0 : target.labels[0];
    return {image, label};
}

std::tuple<cv::Mat, DetectionTarget> COCODataset::get_detection_item(size_t idx) const {
    if (idx >= sample_ids_.size()) {
        throw std::out_of_range("Index out of range: " + std::to_string(idx));
    }

    int img_id = sample_ids_[idx];
    const auto& img_info = images_.at(img_id);
    fs::path img_path = root_ / img_info.at("file_name");

    cv::Mat image = cv::imread(img_path.string());
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path.string());
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Extract annotations
    DetectionTarget target;
    target.image_id = img_id;

    if (annotations_.count(img_id)) {
        for (const auto& ann : annotations_.at(img_id)) {
            // COCO bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
            float x = ann.at("x");
            float y = ann.at("y");
            float w = ann.at("w");
            float h = ann.at("h");
            target.boxes.push_back({x, y, x + w, y + h});
            target.labels.push_back(static_cast<int>(ann.at("category_id")));
        }
    }

    if (transform_) {
        return transform_(image, target);
    }

    return {image, target};
}

size_t COCODataset::num_classes() const {
    return class_names_.size();
}

std::vector<std::string> COCODataset::class_names() const {
    return class_names_;
}

// ============================================================================
// YOLODataset
// ============================================================================

YOLODataset::YOLODataset(
    const std::string& root,
    const std::string& split,
    std::function<std::tuple<cv::Mat, DetectionTarget>(const cv::Mat&, const DetectionTarget&)> transform,
    const std::vector<std::string>& class_names
)
    : root_(root)
    , split_(split)
    , transform_(std::move(transform))
    , class_names_(class_names)
{
    load_dataset();
}

void YOLODataset::load_dataset() {
    fs::path images_dir = root_ / "images" / split_;
    fs::path labels_dir = root_ / "labels" / split_;

    // Try alternative structure
    if (!fs::exists(images_dir)) {
        images_dir = root_ / split_ / "images";
        labels_dir = root_ / split_ / "labels";
    }

    if (!fs::exists(images_dir)) {
        throw std::runtime_error("Images directory not found: " + images_dir.string());
    }

    // Load class names if available
    fs::path names_file = root_ / "classes.txt";
    if (fs::exists(names_file) && class_names_.empty()) {
        std::ifstream file(names_file);
        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty()) {
                class_names_.push_back(line);
            }
        }
    }

    // Find image-label pairs
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(ImageFolderDataset::SUPPORTED_EXTENSIONS.begin(),
                      ImageFolderDataset::SUPPORTED_EXTENSIONS.end(), ext)
            != ImageFolderDataset::SUPPORTED_EXTENSIONS.end())
        {
            fs::path label_path = labels_dir / (entry.path().stem().string() + ".txt");
            if (fs::exists(label_path)) {
                samples_.emplace_back(entry.path(), label_path);
            }
        }
    }

    std::cout << "[YOLODataset] Loaded " << samples_.size()
              << " samples from " << root_.string() << std::endl;
}

size_t YOLODataset::size() const {
    return samples_.size();
}

std::tuple<cv::Mat, int> YOLODataset::get_item(size_t idx) const {
    auto [image, target] = get_detection_item(idx);
    int label = target.labels.empty() ? 0 : target.labels[0];
    return {image, label};
}

std::tuple<cv::Mat, DetectionTarget> YOLODataset::get_detection_item(size_t idx) const {
    if (idx >= samples_.size()) {
        throw std::out_of_range("Index out of range: " + std::to_string(idx));
    }

    const auto& [img_path, label_path] = samples_[idx];

    cv::Mat image = cv::imread(img_path.string());
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path.string());
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int h = image.rows;
    int w = image.cols;

    // Parse YOLO labels
    DetectionTarget target;
    std::ifstream file(label_path);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int class_id;
        float cx, cy, bw, bh;

        if (iss >> class_id >> cx >> cy >> bw >> bh) {
            // Convert normalized YOLO to pixel coordinates
            float x1 = (cx - bw / 2) * w;
            float y1 = (cy - bh / 2) * h;
            float x2 = (cx + bw / 2) * w;
            float y2 = (cy + bh / 2) * h;

            target.boxes.push_back({x1, y1, x2, y2});
            target.labels.push_back(class_id);
        }
    }

    if (transform_) {
        return transform_(image, target);
    }

    return {image, target};
}

size_t YOLODataset::num_classes() const {
    if (!class_names_.empty()) {
        return class_names_.size();
    }

    // Infer from labels (sample first 100)
    int max_class = 0;
    size_t n_samples = std::min(samples_.size(), size_t(100));

    for (size_t i = 0; i < n_samples; ++i) {
        const auto& [_, label_path] = samples_[i];
        std::ifstream file(label_path);
        std::string line;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int class_id;
            if (iss >> class_id) {
                max_class = std::max(max_class, class_id);
            }
        }
    }

    return max_class + 1;
}

std::vector<std::string> YOLODataset::class_names() const {
    return class_names_;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::pair<std::vector<size_t>, std::vector<size_t>> split_dataset(
    const IDataset& dataset,
    float train_ratio,
    int seed
) {
    size_t n = dataset.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t n_train = static_cast<size_t>(n * train_ratio);

    std::vector<size_t> train_indices(indices.begin(), indices.begin() + n_train);
    std::vector<size_t> val_indices(indices.begin() + n_train, indices.end());

    return {train_indices, val_indices};
}

} // namespace train
} // namespace ivit
