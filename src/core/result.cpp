/**
 * @file result.cpp
 * @brief Results class implementation
 */

#include "ivit/core/result.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>

namespace ivit {

// ============================================================================
// Results implementation
// ============================================================================

std::vector<Detection> Results::filter_by_class(
    const std::vector<int>& class_ids
) const {
    std::vector<Detection> filtered;

    for (const auto& det : detections) {
        for (int id : class_ids) {
            if (det.class_id == id) {
                filtered.push_back(det);
                break;
            }
        }
    }

    return filtered;
}

std::vector<Detection> Results::filter_by_confidence(float min_conf) const {
    std::vector<Detection> filtered;

    for (const auto& det : detections) {
        if (det.confidence >= min_conf) {
            filtered.push_back(det);
        }
    }

    return filtered;
}

cv::Mat Results::colorize_mask(const std::map<int, cv::Vec3b>& colormap) const {
    if (segmentation_mask.empty()) {
        throw IVITError("No segmentation mask available");
    }

    cv::Mat colored(segmentation_mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // Generate default colormap if not provided
    std::map<int, cv::Vec3b> cmap = colormap;
    if (cmap.empty()) {
        double min_val, max_val;
        cv::minMaxLoc(segmentation_mask, &min_val, &max_val);

        for (int i = 0; i <= static_cast<int>(max_val); i++) {
            float hue = static_cast<float>(i) / (max_val + 1) * 180;
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 200));
            cv::Mat rgb;
            cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
            cmap[i] = rgb.at<cv::Vec3b>(0, 0);
        }
        cmap[0] = cv::Vec3b(0, 0, 0);  // Background
    }

    // Apply colormap
    for (int y = 0; y < segmentation_mask.rows; y++) {
        for (int x = 0; x < segmentation_mask.cols; x++) {
            int class_id = segmentation_mask.at<int>(y, x);
            if (cmap.count(class_id)) {
                colored.at<cv::Vec3b>(y, x) = cmap.at(class_id);
            }
        }
    }

    return colored;
}

cv::Mat Results::overlay_mask(const cv::Mat& image, float alpha) const {
    cv::Mat colored = colorize_mask();
    cv::Mat overlay;

    // Resize colored mask to match image size if needed
    if (colored.size() != image.size()) {
        cv::resize(colored, colored, image.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::addWeighted(image, 1.0 - alpha, colored, alpha, 0, overlay);
    return overlay;
}

cv::Mat Results::visualize(
    const cv::Mat& image,
    bool show_labels,
    bool show_confidence,
    bool show_boxes,
    bool show_masks
) const {
    cv::Mat vis = image.clone();

    // Color palette
    const std::vector<cv::Scalar> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
        {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
        {128, 0, 0}, {0, 128, 0}, {0, 0, 128},
    };

    // Draw segmentation mask
    if (show_masks && !segmentation_mask.empty()) {
        vis = overlay_mask(vis, 0.5);
    }

    // Draw detections
    if (show_boxes) {
        for (const auto& det : detections) {
            cv::Scalar color = colors[det.class_id % colors.size()];

            // Draw box
            cv::rectangle(vis,
                cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1)),
                cv::Point(static_cast<int>(det.bbox.x2), static_cast<int>(det.bbox.y2)),
                color, 2);

            // Draw label
            if (show_labels || show_confidence) {
                std::ostringstream label_stream;
                if (show_labels) {
                    label_stream << det.label;
                }
                if (show_confidence) {
                    if (show_labels) label_stream << " ";
                    label_stream << std::fixed << std::setprecision(2) << det.confidence;
                }
                std::string label = label_stream.str();

                int baseline;
                cv::Size text_size = cv::getTextSize(label,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

                cv::rectangle(vis,
                    cv::Point(static_cast<int>(det.bbox.x1),
                              static_cast<int>(det.bbox.y1) - text_size.height - 4),
                    cv::Point(static_cast<int>(det.bbox.x1) + text_size.width,
                              static_cast<int>(det.bbox.y1)),
                    color, -1);

                cv::putText(vis, label,
                    cv::Point(static_cast<int>(det.bbox.x1),
                              static_cast<int>(det.bbox.y1) - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
            }

            // Draw instance mask if available
            if (det.mask.has_value() && show_masks) {
                cv::Mat mask_colored;
                cv::Mat mask = det.mask.value();
                cv::cvtColor(mask * 255, mask_colored, cv::COLOR_GRAY2BGR);
                mask_colored.setTo(color, mask > 0);
                cv::addWeighted(vis, 1.0, mask_colored, 0.3, 0, vis);
            }
        }
    }

    // Draw poses
    for (const auto& pose : poses) {
        // Draw keypoints
        for (size_t i = 0; i < pose.keypoints.size(); i++) {
            const auto& kp = pose.keypoints[i];
            if (kp.confidence > 0.5) {
                cv::circle(vis,
                    cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                    3, colors[i % colors.size()], -1);
            }
        }

        // Draw skeleton (COCO format)
        static const std::vector<std::pair<int, int>> skeleton = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4},  // Head
            {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // Arms
            {5, 11}, {6, 12}, {11, 12},  // Torso
            {11, 13}, {13, 15}, {12, 14}, {14, 16},  // Legs
        };

        for (const auto& [i, j] : skeleton) {
            if (i < pose.keypoints.size() && j < pose.keypoints.size()) {
                const auto& kp1 = pose.keypoints[i];
                const auto& kp2 = pose.keypoints[j];
                if (kp1.confidence > 0.5 && kp2.confidence > 0.5) {
                    cv::line(vis,
                        cv::Point(static_cast<int>(kp1.x), static_cast<int>(kp1.y)),
                        cv::Point(static_cast<int>(kp2.x), static_cast<int>(kp2.y)),
                        colors[i % colors.size()], 2);
                }
            }
        }
    }

    return vis;
}

std::string Results::to_json() const {
    std::ostringstream json;
    json << "{\n";

    // Metadata
    json << "  \"inference_time_ms\": " << inference_time_ms << ",\n";
    json << "  \"device_used\": \"" << device_used << "\",\n";
    json << "  \"image_size\": [" << image_size.height << ", " << image_size.width << "],\n";

    // Classifications
    if (!classifications.empty()) {
        json << "  \"classifications\": [\n";
        for (size_t i = 0; i < classifications.size(); i++) {
            const auto& c = classifications[i];
            json << "    {\"class_id\": " << c.class_id
                 << ", \"label\": \"" << c.label
                 << "\", \"score\": " << c.score << "}";
            if (i < classifications.size() - 1) json << ",";
            json << "\n";
        }
        json << "  ],\n";
    }

    // Detections
    if (!detections.empty()) {
        json << "  \"detections\": [\n";
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& d = detections[i];
            json << "    {\"class_id\": " << d.class_id
                 << ", \"label\": \"" << d.label
                 << "\", \"confidence\": " << d.confidence
                 << ", \"bbox\": {\"x1\": " << d.bbox.x1
                 << ", \"y1\": " << d.bbox.y1
                 << ", \"x2\": " << d.bbox.x2
                 << ", \"y2\": " << d.bbox.y2 << "}}";
            if (i < detections.size() - 1) json << ",";
            json << "\n";
        }
        json << "  ]\n";
    }

    json << "}\n";
    return json.str();
}

void Results::save(const std::string& path, const std::string& format) const {
    std::string fmt = format;
    if (fmt == "json" || path.substr(path.length() - 5) == ".json") {
        std::ofstream file(path);
        file << to_json();
    } else if (fmt == "txt" || path.substr(path.length() - 4) == ".txt") {
        // YOLO format
        std::ofstream file(path);
        for (const auto& det : detections) {
            auto [cx, cy, w, h] = det.bbox.to_cxcywh();
            // Normalize by image size
            cx /= image_size.width;
            cy /= image_size.height;
            w /= image_size.width;
            h /= image_size.height;
            file << det.class_id << " " << cx << " " << cy << " " << w << " " << h << "\n";
        }
    } else {
        throw IVITError("Unsupported save format: " + format);
    }
}

} // namespace ivit
