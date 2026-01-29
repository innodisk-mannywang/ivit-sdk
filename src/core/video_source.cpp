/**
 * @file video_source.cpp
 * @brief Video source implementation
 */

#include "ivit/core/video_source.hpp"
#include "ivit/core/common.hpp"
#include <cctype>
#include <algorithm>

namespace ivit {

VideoSource::VideoSource(const std::string& source) : source_(source) {
    // Check if source is a numeric camera index
    bool is_camera = !source.empty() &&
        std::all_of(source.begin(), source.end(), ::isdigit);

    if (is_camera) {
        int camera_id = std::stoi(source);
        if (!cap_.open(camera_id)) {
            throw IVITError("Failed to open camera: " + source);
        }
    } else {
        if (!cap_.open(source)) {
            throw IVITError("Failed to open video source: " + source);
        }
    }
}

cv::Mat VideoSource::read() {
    cv::Mat frame;
    if (cap_.isOpened()) {
        cap_.read(frame);
    }
    return frame;
}

bool VideoSource::is_opened() const {
    return cap_.isOpened();
}

int VideoSource::frame_count() const {
    int count = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    return (count > 0) ? count : -1;
}

double VideoSource::fps() const {
    return cap_.get(cv::CAP_PROP_FPS);
}

void VideoSource::release() {
    cap_.release();
}

} // namespace ivit
