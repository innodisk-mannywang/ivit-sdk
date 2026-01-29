/**
 * @file video_source.hpp
 * @brief Video source for streaming inference
 */

#ifndef IVIT_CORE_VIDEO_SOURCE_HPP
#define IVIT_CORE_VIDEO_SOURCE_HPP

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <string>

namespace ivit {

/**
 * @brief Video source abstraction for files, cameras, and RTSP streams
 */
class VideoSource {
public:
    /**
     * @brief Open a video source
     *
     * @param source File path, camera index as string ("0"), or RTSP URL
     */
    explicit VideoSource(const std::string& source);
    ~VideoSource() = default;

    /**
     * @brief Read the next frame
     * @return Frame image, or empty Mat if end of stream
     */
    cv::Mat read();

    /**
     * @brief Check if the source is opened
     */
    bool is_opened() const;

    /**
     * @brief Get total frame count (for files, -1 for live streams)
     */
    int frame_count() const;

    /**
     * @brief Get FPS of the source
     */
    double fps() const;

    /**
     * @brief Release the video source
     */
    void release();

private:
    cv::VideoCapture cap_;
    std::string source_;
};

} // namespace ivit

#endif // IVIT_CORE_VIDEO_SOURCE_HPP
