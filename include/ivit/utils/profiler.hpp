/**
 * @file profiler.hpp
 * @brief Performance profiling utilities
 */

#ifndef IVIT_UTILS_PROFILER_HPP
#define IVIT_UTILS_PROFILER_HPP

#include "ivit/core/common.hpp"
#include "ivit/core/model.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <memory>

namespace ivit {
namespace utils {

/**
 * @brief Profile report structure
 */
struct ProfileReport {
    std::string model_name;
    std::string device;
    std::string backend;
    std::string precision;
    std::vector<int64_t> input_shape;
    int iterations = 0;

    // Latency statistics (milliseconds)
    float latency_mean = 0;
    float latency_median = 0;
    float latency_std = 0;
    float latency_min = 0;
    float latency_max = 0;
    float latency_p95 = 0;
    float latency_p99 = 0;

    // Throughput
    float throughput_fps = 0;

    // Memory
    float memory_mb = 0;

    /**
     * @brief Format report as string
     */
    std::string to_string() const;

    /**
     * @brief Export to JSON
     */
    std::string to_json() const;

    /**
     * @brief Save to file
     */
    void save(const std::string& path) const;
};

/**
 * @brief Performance profiler
 */
class Profiler {
public:
    Profiler() = default;

    /**
     * @brief Run benchmark on model
     *
     * @param model Model to benchmark
     * @param input_shape Input shape (NCHW)
     * @param iterations Number of iterations
     * @param warmup Warmup iterations
     * @return Profile report
     */
    ProfileReport benchmark(
        std::shared_ptr<Model> model,
        const std::vector<int64_t>& input_shape,
        int iterations = 100,
        int warmup = 10
    );

    /**
     * @brief Start timing
     */
    void start();

    /**
     * @brief Stop timing and record
     */
    void stop();

    /**
     * @brief Get elapsed time in milliseconds
     */
    float elapsed_ms() const;

    /**
     * @brief Reset profiler
     */
    void reset();

    /**
     * @brief Get all recorded times
     */
    const std::vector<float>& times() const { return times_; }

    /**
     * @brief Calculate statistics
     */
    void calculate_stats(
        float& mean,
        float& median,
        float& std_dev,
        float& min,
        float& max,
        float& p95,
        float& p99
    ) const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<float> times_;
};

/**
 * @brief Scoped timer for automatic timing
 */
class ScopedTimer {
public:
    explicit ScopedTimer(Profiler& profiler)
        : profiler_(profiler) {
        profiler_.start();
    }

    ~ScopedTimer() {
        profiler_.stop();
    }

private:
    Profiler& profiler_;
};

} // namespace utils
} // namespace ivit

#endif // IVIT_UTILS_PROFILER_HPP
