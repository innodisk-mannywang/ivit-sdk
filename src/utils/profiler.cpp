/**
 * @file profiler.cpp
 * @brief Performance profiling utilities implementation
 */

#include "ivit/utils/profiler.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <opencv2/core.hpp>

namespace ivit {
namespace utils {

// ============================================================================
// ProfileReport implementation
// ============================================================================

std::string ProfileReport::to_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "=== Performance Report ===\n";
    ss << "Model: " << model_name << "\n";
    ss << "Device: " << device << "\n";
    ss << "Backend: " << backend << "\n";
    ss << "Precision: " << precision << "\n";

    // Format input shape
    ss << "Input Shape: [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        ss << input_shape[i];
        if (i < input_shape.size() - 1) ss << ", ";
    }
    ss << "]\n";

    ss << "Iterations: " << iterations << "\n";
    ss << "\n--- Latency (ms) ---\n";
    ss << "  Mean:   " << latency_mean << "\n";
    ss << "  Median: " << latency_median << "\n";
    ss << "  Std:    " << latency_std << "\n";
    ss << "  Min:    " << latency_min << "\n";
    ss << "  Max:    " << latency_max << "\n";
    ss << "  P95:    " << latency_p95 << "\n";
    ss << "  P99:    " << latency_p99 << "\n";
    ss << "\n--- Throughput ---\n";
    ss << "  FPS: " << throughput_fps << "\n";
    ss << "\n--- Memory ---\n";
    ss << "  Usage: " << memory_mb << " MB\n";
    ss << "==========================\n";

    return ss.str();
}

std::string ProfileReport::to_json() const {
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);

    json << "{\n";
    json << "  \"model_name\": \"" << model_name << "\",\n";
    json << "  \"device\": \"" << device << "\",\n";
    json << "  \"backend\": \"" << backend << "\",\n";
    json << "  \"precision\": \"" << precision << "\",\n";

    // Input shape
    json << "  \"input_shape\": [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        json << input_shape[i];
        if (i < input_shape.size() - 1) json << ", ";
    }
    json << "],\n";

    json << "  \"iterations\": " << iterations << ",\n";

    // Latency statistics
    json << "  \"latency\": {\n";
    json << "    \"mean_ms\": " << latency_mean << ",\n";
    json << "    \"median_ms\": " << latency_median << ",\n";
    json << "    \"std_ms\": " << latency_std << ",\n";
    json << "    \"min_ms\": " << latency_min << ",\n";
    json << "    \"max_ms\": " << latency_max << ",\n";
    json << "    \"p95_ms\": " << latency_p95 << ",\n";
    json << "    \"p99_ms\": " << latency_p99 << "\n";
    json << "  },\n";

    json << "  \"throughput_fps\": " << throughput_fps << ",\n";
    json << "  \"memory_mb\": " << memory_mb << "\n";
    json << "}\n";

    return json.str();
}

void ProfileReport::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw IVITError("Failed to open file for writing: " + path);
    }

    // Determine format from extension
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
        file << to_json();
    } else {
        file << to_string();
    }
}

// ============================================================================
// Profiler implementation
// ============================================================================

ProfileReport Profiler::benchmark(
    std::shared_ptr<Model> model,
    const std::vector<int64_t>& input_shape,
    int iterations,
    int warmup
) {
    if (!model) {
        throw IVITError("Model is null");
    }

    if (input_shape.size() < 4) {
        throw IVITError("Input shape must have at least 4 dimensions (NCHW)");
    }

    // Prepare dummy input
    int batch = static_cast<int>(input_shape[0]);
    int channels = static_cast<int>(input_shape[1]);
    int height = static_cast<int>(input_shape[2]);
    int width = static_cast<int>(input_shape[3]);

    // Create dummy image (CV_8UC3 for 3 channels, CV_8UC1 for 1 channel)
    int cv_type = (channels == 1) ? CV_8UC1 : CV_8UC3;
    cv::Mat dummy_input(height, width, cv_type, cv::Scalar::all(128));

    InferConfig config;
    config.enable_profiling = true;

    // Warmup phase
    for (int i = 0; i < warmup; i++) {
        model->predict(dummy_input, config);
    }

    // Reset profiler
    reset();

    // Benchmark phase
    for (int i = 0; i < iterations; i++) {
        start();
        model->predict(dummy_input, config);
        stop();
    }

    // Build report
    ProfileReport report;
    report.model_name = model->name();
    report.device = model->device();
    report.backend = ivit::to_string(model->backend());
    report.input_shape = input_shape;
    report.iterations = iterations;

    // Get input info for precision
    auto inputs = model->input_info();
    if (!inputs.empty()) {
        report.precision = ivit::to_string(inputs[0].dtype);
    }

    // Calculate statistics
    calculate_stats(
        report.latency_mean,
        report.latency_median,
        report.latency_std,
        report.latency_min,
        report.latency_max,
        report.latency_p95,
        report.latency_p99
    );

    // Calculate throughput
    if (report.latency_mean > 0) {
        report.throughput_fps = 1000.0f / report.latency_mean * batch;
    }

    // Get memory usage
    report.memory_mb = static_cast<float>(model->memory_usage()) / (1024.0f * 1024.0f);

    return report;
}

void Profiler::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Profiler::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_
    );
    times_.push_back(static_cast<float>(duration.count()) / 1000.0f);
}

float Profiler::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - start_time_
    );
    return static_cast<float>(duration.count()) / 1000.0f;
}

void Profiler::reset() {
    times_.clear();
}

void Profiler::calculate_stats(
    float& mean,
    float& median,
    float& std_dev,
    float& min,
    float& max,
    float& p95,
    float& p99
) const {
    if (times_.empty()) {
        mean = median = std_dev = min = max = p95 = p99 = 0;
        return;
    }

    // Create sorted copy for percentile calculations
    std::vector<float> sorted_times = times_;
    std::sort(sorted_times.begin(), sorted_times.end());

    size_t n = sorted_times.size();

    // Min and max
    min = sorted_times.front();
    max = sorted_times.back();

    // Mean
    float sum = std::accumulate(sorted_times.begin(), sorted_times.end(), 0.0f);
    mean = sum / static_cast<float>(n);

    // Median
    if (n % 2 == 0) {
        median = (sorted_times[n/2 - 1] + sorted_times[n/2]) / 2.0f;
    } else {
        median = sorted_times[n/2];
    }

    // Standard deviation
    float sq_sum = 0;
    for (float t : sorted_times) {
        float diff = t - mean;
        sq_sum += diff * diff;
    }
    std_dev = std::sqrt(sq_sum / static_cast<float>(n));

    // Percentiles (using linear interpolation)
    auto percentile = [&sorted_times, n](float p) -> float {
        if (n == 1) return sorted_times[0];

        float rank = p * static_cast<float>(n - 1);
        size_t lower = static_cast<size_t>(std::floor(rank));
        size_t upper = static_cast<size_t>(std::ceil(rank));

        if (lower == upper) {
            return sorted_times[lower];
        }

        float frac = rank - static_cast<float>(lower);
        return sorted_times[lower] * (1.0f - frac) + sorted_times[upper] * frac;
    };

    p95 = percentile(0.95f);
    p99 = percentile(0.99f);
}

} // namespace utils
} // namespace ivit
