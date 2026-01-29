/**
 * @file callback.hpp
 * @brief Callback system for inference events
 */

#ifndef IVIT_CORE_CALLBACK_HPP
#define IVIT_CORE_CALLBACK_HPP

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <mutex>
#include <chrono>
#include <any>

namespace ivit {

// Forward declarations
class Results;

/**
 * @brief Callback event types
 */
enum class CallbackEvent {
    PreProcess,    ///< Before preprocessing
    PostProcess,   ///< After postprocessing
    InferStart,    ///< Before inference
    InferEnd,      ///< After inference
    BatchStart,    ///< Before batch processing
    BatchEnd,      ///< After batch processing
    StreamStart,   ///< When stream starts
    StreamFrame,   ///< After each stream frame
    StreamEnd      ///< When stream ends
};

/**
 * @brief Convert CallbackEvent to string
 */
inline std::string to_string(CallbackEvent event) {
    switch (event) {
        case CallbackEvent::PreProcess:  return "pre_process";
        case CallbackEvent::PostProcess: return "post_process";
        case CallbackEvent::InferStart:  return "infer_start";
        case CallbackEvent::InferEnd:    return "infer_end";
        case CallbackEvent::BatchStart:  return "batch_start";
        case CallbackEvent::BatchEnd:    return "batch_end";
        case CallbackEvent::StreamStart: return "stream_start";
        case CallbackEvent::StreamFrame: return "stream_frame";
        case CallbackEvent::StreamEnd:   return "stream_end";
    }
    return "unknown";
}

/**
 * @brief Parse string to CallbackEvent
 */
inline CallbackEvent callback_event_from_string(const std::string& s) {
    if (s == "pre_process")  return CallbackEvent::PreProcess;
    if (s == "post_process") return CallbackEvent::PostProcess;
    if (s == "infer_start")  return CallbackEvent::InferStart;
    if (s == "infer_end")    return CallbackEvent::InferEnd;
    if (s == "batch_start")  return CallbackEvent::BatchStart;
    if (s == "batch_end")    return CallbackEvent::BatchEnd;
    if (s == "stream_start") return CallbackEvent::StreamStart;
    if (s == "stream_frame") return CallbackEvent::StreamFrame;
    if (s == "stream_end")   return CallbackEvent::StreamEnd;
    throw std::invalid_argument("Unknown callback event: " + s);
}

/**
 * @brief Context passed to callback functions
 */
struct CallbackContext {
    CallbackEvent event;               ///< Event type
    std::string model_name;            ///< Model name
    std::string device;                ///< Device ID
    double latency_ms = 0.0;           ///< Inference latency in milliseconds
    int batch_size = 1;                ///< Batch size
    int frame_index = -1;              ///< Frame index (for streaming)
    const Results* results = nullptr;  ///< Pointer to results (valid during callback)
    std::map<std::string, std::string> metadata; ///< Additional metadata
};

/**
 * @brief Callback entry with priority
 */
struct CallbackEntry {
    std::function<void(const CallbackContext&)> fn;
    int priority = 0;     ///< Higher = called first
    int id = 0;           ///< Unique ID for removal
};

/**
 * @brief Manager for registering and triggering callbacks
 *
 * Thread-safe callback management for inference events.
 */
class CallbackManager {
public:
    CallbackManager() = default;

    /**
     * @brief Register a callback for an event
     *
     * @param event Event type
     * @param callback Function to call
     * @param priority Higher priority callbacks are called first
     * @return Callback ID for later removal
     */
    int register_callback(
        CallbackEvent event,
        std::function<void(const CallbackContext&)> callback,
        int priority = 0
    );

    /**
     * @brief Register a callback using event string name
     */
    int register_callback(
        const std::string& event_name,
        std::function<void(const CallbackContext&)> callback,
        int priority = 0
    );

    /**
     * @brief Remove a specific callback by ID
     *
     * @param event Event type
     * @param callback_id ID returned by register_callback
     * @return true if removed
     */
    bool unregister_callback(CallbackEvent event, int callback_id);

    /**
     * @brief Remove all callbacks for an event
     *
     * @param event Event type
     * @return Number of callbacks removed
     */
    int unregister_all(CallbackEvent event);

    /**
     * @brief Remove all callbacks for an event by string name
     */
    int unregister_all(const std::string& event_name);

    /**
     * @brief Trigger all callbacks for an event
     *
     * Callbacks are called in priority order (highest first).
     *
     * @param ctx Callback context
     */
    void trigger(const CallbackContext& ctx);

    /**
     * @brief Check if any callbacks are registered for an event
     */
    bool has_callbacks(CallbackEvent event) const;

    /**
     * @brief Get number of registered callbacks for an event
     */
    size_t callback_count(CallbackEvent event) const;

    /**
     * @brief Clear all callbacks
     */
    void clear();

private:
    mutable std::mutex mutex_;
    std::map<CallbackEvent, std::vector<CallbackEntry>> callbacks_;
    int next_id_ = 1;
};

} // namespace ivit

#endif // IVIT_CORE_CALLBACK_HPP
