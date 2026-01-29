/**
 * @file callback.cpp
 * @brief Callback system implementation
 */

#include "ivit/core/callback.hpp"
#include <algorithm>

namespace ivit {

int CallbackManager::register_callback(
    CallbackEvent event,
    std::function<void(const CallbackContext&)> callback,
    int priority
) {
    std::lock_guard<std::mutex> lock(mutex_);
    int id = next_id_++;
    callbacks_[event].push_back({std::move(callback), priority, id});

    // Sort by priority (descending)
    auto& entries = callbacks_[event];
    std::stable_sort(entries.begin(), entries.end(),
        [](const CallbackEntry& a, const CallbackEntry& b) {
            return a.priority > b.priority;
        });

    return id;
}

int CallbackManager::register_callback(
    const std::string& event_name,
    std::function<void(const CallbackContext&)> callback,
    int priority
) {
    return register_callback(
        callback_event_from_string(event_name),
        std::move(callback),
        priority
    );
}

bool CallbackManager::unregister_callback(CallbackEvent event, int callback_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = callbacks_.find(event);
    if (it == callbacks_.end()) return false;

    auto& entries = it->second;
    auto entry_it = std::find_if(entries.begin(), entries.end(),
        [callback_id](const CallbackEntry& e) { return e.id == callback_id; });

    if (entry_it == entries.end()) return false;
    entries.erase(entry_it);
    return true;
}

int CallbackManager::unregister_all(CallbackEvent event) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = callbacks_.find(event);
    if (it == callbacks_.end()) return 0;
    int count = static_cast<int>(it->second.size());
    callbacks_.erase(it);
    return count;
}

int CallbackManager::unregister_all(const std::string& event_name) {
    return unregister_all(callback_event_from_string(event_name));
}

void CallbackManager::trigger(const CallbackContext& ctx) {
    std::vector<CallbackEntry> entries_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = callbacks_.find(ctx.event);
        if (it == callbacks_.end()) return;
        entries_copy = it->second;
    }

    for (const auto& entry : entries_copy) {
        entry.fn(ctx);
    }
}

bool CallbackManager::has_callbacks(CallbackEvent event) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = callbacks_.find(event);
    return it != callbacks_.end() && !it->second.empty();
}

size_t CallbackManager::callback_count(CallbackEvent event) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = callbacks_.find(event);
    if (it == callbacks_.end()) return 0;
    return it->second.size();
}

void CallbackManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.clear();
}

} // namespace ivit
