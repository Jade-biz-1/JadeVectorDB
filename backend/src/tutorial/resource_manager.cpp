#include "resource_manager.h"
#include <algorithm>
#include <iostream>

namespace tutorial {

ResourceManager::ResourceManager() {
    // Set up a background thread to clean up expired sessions periodically
    std::thread([this]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::minutes(5)); // Clean up every 5 minutes
            cleanupExpiredSessions();
        }
    }).detach();
}

void ResourceManager::cleanupExpiredSessions() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::system_clock::now();
    for (auto it = session_usages_.begin(); it != session_usages_.end();) {
        if (now - it->second.last_request_time > limits_.session_timeout) {
            it = session_usages_.erase(it);
        } else {
            ++it;
        }
    }
}

bool ResourceManager::isRequestAllowed(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if session exists, if not, create a new one
    auto it = session_usages_.find(session_id);
    if (it == session_usages_.end()) {
        session_usages_[session_id] = ResourceUsage();
        it = session_usages_.find(session_id);
    }
    
    // Update the last request time
    it->second.last_request_time = std::chrono::system_clock::now();
    
    // Check rate limits
    auto now = std::chrono::system_clock::now();
    auto one_minute_ago = now - std::chrono::minutes(1);
    
    // Count requests in the last minute
    size_t recent_requests = 0;
    // For a real implementation, we'd track timestamps of all requests
    // For the tutorial, we'll just check the total count within the minute window
    // In a real implementation, we would store request timestamps per session
    
    // For this tutorial implementation, we'll do a simplified check
    if (it->second.api_calls_made >= limits_.max_api_calls_per_minute) {
        std::cout << "Rate limit exceeded for session " << session_id << std::endl;
        return false;
    }
    
    // Check other limits
    if (it->second.vectors_stored >= limits_.max_vectors_per_session) {
        std::cout << "Vector storage limit exceeded for session " << session_id << std::endl;
        return false;
    }
    
    if (it->second.databases_created >= limits_.max_databases_per_session) {
        std::cout << "Database creation limit exceeded for session " << session_id << std::endl;
        return false;
    }
    
    if (it->second.memory_used_bytes >= limits_.max_memory_per_session_bytes) {
        std::cout << "Memory usage limit exceeded for session " << session_id << std::endl;
        return false;
    }
    
    // Update API call count
    it->second.api_calls_made++;
    
    return true;
}

void ResourceManager::recordRequest(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // This is called after the request is confirmed to be allowed
    // The count is already updated in isRequestAllowed
    auto it = session_usages_.find(session_id);
    if (it != session_usages_.end()) {
        it->second.last_request_time = std::chrono::system_clock::now();
    }
}

bool ResourceManager::isSessionOverLimit(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = session_usages_.find(session_id);
    if (it == session_usages_.end()) {
        return false; // If session doesn't exist, it's not over limit
    }
    
    // Check all limits
    if (it->second.vectors_stored >= limits_.max_vectors_per_session) {
        return true;
    }
    
    if (it->second.databases_created >= limits_.max_databases_per_session) {
        return true;
    }
    
    if (it->second.memory_used_bytes >= limits_.max_memory_per_session_bytes) {
        return true;
    }
    
    return false;
}

bool ResourceManager::updateVectorStorage(const std::string& session_id, size_t vector_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = session_usages_.find(session_id);
    if (it == session_usages_.end()) {
        session_usages_[session_id] = ResourceUsage();
        it = session_usages_.find(session_id);
    }
    
    // Calculate memory usage based on vector size (assuming 4 bytes per float32)
    size_t memory_increase = vector_size * sizeof(float);
    
    // Check if this would exceed limits
    if (it->second.vectors_stored + 1 > limits_.max_vectors_per_session) {
        return false;
    }
    
    if (it->second.memory_used_bytes + memory_increase > limits_.max_memory_per_session_bytes) {
        return false;
    }
    
    // Update resource usage
    it->second.vectors_stored++;
    it->second.memory_used_bytes += memory_increase;
    
    return true;
}

bool ResourceManager::updateDatabaseCreation(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = session_usages_.find(session_id);
    if (it == session_usages_.end()) {
        session_usages_[session_id] = ResourceUsage();
        it = session_usages_.find(session_id);
    }
    
    // Check if this would exceed limits
    if (it->second.databases_created + 1 > limits_.max_databases_per_session) {
        return false;
    }
    
    // Update resource usage
    it->second.databases_created++;
    
    return true;
}

ResourceUsage ResourceManager::getResourceUsage(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = session_usages_.find(session_id);
    if (it != session_usages_.end()) {
        return it->second;
    }
    
    return ResourceUsage(); // Return default if not found
}

void ResourceManager::resetSession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    session_usages_.erase(session_id);
}

ResourceLimits ResourceManager::getResourceLimits() const {
    return limits_;
}

// Singleton access
std::shared_ptr<ResourceManager> getResourceManager() {
    static std::shared_ptr<ResourceManager> instance = std::make_shared<ResourceManager>();
    return instance;
}

} // namespace tutorial