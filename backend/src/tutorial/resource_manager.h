#ifndef TUTORIAL_RESOURCE_MANAGER_H
#define TUTORIAL_RESOURCE_MANAGER_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>

namespace tutorial {

// Structure to track resource usage for a session
struct ResourceUsage {
    size_t api_calls_made = 0;
    size_t vectors_stored = 0;
    size_t databases_created = 0;
    size_t memory_used_bytes = 0;
    std::chrono::system_clock::time_point last_request_time;
    
    ResourceUsage() : last_request_time(std::chrono::system_clock::now()) {}
};

// Resource limits for tutorial environment
struct ResourceLimits {
    size_t max_api_calls_per_minute = 60;      // 1 call per second average
    size_t max_vectors_per_session = 1000;     // Max 1000 vectors per session
    size_t max_databases_per_session = 10;     // Max 10 databases per session
    size_t max_memory_per_session_bytes = 100 * 1024 * 1024; // 100 MB max
    std::chrono::minutes session_timeout{30};  // 30-minute session timeout
};

class ResourceManager {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ResourceUsage> session_usages_;
    ResourceLimits limits_;
    
    // Clean up expired sessions
    void cleanupExpiredSessions();
    
public:
    ResourceManager();
    ~ResourceManager() = default;
    
    // Check if a session is allowed to make a request
    bool isRequestAllowed(const std::string& session_id);
    
    // Record a request for a session
    void recordRequest(const std::string& session_id);
    
    // Check if a session has exceeded its limits
    bool isSessionOverLimit(const std::string& session_id);
    
    // Update resource usage for vector storage operation
    bool updateVectorStorage(const std::string& session_id, size_t vector_size);
    
    // Update resource usage for database creation
    bool updateDatabaseCreation(const std::string& session_id);
    
    // Get current resource usage for a session
    ResourceUsage getResourceUsage(const std::string& session_id);
    
    // Reset a session's resource usage (for testing)
    void resetSession(const std::string& session_id);
    
    // Get current resource limits
    ResourceLimits getResourceLimits() const;
};

// Singleton access to resource manager
std::shared_ptr<ResourceManager> getResourceManager();

} // namespace tutorial

#endif // TUTORIAL_RESOURCE_MANAGER_H