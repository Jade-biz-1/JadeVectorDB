#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <atomic>
#include <thread>
#include <condition_variable>

namespace jadevectordb {

// Forward declaration
class MemoryMappedVectorStore;

/**
 * @brief Compaction statistics
 */
struct CompactionStats {
    uint64_t total_vectors = 0;
    uint64_t active_vectors = 0;
    uint64_t deleted_vectors = 0;
    uint64_t bytes_before = 0;
    uint64_t bytes_after = 0;
    uint64_t bytes_reclaimed = 0;
    double duration_seconds = 0.0;
    bool success = false;
    std::string error_message;
};

/**
 * @brief Compaction policy configuration
 */
struct CompactionPolicy {
    // Size-based trigger
    uint64_t min_file_size_bytes = 10 * 1024 * 1024;  // 10 MB minimum
    double min_deleted_ratio = 0.2;  // 20% deleted vectors
    
    // Time-based trigger
    uint64_t max_age_since_last_compaction_seconds = 24 * 3600;  // 24 hours
    
    // Resource limits
    uint64_t max_compaction_duration_seconds = 300;  // 5 minutes
    bool allow_concurrent_compaction = false;  // One at a time
    
    // Background compaction
    bool enable_background_compaction = true;
    uint64_t background_check_interval_seconds = 3600;  // 1 hour
};

/**
 * @brief Vector file compactor for space reclamation
 * 
 * Compacts vector files by:
 * - Removing deleted vector entries
 * - Defragmenting vector data
 * - Rewriting file with only active vectors
 * - Atomic replacement of old file with compacted file
 * 
 * Thread-safe and can run in background.
 */
class VectorFileCompactor {
public:
    /**
     * @brief Constructor
     * @param store Reference to memory-mapped vector store
     * @param policy Compaction policy configuration
     */
    explicit VectorFileCompactor(MemoryMappedVectorStore& store,
                                const CompactionPolicy& policy = CompactionPolicy{});
    
    /**
     * @brief Destructor - stops background compaction thread
     */
    ~VectorFileCompactor();

    // Disable copy and move
    VectorFileCompactor(const VectorFileCompactor&) = delete;
    VectorFileCompactor& operator=(const VectorFileCompactor&) = delete;

    /**
     * @brief Compact a specific database file
     * @param database_id Database identifier
     * @param force Force compaction even if policy not met
     * @return Compaction statistics
     */
    CompactionStats compact_database(const std::string& database_id, bool force = false);

    /**
     * @brief Check if database needs compaction
     * @param database_id Database identifier
     * @return true if compaction recommended
     */
    bool needs_compaction(const std::string& database_id) const;

    /**
     * @brief Start background compaction thread
     * 
     * Periodically checks all databases and compacts those
     * that meet the compaction policy criteria.
     */
    void start_background_compaction();

    /**
     * @brief Stop background compaction thread
     */
    void stop_background_compaction();

    /**
     * @brief Check if background compaction is running
     */
    bool is_background_compaction_running() const;

    /**
     * @brief Get compaction policy
     */
    const CompactionPolicy& get_policy() const { return policy_; }

    /**
     * @brief Update compaction policy
     */
    void set_policy(const CompactionPolicy& policy);

    /**
     * @brief Set callback for compaction events
     * @param callback Function called when compaction completes
     */
    void set_compaction_callback(std::function<void(const std::string&, const CompactionStats&)> callback);

private:
    // Reference to vector store
    MemoryMappedVectorStore& store_;
    
    // Compaction policy
    CompactionPolicy policy_;
    std::mutex policy_mutex_;
    
    // Background compaction
    std::atomic<bool> background_running_{false};
    std::unique_ptr<std::thread> background_thread_;
    std::condition_variable background_cv_;
    std::mutex background_mutex_;
    
    // Compaction callback
    std::function<void(const std::string&, const CompactionStats&)> compaction_callback_;
    std::mutex callback_mutex_;
    
    // Compaction state
    std::atomic<bool> compaction_in_progress_{false};
    std::string current_database_being_compacted_;
    std::mutex compaction_mutex_;
    
    // Helper methods
    void background_compaction_loop();
    std::vector<std::string> get_databases_needing_compaction() const;
    CompactionStats perform_compaction(const std::string& database_id);
    
    // File operations
    bool create_compacted_file(const std::string& database_id,
                               const std::string& temp_file_path);
    bool atomic_replace_file(const std::string& old_path,
                            const std::string& new_path);
    
    // Metrics collection
    void collect_compaction_metrics(const std::string& database_id,
                                   CompactionStats& stats) const;
    std::string get_vector_file_path(const std::string& database_id) const;
};

} // namespace jadevectordb
