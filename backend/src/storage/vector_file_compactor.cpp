#include "vector_file_compactor.h"
#include "memory_mapped_vector_store.h"
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

namespace jadevectordb {

VectorFileCompactor::VectorFileCompactor(MemoryMappedVectorStore& store,
                                       const CompactionPolicy& policy)
    : store_(store), policy_(policy) {
}

VectorFileCompactor::~VectorFileCompactor() {
    stop_background_compaction();
}

CompactionStats VectorFileCompactor::compact_database(const std::string& database_id, bool force) {
    std::lock_guard<std::mutex> lock(compaction_mutex_);
    
    if (compaction_in_progress_ && !force) {
        CompactionStats stats;
        stats.success = false;
        stats.error_message = "Compaction already in progress for another database";
        return stats;
    }
    
    if (!force && !needs_compaction(database_id)) {
        CompactionStats stats;
        stats.success = false;
        stats.error_message = "Database does not meet compaction policy criteria";
        return stats;
    }
    
    compaction_in_progress_ = true;
    current_database_being_compacted_ = database_id;
    
    auto stats = perform_compaction(database_id);
    
    compaction_in_progress_ = false;
    current_database_being_compacted_.clear();
    
    // Notify callback if set
    {
        std::lock_guard<std::mutex> cb_lock(callback_mutex_);
        if (compaction_callback_) {
            compaction_callback_(database_id, stats);
        }
    }
    
    return stats;
}

bool VectorFileCompactor::needs_compaction(const std::string& database_id) const {
    // Check if database exists
    if (!store_.has_database(database_id)) {
        return false;
    }
    
    // Get database statistics
    size_t active_count = store_.list_vector_ids(database_id).size();  // Count active vectors
    size_t deleted_count = store_.get_deleted_count(database_id);
    
    // Total vectors includes active + deleted
    size_t total_vectors = active_count + deleted_count;
    
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(policy_mutex_));
    
    // Check minimum file size - need at least some vectors to compact
    if (total_vectors < 10) {
        return false;  // Too small to bother compacting
    }
    
    // Check if we have any deleted vectors
    if (deleted_count == 0) {
        return false;  // Nothing to compact
    }
    
    // Calculate deleted ratio
    double deleted_ratio = static_cast<double>(deleted_count) / static_cast<double>(total_vectors);
    
    // Compare against policy threshold
    return deleted_ratio >= policy_.min_deleted_ratio;
}

void VectorFileCompactor::start_background_compaction() {
    if (background_running_) {
        return;  // Already running
    }
    
    {
        std::lock_guard<std::mutex> lock(policy_mutex_);
        if (!policy_.enable_background_compaction) {
            return;  // Background compaction disabled
        }
    }
    
    background_running_ = true;
    background_thread_ = std::make_unique<std::thread>(
        &VectorFileCompactor::background_compaction_loop, this);
}

void VectorFileCompactor::stop_background_compaction() {
    if (!background_running_) {
        return;  // Not running
    }
    
    background_running_ = false;
    background_cv_.notify_all();
    
    if (background_thread_ && background_thread_->joinable()) {
        background_thread_->join();
    }
    background_thread_.reset();
}

bool VectorFileCompactor::is_background_compaction_running() const {
    return background_running_;
}

void VectorFileCompactor::set_policy(const CompactionPolicy& policy) {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    policy_ = policy;
}

void VectorFileCompactor::set_compaction_callback(
    std::function<void(const std::string&, const CompactionStats&)> callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    compaction_callback_ = std::move(callback);
}

void VectorFileCompactor::background_compaction_loop() {
    while (background_running_) {
        try {
            // Get databases that need compaction
            auto databases = get_databases_needing_compaction();
            
            // Compact each database that needs it
            for (const auto& db_id : databases) {
                if (!background_running_) {
                    break;
                }
                
                try {
                    auto stats = compact_database(db_id, false);
                    if (stats.success) {
                        std::cout << "[Compactor] Successfully compacted database " << db_id
                                 << ", reclaimed " << stats.bytes_reclaimed << " bytes" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Compactor] Error compacting database " << db_id
                             << ": " << e.what() << std::endl;
                }
            }
            
            // Wait for next check interval
            std::unique_lock<std::mutex> lock(background_mutex_);
            uint64_t interval_seconds;
            {
                std::lock_guard<std::mutex> policy_lock(policy_mutex_);
                interval_seconds = policy_.background_check_interval_seconds;
            }
            
            background_cv_.wait_for(lock, std::chrono::seconds(interval_seconds),
                                   [this] { return !background_running_; });
        } catch (const std::exception& e) {
            std::cerr << "[Compactor] Background compaction error: " << e.what() << std::endl;
            
            // Wait a bit before retrying
            std::unique_lock<std::mutex> lock(background_mutex_);
            background_cv_.wait_for(lock, std::chrono::seconds(60),
                                   [this] { return !background_running_; });
        }
    }
}

std::vector<std::string> VectorFileCompactor::get_databases_needing_compaction() const {
    std::vector<std::string> result;
    
    // Get all databases from the store
    auto all_databases = store_.list_databases();
    
    // Check each database to see if it needs compaction
    for (const auto& database_id : all_databases) {
        if (needs_compaction(database_id)) {
            result.push_back(database_id);
        }
    }
    
    return result;
}

CompactionStats VectorFileCompactor::perform_compaction(const std::string& database_id) {
    CompactionStats stats;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Collect pre-compaction metrics
        collect_compaction_metrics(database_id, stats);
        
        // Flush any pending changes
        store_.flush(database_id, true);
        
        // Get all ACTIVE vector IDs before closing file
        auto active_ids = store_.list_vector_ids(database_id);
        int dimension = store_.get_dimension(database_id);
        
        if (active_ids.empty()) {
            // No active vectors, just mark success
            stats.bytes_after = stats.bytes_before;
            stats.bytes_reclaimed = 0;
            stats.success = true;
            auto end_time = std::chrono::steady_clock::now();
            stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
            return stats;
        }
        
        // Retrieve all active vectors while file is still open
        std::vector<std::pair<std::string, std::vector<float>>> vectors_to_keep;
        vectors_to_keep.reserve(active_ids.size());
        
        for (const auto& vid : active_ids) {
            auto vec_data = store_.retrieve_vector(database_id, vid);
            if (vec_data.has_value()) {
                vectors_to_keep.emplace_back(vid, std::move(vec_data.value()));
            }
        }
        
        // Close the original file
        store_.close_vector_file(database_id, true);
        
        // Backup original file
        std::string original_file = get_vector_file_path(database_id);
        std::string backup_file = original_file + ".backup";
        
        try {
            fs::copy_file(original_file, backup_file, fs::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            stats.success = false;
            stats.error_message = std::string("Failed to backup original file: ") + e.what();
            store_.open_vector_file(database_id);  // Reopen original
            return stats;
        }
        
        // Delete original file
        try {
            fs::remove(original_file);
        } catch (const std::exception& e) {
            stats.success = false;
            stats.error_message = std::string("Failed to remove original file: ") + e.what();
            store_.open_vector_file(database_id);  // Reopen original
            return stats;
        }
        
        // Create new compacted file with exact capacity for active vectors
        size_t new_capacity = std::max(vectors_to_keep.size() * 2, size_t(100));  // 2x for growth
        
        if (!store_.create_vector_file(database_id, dimension, new_capacity)) {
            stats.success = false;
            stats.error_message = "Failed to create new compacted file";
            
            // Restore from backup
            try {
                fs::rename(backup_file, original_file);
                store_.open_vector_file(database_id);
            } catch (...) {
                // Critical failure
            }
            return stats;
        }
        
        // Store all active vectors into new file
        // This will automatically rebuild the indices as vectors are stored
        size_t stored_count = 0;
        for (const auto& [vid, vec_data] : vectors_to_keep) {
            if (store_.store_vector(database_id, vid, vec_data)) {
                stored_count++;
            } else {
                std::cerr << "Failed to store vector: " << vid << std::endl;
            }
        }
        
        std::cerr << "Stored " << stored_count << " / " << vectors_to_keep.size() << " vectors" << std::endl;
        
        // Flush to ensure all data is written
        store_.flush(database_id, true);
        
        if (stored_count != vectors_to_keep.size()) {
            stats.success = false;
            stats.error_message = "Failed to store all vectors during compaction (" + 
                                std::to_string(stored_count) + "/" + std::to_string(vectors_to_keep.size()) + ")";
            
            // Restore from backup
            store_.close_vector_file(database_id, false);
            try {
                fs::remove(original_file);
                fs::rename(backup_file, original_file);
                store_.open_vector_file(database_id);
            } catch (...) {
                // Critical failure
            }
            return stats;
        }
        
        // Success! Remove backup
        try {
            fs::remove(backup_file);
        } catch (...) {
            // Backup removal failed, but compaction succeeded
        }
        
        // Collect post-compaction metrics
        CompactionStats post_stats;
        collect_compaction_metrics(database_id, post_stats);
        
        stats.bytes_after = post_stats.bytes_before;
        stats.bytes_reclaimed = stats.bytes_before - stats.bytes_after;
        stats.active_vectors = post_stats.active_vectors;
        stats.success = true;
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
    } catch (const std::exception& e) {
        stats.success = false;
        stats.error_message = std::string("Compaction exception: ") + e.what();
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    }
    
    return stats;
}

bool VectorFileCompactor::create_compacted_file(const std::string& database_id,
                                               const std::string& temp_file_path) {
    try {
        // Get all active vector IDs
        auto vector_ids = store_.list_vector_ids(database_id);
        int dimension = store_.get_dimension(database_id);
        
        if (dimension == 0) {
            return false;
        }
        
        // If no vectors, still create empty file
        if (vector_ids.empty()) {
            // Create minimal file
            std::ofstream ofs(temp_file_path, std::ios::binary);
            return ofs.good();
        }
        
        // For now, copy the entire file and let the store handle compaction internally
        // A full implementation would:
        // 1. Create new memory-mapped file at temp_file_path
        // 2. Copy only active vectors (using list_vector_ids)
        // 3. Write compacted file with no deleted entries
        
        // Simple approach: copy original file (compaction happens on rewrite)
        std::string original_file = get_vector_file_path(database_id);
        
        std::ifstream src(original_file, std::ios::binary);
        std::ofstream dst(temp_file_path, std::ios::binary);
        
        if (!src || !dst) {
            return false;
        }
        
        dst << src.rdbuf();
        
        return dst.good();
    } catch (const std::exception& e) {
        std::cerr << "[Compactor] Error creating compacted file: " << e.what() << std::endl;
        return false;
    }
}

bool VectorFileCompactor::atomic_replace_file(const std::string& old_path,
                                             const std::string& new_path) {
    try {
        // Create backup of original file
        std::string backup_path = old_path + ".backup";
        
        // Copy original to backup
        fs::copy_file(old_path, backup_path, fs::copy_options::overwrite_existing);
        
        // Rename compacted file to original
        fs::rename(new_path, old_path);
        
        // Remove backup if replacement successful
        fs::remove(backup_path);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Compactor] Error in atomic file replacement: " << e.what() << std::endl;
        return false;
    }
}

void VectorFileCompactor::collect_compaction_metrics(const std::string& database_id,
                                                    CompactionStats& stats) const {
    try {
        stats.active_vectors = store_.get_vector_count(database_id);
        stats.deleted_vectors = store_.get_deleted_count(database_id);
        stats.total_vectors = stats.active_vectors + stats.deleted_vectors;
        
        // Get file size
        std::string file_path = get_vector_file_path(database_id);
        if (fs::exists(file_path)) {
            stats.bytes_before = fs::file_size(file_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "[Compactor] Error collecting metrics: " << e.what() << std::endl;
    }
}

std::string VectorFileCompactor::get_vector_file_path(const std::string& database_id) const {
    // Get the actual path from the store to avoid hardcoded paths
    return store_.get_database_vector_file_path(database_id);
}

} // namespace jadevectordb
