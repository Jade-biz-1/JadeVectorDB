#include "snapshot_manager.h"
#include "memory_mapped_vector_store.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace jadevectordb {

namespace fs = std::filesystem;

SnapshotManager::SnapshotManager(MemoryMappedVectorStore& store,
                                 const std::string& snapshot_directory)
    : store_(store)
    , snapshot_directory_(snapshot_directory) {
    
    // Create snapshot directory
    fs::create_directories(snapshot_directory_);
}

SnapshotManager::~SnapshotManager() = default;

std::string SnapshotManager::generate_snapshot_id() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return "snapshot_" + std::to_string(ms);
}

std::string SnapshotManager::get_snapshot_data_path(const std::string& snapshot_id) const {
    return snapshot_directory_ + "/" + snapshot_id + ".snap";
}

std::string SnapshotManager::get_snapshot_meta_path(const std::string& snapshot_id) const {
    return snapshot_directory_ + "/" + snapshot_id + ".meta";
}

std::string SnapshotManager::calculate_file_checksum(const std::string& file_path) const {
    // Simple checksum - in production, use SHA-256
    std::ifstream file(file_path, std::ios::binary);
    if (!file) return "";
    
    uint64_t checksum = 0;
    char buffer[4096];
    
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        size_t bytes_read = file.gcount();
        for (size_t i = 0; i < bytes_read; i++) {
            checksum = checksum * 31 + static_cast<uint8_t>(buffer[i]);
        }
    }
    
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << checksum;
    return ss.str();
}

bool SnapshotManager::write_metadata(const SnapshotMetadata& metadata) {
    std::string meta_path = get_snapshot_meta_path(metadata.snapshot_id);
    std::ofstream meta_file(meta_path);
    
    if (!meta_file) return false;
    
    meta_file << "snapshot_id=" << metadata.snapshot_id << "\n";
    meta_file << "database_id=" << metadata.database_id << "\n";
    meta_file << "timestamp=" << metadata.timestamp << "\n";
    meta_file << "vector_count=" << metadata.vector_count << "\n";
    meta_file << "file_size=" << metadata.file_size << "\n";
    meta_file << "checksum=" << metadata.checksum << "\n";
    meta_file << "description=" << metadata.description << "\n";
    
    return meta_file.good();
}

std::optional<SnapshotMetadata> SnapshotManager::read_metadata(const std::string& meta_path) {
    std::ifstream meta_file(meta_path);
    if (!meta_file) return std::nullopt;
    
    SnapshotMetadata metadata;
    std::string line;
    
    while (std::getline(meta_file, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        if (key == "snapshot_id") metadata.snapshot_id = value;
        else if (key == "database_id") metadata.database_id = value;
        else if (key == "timestamp") metadata.timestamp = std::stoull(value);
        else if (key == "vector_count") metadata.vector_count = std::stoull(value);
        else if (key == "file_size") metadata.file_size = std::stoull(value);
        else if (key == "checksum") metadata.checksum = value;
        else if (key == "description") metadata.description = value;
    }
    
    return metadata;
}

SnapshotStats SnapshotManager::create_snapshot(const std::string& database_id,
                                               const std::string& description) {
    SnapshotStats stats;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Generate snapshot ID
        std::string snapshot_id = generate_snapshot_id();
        
        // Get database file path
        std::string db_file_path = store_.get_database_vector_file_path(database_id);
        
        if (!fs::exists(db_file_path)) {
            stats.error_message = "Database file not found: " + db_file_path;
            return stats;
        }
        
        // Flush database to ensure consistency
        store_.flush(database_id, true);
        
        // Copy database file to snapshot
        std::string snapshot_path = get_snapshot_data_path(snapshot_id);
        fs::copy_file(db_file_path, snapshot_path, fs::copy_options::overwrite_existing);
        
        // Get file info
        auto file_size = fs::file_size(snapshot_path);
        auto vector_count = store_.get_vector_count(database_id);
        
        // Calculate checksum
        std::string checksum = calculate_file_checksum(snapshot_path);
        
        // Create metadata
        SnapshotMetadata metadata;
        metadata.snapshot_id = snapshot_id;
        metadata.database_id = database_id;
        metadata.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        metadata.vector_count = vector_count;
        metadata.file_size = file_size;
        metadata.checksum = checksum;
        metadata.description = description;
        
        // Write metadata
        if (!write_metadata(metadata)) {
            stats.error_message = "Failed to write metadata";
            fs::remove(snapshot_path);
            return stats;
        }
        
        // Update stats
        stats.vectors_snapshotted = vector_count;
        stats.bytes_written = file_size;
        stats.metadata = metadata;
        stats.success = true;
        
    } catch (const std::exception& e) {
        stats.error_message = std::string("Exception: ") + e.what();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    return stats;
}

SnapshotRestoreStats SnapshotManager::restore_from_snapshot(const std::string& snapshot_id,
                                                            const std::string& target_database_id) {
    SnapshotRestoreStats stats;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Get snapshot paths
        std::string snapshot_path = get_snapshot_data_path(snapshot_id);
        std::string meta_path = get_snapshot_meta_path(snapshot_id);
        
        if (!fs::exists(snapshot_path)) {
            stats.error_message = "Snapshot file not found: " + snapshot_path;
            return stats;
        }
        
        if (!fs::exists(meta_path)) {
            stats.error_message = "Snapshot metadata not found: " + meta_path;
            return stats;
        }
        
        // Read metadata
        auto metadata = read_metadata(meta_path);
        if (!metadata) {
            stats.error_message = "Failed to read snapshot metadata";
            return stats;
        }
        
        // Verify checksum
        std::string current_checksum = calculate_file_checksum(snapshot_path);
        if (current_checksum != metadata->checksum) {
            stats.error_message = "Snapshot checksum mismatch - data may be corrupted";
            return stats;
        }
        
        // Close target database if open
        store_.close_vector_file(target_database_id, true);
        
        // Get target database file path
        std::string target_path = store_.get_database_vector_file_path(target_database_id);
        
        // Create target directory if needed
        fs::create_directories(fs::path(target_path).parent_path());
        
        // Copy snapshot to target location
        fs::copy_file(snapshot_path, target_path, fs::copy_options::overwrite_existing);
        
        // Reopen database
        if (!store_.open_vector_file(target_database_id)) {
            stats.error_message = "Failed to open restored database";
            return stats;
        }
        
        // Update stats
        stats.vectors_restored = metadata->vector_count;
        stats.bytes_read = metadata->file_size;
        stats.success = true;
        
    } catch (const std::exception& e) {
        stats.error_message = std::string("Exception: ") + e.what();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    return stats;
}

std::vector<SnapshotMetadata> SnapshotManager::list_snapshots(const std::string& database_id) {
    std::vector<SnapshotMetadata> snapshots;
    
    if (!fs::exists(snapshot_directory_)) {
        return snapshots;
    }
    
    for (const auto& entry : fs::directory_iterator(snapshot_directory_)) {
        if (entry.path().extension() == ".meta") {
            auto metadata = read_metadata(entry.path().string());
            if (metadata) {
                // Filter by database_id if specified
                if (database_id.empty() || metadata->database_id == database_id) {
                    snapshots.push_back(*metadata);
                }
            }
        }
    }
    
    // Sort by timestamp (most recent first)
    std::sort(snapshots.begin(), snapshots.end(),
             [](const SnapshotMetadata& a, const SnapshotMetadata& b) {
                 return a.timestamp > b.timestamp;
             });
    
    return snapshots;
}

std::optional<SnapshotMetadata> SnapshotManager::get_snapshot_metadata(const std::string& snapshot_id) {
    std::string meta_path = get_snapshot_meta_path(snapshot_id);
    return read_metadata(meta_path);
}

bool SnapshotManager::delete_snapshot(const std::string& snapshot_id) {
    std::string snapshot_path = get_snapshot_data_path(snapshot_id);
    std::string meta_path = get_snapshot_meta_path(snapshot_id);
    
    std::error_code ec;
    bool data_deleted = fs::remove(snapshot_path, ec);
    bool meta_deleted = fs::remove(meta_path, ec);
    
    return data_deleted && meta_deleted;
}

bool SnapshotManager::verify_snapshot(const std::string& snapshot_id) {
    std::string snapshot_path = get_snapshot_data_path(snapshot_id);
    std::string meta_path = get_snapshot_meta_path(snapshot_id);
    
    if (!fs::exists(snapshot_path) || !fs::exists(meta_path)) {
        return false;
    }
    
    auto metadata = read_metadata(meta_path);
    if (!metadata) {
        return false;
    }
    
    // Verify checksum
    std::string current_checksum = calculate_file_checksum(snapshot_path);
    return current_checksum == metadata->checksum;
}

int SnapshotManager::cleanup_old_snapshots(const std::string& database_id, int keep_count) {
    auto snapshots = list_snapshots(database_id);
    
    if (snapshots.size() <= static_cast<size_t>(keep_count)) {
        return 0;
    }
    
    int deleted_count = 0;
    for (size_t i = keep_count; i < snapshots.size(); i++) {
        if (delete_snapshot(snapshots[i].snapshot_id)) {
            deleted_count++;
        }
    }
    
    return deleted_count;
}

uint64_t SnapshotManager::get_total_snapshot_size() {
    uint64_t total_size = 0;
    
    if (!fs::exists(snapshot_directory_)) {
        return 0;
    }
    
    for (const auto& entry : fs::directory_iterator(snapshot_directory_)) {
        if (entry.is_regular_file()) {
            total_size += fs::file_size(entry);
        }
    }
    
    return total_size;
}

} // namespace jadevectordb
