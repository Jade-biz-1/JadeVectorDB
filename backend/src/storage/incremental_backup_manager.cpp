#include "incremental_backup_manager.h"
#include "memory_mapped_vector_store.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>

namespace fs = std::filesystem;

namespace jadevectordb {

IncrementalBackupManager::IncrementalBackupManager(MemoryMappedVectorStore& store,
                                                 const std::string& backup_directory)
    : store_(store), backup_directory_(backup_directory) {
    // Create backup directory if it doesn't exist
    fs::create_directories(backup_directory_);
}

IncrementalBackupManager::~IncrementalBackupManager() {
    // Cleanup if needed
}

BackupStats IncrementalBackupManager::create_full_backup(const std::string& database_id) {
    BackupStats stats;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Flush any pending changes
        store_.flush(database_id, true);
        
        // Get all vector IDs
        auto vector_ids = store_.list_vector_ids(database_id);
        
        if (vector_ids.empty()) {
            stats.success = false;
            stats.error_message = "No vectors to backup";
            return stats;
        }
        
        // Generate backup ID
        std::string backup_id = generate_backup_id();
        std::string backup_path = get_backup_path(backup_id);
        
        // Write backup file
        bool written = write_backup_file(backup_path, database_id, vector_ids);
        if (!written) {
            stats.success = false;
            stats.error_message = "Failed to write backup file";
            return stats;
        }
        
        // Calculate checksum
        std::string checksum = calculate_checksum(backup_path);
        
        // Create metadata
        BackupMetadata metadata;
        metadata.backup_id = backup_id;
        metadata.database_id = database_id;
        metadata.timestamp = get_current_timestamp();
        metadata.is_full_backup = true;
        metadata.parent_backup_id = "";
        metadata.vector_count = vector_ids.size();
        metadata.size_bytes = fs::file_size(backup_path);
        metadata.checksum = checksum;
        
        // Write metadata
        if (!write_metadata(metadata)) {
            stats.success = false;
            stats.error_message = "Failed to write backup metadata";
            return stats;
        }
        
        // Update backup state
        {
            std::lock_guard<std::mutex> lock(backup_state_mutex_);
            last_backup_time_[database_id] = metadata.timestamp;
            last_backup_id_[database_id] = backup_id;
        }
        
        // Clear change tracking after successful backup
        clear_change_tracking(database_id);
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        stats.vectors_backed_up = vector_ids.size();
        stats.bytes_written = metadata.size_bytes;
        stats.success = true;
        stats.metadata = metadata;
        
    } catch (const std::exception& e) {
        stats.success = false;
        stats.error_message = std::string("Backup exception: ") + e.what();
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    }
    
    return stats;
}

BackupStats IncrementalBackupManager::create_incremental_backup(const std::string& database_id) {
    BackupStats stats;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Check if we have a base backup
        std::string parent_backup_id;
        {
            std::lock_guard<std::mutex> lock(backup_state_mutex_);
            auto it = last_backup_id_.find(database_id);
            if (it == last_backup_id_.end()) {
                // No previous backup, create full backup instead
                return create_full_backup(database_id);
            }
            parent_backup_id = it->second;
        }
        
        // Get changed vectors
        std::vector<std::string> changed_vector_ids;
        {
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            auto it = changed_vectors_.find(database_id);
            if (it != changed_vectors_.end()) {
                changed_vector_ids.assign(it->second.begin(), it->second.end());
            }
        }
        
        if (changed_vector_ids.empty()) {
            stats.success = false;
            stats.error_message = "No changes to backup";
            return stats;
        }
        
        // Flush pending changes
        store_.flush(database_id, true);
        
        // Generate backup ID
        std::string backup_id = generate_backup_id();
        std::string backup_path = get_backup_path(backup_id);
        
        // Write incremental backup file
        bool written = write_backup_file(backup_path, database_id, changed_vector_ids);
        if (!written) {
            stats.success = false;
            stats.error_message = "Failed to write incremental backup file";
            return stats;
        }
        
        // Calculate checksum
        std::string checksum = calculate_checksum(backup_path);
        
        // Create metadata
        BackupMetadata metadata;
        metadata.backup_id = backup_id;
        metadata.database_id = database_id;
        metadata.timestamp = get_current_timestamp();
        metadata.is_full_backup = false;
        metadata.parent_backup_id = parent_backup_id;
        metadata.vector_count = changed_vector_ids.size();
        metadata.size_bytes = fs::file_size(backup_path);
        metadata.checksum = checksum;
        
        // Write metadata
        if (!write_metadata(metadata)) {
            stats.success = false;
            stats.error_message = "Failed to write backup metadata";
            return stats;
        }
        
        // Update backup state
        {
            std::lock_guard<std::mutex> lock(backup_state_mutex_);
            last_backup_time_[database_id] = metadata.timestamp;
            last_backup_id_[database_id] = backup_id;
        }
        
        // Clear change tracking after successful backup
        clear_change_tracking(database_id);
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        stats.vectors_backed_up = changed_vector_ids.size();
        stats.bytes_written = metadata.size_bytes;
        stats.success = true;
        stats.metadata = metadata;
        
    } catch (const std::exception& e) {
        stats.success = false;
        stats.error_message = std::string("Incremental backup exception: ") + e.what();
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    }
    
    return stats;
}

RestoreStats IncrementalBackupManager::restore_from_backup(const std::string& backup_id,
                                                          const std::string& target_database_id) {
    RestoreStats stats;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Get backup chain
        auto backup_chain = get_backup_chain(backup_id);
        
        if (backup_chain.empty()) {
            stats.success = false;
            stats.error_message = "Invalid backup chain";
            return stats;
        }
        
        // Restore each backup in the chain
        for (const auto& bid : backup_chain) {
            std::string backup_path = get_backup_path(bid);
            
            // Verify backup
            if (!verify_backup(bid)) {
                stats.success = false;
                stats.error_message = "Backup verification failed for " + bid;
                return stats;
            }
            
            // Read backup file
            std::vector<std::pair<std::string, std::vector<float>>> vectors;
            if (!read_backup_file(backup_path, vectors)) {
                stats.success = false;
                stats.error_message = "Failed to read backup file " + bid;
                return stats;
            }
            
            // Store vectors
            for (const auto& [vector_id, values] : vectors) {
                if (!store_.store_vector(target_database_id, vector_id, values)) {
                    std::cerr << "[Backup] Warning: Failed to restore vector " << vector_id << std::endl;
                }
            }
            
            stats.vectors_restored += vectors.size();
        }
        
        // Flush restored data
        store_.flush(target_database_id, true);
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        stats.success = true;
        
    } catch (const std::exception& e) {
        stats.success = false;
        stats.error_message = std::string("Restore exception: ") + e.what();
        
        auto end_time = std::chrono::steady_clock::now();
        stats.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    }
    
    return stats;
}

std::vector<BackupMetadata> IncrementalBackupManager::list_backups(const std::string& database_id) const {
    std::vector<BackupMetadata> result;
    
    try {
        for (const auto& entry : fs::directory_iterator(backup_directory_)) {
            if (entry.path().extension() == ".meta") {
                std::string backup_id = entry.path().stem().string();
                auto metadata = read_metadata(backup_id);
                
                if (metadata && (database_id.empty() || metadata->database_id == database_id)) {
                    result.push_back(*metadata);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error listing backups: " << e.what() << std::endl;
    }
    
    return result;
}

bool IncrementalBackupManager::delete_backup(const std::string& backup_id) {
    try {
        std::string backup_path = get_backup_path(backup_id);
        std::string metadata_path = get_metadata_path(backup_id);
        
        fs::remove(backup_path);
        fs::remove(metadata_path);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error deleting backup: " << e.what() << std::endl;
        return false;
    }
}

bool IncrementalBackupManager::verify_backup(const std::string& backup_id) const {
    try {
        auto metadata = read_metadata(backup_id);
        if (!metadata) {
            return false;
        }
        
        std::string backup_path = get_backup_path(backup_id);
        std::string checksum = calculate_checksum(backup_path);
        
        return checksum == metadata->checksum;
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error verifying backup: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> IncrementalBackupManager::get_backup_chain(const std::string& backup_id) const {
    std::vector<std::string> chain;
    
    try {
        std::string current_id = backup_id;
        
        while (!current_id.empty()) {
            chain.insert(chain.begin(), current_id);
            
            auto metadata = read_metadata(current_id);
            if (!metadata) {
                return {};  // Invalid chain
            }
            
            if (metadata->is_full_backup) {
                break;  // Reached the base backup
            }
            
            current_id = metadata->parent_backup_id;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error building backup chain: " << e.what() << std::endl;
        return {};
    }
    
    return chain;
}

void IncrementalBackupManager::enable_change_tracking(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(tracking_mutex_);
    tracking_enabled_[database_id] = true;
}

void IncrementalBackupManager::disable_change_tracking(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(tracking_mutex_);
    tracking_enabled_[database_id] = false;
}

bool IncrementalBackupManager::is_change_tracking_enabled(const std::string& database_id) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(tracking_mutex_));
    auto it = tracking_enabled_.find(database_id);
    return it != tracking_enabled_.end() && it->second;
}

void IncrementalBackupManager::record_vector_change(const std::string& database_id,
                                                   const std::string& vector_id) {
    std::lock_guard<std::mutex> lock(tracking_mutex_);
    
    auto it = tracking_enabled_.find(database_id);
    if (it != tracking_enabled_.end() && it->second) {
        changed_vectors_[database_id].insert(vector_id);
    }
}

void IncrementalBackupManager::clear_change_tracking(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(tracking_mutex_);
    changed_vectors_[database_id].clear();
}

std::string IncrementalBackupManager::generate_backup_id() const {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    // Add counter to ensure uniqueness even if backups created in same millisecond
    uint64_t counter = backup_counter_.fetch_add(1);
    
    std::ostringstream oss;
    oss << "backup_" << timestamp;
    if (counter > 0) {
        oss << "_" << counter;
    }
    return oss.str();
}

std::string IncrementalBackupManager::get_backup_path(const std::string& backup_id) const {
    return backup_directory_ + "/" + backup_id + ".jvdb";
}

std::string IncrementalBackupManager::get_metadata_path(const std::string& backup_id) const {
    return backup_directory_ + "/" + backup_id + ".meta";
}

bool IncrementalBackupManager::write_backup_file(const std::string& backup_path,
                                                 const std::string& database_id,
                                                 const std::vector<std::string>& vector_ids) {
    try {
        std::ofstream file(backup_path, std::ios::binary);
        if (!file) {
            return false;
        }
        
        // Retrieve and write vectors
        for (const auto& vector_id : vector_ids) {
            auto values_opt = store_.retrieve_vector(database_id, vector_id);
            if (!values_opt) {
                continue;  // Skip if vector not found
            }
            
            const auto& values = *values_opt;
            
            // Write vector ID length and ID
            uint32_t id_len = vector_id.size();
            file.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
            file.write(vector_id.data(), id_len);
            
            // Write vector dimension
            uint32_t dim = values.size();
            file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            
            // Write vector values
            file.write(reinterpret_cast<const char*>(values.data()), dim * sizeof(float));
        }
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error writing backup file: " << e.what() << std::endl;
        return false;
    }
}

bool IncrementalBackupManager::read_backup_file(const std::string& backup_path,
                                               std::vector<std::pair<std::string, std::vector<float>>>& vectors) const {
    try {
        std::ifstream file(backup_path, std::ios::binary);
        if (!file) {
            return false;
        }
        
        while (file.peek() != EOF) {
            // Read vector ID
            uint32_t id_len;
            file.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
            
            std::string vector_id(id_len, '\0');
            file.read(&vector_id[0], id_len);
            
            // Read vector dimension
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            
            // Read vector values
            std::vector<float> values(dim);
            file.read(reinterpret_cast<char*>(values.data()), dim * sizeof(float));
            
            vectors.emplace_back(std::move(vector_id), std::move(values));
        }
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error reading backup file: " << e.what() << std::endl;
        return false;
    }
}

bool IncrementalBackupManager::write_metadata(const BackupMetadata& metadata) const {
    try {
        std::string meta_path = get_metadata_path(metadata.backup_id);
        std::ofstream file(meta_path);
        
        if (!file) {
            return false;
        }
        
        // Write metadata as JSON-like format
        file << "backup_id:" << metadata.backup_id << "\n";
        file << "database_id:" << metadata.database_id << "\n";
        file << "timestamp:" << metadata.timestamp << "\n";
        file << "is_full_backup:" << (metadata.is_full_backup ? "true" : "false") << "\n";
        file << "parent_backup_id:" << metadata.parent_backup_id << "\n";
        file << "vector_count:" << metadata.vector_count << "\n";
        file << "size_bytes:" << metadata.size_bytes << "\n";
        file << "checksum:" << metadata.checksum << "\n";
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error writing metadata: " << e.what() << std::endl;
        return false;
    }
}

std::optional<BackupMetadata> IncrementalBackupManager::read_metadata(const std::string& backup_id) const {
    try {
        std::string meta_path = get_metadata_path(backup_id);
        std::ifstream file(meta_path);
        
        if (!file) {
            return std::nullopt;
        }
        
        BackupMetadata metadata;
        std::string line;
        
        while (std::getline(file, line)) {
            size_t pos = line.find(':');
            if (pos == std::string::npos) continue;
            
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            if (key == "backup_id") metadata.backup_id = value;
            else if (key == "database_id") metadata.database_id = value;
            else if (key == "timestamp") metadata.timestamp = std::stoull(value);
            else if (key == "is_full_backup") metadata.is_full_backup = (value == "true");
            else if (key == "parent_backup_id") metadata.parent_backup_id = value;
            else if (key == "vector_count") metadata.vector_count = std::stoull(value);
            else if (key == "size_bytes") metadata.size_bytes = std::stoull(value);
            else if (key == "checksum") metadata.checksum = value;
        }
        
        file.close();
        return metadata;
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error reading metadata: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::string IncrementalBackupManager::calculate_checksum(const std::string& file_path) const {
    try {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            return "";
        }
        
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        
        char buffer[8192];
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            SHA256_Update(&sha256, buffer, file.gcount());
        }
        
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256_Final(hash, &sha256);
        
        std::ostringstream oss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        
        return oss.str();
    } catch (const std::exception& e) {
        std::cerr << "[Backup] Error calculating checksum: " << e.what() << std::endl;
        return "";
    }
}

uint64_t IncrementalBackupManager::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
}

} // namespace jadevectordb
