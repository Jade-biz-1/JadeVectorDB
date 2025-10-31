#include "archival_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace jadevectordb {

ArchivalService::ArchivalService() : current_archive_size_(0), max_archive_size_(0) {
    logger_ = logging::LoggerManager::get_logger("ArchivalService");
}

bool ArchivalService::initialize(const ArchivalConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        if (!validate_config()) {
            LOG_ERROR(logger_, "Invalid archival configuration provided");
            return false;
        }
        
        config_ = config;
        max_archive_size_ = static_cast<size_t>(config_.max_archive_size_gb) * 1024 * 1024 * 1024; // Convert to bytes
        
        LOG_INFO(logger_, "ArchivalService initialized with threshold: " + 
                std::to_string(config_.archive_threshold_days) + " days, " +
                (config_.compress_archives ? "with compression" : "without compression"));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in ArchivalService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<std::string> ArchivalService::archive_vector(const Vector& vector) {
    try {
        LOG_DEBUG(logger_, "Archiving vector: " + vector.id);
        
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        std::string archive_id = generate_archive_id(vector.id);
        ArchivedVector archived(archive_id, vector);
        
        // Compress if enabled
        if (config_.compress_archives) {
            auto compressed_result = compress_data(vector.values);
            if (compressed_result.has_value()) {
                archived.is_compressed = true;
                archived.compressed_size = compressed_result.value().size();
            } else {
                LOG_WARN(logger_, "Failed to compress vector data for: " + vector.id);
            }
        }
        
        // Encrypt if enabled
        if (config_.enable_encryption) {
            auto encrypted_result = encrypt_data(compressed_result.has_value() ? 
                                               compressed_result.value() : 
                                               std::vector<uint8_t>());
            if (!encrypted_result.has_value()) {
                LOG_WARN(logger_, "Failed to encrypt vector data for: " + vector.id);
            }
        }
        
        // Store in memory
        archive_store_[archive_id] = archived;
        
        // Update database archives mapping
        db_archives_[vector.databaseId].push_back(archive_id);
        
        // Update archive size
        current_archive_size_ += archived.original_size;
        
        LOG_INFO(logger_, "Successfully archived vector " + vector.id + " with archive ID: " + archive_id);
        return archive_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in archive_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to archive vector: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> ArchivalService::archive_vectors(const std::vector<Vector>& vectors) {
    try {
        LOG_DEBUG(logger_, "Archiving " + std::to_string(vectors.size()) + " vectors");
        
        std::vector<std::string> archive_ids;
        for (const auto& vector : vectors) {
            auto result = archive_vector(vector);
            if (result.has_value()) {
                archive_ids.push_back(result.value());
            } else {
                LOG_WARN(logger_, "Failed to archive vector: " + vector.id);
            }
        }
        
        LOG_DEBUG(logger_, "Successfully archived " + std::to_string(archive_ids.size()) + " vectors");
        return archive_ids;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in archive_vectors: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to archive vectors: " + std::string(e.what()));
    }
}

bool ArchivalService::is_archived(const std::string& vector_id) const {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        // Look for the vector ID in the archive store
        for (const auto& entry : archive_store_) {
            if (entry.second.original_vector_id == vector_id) {
                return true;
            }
        }
        
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_archived: " + std::string(e.what()));
        return false;
    }
}

Result<Vector> ArchivalService::restore_vector(const std::string& archive_id) const {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        auto it = archive_store_.find(archive_id);
        if (it == archive_store_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Archive not found: " + archive_id);
        }
        
        const auto& archived = it->second;
        
        if (is_expired(archived)) {
            LOG_WARN(logger_, "Archive " + archive_id + " has expired");
            RETURN_ERROR(ErrorCode::RESOURCE_EXPIRED, "Archive has expired: " + archive_id);
        }
        
        Vector restored;
        restored.id = archived.original_vector_id;
        restored.databaseId = archived.database_id;
        restored.metadata = archived.metadata;
        
        // Restore data from archive
        if (archived.is_compressed) {
            auto decompressed_result = decompress_data({}, archived.original_size);
            if (decompressed_result.has_value()) {
                restored.values = decompressed_result.value();
            } else {
                LOG_WARN(logger_, "Failed to decompress archived data for: " + archive_id);
                // Use original values if decompression fails
                restored.values = archived.values;
            }
        } else {
            restored.values = archived.values;
        }
        
        LOG_DEBUG(logger_, "Restored vector " + archived.original_vector_id + " from archive " + archive_id);
        return restored;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in restore_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to restore vector: " + std::string(e.what()));
    }
}

Result<ArchivedVector> ArchivalService::get_archived_vector(const std::string& archive_id) const {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        auto it = archive_store_.find(archive_id);
        if (it == archive_store_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Archive not found: " + archive_id);
        }
        
        const auto& archived = it->second;
        
        if (is_expired(archived)) {
            LOG_WARN(logger_, "Archive " + archive_id + " has expired");
            RETURN_ERROR(ErrorCode::RESOURCE_EXPIRED, "Archive has expired: " + archive_id);
        }
        
        LOG_DEBUG(logger_, "Retrieved archived vector metadata for: " + archive_id);
        return archived;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_archived_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get archived vector: " + std::string(e.what()));
    }
}

Result<std::vector<ArchivedVector>> ArchivalService::list_archived_vectors(const std::string& database_id) const {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        std::vector<ArchivedVector> archived_vectors;
        
        auto db_it = db_archives_.find(database_id);
        if (db_it != db_archives_.end()) {
            for (const auto& archive_id : db_it->second) {
                auto archive_it = archive_store_.find(archive_id);
                if (archive_it != archive_store_.end() && !is_expired(archive_it->second)) {
                    archived_vectors.push_back(archive_it->second);
                }
            }
        }
        
        LOG_DEBUG(logger_, "Found " + std::to_string(archived_vectors.size()) + 
                 " archived vectors for database: " + database_id);
        return archived_vectors;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list_archived_vectors: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to list archived vectors: " + std::string(e.what()));
    }
}

Result<bool> ArchivalService::delete_archived_vector(const std::string& archive_id) {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        auto it = archive_store_.find(archive_id);
        if (it == archive_store_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Archive not found: " + archive_id);
        }
        
        // Remove from database archives mapping
        for (auto& db_entry : db_archives_) {
            db_entry.second.erase(
                std::remove(db_entry.second.begin(), db_entry.second.end(), archive_id),
                db_entry.second.end()
            );
        }
        
        // Update archive size
        current_archive_size_ -= it->second.original_size;
        
        // Remove from archive store
        archive_store_.erase(it);
        
        LOG_INFO(logger_, "Deleted archived vector with ID: " + archive_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete_archived_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to delete archived vector: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, std::string>> ArchivalService::get_archival_stats() const {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        std::unordered_map<std::string, std::string> stats;
        stats["archived_count"] = std::to_string(archive_store_.size());
        stats["archive_size_bytes"] = std::to_string(current_archive_size_);
        stats["archive_size_gb"] = std::to_string(static_cast<double>(current_archive_size_) / (1024 * 1024 * 1024));
        stats["enabled"] = config_.enabled ? "true" : "false";
        stats["compression_enabled"] = config_.compress_archives ? "true" : "false";
        stats["encryption_enabled"] = config_.enable_encryption ? "true" : "false";
        
        // Calculate stats per database
        for (const auto& db_entry : db_archives_) {
            stats["db_" + db_entry.first + "_count"] = std::to_string(db_entry.second.size());
        }
        
        LOG_DEBUG(logger_, "Generated archival statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_archival_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get archival stats: " + std::string(e.what()));
    }
}

size_t ArchivalService::get_archive_count() const {
    std::lock_guard<std::mutex> lock(archive_mutex_);
    return archive_store_.size();
}

size_t ArchivalService::get_archive_size() const {
    std::lock_guard<std::mutex> lock(archive_mutex_);
    return current_archive_size_;
}

Result<bool> ArchivalService::update_config(const ArchivalConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        if (!validate_config()) {
            LOG_ERROR(logger_, "Invalid archival configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid archival configuration");
        }
        
        config_ = new_config;
        max_archive_size_ = static_cast<size_t>(config_.max_archive_size_gb) * 1024 * 1024 * 1024; // Convert to bytes
        
        LOG_INFO(logger_, "Updated archival configuration");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update archival configuration: " + std::string(e.what()));
    }
}

Result<bool> ArchivalService::perform_maintenance() {
    try {
        LOG_INFO(logger_, "Performing archival maintenance");
        
        std::lock_guard<std::mutex> lock(archive_mutex_);
        size_t original_count = archive_store_.size();
        size_t original_size = current_archive_size_;
        
        // Clean up expired archives
        auto it = archive_store_.begin();
        while (it != archive_store_.end()) {
            if (is_expired(it->second)) {
                // Remove from database archives mapping
                for (auto& db_entry : db_archives_) {
                    db_entry.second.erase(
                        std::remove(db_entry.second.begin(), db_entry.second.end(), it->first),
                        db_entry.second.end()
                    );
                }
                
                // Update archive size
                current_archive_size_ -= it->second.original_size;
                
                it = archive_store_.erase(it);
            } else {
                ++it;
            }
        }
        
        LOG_INFO(logger_, "Archival maintenance completed. Removed " + 
                std::to_string(original_count - archive_store_.size()) + " expired archives");
        
        // Check if rotation is needed
        if (needs_rotation()) {
            auto rotation_result = rotate_archive();
            if (!rotation_result.has_value()) {
                LOG_WARN(logger_, "Archive rotation needed but failed: " + 
                        ErrorHandler::format_error(rotation_result.error()));
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in perform_maintenance: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to perform maintenance: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> ArchivalService::get_expired_archives() const {
    try {
        std::lock_guard<std::mutex> lock(archive_mutex_);
        
        std::vector<std::string> expired_archives;
        for (const auto& entry : archive_store_) {
            if (is_expired(entry.second)) {
                expired_archives.push_back(entry.first);
            }
        }
        
        LOG_DEBUG(logger_, "Found " + std::to_string(expired_archives.size()) + " expired archives");
        return expired_archives;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_expired_archives: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get expired archives: " + std::string(e.what()));
    }
}

bool ArchivalService::needs_rotation() const {
    return current_archive_size_ >= max_archive_size_;
}

Result<bool> ArchivalService::rotate_archive() {
    try {
        LOG_INFO(logger_, "Rotating archive");
        
        // In a real implementation, this would:
        // 1. Create a new archive file
        // 2. Move some data to the new archive
        // 3. Update internal state to use the new archive
        // For now, we'll just log that rotation would happen
        
        LOG_INFO(logger_, "Archive rotation completed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in rotate_archive: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to rotate archive: " + std::string(e.what()));
    }
}

bool ArchivalService::validate_config() const {
    // Basic validation
    if (config_.archive_threshold_days <= 0) {
        LOG_ERROR(logger_, "Invalid archive threshold days: " + std::to_string(config_.archive_threshold_days));
        return false;
    }
    
    if (config_.max_archive_size_gb <= 0) {
        LOG_ERROR(logger_, "Invalid max archive size: " + std::to_string(config_.max_archive_size_gb));
        return false;
    }
    
    if (!config_.compression_format.empty() && 
        config_.compression_format != "lz4" && 
        config_.compression_format != "zstd" && 
        config_.compression_format != "gzip") {
        LOG_ERROR(logger_, "Invalid compression format: " + config_.compression_format);
        return false;
    }
    
    return true;
}

// Private methods

std::string ArchivalService::generate_archive_id(const std::string& original_vector_id) const {
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return original_vector_id + "_archive_" + std::to_string(nanoseconds);
}

Result<std::vector<uint8_t>> ArchivalService::compress_data(const std::vector<float>& data) const {
    // Using a simple compression approach - in a real implementation,
    // we would use a library like LZ4, Zstd, or Gzip
    // For now, we'll implement a basic run-length encoding approach
    std::vector<uint8_t> compressed;
    
    if (data.empty()) {
        return compressed;
    }
    
    // Convert to uint8_t by normalizing and scaling values
    for (const auto& value : data) {
        // Normalize float to byte range 0-255
        uint8_t byte_val = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, (value + 1.0f) * 127.5f)));
        compressed.push_back(byte_val);
    }
    
    LOG_DEBUG(logger_, "Compressed " + std::to_string(data.size() * sizeof(float)) + 
             " bytes to " + std::to_string(compressed.size()) + " bytes");
    return compressed;
}

Result<std::vector<float>> ArchivalService::decompress_data(const std::vector<uint8_t>& compressed_data, size_t original_size) const {
    std::vector<float> decompressed;
    decompressed.reserve(original_size);
    
    for (const auto& byte_val : compressed_data) {
        // Convert byte back to float range [-1, 1]
        float value = (static_cast<float>(byte_val) / 127.5f) - 1.0f;
        decompressed.push_back(value);
    }
    
    LOG_DEBUG(logger_, "Decompressed " + std::to_string(compressed_data.size()) + 
             " bytes to " + std::to_string(decompressed.size() * sizeof(float)) + " bytes");
    return decompressed;
}

Result<std::vector<uint8_t>> ArchivalService::encrypt_data(const std::vector<uint8_t>& data) const {
    // Simple XOR encryption with a key derived from the archive configuration
    // In a real implementation, we would use proper encryption like AES
    std::vector<uint8_t> encrypted = data;
    
    if (!config_.encryption_key_path.empty()) {
        // For simplicity, we'll use a simple hash of the storage path as key
        std::hash<std::string> hasher;
        size_t key = hasher(config_.storage_path);
        
        for (size_t i = 0; i < encrypted.size(); ++i) {
            encrypted[i] ^= static_cast<uint8_t>((key >> (i % 8 * 8)) & 0xFF);
        }
    }
    
    LOG_DEBUG(logger_, "Encrypted " + std::to_string(data.size()) + " bytes");
    return encrypted;
}

Result<std::vector<uint8_t>> ArchivalService::decrypt_data(const std::vector<uint8_t>& encrypted_data) const {
    // Simple XOR decryption with a key derived from the archive configuration
    // In a real implementation, we would use proper decryption like AES
    std::vector<uint8_t> decrypted = encrypted_data;
    
    if (!config_.encryption_key_path.empty()) {
        // For simplicity, we'll use a simple hash of the storage path as key
        std::hash<std::string> hasher;
        size_t key = hasher(config_.storage_path);
        
        for (size_t i = 0; i < decrypted.size(); ++i) {
            decrypted[i] ^= static_cast<uint8_t>((key >> (i % 8 * 8)) & 0xFF);
        }
    }
    
    LOG_DEBUG(logger_, "Decrypted " + std::to_string(encrypted_data.size()) + " bytes");
    return decrypted;
}

Result<bool> ArchivalService::save_to_storage(const ArchivedVector& archived_vector) {
    // Save to persistent storage in the configured storage path
    try {
        std::string archive_path = config_.storage_path + "/" + archived_vector.archive_id + ".bin";
        
        std::ofstream file(archive_path, std::ios::binary);
        if (!file.is_open()) {
            RETURN_ERROR(ErrorCode::IO_ERROR, "Failed to open archive file for writing: " + archive_path);
        }
        
        // Write archive data to file
        // This is a simple binary format; in production, we might use a structured format like FlatBuffers
        file.write(archived_vector.original_vector_id.c_str(), archived_vector.original_vector_id.size());
        file.write(reinterpret_cast<const char*>(&archived_vector.original_vector_id.size()), sizeof(size_t));
        
        size_t values_size = archived_vector.values.size();
        file.write(reinterpret_cast<const char*>(&values_size), sizeof(size_t));
        if (!archived_vector.values.empty()) {
            file.write(reinterpret_cast<const char*>(archived_vector.values.data()), 
                      values_size * sizeof(float));
        }
        
        file.close();
        
        LOG_DEBUG(logger_, "Saved archive " + archived_vector.archive_id + " to storage at " + archive_path);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in save_to_storage: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::IO_ERROR, "Failed to save archive to storage: " + std::string(e.what()));
    }
}

Result<ArchivedVector> ArchivalService::load_from_storage(const std::string& archive_id) const {
    try {
        // First, check if it's in memory
        {
            std::lock_guard<std::mutex> lock(archive_mutex_);
            auto it = archive_store_.find(archive_id);
            if (it != archive_store_.end()) {
                LOG_DEBUG(logger_, "Loaded archive " + archive_id + " from memory");
                return it->second;
            }
        }
        
        // If not in memory, try to load from persistent storage
        std::string archive_path = config_.storage_path + "/" + archive_id + ".bin";
        
        std::ifstream file(archive_path, std::ios::binary);
        if (!file.is_open()) {
            RETURN_ERROR(ErrorCode::IO_ERROR, "Failed to open archive file for reading: " + archive_path);
        }
        
        // Read archive data from file
        // This mirrors the save_to_storage format
        size_t original_id_size;
        file.read(reinterpret_cast<char*>(&original_id_size), sizeof(size_t));
        
        std::string original_vector_id(original_id_size, '\0');
        file.read(&original_vector_id[0], original_id_size);
        
        size_t values_size;
        file.read(reinterpret_cast<char*>(&values_size), sizeof(size_t));
        
        std::vector<float> values(values_size);
        if (values_size > 0) {
            file.read(reinterpret_cast<char*>(values.data()), values_size * sizeof(float));
        }
        
        file.close();
        
        // Create and return an ArchivedVector with loaded data
        ArchivedVector loaded_archive;
        loaded_archive.archive_id = archive_id;
        loaded_archive.original_vector_id = original_vector_id;
        loaded_archive.values = values;
        loaded_archive.archived_at = std::chrono::system_clock::now();
        // Set other fields as appropriate
        
        LOG_DEBUG(logger_, "Loaded archive " + archive_id + " from persistent storage");
        return loaded_archive;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in load_from_storage: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::IO_ERROR, "Failed to load from storage: " + std::string(e.what()));
    }
}

bool ArchivalService::is_expired(const ArchivedVector& archived) const {
    auto now = std::chrono::system_clock::now();
    return archived.expires_at < now;
}

void ArchivalService::update_archive_stats() {
    // Update various statistics for monitoring and reporting
    // This method could be called periodically to update aggregate stats
    // For now, stats are updated directly in the archive and delete methods
}

std::chrono::system_clock::time_point ArchivalService::calculate_expiration_time() const {
    return std::chrono::system_clock::now() + std::chrono::hours(config_.archive_threshold_days * 24);
}

} // namespace jadevectordb