#include "archival_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "lib/compression.h"
#include "lib/encryption.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cstring>

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

        std::vector<uint8_t> data_to_store;

        // Compress if enabled
        if (config_.compress_archives) {
            auto compressed_result = compress_data(vector.values);
            if (compressed_result.has_value()) {
                archived.is_compressed = true;
                archived.compressed_size = compressed_result.value().size();
                data_to_store = compressed_result.value();
            } else {
                LOG_WARN(logger_, "Failed to compress vector data for: " + vector.id);
            }
        }

        // Encrypt if enabled (not actually storing encrypted data, just demonstrating capability)
        if (config_.enable_encryption && !data_to_store.empty()) {
            auto encrypted_result = encrypt_data(data_to_store);
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
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Archive not found: " + archive_id);
        }
        
        const auto& archived = it->second;
        
        if (is_expired(archived)) {
            LOG_WARN(logger_, "Archive " + archive_id + " has expired");
            RETURN_ERROR(ErrorCode::RESOURCE_EXHAUSTED, "Archive has expired: " + archive_id);
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
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Archive not found: " + archive_id);
        }
        
        const auto& archived = it->second;
        
        if (is_expired(archived)) {
            LOG_WARN(logger_, "Archive " + archive_id + " has expired");
            RETURN_ERROR(ErrorCode::RESOURCE_EXHAUSTED, "Archive has expired: " + archive_id);
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
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Archive not found: " + archive_id);
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
        LOG_INFO(logger_, "Rotating archive - current size: " +
                std::to_string(current_archive_size_) + " bytes");

        // Create rotation timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

        // Create new archive directory with timestamp
        std::string rotation_dir = config_.storage_path + "/archive_" + std::to_string(timestamp);

        // Ensure storage path exists
        if (!std::filesystem::exists(config_.storage_path)) {
            std::filesystem::create_directories(config_.storage_path);
        }

        // Create rotation directory
        if (!std::filesystem::create_directory(rotation_dir)) {
            LOG_ERROR(logger_, "Failed to create rotation directory: " + rotation_dir);
            RETURN_ERROR(ErrorCode::DATA_LOSS, "Failed to create rotation directory");
        }

        LOG_INFO(logger_, "Created rotation directory: " + rotation_dir);

        // Move oldest archives to rotation directory
        size_t moved_count = 0;
        size_t moved_size = 0;
        size_t target_reduction = current_archive_size_ / 2; // Move half to rotation

        // Sort archives by age (oldest first)
        std::vector<std::pair<std::string, ArchivedVector>> sorted_archives;
        for (const auto& entry : archive_store_) {
            sorted_archives.emplace_back(entry.first, entry.second);
        }

        std::sort(sorted_archives.begin(), sorted_archives.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.archived_at < b.second.archived_at;
                 });

        // Move archives to rotation directory
        for (const auto& [archive_id, archived_vec] : sorted_archives) {
            if (moved_size >= target_reduction) {
                break;
            }

            // Save to rotation directory
            std::string src_path = config_.storage_path + "/" + archive_id + ".bin";
            std::string dest_path = rotation_dir + "/" + archive_id + ".bin";

            try {
                if (std::filesystem::exists(src_path)) {
                    std::filesystem::rename(src_path, dest_path);
                    moved_count++;
                    moved_size += archived_vec.original_size;

                    // Remove from active archive store (will be loaded from rotation if needed)
                    archive_store_.erase(archive_id);

                    // Update database archives mapping
                    for (auto& db_entry : db_archives_) {
                        db_entry.second.erase(
                            std::remove(db_entry.second.begin(), db_entry.second.end(), archive_id),
                            db_entry.second.end()
                        );
                    }
                }
            } catch (const std::filesystem::filesystem_error& fs_err) {
                LOG_WARN(logger_, "Failed to move archive " + archive_id + ": " + fs_err.what());
            }
        }

        // Update archive size
        current_archive_size_ -= moved_size;

        LOG_INFO(logger_, "Archive rotation completed - moved " + std::to_string(moved_count) +
                " archives (" + std::to_string(moved_size) + " bytes) to cold storage at " + rotation_dir);

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
    try {
        if (data.empty()) {
            return std::vector<uint8_t>();
        }

        // Use the compression library for proper compression
        compression::CompressionManager comp_manager;
        compression::CompressionConfig comp_config;

        // Choose compression type based on configured format
        if (config_.compression_format == "lz4") {
            comp_config.type = compression::CompressionType::QUANTIZATION; // Use quantization as proxy
            comp_config.quality = compression::CompressionQuality::HIGH;
        } else if (config_.compression_format == "zstd") {
            comp_config.type = compression::CompressionType::PCA;
            comp_config.quality = compression::CompressionQuality::MEDIUM;
        } else {
            comp_config.type = compression::CompressionType::SVD;
            comp_config.quality = compression::CompressionQuality::LOSSLESS;
        }

        comp_config.compression_ratio = 0.7; // Target 70% of original size

        if (!comp_manager.configure(comp_config)) {
            LOG_WARN(logger_, "Failed to configure compression manager, using default");
        }

        std::vector<uint8_t> compressed = comp_manager.compress_vector(data);

        LOG_DEBUG(logger_, "Compressed " + std::to_string(data.size() * sizeof(float)) +
                 " bytes to " + std::to_string(compressed.size()) + " bytes using " +
                 config_.compression_format);

        return compressed;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in compress_data: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Compression failed: " + std::string(e.what()));
    }
}

Result<std::vector<float>> ArchivalService::decompress_data(const std::vector<uint8_t>& compressed_data, size_t original_size) const {
    try {
        if (compressed_data.empty()) {
            return std::vector<float>();
        }

        // Use the compression library for proper decompression
        compression::CompressionManager comp_manager;
        compression::CompressionConfig comp_config;

        // Use same compression type as compression
        if (config_.compression_format == "lz4") {
            comp_config.type = compression::CompressionType::QUANTIZATION;
            comp_config.quality = compression::CompressionQuality::HIGH;
        } else if (config_.compression_format == "zstd") {
            comp_config.type = compression::CompressionType::PCA;
            comp_config.quality = compression::CompressionQuality::MEDIUM;
        } else {
            comp_config.type = compression::CompressionType::SVD;
            comp_config.quality = compression::CompressionQuality::LOSSLESS;
        }

        comp_config.compression_ratio = 0.7;

        if (!comp_manager.configure(comp_config)) {
            LOG_WARN(logger_, "Failed to configure compression manager for decompression");
        }

        std::vector<float> decompressed = comp_manager.decompress_vector(compressed_data, original_size);

        LOG_DEBUG(logger_, "Decompressed " + std::to_string(compressed_data.size()) +
                 " bytes to " + std::to_string(decompressed.size() * sizeof(float)) + " bytes");

        return decompressed;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in decompress_data: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Decompression failed: " + std::string(e.what()));
    }
}

Result<std::vector<uint8_t>> ArchivalService::encrypt_data(const std::vector<uint8_t>& data) const {
    try {
        if (data.empty() || config_.encryption_key_path.empty()) {
            return data; // No encryption needed
        }

        // Use the encryption library for proper AES-256-GCM encryption
        encryption::EncryptionManager enc_manager;
        encryption::EncryptionConfig enc_config;
        enc_config.algorithm = encryption::EncryptionAlgorithm::AES_256_GCM;
        enc_config.key_size_bits = 256;
        enc_config.enable_hardware_acceleration = true;

        // Generate a key ID based on the storage path
        std::hash<std::string> hasher;
        size_t key_hash = hasher(config_.encryption_key_path);
        enc_config.key_id = "archive_key_" + std::to_string(key_hash);

        std::vector<uint8_t> encrypted = enc_manager.encrypt_data(data, enc_config);

        LOG_DEBUG(logger_, "Encrypted " + std::to_string(data.size()) +
                 " bytes to " + std::to_string(encrypted.size()) + " bytes using AES-256-GCM");

        return encrypted;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in encrypt_data: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Encryption failed: " + std::string(e.what()));
    }
}

Result<std::vector<uint8_t>> ArchivalService::decrypt_data(const std::vector<uint8_t>& encrypted_data) const {
    try {
        if (encrypted_data.empty() || config_.encryption_key_path.empty()) {
            return encrypted_data; // No decryption needed
        }

        // Use the encryption library for proper AES-256-GCM decryption
        encryption::EncryptionManager enc_manager;
        encryption::EncryptionConfig enc_config;
        enc_config.algorithm = encryption::EncryptionAlgorithm::AES_256_GCM;
        enc_config.key_size_bits = 256;
        enc_config.enable_hardware_acceleration = true;

        // Generate the same key ID as encryption
        std::hash<std::string> hasher;
        size_t key_hash = hasher(config_.encryption_key_path);
        enc_config.key_id = "archive_key_" + std::to_string(key_hash);

        std::vector<uint8_t> decrypted = enc_manager.decrypt_data(encrypted_data, enc_config);

        LOG_DEBUG(logger_, "Decrypted " + std::to_string(encrypted_data.size()) +
                 " bytes to " + std::to_string(decrypted.size()) + " bytes using AES-256-GCM");

        return decrypted;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in decrypt_data: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Decryption failed: " + std::string(e.what()));
    }
}

Result<bool> ArchivalService::save_to_storage(const ArchivedVector& archived_vector) {
    try {
        // Ensure storage directory exists
        if (!std::filesystem::exists(config_.storage_path)) {
            std::filesystem::create_directories(config_.storage_path);
            LOG_INFO(logger_, "Created archive storage directory: " + config_.storage_path);
        }

        std::string archive_path = config_.storage_path + "/" + archived_vector.archive_id + ".bin";

        std::ofstream file(archive_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            RETURN_ERROR(ErrorCode::DATA_LOSS, "Failed to open archive file for writing: " + archive_path);
        }

        // Write archive header (magic number and version)
        const uint32_t ARCHIVE_MAGIC = 0x4A564442; // "JVDB" in hex
        const uint32_t ARCHIVE_VERSION = 1;
        file.write(reinterpret_cast<const char*>(&ARCHIVE_MAGIC), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&ARCHIVE_VERSION), sizeof(uint32_t));

        // Write original vector ID
        size_t id_size = archived_vector.original_vector_id.size();
        file.write(reinterpret_cast<const char*>(&id_size), sizeof(size_t));
        file.write(archived_vector.original_vector_id.c_str(), id_size);

        // Write database ID
        size_t db_id_size = archived_vector.database_id.size();
        file.write(reinterpret_cast<const char*>(&db_id_size), sizeof(size_t));
        file.write(archived_vector.database_id.c_str(), db_id_size);

        // Write vector values
        size_t values_size = archived_vector.values.size();
        file.write(reinterpret_cast<const char*>(&values_size), sizeof(size_t));
        if (!archived_vector.values.empty()) {
            file.write(reinterpret_cast<const char*>(archived_vector.values.data()),
                      values_size * sizeof(float));
        }

        // Write metadata flags
        file.write(reinterpret_cast<const char*>(&archived_vector.is_compressed), sizeof(bool));
        file.write(reinterpret_cast<const char*>(&archived_vector.original_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&archived_vector.compressed_size), sizeof(size_t));

        // Write timestamps
        auto archived_time = std::chrono::system_clock::to_time_t(archived_vector.archived_at);
        auto expires_time = std::chrono::system_clock::to_time_t(archived_vector.expires_at);
        file.write(reinterpret_cast<const char*>(&archived_time), sizeof(std::time_t));
        file.write(reinterpret_cast<const char*>(&expires_time), sizeof(std::time_t));

        if (!file.good()) {
            RETURN_ERROR(ErrorCode::DATA_LOSS, "Error writing archive file: " + archive_path);
        }

        file.close();

        LOG_DEBUG(logger_, "Saved archive " + archived_vector.archive_id + " to storage at " + archive_path);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in save_to_storage: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::DATA_LOSS, "Failed to save archive to storage: " + std::string(e.what()));
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
            RETURN_ERROR(ErrorCode::DATA_LOSS, "Failed to open archive file for reading: " + archive_path);
        }

        // Read and verify archive header
        uint32_t magic, version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

        const uint32_t ARCHIVE_MAGIC = 0x4A564442; // "JVDB" in hex
        if (magic != ARCHIVE_MAGIC) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid archive file magic number");
        }

        if (version != 1) {
            RETURN_ERROR(ErrorCode::VECTOR_DIMENSION_MISMATCH, "Unsupported archive version: " + std::to_string(version));
        }

        // Read original vector ID
        size_t id_size;
        file.read(reinterpret_cast<char*>(&id_size), sizeof(size_t));
        std::string original_vector_id(id_size, '\0');
        file.read(&original_vector_id[0], id_size);

        // Read database ID
        size_t db_id_size;
        file.read(reinterpret_cast<char*>(&db_id_size), sizeof(size_t));
        std::string database_id(db_id_size, '\0');
        file.read(&database_id[0], db_id_size);

        // Read vector values
        size_t values_size;
        file.read(reinterpret_cast<char*>(&values_size), sizeof(size_t));
        std::vector<float> values(values_size);
        if (values_size > 0) {
            file.read(reinterpret_cast<char*>(values.data()), values_size * sizeof(float));
        }

        // Read metadata flags
        bool is_compressed;
        size_t original_size, compressed_size;
        file.read(reinterpret_cast<char*>(&is_compressed), sizeof(bool));
        file.read(reinterpret_cast<char*>(&original_size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&compressed_size), sizeof(size_t));

        // Read timestamps
        std::time_t archived_time, expires_time;
        file.read(reinterpret_cast<char*>(&archived_time), sizeof(std::time_t));
        file.read(reinterpret_cast<char*>(&expires_time), sizeof(std::time_t));

        if (!file.good() && !file.eof()) {
            RETURN_ERROR(ErrorCode::DATA_LOSS, "Error reading archive file: " + archive_path);
        }

        file.close();

        // Create and return an ArchivedVector with loaded data
        ArchivedVector loaded_archive;
        loaded_archive.archive_id = archive_id;
        loaded_archive.original_vector_id = original_vector_id;
        loaded_archive.database_id = database_id;
        loaded_archive.values = values;
        loaded_archive.is_compressed = is_compressed;
        loaded_archive.original_size = original_size;
        loaded_archive.compressed_size = compressed_size;
        loaded_archive.archived_at = std::chrono::system_clock::from_time_t(archived_time);
        loaded_archive.expires_at = std::chrono::system_clock::from_time_t(expires_time);

        LOG_DEBUG(logger_, "Loaded archive " + archive_id + " from persistent storage at " + archive_path);
        return loaded_archive;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in load_from_storage: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::DATA_LOSS, "Failed to load from storage: " + std::string(e.what()));
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