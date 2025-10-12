#ifndef JADEVECTORDB_ARCHIVAL_SERVICE_H
#define JADEVECTORDB_ARCHIVAL_SERVICE_H

#include "models/vector.h"
#include "models/database.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace jadevectordb {

// Information about an archived vector
struct ArchivedVector {
    std::string archive_id;
    std::string original_vector_id;
    std::string database_id;
    std::vector<float> values;
    std::unordered_map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point archived_at;
    std::chrono::system_clock::time_point expires_at;  // When it will be deleted permanently
    std::string restored_from;  // Archive ID this was restored from (if applicable)
    bool is_compressed;  // Whether the vector data is compressed
    size_t original_size;  // Size of the original vector (for metrics)
    size_t compressed_size;  // Size when compressed (if applicable)
    
    ArchivedVector() : is_compressed(false), original_size(0), compressed_size(0) {}
    ArchivedVector(const std::string& archive_id, const Vector& original)
        : archive_id(archive_id), original_vector_id(original.id), 
          database_id(original.databaseId), values(original.values),
          metadata(original.metadata), archived_at(std::chrono::system_clock::now()),
          is_compressed(false), original_size(original.values.size() * sizeof(float)), 
          compressed_size(0) {
        // Set expiration based on retention policy
        expires_at = archived_at + std::chrono::hours(24 * 365); // Default: 1 year
    }
};

// Configuration for archival
struct ArchivalConfig {
    bool enabled = true;                    // Whether archival is enabled
    int archive_threshold_days = 180;      // Age in days when vectors get archived
    std::string storage_path = "./archive"; // Path to store archived data
    bool compress_archives = true;         // Whether to compress archived data
    std::string compression_format = "lz4"; // Compression format ("lz4", "zstd", "gzip")
    int max_archive_size_gb = 100;         // Maximum size of archive before rotation
    bool enable_encryption = false;        // Whether to encrypt archived data
    std::string encryption_key_path = "";  // Path to encryption key file
    
    ArchivalConfig() = default;
};

/**
 * @brief Service to handle archiving of vector data
 * 
 * This service manages the archival of vector data that meets retention criteria,
 * moving it to long-term storage while maintaining accessibility.
 */
class ArchivalService {
private:
    std::shared_ptr<logging::Logger> logger_;
    ArchivalConfig config_;
    std::unordered_map<std::string, ArchivedVector> archive_store_; // archive_id -> archived_vector
    std::unordered_map<std::string, std::vector<std::string>> db_archives_; // database_id -> [archive_ids]
    std::mutex archive_mutex_;
    
    size_t current_archive_size_;
    size_t max_archive_size_;
    
public:
    explicit ArchivalService();
    ~ArchivalService() = default;
    
    // Initialize the archival service with configuration
    bool initialize(const ArchivalConfig& config);
    
    // Archive a vector
    Result<std::string> archive_vector(const Vector& vector);
    
    // Archive multiple vectors at once
    Result<std::vector<std::string>> archive_vectors(const std::vector<Vector>& vectors);
    
    // Check if a vector is archived
    bool is_archived(const std::string& vector_id) const;
    
    // Restore an archived vector
    Result<Vector> restore_vector(const std::string& archive_id) const;
    
    // Get an archived vector without restoring it (metadata only)
    Result<ArchivedVector> get_archived_vector(const std::string& archive_id) const;
    
    // List all archived vectors for a database
    Result<std::vector<ArchivedVector>> list_archived_vectors(const std::string& database_id) const;
    
    // Delete archived vector (permanent deletion)
    Result<bool> delete_archived_vector(const std::string& archive_id);
    
    // Get archival statistics
    Result<std::unordered_map<std::string, std::string>> get_archival_stats() const;
    
    // Get the number of archived items
    size_t get_archive_count() const;
    
    // Get the total size of archived data
    size_t get_archive_size() const;
    
    // Update archival configuration
    Result<bool> update_config(const ArchivalConfig& new_config);
    
    // Perform archive maintenance (cleanup expired, rotate if needed)
    Result<bool> perform_maintenance();
    
    // Get all archives that have expired based on retention policy
    Result<std::vector<std::string>> get_expired_archives() const;
    
    // Check if archival storage needs rotation due to size
    bool needs_rotation() const;
    
    // Rotate the archive (create new archive file)
    Result<bool> rotate_archive();
    
    // Validate archival configuration
    bool validate_config() const;

private:
    // Internal helper methods
    
    // Generate a unique archive ID
    std::string generate_archive_id(const std::string& original_vector_id) const;
    
    // Compress vector data if compression is enabled
    Result<std::vector<uint8_t>> compress_data(const std::vector<float>& data) const;
    
    // Decompress vector data if it was archived compressed
    Result<std::vector<float>> decompress_data(const std::vector<uint8_t>& compressed_data, size_t original_size) const;
    
    // Encrypt archived data if encryption is enabled
    Result<std::vector<uint8_t>> encrypt_data(const std::vector<uint8_t>& data) const;
    
    // Decrypt archived data if it was archived encrypted
    Result<std::vector<uint8_t>> decrypt_data(const std::vector<uint8_t>& encrypted_data) const;
    
    // Save archive to persistent storage
    Result<bool> save_to_storage(const ArchivedVector& archived_vector);
    
    // Load archive from persistent storage
    Result<ArchivedVector> load_from_storage(const std::string& archive_id) const;
    
    // Check if an archive has expired based on retention policy
    bool is_expired(const ArchivedVector& archived) const;
    
    // Update archive statistics after operations
    void update_archive_stats();
    
    // Calculate retention expiration time based on policy
    std::chrono::system_clock::time_point calculate_expiration_time() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_ARCHIVAL_SERVICE_H