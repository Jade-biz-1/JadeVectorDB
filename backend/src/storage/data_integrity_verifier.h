#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

namespace jadevectordb {

// Forward declaration
class MemoryMappedVectorStore;

/**
 * @brief Result of an integrity check
 */
struct IntegrityCheckResult {
    bool passed = true;
    std::string database_id;
    uint64_t vectors_checked = 0;
    uint64_t corrupted_vectors = 0;
    uint64_t index_errors = 0;
    uint64_t free_list_errors = 0;
    std::vector<std::string> error_messages;
    
    // Detailed corruption info
    std::vector<uint64_t> corrupted_vector_ids;
    std::vector<uint64_t> orphaned_index_entries;  // Index points to invalid data
    std::vector<uint64_t> missing_index_entries;   // Data has no index entry
    
    void add_error(const std::string& error) {
        error_messages.push_back(error);
        passed = false;
    }
};

/**
 * @brief Data integrity verification and repair
 * 
 * Verifies database integrity:
 * - Vector data checksums
 * - Index consistency (all entries point to valid data)
 * - Free list validity (no overlaps, correct sizes)
 * - File structure integrity
 * 
 * Can detect:
 * - Corrupted vector data
 * - Orphaned index entries
 * - Missing index entries
 * - Free list corruption
 * - File truncation
 */
class DataIntegrityVerifier {
public:
    /**
     * @brief Construct verifier for a database
     * @param store The vector store to verify
     */
    explicit DataIntegrityVerifier(MemoryMappedVectorStore* store);
    
    ~DataIntegrityVerifier() = default;
    
    // Disable copy and move
    DataIntegrityVerifier(const DataIntegrityVerifier&) = delete;
    DataIntegrityVerifier& operator=(const DataIntegrityVerifier&) = delete;
    
    /**
     * @brief Perform full integrity check
     * @param database_id Database to verify
     * @param deep_check If true, verify vector data checksums (slow)
     * @return Check results
     */
    IntegrityCheckResult verify_database(const std::string& database_id, bool deep_check = false);
    
    /**
     * @brief Verify vector data checksums
     * @param database_id Database to verify
     * @return Check results
     */
    IntegrityCheckResult verify_vector_checksums(const std::string& database_id);
    
    /**
     * @brief Verify index consistency
     * @param database_id Database to verify
     * @return Check results
     */
    IntegrityCheckResult verify_index_consistency(const std::string& database_id);
    
    /**
     * @brief Verify free list integrity
     * @param database_id Database to verify
     * @return Check results
     */
    IntegrityCheckResult verify_free_list(const std::string& database_id);
    
    /**
     * @brief Attempt to repair detected issues
     * @param database_id Database to repair
     * @param result Previous check result
     * @param create_backup If true, create backup before repair
     * @return True if repair successful
     */
    bool repair_database(const std::string& database_id, 
                        const IntegrityCheckResult& result,
                        bool create_backup = true);
    
    /**
     * @brief Rebuild index from vector data
     * @param database_id Database to rebuild
     * @return True if successful
     */
    bool rebuild_index(const std::string& database_id);
    
    /**
     * @brief Rebuild free list by scanning allocated space
     * @param database_id Database to rebuild
     * @return True if successful
     */
    bool rebuild_free_list(const std::string& database_id);
    
    /**
     * @brief Set progress callback
     * @param callback Called periodically with (current, total)
     */
    void set_progress_callback(std::function<void(uint64_t, uint64_t)> callback);
    
private:
    MemoryMappedVectorStore* store_;
    std::function<void(uint64_t, uint64_t)> progress_callback_;
    
    // Helper: Calculate checksum for vector data
    uint32_t calculate_vector_checksum(const void* data, size_t size);
    
    // Helper: Check if offset is within file bounds
    bool is_valid_offset(const std::string& database_id, uint64_t offset, uint64_t size);
    
    // Helper: Report progress
    void report_progress(uint64_t current, uint64_t total);
};

} // namespace jadevectordb
