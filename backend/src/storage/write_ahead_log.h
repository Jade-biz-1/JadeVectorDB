#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <cstdint>
#include <optional>
#include <functional>

namespace jadevectordb {

/**
 * @brief WAL entry types
 */
enum class WALEntryType : uint8_t {
    VECTOR_STORE = 1,    // Store a new vector
    VECTOR_UPDATE = 2,   // Update existing vector
    VECTOR_DELETE = 3,   // Delete a vector
    INDEX_RESIZE = 4,    // Index resize operation
    CHECKPOINT = 5,      // Checkpoint marker
    COMMIT = 6          // Transaction commit
};

/**
 * @brief WAL entry header (32 bytes, aligned)
 */
struct alignas(32) WALEntryHeader {
    uint32_t magic_number;      // 0x57414C31 ("WAL1")
    WALEntryType entry_type;    // Type of operation
    uint8_t reserved1[3];       // Padding
    uint64_t sequence_number;   // Monotonically increasing
    uint64_t timestamp;         // Unix timestamp in microseconds
    uint64_t data_size;         // Size of entry data in bytes
    uint32_t checksum;          // CRC32 checksum of data
    uint8_t reserved2[4];       // Padding to 32 bytes
};

/**
 * @brief WAL entry data for vector operations
 */
struct WALVectorEntry {
    char database_id[64];       // Database identifier (null-terminated)
    char vector_id[64];         // Vector identifier (null-terminated)
    uint32_t dimension;         // Vector dimension
    uint32_t reserved;          // Padding
    // Followed by: float values[dimension]
};

/**
 * @brief Write-Ahead Log for crash recovery
 * 
 * Provides durability guarantees:
 * - All operations logged before execution
 * - Sequential log file format
 * - Checksum verification
 * - Replay on startup for crash recovery
 * - Periodic checkpoints to truncate log
 */
class WriteAheadLog {
public:
    /**
     * @brief Constructor
     * @param database_id Database identifier
     * @param wal_directory Directory for WAL files
     */
    explicit WriteAheadLog(const std::string& database_id,
                          const std::string& wal_directory);
    
    /**
     * @brief Destructor - ensures log is flushed
     */
    ~WriteAheadLog();

    // Disable copy and move
    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

    /**
     * @brief Initialize WAL - opens or creates log file
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Log a vector store operation
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     * @param values Vector data
     * @return true if logged successfully
     */
    bool log_vector_store(const std::string& database_id,
                         const std::string& vector_id,
                         const std::vector<float>& values);

    /**
     * @brief Log a vector update operation
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     * @param values New vector data
     * @return true if logged successfully
     */
    bool log_vector_update(const std::string& database_id,
                          const std::string& vector_id,
                          const std::vector<float>& values);

    /**
     * @brief Log a vector delete operation
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     * @return true if logged successfully
     */
    bool log_vector_delete(const std::string& database_id,
                          const std::string& vector_id);

    /**
     * @brief Log an index resize operation
     * @param database_id Database identifier
     * @param old_capacity Old index capacity
     * @param new_capacity New index capacity
     * @return true if logged successfully
     */
    bool log_index_resize(const std::string& database_id,
                         size_t old_capacity,
                         size_t new_capacity);

    /**
     * @brief Write a checkpoint marker
     * @return true if successful
     */
    bool write_checkpoint();

    /**
     * @brief Flush all pending writes to disk
     * @param sync If true, also sync to physical storage
     * @return true if successful
     */
    bool flush(bool sync = false);

    /**
     * @brief Replay WAL entries for crash recovery
     * @param replay_callback Callback to execute each entry
     * @return Number of entries replayed, or -1 on error
     */
    int replay(std::function<bool(const WALEntryHeader&, const std::vector<uint8_t>&)> replay_callback);

    /**
     * @brief Truncate log after successful checkpoint
     * @return true if successful
     */
    bool truncate();

    /**
     * @brief Get current log size in bytes
     * @return Log file size
     */
    size_t get_log_size() const;

    /**
     * @brief Get number of entries in log
     * @return Entry count
     */
    uint64_t get_entry_count() const;

    /**
     * @brief Check if log needs checkpoint
     * @param size_threshold Size threshold in bytes (default: 100MB)
     * @return true if checkpoint recommended
     */
    bool needs_checkpoint(size_t size_threshold = 100 * 1024 * 1024) const;

private:
    std::string database_id_;
    std::string wal_directory_;
    std::string wal_file_path_;
    std::fstream wal_file_;
    std::mutex wal_mutex_;
    
    uint64_t sequence_number_;
    uint64_t entry_count_;
    size_t log_size_;
    
    static constexpr uint32_t WAL_MAGIC = 0x57414C31;  // "WAL1"
    
    // Helper methods
    std::string get_wal_file_path() const;
    bool write_entry(WALEntryType type, const void* data, size_t size);
    uint32_t calculate_checksum(const void* data, size_t size) const;
    bool verify_checksum(const WALEntryHeader& header, const void* data) const;
    uint64_t get_timestamp_micros() const;
};

} // namespace jadevectordb
