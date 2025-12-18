#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <optional>
#include <cstdint>
#include <functional>

namespace jadevectordb {

// Forward declarations
struct VectorFileHeader;
struct VectorIndexEntry;
class FileHandle;

/**
 * @brief Memory-mapped vector store for high-performance persistent storage
 * 
 * Uses memory-mapped files to store vector embeddings with:
 * - SIMD-aligned memory (32-byte for AVX operations)
 * - Binary serialization format with fast lookup
 * - Thread-safe concurrent access
 * - Automatic file growth and space management
 * - Cross-platform support (Unix mmap, Windows CreateFileMapping)
 */
class MemoryMappedVectorStore {
public:
    /**
     * @brief Construct vector store with storage directory
     * @param storage_path Base directory for vector files (e.g., "/data/vectors")
     */
    explicit MemoryMappedVectorStore(const std::string& storage_path);
    
    /**
     * @brief Destructor - closes all open files and flushes changes
     */
    ~MemoryMappedVectorStore();

    // Disable copy (move only)
    MemoryMappedVectorStore(const MemoryMappedVectorStore&) = delete;
    MemoryMappedVectorStore& operator=(const MemoryMappedVectorStore&) = delete;

    /**
     * @brief Create a new vector file for a database
     * @param database_id Unique database identifier
     * @param dimension Vector dimension (e.g., 384, 768, 1536)
     * @param initial_capacity Initial capacity in number of vectors
     * @return true if file created successfully
     */
    bool create_vector_file(const std::string& database_id, 
                           int dimension, 
                           size_t initial_capacity = 1000);

    /**
     * @brief Open existing vector file for a database (lazy loading)
     * @param database_id Unique database identifier
     * @return true if file opened successfully
     */
    bool open_vector_file(const std::string& database_id);

    /**
     * @brief Close vector file for a database
     * @param database_id Unique database identifier
     * @param flush_changes If true, flush changes to disk before closing
     */
    void close_vector_file(const std::string& database_id, bool flush_changes = true);

    /**
     * @brief Store a vector in the database
     * @param database_id Database identifier
     * @param vector_id Unique vector identifier
     * @param values Vector values (dimension must match database)
     * @return true if stored successfully
     */
    bool store_vector(const std::string& database_id,
                     const std::string& vector_id,
                     const std::vector<float>& values);

    /**
     * @brief Retrieve a vector from the database
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     * @return Vector values if found, std::nullopt otherwise
     */
    std::optional<std::vector<float>> retrieve_vector(const std::string& database_id,
                                                      const std::string& vector_id);

    /**
     * @brief Update an existing vector
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     * @param new_values New vector values (dimension must match)
     * @return true if updated successfully
     */
    bool update_vector(const std::string& database_id,
                      const std::string& vector_id,
                      const std::vector<float>& new_values);

    /**
     * @brief Delete a vector (soft delete with flag)
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     * @return true if deleted successfully
     */
    bool delete_vector(const std::string& database_id,
                      const std::string& vector_id);

    /**
     * @brief Store multiple vectors in batch (optimized)
     * @param database_id Database identifier
     * @param vectors Vector of (vector_id, values) pairs
     * @return Number of vectors successfully stored
     */
    size_t batch_store(const std::string& database_id,
                      const std::vector<std::pair<std::string, std::vector<float>>>& vectors);

    /**
     * @brief Retrieve multiple vectors in batch
     * @param database_id Database identifier
     * @param vector_ids Vector identifiers to retrieve
     * @return Vector of optional values (nullopt if not found)
     */
    std::vector<std::optional<std::vector<float>>> batch_retrieve(
        const std::string& database_id,
        const std::vector<std::string>& vector_ids);

    /**
     * @brief Flush changes to disk for a database
     * @param database_id Database identifier
     * @param sync If true, use synchronous flush (msync MS_SYNC)
     */
    void flush(const std::string& database_id, bool sync = false);

    /**
     * @brief Flush all open databases
     * @param sync If true, use synchronous flush
     */
    void flush_all(bool sync = false);

    /**
     * @brief Delete all vector data for a database
     * @param database_id Database identifier
     * @return true if deleted successfully
     */
    bool delete_database_vectors(const std::string& database_id);

    /**
     * @brief Get vector count for a database
     * @param database_id Database identifier
     * @return Number of active vectors (excluding deleted)
     */
    size_t get_vector_count(const std::string& database_id);

    /**
     * @brief Get dimension for a database
     * @param database_id Database identifier
     * @return Vector dimension, or 0 if not found
     */
    int get_dimension(const std::string& database_id);

    /**
     * @brief Check if database has vector file
     * @param database_id Database identifier
     * @return true if vector file exists
     */
    bool has_database(const std::string& database_id);

    /**
     * @brief List all vector IDs in a database
     * @param database_id Database identifier
     * @return Vector of vector IDs (active vectors only)
     */
    std::vector<std::string> list_vector_ids(const std::string& database_id);

    /**
     * @brief Get count of deleted vectors in a database
     * @param database_id Database identifier
     * @return Number of deleted vectors
     */
    size_t get_deleted_count(const std::string& database_id);

    /**
     * @brief Get the file path for a database's vector file
     * @param database_id Database identifier
     * @return Full path to the vector file
     */
    std::string get_database_vector_file_path(const std::string& database_id) const;

private:
    // Storage configuration
    std::string storage_path_;
    
    // Open file handles and memory mappings
    struct DatabaseFile {
        std::unique_ptr<FileHandle> handle;
        void* mapped_memory = nullptr;
        size_t file_size = 0;
        size_t mapped_size = 0;
        VectorFileHeader* header = nullptr;
        std::mutex mutex;  // Per-database mutex for thread safety
        uint64_t last_access_time = 0;  // For LRU eviction
        
        // Vector ID index for tracking active/deleted vectors
        std::unordered_map<std::string, bool> vector_id_index_;  // true=active, false=deleted
    };
    
    std::unordered_map<std::string, std::unique_ptr<DatabaseFile>> open_files_;
    std::mutex files_mutex_;  // Protects open_files_ map
    
    // LRU configuration
    size_t max_open_files_ = 100;
    
    // Helper methods
    std::string get_database_directory(const std::string& database_id) const;
    std::string get_vector_file_path(const std::string& database_id) const;
    
    DatabaseFile* get_or_open_file(const std::string& database_id);
    bool ensure_capacity(DatabaseFile* db_file, size_t additional_vectors);
    bool resize_file(DatabaseFile* db_file, size_t new_size);
    void evict_lru_file();
    
    // Vector index operations
    VectorIndexEntry* find_index_entry(DatabaseFile* db_file, const std::string& vector_id);
    VectorIndexEntry* allocate_index_entry(DatabaseFile* db_file, const std::string& vector_id);
    void rebuild_vector_id_index(DatabaseFile* db_file);
    uint64_t hash_vector_id(const std::string& vector_id) const;
    
    // Memory operations
    void* get_vector_data_pointer(DatabaseFile* db_file, uint64_t offset);
    uint64_t allocate_vector_space(DatabaseFile* db_file, size_t size);
    
    // Platform-specific helpers
    void* map_file(FileHandle* handle, size_t size);
    void unmap_file(void* addr, size_t size);
    bool sync_file(void* addr, size_t size, bool synchronous);
};

/**
 * @brief Vector file header (128 bytes, 64-byte aligned)
 */
struct alignas(64) VectorFileHeader {
    uint32_t magic_number;      // 0x4A564442 ("JVDB")
    uint32_t version;           // Format version (currently 1)
    uint32_t dimension;         // Vector dimension
    uint32_t reserved1;         // Padding for alignment
    uint64_t vector_count;      // Total vectors (including deleted)
    uint64_t active_count;      // Active vectors (excluding deleted)
    uint64_t index_offset;      // Byte offset to vector index
    uint64_t data_offset;       // Byte offset to vector data
    uint64_t vector_ids_offset; // Byte offset to vector ID strings
    uint64_t index_capacity;    // Max index entries
    uint64_t data_capacity;     // Data section size in bytes
    uint8_t reserved2[48];      // Reserved for future use (padding to 128 bytes)
};

/**
 * @brief Vector index entry (64 bytes per vector, 64-byte aligned)
 */
struct alignas(64) VectorIndexEntry {
    uint64_t vector_id_hash;    // Hash of vector ID string
    uint64_t data_offset;       // Byte offset in data section
    uint64_t string_offset;     // Byte offset to vector ID string
    uint32_t size;              // Vector size in bytes
    uint32_t flags;             // Status flags
    uint64_t timestamp;         // Creation/modification timestamp
    uint64_t reserved[3];       // Reserved for future use (padding to 64 bytes)
    
    // Flags
    static constexpr uint32_t FLAG_ACTIVE = 0x01;
    static constexpr uint32_t FLAG_DELETED = 0x02;
};

/**
 * @brief Cross-platform file handle wrapper
 */
class FileHandle {
public:
    FileHandle() = default;
    ~FileHandle();
    
    bool open(const std::string& path, bool create);
    void close();
    bool is_open() const;
    bool resize(size_t new_size);
    size_t get_size() const;
    
#ifdef _WIN32
    void* native_handle() const { return handle_; }
private:
    void* handle_ = nullptr;  // HANDLE on Windows
#else
    int native_handle() const { return fd_; }
private:
    int fd_ = -1;  // File descriptor on Unix
#endif
    
    std::string path_;
};

} // namespace jadevectordb
