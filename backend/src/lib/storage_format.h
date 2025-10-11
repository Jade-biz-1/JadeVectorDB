#ifndef JADEVECTORDB_STORAGE_FORMAT_H
#define JADEVECTORDB_STORAGE_FORMAT_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include "models/vector.h"
#include "models/database.h"
#include "models/index.h"
#include "models/embedding_model.h"

namespace jadevectordb {

// Custom binary storage format implementation
namespace storage_format {

    // Storage format constants
    constexpr uint32_t FORMAT_VERSION = 1;
    constexpr uint32_t MAGIC_NUMBER = 0x4A444256; // "JDBV" in hex
    
    // Header structure for storage files
    struct StorageHeader {
        uint32_t magic_number;      // Magic number to identify file format
        uint32_t format_version;     // Format version
        uint64_t timestamp;         // Creation/modification timestamp
        uint32_t data_type;         // Type of data stored (VECTOR, DATABASE, INDEX, etc.)
        uint32_t reserved;          // Reserved for future use
        uint64_t data_size;         // Size of the data section
        uint64_t checksum;          // Checksum for data integrity
    };
    
    // Data type enumeration
    enum DataType : uint32_t {
        VECTOR_DATA = 1,
        DATABASE_DATA = 2,
        INDEX_DATA = 3,
        EMBEDDING_MODEL_DATA = 4,
        BATCH_VECTOR_DATA = 5
    };
    
    // Vector storage format
    struct VectorStorageFormat {
        std::string id;
        uint32_t dimension;
        std::vector<float> values;
        std::string metadata_json;  // Serialized metadata as JSON
        uint64_t created_timestamp;
        uint64_t updated_timestamp;
        uint32_t version;
        bool deleted;
    };
    
    // Database storage format
    struct DatabaseStorageFormat {
        std::string database_id;
        std::string name;
        std::string description;
        uint32_t vector_dimension;
        std::string index_type;
        std::string index_parameters_json;
        std::string sharding_config_json;
        std::string replication_config_json;
        std::string embedding_models_json;
        std::string metadata_schema_json;
        std::string retention_policy_json;
        std::string access_control_json;
        uint64_t created_timestamp;
        uint64_t updated_timestamp;
    };
    
    // Index storage format
    struct IndexStorageFormat {
        std::string index_id;
        std::string database_id;
        std::string type;
        std::string parameters_json;
        std::string status;
        uint64_t created_timestamp;
        uint64_t updated_timestamp;
        uint64_t vector_count;
        uint64_t size_bytes;
    };
    
    // Embedding model storage format
    struct EmbeddingModelStorageFormat {
        std::string model_id;
        std::string name;
        std::string version;
        std::string provider;
        std::string input_type;
        uint32_t output_dimension;
        std::string parameters_json;
        std::string status;
    };
    
    // Storage file manager
    class StorageFileManager {
    private:
        std::string file_path_;
        bool is_open_;
        
    public:
        explicit StorageFileManager(const std::string& file_path);
        ~StorageFileManager();
        
        // File operations
        bool create_file();
        bool open_file();
        bool close_file();
        bool is_open() const;
        
        // Write operations
        bool write_vector(const Vector& vector);
        bool write_database(const Database& database);
        bool write_index(const Index& index);
        bool write_embedding_model(const EmbeddingModel& model);
        bool write_vector_batch(const std::vector<Vector>& vectors);
        
        // Read operations
        Vector read_vector(const std::string& id);
        Database read_database(const std::string& database_id);
        Index read_index(const std::string& index_id);
        EmbeddingModel read_embedding_model(const std::string& model_id);
        std::vector<Vector> read_vector_batch(const std::vector<std::string>& ids);
        
        // Utility functions
        static uint64_t calculate_checksum(const uint8_t* data, size_t size);
        static StorageHeader create_header(DataType data_type, uint64_t data_size);
        static bool verify_header(const StorageHeader& header);
        static std::string get_file_extension();
        
        // Memory mapping support
        void* map_file_region(size_t offset, size_t length);
        bool unmap_file_region(void* addr, size_t length);
    };
    
    // Conversion functions
    VectorStorageFormat convert_to_storage_format(const Vector& vector);
    Vector convert_from_storage_format(const VectorStorageFormat& storage_format);
    
    DatabaseStorageFormat convert_to_storage_format(const Database& database);
    Database convert_from_storage_format(const DatabaseStorageFormat& storage_format);
    
    IndexStorageFormat convert_to_storage_format(const Index& index);
    Index convert_from_storage_format(const IndexStorageFormat& storage_format);
    
    EmbeddingModelStorageFormat convert_to_storage_format(const EmbeddingModel& model);
    EmbeddingModel convert_from_storage_format(const EmbeddingModelStorageFormat& storage_format);
    
    // Batch conversion functions
    std::vector<VectorStorageFormat> convert_to_storage_format_batch(const std::vector<Vector>& vectors);
    std::vector<Vector> convert_from_storage_format_batch(const std::vector<VectorStorageFormat>& storage_formats);
    
    // Utility functions for storage optimization
    size_t estimate_storage_size(const Vector& vector);
    size_t estimate_storage_size(const Database& database);
    size_t estimate_storage_size(const Index& index);
    size_t estimate_storage_size(const EmbeddingModel& model);
    
    // Compression utilities
    std::vector<uint8_t> compress_data(const uint8_t* data, size_t size);
    std::vector<uint8_t> decompress_data(const uint8_t* compressed_data, size_t compressed_size, size_t original_size);

} // namespace storage_format

} // namespace jadevectordb

#endif // JADEVECTORDB_STORAGE_FORMAT_H