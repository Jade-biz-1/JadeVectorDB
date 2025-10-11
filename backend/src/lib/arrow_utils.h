#ifndef JADEVECTORDB_ARROW_UTILS_H
#define JADEVECTORDB_ARROW_UTILS_H

#include <string>
#include <vector>
#include <memory>
#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include "models/vector.h"
#include "models/database.h"

namespace jadevectordb {

// Apache Arrow utilities for in-memory operations
namespace arrow_utils {

    // Arrow schema definitions
    class ArrowSchemaFactory {
    public:
        // Create schema for vector data
        static std::shared_ptr<arrow::Schema> create_vector_schema();
        
        // Create schema for database metadata
        static std::shared_ptr<arrow::Schema> create_database_schema();
        
        // Create schema for index metadata
        static std::shared_ptr<arrow::Schema> create_index_schema();
        
        // Create schema for embedding model metadata
        static std::shared_ptr<arrow::Schema> create_embedding_model_schema();
        
        // Create schema for batch vector operations
        static std::shared_ptr<arrow::Schema> create_vector_batch_schema();
    };
    
    // Arrow data conversion utilities
    class ArrowConverter {
    public:
        // Convert Vector to Arrow RecordBatch
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> vector_to_record_batch(
            const Vector& vector);
        
        // Convert Arrow RecordBatch to Vector
        static arrow::Result<Vector> record_batch_to_vector(
            const std::shared_ptr<arrow::RecordBatch>& batch, int row_index = 0);
        
        // Convert vector of Vectors to Arrow RecordBatch
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> vector_batch_to_record_batch(
            const std::vector<Vector>& vectors);
        
        // Convert Arrow RecordBatch to vector of Vectors
        static arrow::Result<std::vector<Vector>> record_batch_to_vector_batch(
            const std::shared_ptr<arrow::RecordBatch>& batch);
        
        // Convert Database to Arrow RecordBatch
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> database_to_record_batch(
            const Database& database);
        
        // Convert Arrow RecordBatch to Database
        static arrow::Result<Database> record_batch_to_database(
            const std::shared_ptr<arrow::RecordBatch>& batch, int row_index = 0);
        
        // Convert Index to Arrow RecordBatch
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> index_to_record_batch(
            const Index& index);
        
        // Convert Arrow RecordBatch to Index
        static arrow::Result<Index> record_batch_to_index(
            const std::shared_ptr<arrow::RecordBatch>& batch, int row_index = 0);
    };
    
    // Arrow memory management utilities
    class ArrowMemoryManager {
    public:
        // Create memory pool with custom configuration
        static std::shared_ptr<arrow::MemoryPool> create_memory_pool(
            size_t initial_capacity = 1024 * 1024); // 1MB default
        
        // Get default memory pool
        static std::shared_ptr<arrow::MemoryPool> get_default_pool();
        
        // Allocate buffer with alignment
        static arrow::Result<std::shared_ptr<arrow::Buffer>> allocate_aligned_buffer(
            size_t size, size_t alignment = 64);
        
        // Allocate buffer for vector data
        static arrow::Result<std::shared_ptr<arrow::Buffer>> allocate_vector_buffer(
            size_t vector_count, size_t vector_dimension);
    };
    
    // Arrow I/O utilities
    class ArrowIOUtils {
    public:
        // Write RecordBatch to file
        static arrow::Status write_record_batch_to_file(
            const std::shared_ptr<arrow::RecordBatch>& batch,
            const std::string& file_path);
        
        // Read RecordBatch from file
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> read_record_batch_from_file(
            const std::string& file_path);
        
        // Write Table to file (IPC format)
        static arrow::Status write_table_to_ipc_file(
            const std::shared_ptr<arrow::Table>& table,
            const std::string& file_path);
        
        // Read Table from file (IPC format)
        static arrow::Result<std::shared_ptr<arrow::Table>> read_table_from_ipc_file(
            const std::string& file_path);
        
        // Stream RecordBatch to socket/network
        static arrow::Status stream_record_batch(
            const std::shared_ptr<arrow::RecordBatch>& batch,
            arrow::ipc::internal::IpcWriteOptions options = arrow::ipc::IpcWriteOptions::Defaults());
    };
    
    // Arrow computation utilities
    class ArrowComputeUtils {
    public:
        // Compute cosine similarity between two vector columns
        static arrow::Result<std::shared_ptr<arrow::DoubleArray>> compute_cosine_similarity(
            const std::shared_ptr<arrow::DoubleArray>& vectors_a,
            const std::shared_ptr<arrow::DoubleArray>& vectors_b);
        
        // Compute Euclidean distance between two vector columns
        static arrow::Result<std::shared_ptr<arrow::DoubleArray>> compute_euclidean_distance(
            const std::shared_ptr<arrow::DoubleArray>& vectors_a,
            const std::shared_ptr<arrow::DoubleArray>& vectors_b);
        
        // Normalize vectors (L2 normalization)
        static arrow::Result<std::shared_ptr<arrow::DoubleArray>> normalize_vectors(
            const std::shared_ptr<arrow::DoubleArray>& vectors);
        
        // Sort RecordBatch by column
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> sort_record_batch(
            const std::shared_ptr<arrow::RecordBatch>& batch,
            const std::string& column_name,
            arrow::SortOrder order = arrow::SortOrder::Ascending);
        
        // Filter RecordBatch by condition
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> filter_record_batch(
            const std::shared_ptr<arrow::RecordBatch>& batch,
            const std::string& column_name,
            const std::string& condition_value);
    };
    
    // Arrow serialization utilities
    class ArrowSerializationUtils {
    public:
        // Serialize RecordBatch to buffer
        static arrow::Result<std::shared_ptr<arrow::Buffer>> serialize_record_batch(
            const std::shared_ptr<arrow::RecordBatch>& batch);
        
        // Deserialize RecordBatch from buffer
        static arrow::Result<std::shared_ptr<arrow::RecordBatch>> deserialize_record_batch(
            const std::shared_ptr<arrow::Buffer>& buffer);
        
        // Serialize schema to buffer
        static arrow::Result<std::shared_ptr<arrow::Buffer>> serialize_schema(
            const std::shared_ptr<arrow::Schema>& schema);
        
        // Deserialize schema from buffer
        static arrow::Result<std::shared_ptr<arrow::Schema>> deserialize_schema(
            const std::shared_ptr<arrow::Buffer>& buffer);
    };

} // namespace arrow_utils

} // namespace jadevectordb

#endif // JADEVECTORDB_ARROW_UTILS_H