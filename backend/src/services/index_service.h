#ifndef JADEVECTORDB_INDEX_SERVICE_H
#define JADEVECTORDB_INDEX_SERVICE_H

#include "models/index.h"
#include "services/index/hnsw_index.h"
#include "services/index/ivf_index.h"
#include "services/index/lsh_index.h"
#include "services/index/flat_index.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace jadevectordb {

// Enum for index types
enum class IndexType {
    HNSW,
    IVF,
    LSH,
    FLAT
};

// Configuration for index creation
struct IndexConfig {
    IndexType type;
    std::string database_id;
    std::unordered_map<std::string, std::string> parameters;  // Type-specific parameters
    
    IndexConfig() : type(IndexType::FLAT) {}
    IndexConfig(IndexType t, const std::string& db_id) : type(t), database_id(db_id) {}
};

/**
 * @brief Service to manage different index types with configurable parameters
 * 
 * This service provides a unified interface for creating, managing, and using
 * different types of vector indices, including HNSW, IVF, LSH, and Flat.
 */
class IndexService {
private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Storage for different index types
    std::unordered_map<std::string, std::unique_ptr<HnswIndex>> hnsw_indexes_;
    std::unordered_map<std::string, std::unique_ptr<IvfIndex>> ivf_indexes_;
    std::unordered_map<std::string, std::unique_ptr<LshIndex>> lsh_indexes_;
    std::unordered_map<std::string, std::unique_ptr<FlatIndex>> flat_indexes_;
    
    // Index metadata
    std::unordered_map<std::string, Index> index_metadata_;  // index_id -> Index object
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit IndexService();
    ~IndexService() = default;
    
    // Initialize the service
    bool initialize();
    
    // Create a new index based on the configuration
    Result<std::string> create_index(const IndexConfig& config);
    
    // Create an HNSW index
    Result<std::string> create_hnsw_index(const std::string& database_id,
                                        const HnswIndex::HnswParams& params = HnswIndex::HnswParams{});
    
    // Create an IVF index
    Result<std::string> create_ivf_index(const std::string& database_id,
                                       const IvfIndex::IvfParams& params = IvfIndex::IvfParams{});
    
    // Create an LSH index
    Result<std::string> create_lsh_index(const std::string& database_id,
                                       const LshIndex::LshParams& params = LshIndex::LshParams{});
    
    // Create a Flat index
    Result<std::string> create_flat_index(const std::string& database_id,
                                        const FlatIndex::FlatParams& params = FlatIndex::FlatParams{});
    
    // Add a vector to an index
    Result<bool> add_vector_to_index(const std::string& index_id,
                                   int vector_id,
                                   const std::vector<float>& vector);
    
    // Search in an index
    Result<std::vector<std::pair<int, float>>> search_in_index(const std::string& index_id,
                                                              const std::vector<float>& query,
                                                              int k = 10,
                                                              float threshold = 0.0f);
    
    // Build an index from vectors
    Result<bool> build_index(const std::string& index_id,
                           const std::vector<std::pair<int, std::vector<float>>>& vectors);
    
    // Get index by ID
    Result<Index> get_index(const std::string& index_id) const;
    
    // Update a vector in an index
    Result<bool> update_vector_in_index(const std::string& index_id,
                                      int vector_id,
                                      const std::vector<float>& new_vector);
    
    // Remove a vector from an index
    Result<bool> remove_vector_from_index(const std::string& index_id, int vector_id);
    
    // Check if index exists
    bool index_exists(const std::string& index_id) const;
    
    // Delete an index
    Result<bool> delete_index(const std::string& index_id);
    
    // Get all indexes for a database
    Result<std::vector<Index>> get_indexes_for_database(const std::string& database_id) const;
    
    // Get index statistics
    Result<std::unordered_map<std::string, std::string>> get_index_stats(const std::string& index_id) const;
    
    // Update index configuration
    Result<bool> update_index_config(const std::string& index_id,
                                   const std::unordered_map<std::string, std::string>& new_params);
    
    // Get supported index types
    std::vector<std::string> get_supported_index_types() const;
    
    // Get all active indexes
    Result<std::vector<std::string>> get_all_index_ids() const;
    
    // Rebuild an index
    Result<bool> rebuild_index(const std::string& index_id);

private:
    // Helper methods
    
    // Generate a unique index ID
    std::string generate_index_id() const;
    
    // Convert string index type to enum
    Result<IndexType> string_to_index_type(const std::string& type_str) const;
    
    // Convert enum index type to string
    std::string index_type_to_string(IndexType type) const;
    
    // Validate index parameters
    Result<bool> validate_index_params(IndexType type, const std::unordered_map<std::string, std::string>& params) const;
    
    // Convert string parameters to HNSW parameters
    Result<HnswIndex::HnswParams> parse_hnsw_params(const std::unordered_map<std::string, std::string>& params) const;
    
    // Convert string parameters to IVF parameters
    Result<IvfIndex::IvfParams> parse_ivf_params(const std::unordered_map<std::string, std::string>& params) const;
    
    // Convert string parameters to LSH parameters
    Result<LshIndex::LshParams> parse_lsh_params(const std::unordered_map<std::string, std::string>& params) const;
    
    // Convert string parameters to Flat parameters
    Result<FlatIndex::FlatParams> parse_flat_params(const std::unordered_map<std::string, std::string>& params) const;
    
    // Initialize index metadata from config
    Index create_index_metadata(const std::string& index_id, const IndexConfig& config) const;
    
    // Clean up an index by type
    void cleanup_index(const std::string& index_id, IndexType type);
    
    // Get index type from index ID
    Result<IndexType> get_index_type(const std::string& index_id) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_INDEX_SERVICE_H