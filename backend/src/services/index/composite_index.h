#ifndef JADEVECTORDB_COMPOSITE_INDEX_H
#define JADEVECTORDB_COMPOSITE_INDEX_H

#include "models/index.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/index/hnsw_index.h"
#include "services/index/ivf_index.h"
#include "services/index/lsh_index.h"
#include "services/index/flat_index.h"
#include "services/index/pq_index.h"
#include "services/index/opq_index.h"
#include "services/index/sq_index.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <shared_mutex>
#include <random>

namespace jadevectordb {

// Enum for supported index types in composite index
enum class CompositeIndexType {
    HNSW,
    IVF,
    LSH,
    FLAT,
    PQ,
    OPQ,
    SQ
};

// Structure to represent a component index in the composite index
struct CompositeIndexComponent {
    std::string id;                    // Unique identifier for this component
    CompositeIndexType type;          // Type of index
    std::unique_ptr<void, void(*)(void*)> index_ptr;  // Pointer to the actual index
    std::unordered_map<std::string, std::string> params;  // Parameters for this component
    float weight;                     // Weight for this component during search (for weighted fusion)
    
    // Constructor
    CompositeIndexComponent(const std::string& component_id, 
                           CompositeIndexType index_type, 
                           void* index, 
                           const std::unordered_map<std::string, std::string>& index_params,
                           float component_weight = 1.0f)
        : id(component_id), type(index_type), 
          index_ptr(index, [](void* p){ /* Deleter will be set based on actual type */ }),
          params(index_params), weight(component_weight) {}
};

/**
 * @brief Implementation of Composite Index that combines multiple index types
 * 
 * This implementation allows combining different index algorithms to leverage
 * their respective strengths. It supports multiple search strategies:
 * - Independent search: Search each component separately
 * - Fused search: Combine results from multiple components using various fusion methods
 * - Hierarchical search: Use one index to filter candidates, another to refine
 */
class CompositeIndex {
public:
    struct CompositeIndexParams {
        // How to combine results from different indices
        enum FusionMethod {
            RRF,        // Reciprocal Rank Fusion
            WEIGHTED,   // Weighted score combination
            SIMPLE,     // Simple combination without weights
            HYBRID      // Hierarchical combination
        };
        
        FusionMethod fusion_method;
        int rrf_k;  // Parameter for RRF (used for rank normalization)
        bool enable_filtering;  // Whether to allow filtering during search
        bool allow_multiple_searches;  // Whether to allow searching all components
        
        // Constructor
        CompositeIndexParams() : fusion_method(RRF), rrf_k(60), 
                                enable_filtering(true), allow_multiple_searches(true) {}
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    CompositeIndexParams params_;
    
    // Data structures
    std::vector<CompositeIndexComponent> components_;  // The component indices
    std::unordered_map<std::string, size_t> id_to_idx_map_;  // Map from component ID to index in components_
    
    // Current state
    int dimension_ = 0;
    bool is_built_ = false;  // Whether all component indices are built
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit CompositeIndex(const CompositeIndexParams& params = CompositeIndexParams{});
    ~CompositeIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const CompositeIndexParams& params);
    
    // Add a component index to the composite index
    Result<bool> add_component_index(const std::string& component_id,
                                    CompositeIndexType type,
                                    const std::unordered_map<std::string, std::string>& params,
                                    float weight = 1.0f);
    
    // Add a vector to all component indices
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build all component indices
    Result<bool> build();
    
    // Search for similar vectors using the composite approach
    Result<std::vector<std::pair<int, float>>> search(const std::vector<float>& query,
                                                     int k = 10,
                                                     float threshold = 0.0f) const;
    
    // Build the index from a set of vectors (batch operation)
    Result<bool> build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors);
    
    // Check if the index contains a specific vector
    bool contains(int vector_id) const;
    
    // Remove a vector from all component indices
    Result<bool> remove_vector(int vector_id);
    
    // Update a vector in all component indices
    Result<bool> update_vector(int vector_id, const std::vector<float>& new_vector);
    
    // Get the number of vectors in the composite index
    size_t size() const;
    
    // Get index statistics
    Result<std::unordered_map<std::string, std::string>> get_stats() const;
    
    // Clear the index and all components
    void clear();
    
    // Check if the index is empty
    bool empty() const;
    
    // Get the dimension of vectors in the index
    int get_dimension() const;
    
    // Get number of component indices
    size_t get_num_components() const;
    
    // Get the fusion method being used
    CompositeIndexParams::FusionMethod get_fusion_method() const;

private:
    // Internal implementation methods
    
    // Perform search on a single component index
    Result<std::vector<std::pair<int, float>>> search_component(
        size_t component_idx, 
        const std::vector<float>& query, 
        int k, 
        float threshold) const;
    
    // Fuse results from multiple components using RRF (Reciprocal Rank Fusion)
    std::vector<std::pair<int, float>> fuse_results_rrf(
        const std::vector<std::vector<std::pair<int, float>>>& component_results) const;
    
    // Fuse results from multiple components using weighted scoring
    std::vector<std::pair<int, float>> fuse_results_weighted(
        const std::vector<std::vector<std::pair<int, float>>>& component_results) const;
    
    // Fuse results from multiple components using simple combination
    std::vector<std::pair<int, float>> fuse_results_simple(
        const std::vector<std::vector<std::pair<int, float>>>& component_results) const;
    
    // Compute squared distance between two vectors
    float compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Validate index state
    bool validate() const;
    
    // Get the size of a specific component index
    size_t get_component_size(size_t component_idx) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_COMPOSITE_INDEX_H