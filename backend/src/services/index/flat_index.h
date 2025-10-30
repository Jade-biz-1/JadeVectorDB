#ifndef JADEVECTORDB_FLAT_INDEX_H
#define JADEVECTORDB_FLAT_INDEX_H

#include "models/index.h"
#include "lib/error_handling.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace jadevectordb {

namespace logging {
    class Logger;
}


/**
 * @brief Implementation of Flat (Linear) index algorithm
 * 
 * This implementation provides brute-force similarity search by comparing
 * the query vector with all stored vectors. While not efficient for large
 * datasets, it provides exact search results.
 */
class FlatIndex {
public:
    struct FlatParams {
        // For flat index, parameters are minimal
        // It's mainly used as a baseline for comparison
        bool normalize_vectors;  // Whether to normalize vectors before storage
        
        // Constructor
        FlatParams() : normalize_vectors(false) {}
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    FlatParams params_;
    
    // Data structures
    std::vector<std::pair<int, std::vector<float>>> vectors_;  // Vector ID and data
    std::unordered_map<int, size_t> id_to_idx_;               // Maps vector ID to index in vectors_
    std::vector<float> norms_;                                // L2 norms of vectors (if normalized)
    
    // Current state
    int dimension_ = 0;
    bool is_built_ = false;
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit FlatIndex(const FlatParams& params = FlatParams{});
    ~FlatIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const FlatParams& params);
    
    // Add a vector to the index
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build the index structure (for flat index, this is mostly a no-op)
    Result<bool> build();
    
    // Search for similar vectors using brute-force approach
    Result<std::vector<std::pair<int, float>>> search(const std::vector<float>& query,
                                                     int k = 10,
                                                     float threshold = 0.0f) const;
    
    // Search for similar vectors with custom distance function
    Result<std::vector<std::pair<int, float>>> search_with_distance(
        const std::vector<float>& query,
        int k,
        float threshold,
        std::function<float(const std::vector<float>&, const std::vector<float>&)> distance_func) const;
    
    // Build the index from a set of vectors (batch operation)
    Result<bool> build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors);
    
    // Check if the index contains a specific vector
    bool contains(int vector_id) const;
    
    // Remove a vector from the index
    Result<bool> remove_vector(int vector_id);
    
    // Update a vector in the index
    Result<bool> update_vector(int vector_id, const std::vector<float>& new_vector);
    
    // Get the number of vectors in the index
    size_t size() const;
    
    // Get index statistics
    Result<std::unordered_map<std::string, std::string>> get_stats() const;
    
    // Clear the index
    void clear();
    
    // Check if the index is empty
    bool empty() const;
    
    // Get the dimension of vectors in the index
    int get_dimension() const;
    
    // Compute cosine similarity between two vectors
    float compute_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Compute Euclidean distance between two vectors
    float compute_euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Compute dot product between two vectors
    float compute_dot_product(const std::vector<float>& a, const std::vector<float>& b) const;

private:
    // Internal implementation methods
    
    // Normalize a vector in place
    std::vector<float> normalize_vector(const std::vector<float>& vector) const;
    
    // Compute L2 norm of a vector
    float compute_l2_norm(const std::vector<float>& vector) const;
    
    // Validate index state
    bool validate() const;
    
    // Add a single vector to internal storage
    void add_vector_internal(int vector_id, const std::vector<float>& vector);
    
    // Get vector by ID
    Result<std::vector<float>> get_vector_by_id(int vector_id) const;
    
    // Update a single vector in internal storage
    void update_vector_internal(int vector_id, const std::vector<float>& new_vector);
    
    // Remove vector by ID
    void remove_vector_internal(int vector_id);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_FLAT_INDEX_H