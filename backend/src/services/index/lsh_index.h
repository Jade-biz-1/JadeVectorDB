#ifndef JADEVECTORDB_LSH_INDEX_H
#define JADEVECTORDB_LSH_INDEX_H

#include "models/index.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace jadevectordb {

// Represents a hash function for LSH
struct LshHashFunction {
    std::vector<float> a;  // Random vector for projection
    float b;               // Random offset
    float w;               // Width of hash bucket
    
    LshHashFunction(int dimension, float bucket_width, std::mt19937& rng);
};

/**
 * @brief Implementation of Locality Sensitive Hashing (LSH) index algorithm
 * 
 * This implementation provides approximate nearest neighbor search using
 * random projections to hash similar vectors into the same buckets.
 */
class LshIndex {
public:
    struct LshParams {
        int num_tables = 10;            // Number of hash tables
        int num_projections = 16;       // Number of projections per table
        float bucket_width = 4.0f;      // Width of hash bucket
        int random_seed = 100;          // Seed for random projections
        
        // Constructor
        LshParams() = default;
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    LshParams params_;
    
    // Data structures
    std::vector<std::vector<LshHashFunction>> hash_functions_;  // [table_idx][proj_idx]
    std::vector<std::unordered_map<std::string, std::vector<int>>> hash_tables_; // [table_idx]
    std::unordered_map<int, std::vector<float>> vectors_;       // Vector ID -> vector data
    std::vector<std::mutex> table_locks_;                       // Mutex for each hash table
    
    // Current state
    int dimension_ = 0;
    bool is_built_ = false;
    
    // Random generator
    std::mt19937 rng_;
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit LshIndex(const LshParams& params = LshParams{});
    ~LshIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const LshParams& params);
    
    // Add a vector to the index
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build the index structure
    Result<bool> build();
    
    // Search for similar vectors
    Result<std::vector<std::pair<int, float>>> search(const std::vector<float>& query,
                                                     int k = 10,
                                                     float threshold = 0.0f) const;
    
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
    
    // Get the number of hash tables
    int get_num_tables() const;
    
    // Get the number of projections per table
    int get_num_projections() const;

private:
    // Internal implementation methods
    
    // Compute hash value for a vector using a specific hash function
    int compute_hash(const std::vector<float>& vector, const LshHashFunction& hash_func) const;
    
    // Generate a hash key for a vector using all hash functions in a table
    std::string generate_hash_key(const std::vector<float>& vector, int table_idx) const;
    
    // Compute exact distance between two vectors
    float compute_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Compute cosine similarity between two vectors
    float compute_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Add a vector to all hash tables
    void add_vector_to_tables(int vector_id, const std::vector<float>& vector);
    
    // Remove a vector from all hash tables
    void remove_vector_from_tables(int vector_id, const std::vector<float>& vector);
    
    // Validate index state
    bool validate() const;
    
    // Recompute hashes for all vectors (used after parameter changes)
    Result<bool> recompute_all_hashes();
    
    // Generate random projection vectors
    void generate_random_projections();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_LSH_INDEX_H