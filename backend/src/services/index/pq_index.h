#ifndef JADEVECTORDB_PQ_INDEX_H
#define JADEVECTORDB_PQ_INDEX_H

#include "models/index.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <shared_mutex>
#include <random>

namespace jadevectordb {

/**
 * @brief Implementation of Product Quantization (PQ) index algorithm
 * 
 * This implementation provides efficient similarity search using vector quantization.
 * It divides high-dimensional vectors into subvectors and quantizes each subvector
 * separately, resulting in a compressed representation that enables fast approximate
 * similarity search.
 */
class PqIndex {
public:
    struct PqParams {
        int subvector_dimension;          // Dimension of each subvector
        int num_centroids;                // Number of centroids for each subvector (typically 256 for 8-bit code)
        int num_subvectors;               // Number of subvectors (computed from vector dimension and subvector_dimension)
        bool use_residual;                // Whether to use residual quantization
        int max_iterations;               // Max iterations for k-means clustering
        float tolerance;                  // Tolerance for k-means convergence
        int random_seed;                  // Random seed for initialization
        
        // Constructor
        PqParams() : subvector_dimension(8), num_centroids(256), num_subvectors(0), 
                     use_residual(false), max_iterations(100), 
                     tolerance(1e-4f), random_seed(100) {}
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    PqParams params_;
    
    // Data structures
    std::vector<std::vector<std::vector<float>>> subvector_centroids_;  // [subvector_idx][centroid_idx][subvector_dimension]
    std::unordered_map<int, std::vector<uint8_t>> pq_codes_;            // Vector ID -> PQ codes
    std::unordered_map<int, std::vector<float>> original_vectors_;      // Vector ID -> original vector (for reconstruction if needed)
    
    // Current state
    int dimension_ = 0;
    bool is_trained_ = false;      // Whether the subvector centroids are trained
    bool is_built_ = false;        // Whether the index is built with vectors
    
    // Random generator
    mutable std::mt19937 rng_;
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit PqIndex(const PqParams& params = PqParams{});
    ~PqIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const PqParams& params);
    
    // Add a vector to the index (will be encoded during build phase)
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build the index (train subvector centroids, encode vectors)
    Result<bool> build();
    
    // Search for similar vectors using PQ approximation
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
    
    // Get the number of subvectors
    int get_num_subvectors() const;
    
    // Get the dimension of each subvector
    int get_subvector_dimension() const;
    
    // Encode a vector using the trained subvector centroids
    std::vector<uint8_t> encode_vector(const std::vector<float>& vector) const;
    
    // Decode a PQ code to reconstruct an approximate vector
    std::vector<float> decode_code(const std::vector<uint8_t>& code) const;

private:
    // Internal implementation methods
    
    // Train subvector centroids using k-means clustering
    Result<bool> train_subvector_centroids(const std::vector<std::vector<float>>& vectors);
    
    // Split a vector into subvectors
    std::vector<std::vector<float>> split_into_subvectors(const std::vector<float>& vector) const;
    
    // Reconstruct a vector from its PQ code
    std::vector<float> reconstruct_from_code(const std::vector<uint8_t>& code) const;
    
    // Compute distance in the PQ space (asymmetric distance)
    float compute_pq_distance(const std::vector<uint8_t>& code1, const std::vector<uint8_t>& code2) const;
    
    // Compute distance between a query vector and a PQ code (asymmetric distance)
    float compute_asymmetric_distance(const std::vector<float>& query, const std::vector<uint8_t>& code) const;
    
    // Perform k-means clustering on subvectors to get centroids
    std::vector<std::vector<float>> perform_kmeans_clustering_on_subvectors(
        const std::vector<std::vector<float>>& subvectors, int k) const;
    
    // Compute squared distance between two vectors
    float compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Validate index state
    bool validate() const;
    
    // Initialize random generator
    void initialize_random_generator();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_PQ_INDEX_H