#ifndef JADEVECTORDB_OPQ_INDEX_H
#define JADEVECTORDB_OPQ_INDEX_H

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
 * @brief Implementation of Optimized Product Quantization (OPQ) index algorithm
 * 
 * This implementation extends Product Quantization by learning a rotation matrix
 * to minimize the quantization error. The rotation aligns the principal axes
 * of the data with the quantization boundaries, resulting in better approximation
 * of the original vectors.
 */
class OpqIndex {
public:
    struct OpqParams {
        int subvector_dimension;          // Dimension of each subvector
        int num_centroids;                // Number of centroids for each subvector (typically 256 for 8-bit code)
        int num_subvectors;               // Number of subvectors (computed from vector dimension and subvector_dimension)
        int rotation_optimization_iterations;  // Number of iterations for rotation optimization
        bool use_residual;                // Whether to use residual quantization
        int max_iterations;               // Max iterations for k-means clustering
        float tolerance;                  // Tolerance for k-means convergence
        int random_seed;                  // Random seed for initialization
        
        // Constructor
        OpqParams() : subvector_dimension(8), num_centroids(256), num_subvectors(0), 
                     rotation_optimization_iterations(10), use_residual(false), 
                     max_iterations(100), tolerance(1e-4f), random_seed(100) {}
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    OpqParams params_;
    
    // Data structures
    std::vector<std::vector<std::vector<float>>> subvector_centroids_;  // [subvector_idx][centroid_idx][subvector_dimension]
    std::vector<std::vector<float>> rotation_matrix_;                   // Rotation matrix for OPQ
    std::vector<std::vector<float>> inverse_rotation_matrix_;           // Inverse of rotation matrix
    std::unordered_map<int, std::vector<uint8_t>> pq_codes_;            // Vector ID -> PQ codes
    std::unordered_map<int, std::vector<float>> original_vectors_;      // Vector ID -> original vector (for reconstruction if needed)
    
    // Current state
    int dimension_ = 0;
    bool is_trained_ = false;      // Whether the rotation matrix and subvector centroids are trained
    bool is_built_ = false;        // Whether the index is built with vectors
    
    // Random generator
    mutable std::mt19937 rng_;
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit OpqIndex(const OpqParams& params = OpqParams{});
    ~OpqIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const OpqParams& params);
    
    // Add a vector to the index (will be encoded during build phase)
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build the index (train rotation matrix, subvector centroids, encode vectors)
    Result<bool> build();
    
    // Search for similar vectors using OPQ approximation
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
    
    // Apply rotation to a vector
    std::vector<float> apply_rotation(const std::vector<float>& vector) const;
    
    // Apply inverse rotation to a vector
    std::vector<float> apply_inverse_rotation(const std::vector<float>& vector) const;
    
    // Encode a vector using the trained rotation and subvector centroids
    std::vector<uint8_t> encode_vector(const std::vector<float>& vector) const;
    
    // Decode a PQ code to reconstruct an approximate vector
    std::vector<float> decode_code(const std::vector<uint8_t>& code) const;

private:
    // Internal implementation methods
    
    // Learn the rotation matrix for OPQ
    Result<bool> learn_rotation_matrix(std::vector<std::vector<float>>& vectors);
    
    // Initialize the rotation matrix (e.g., as identity matrix)
    void initialize_rotation_matrix();
    
    // Optimize the rotation matrix using the current centroids
    void optimize_rotation_matrix(const std::vector<std::vector<float>>& vectors);
    
    // Train subvector centroids using k-means clustering on rotated vectors
    Result<bool> train_subvector_centroids(const std::vector<std::vector<float>>& vectors);
    
    // Split a vector into subvectors
    std::vector<std::vector<float>> split_into_subvectors(const std::vector<float>& vector) const;
    
    // Reconstruct a vector from its PQ code
    std::vector<float> reconstruct_from_code(const std::vector<uint8_t>& code) const;
    
    // Compute distance in the PQ space (asymmetric distance using rotated vectors)
    float compute_pq_distance(const std::vector<uint8_t>& code1, const std::vector<uint8_t>& code2) const;
    
    // Compute distance between a rotated query vector and a PQ code (asymmetric distance)
    float compute_asymmetric_distance(const std::vector<float>& query, const std::vector<uint8_t>& code) const;
    
    // Perform k-means clustering on subvectors to get centroids
    std::vector<std::vector<float>> perform_kmeans_clustering_on_subvectors(
        const std::vector<std::vector<float>>& subvectors, int k) const;
    
    // Compute squared distance between two vectors
    float compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Compute the covariance matrix of a set of vectors
    std::vector<std::vector<float>> compute_covariance_matrix(const std::vector<std::vector<float>>& vectors) const;
    
    // Perform SVD (Singular Value Decomposition) - simplified implementation
    void compute_svd(const std::vector<std::vector<float>>& matrix, 
                     std::vector<std::vector<float>>& U,
                     std::vector<float>& S,
                     std::vector<std::vector<float>>& V) const;
    
    // Multiply two matrices
    std::vector<std::vector<float>> multiply_matrices(const std::vector<std::vector<float>>& A,
                                                     const std::vector<std::vector<float>>& B) const;
    
    // Multiply matrix and vector
    std::vector<float> multiply_matrix_vector(const std::vector<std::vector<float>>& matrix,
                                             const std::vector<float>& vector) const;
    
    // Transpose a matrix
    std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix) const;
    
    // Validate index state
    bool validate() const;
    
    // Initialize random generator
    void initialize_random_generator();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_OPQ_INDEX_H