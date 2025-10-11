#ifndef JADEVECTORDB_SIMD_OPS_H
#define JADEVECTORDB_SIMD_OPS_H

#include <vector>
#include <cstddef>
#include <Eigen/Dense>

namespace jadevectordb {

// SIMD-optimized vector operations using Eigen library
namespace simd_ops {

    // Compute cosine similarity between two vectors
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    
    // Compute Euclidean distance between two vectors
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b);
    
    // Compute dot product of two vectors
    float dot_product(const std::vector<float>& a, const std::vector<float>& b);
    
    // Normalize a vector (L2 normalization)
    std::vector<float> normalize(const std::vector<float>& vec);
    
    // Compute sum of two vectors
    std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b);
    
    // Compute difference of two vectors
    std::vector<float> vector_subtract(const std::vector<float>& a, const std::vector<float>& b);
    
    // Multiply vector by scalar
    std::vector<float> vector_multiply_scalar(const std::vector<float>& vec, float scalar);
    
    // Compute magnitude (L2 norm) of a vector
    float magnitude(const std::vector<float>& vec);
    
    // Compute batch cosine similarities
    std::vector<float> batch_cosine_similarity(const std::vector<float>& query, 
                                               const std::vector<std::vector<float>>& candidates);
    
    // SIMD-optimized operations using Eigen
    using VectorXf = Eigen::VectorXf;
    using MatrixXf = Eigen::MatrixXf;
    
    // Convert std::vector to Eigen Vector
    VectorXf to_eigen_vector(const std::vector<float>& vec);
    
    // Convert Eigen Vector to std::vector
    std::vector<float> from_eigen_vector(const VectorXf& eigen_vec);
    
    // Eigen-based cosine similarity
    float eigen_cosine_similarity(const VectorXf& a, const VectorXf& b);
    
    // Eigen-based batch cosine similarity
    std::vector<float> eigen_batch_cosine_similarity(const VectorXf& query,
                                                     const MatrixXf& candidates);
    
    // SIMD-optimized matrix-vector multiplication
    VectorXf matrix_vector_multiply(const MatrixXf& matrix, const VectorXf& vector);
    
    // SIMD-optimized vector addition
    VectorXf eigen_vector_add(const VectorXf& a, const VectorXf& b);
    
    // SIMD-optimized vector subtraction
    VectorXf eigen_vector_subtract(const VectorXf& a, const VectorXf& b);
    
    // SIMD-optimized scalar multiplication
    VectorXf eigen_vector_multiply_scalar(const VectorXf& vec, float scalar);

} // namespace simd_ops

} // namespace jadevectordb

#endif // JADEVECTORDB_SIMD_OPS_H