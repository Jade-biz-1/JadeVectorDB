#include "simd_ops.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>

namespace jadevectordb {

namespace simd_ops {

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension for similarity calculation");
        }
        
        // Convert to Eigen vectors for SIMD optimization
        VectorXf va = to_eigen_vector(a);
        VectorXf vb = to_eigen_vector(b);
        
        return eigen_cosine_similarity(va, vb);
    }
    
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension for distance calculation");
        }
        
        VectorXf va = to_eigen_vector(a);
        VectorXf vb = to_eigen_vector(b);
        
        VectorXf diff = va - vb;
        return std::sqrt(diff.squaredNorm());
    }
    
    float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension for dot product calculation");
        }
        
        VectorXf va = to_eigen_vector(a);
        VectorXf vb = to_eigen_vector(b);
        
        return va.dot(vb);
    }
    
    std::vector<float> normalize(const std::vector<float>& vec) {
        if (vec.empty()) {
            return vec;
        }
        
        VectorXf v = to_eigen_vector(vec);
        float norm = v.norm();
        
        if (norm == 0.0f) {
            return vec;
        }
        
        v /= norm;
        return from_eigen_vector(v);
    }
    
    std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension for addition");
        }
        
        VectorXf va = to_eigen_vector(a);
        VectorXf vb = to_eigen_vector(b);
        
        VectorXf result = va + vb;
        return from_eigen_vector(result);
    }
    
    std::vector<float> vector_subtract(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension for subtraction");
        }
        
        VectorXf va = to_eigen_vector(a);
        VectorXf vb = to_eigen_vector(b);
        
        VectorXf result = va - vb;
        return from_eigen_vector(result);
    }
    
    std::vector<float> vector_multiply_scalar(const std::vector<float>& vec, float scalar) {
        VectorXf v = to_eigen_vector(vec);
        VectorXf result = v * scalar;
        return from_eigen_vector(result);
    }
    
    float magnitude(const std::vector<float>& vec) {
        VectorXf v = to_eigen_vector(vec);
        return v.norm();
    }
    
    std::vector<float> batch_cosine_similarity(const std::vector<float>& query, 
                                                const std::vector<std::vector<float>>& candidates) {
        if (query.empty() || candidates.empty()) {
            return {};
        }
        
        // Convert query to Eigen vector
        VectorXf q = to_eigen_vector(query);
        
        // Convert candidates to Eigen matrix
        size_t dim = query.size();
        MatrixXf candidate_matrix(dim, candidates.size());
        
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].size() != dim) {
                throw std::invalid_argument("All candidate vectors must have the same dimension as query");
            }
            candidate_matrix.col(i) = to_eigen_vector(candidates[i]);
        }
        
        return eigen_batch_cosine_similarity(q, candidate_matrix);
    }
    
    // Convert std::vector to Eigen Vector
    VectorXf to_eigen_vector(const std::vector<float>& vec) {
        return Eigen::Map<const VectorXf>(vec.data(), vec.size());
    }
    
    // Convert Eigen Vector to std::vector
    std::vector<float> from_eigen_vector(const VectorXf& eigen_vec) {
        return std::vector<float>(eigen_vec.data(), eigen_vec.data() + eigen_vec.size());
    }
    
    // Eigen-based cosine similarity
    float eigen_cosine_similarity(const VectorXf& a, const VectorXf& b) {
        float dot = a.dot(b);
        float norm_a = a.norm();
        float norm_b = b.norm();
        
        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }
        
        return dot / (norm_a * norm_b);
    }
    
    // Eigen-based batch cosine similarity
    std::vector<float> eigen_batch_cosine_similarity(const VectorXf& query,
                                                      const MatrixXf& candidates) {
        std::vector<float> similarities(candidates.cols());
        
        float query_norm = query.norm();
        if (query_norm == 0.0f) {
            return std::vector<float>(candidates.cols(), 0.0f);
        }
        
        // Compute dot products
        VectorXf dot_products = candidates.transpose() * query;
        
        // Compute norms for each candidate
        VectorXf candidate_norms(candidates.cols());
        for (int i = 0; i < candidates.cols(); ++i) {
            candidate_norms(i) = candidates.col(i).norm();
        }
        
        // Compute similarities
        for (int i = 0; i < candidates.cols(); ++i) {
            if (candidate_norms(i) == 0.0f) {
                similarities[i] = 0.0f;
            } else {
                similarities[i] = dot_products(i) / (query_norm * candidate_norms(i));
            }
        }
        
        return similarities;
    }
    
    // SIMD-optimized matrix-vector multiplication
    VectorXf matrix_vector_multiply(const MatrixXf& matrix, const VectorXf& vector) {
        return matrix * vector;
    }
    
    // SIMD-optimized vector addition
    VectorXf eigen_vector_add(const VectorXf& a, const VectorXf& b) {
        return a + b;
    }
    
    // SIMD-optimized vector subtraction
    VectorXf eigen_vector_subtract(const VectorXf& a, const VectorXf& b) {
        return a - b;
    }
    
    // SIMD-optimized scalar multiplication
    VectorXf eigen_vector_multiply_scalar(const VectorXf& vec, float scalar) {
        return vec * scalar;
    }

} // namespace simd_ops

} // namespace jadevectordb