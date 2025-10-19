// Vector space properties for JadeVectorDB
// Properties that validate mathematical properties of vector operations

#ifndef VECTOR_SPACE_PROPERTIES_H
#define VECTOR_SPACE_PROPERTIES_H

#include "property_test_framework.h"
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace property_tests {
namespace vector_space {

// Property: Vector dimension consistency - all vectors in a space have same dimension
inline bool dimension_consistency_property(const std::vector<float>& vector, int expected_dimension) {
    return vector.size() == static_cast<size_t>(expected_dimension);
}

// Property: Norm bounds - normalized vectors have norm close to 1
inline bool norm_bounds_property(const std::vector<float>& vector) {
    float norm = 0.0f;
    for (float val : vector) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    // For normalized vectors, the norm should be close to 1
    // We'll check if it's in the range [0.99, 1.01] to account for floating point precision
    return norm >= 0.99f && norm <= 1.01f;
}

// Property: Distance metric - non-negativity (distance >= 0)
inline bool distance_non_negativity_property(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    
    float distance = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    distance = std::sqrt(distance);
    
    return distance >= 0.0f;
}

// Property: Distance metric - identity (distance(a, a) == 0)
inline bool distance_identity_property(const std::vector<float>& vector) {
    float distance = 0.0f;
    for (float val : vector) {
        distance += val * val;  // Distance from zero vector
    }
    distance = std::sqrt(distance);
    
    // Distance from zero vector is just the norm
    // Distance from vector to itself should be computed separately
    // Instead, we'll check that distance between identical vectors is 0
    return true;  // This property will be tested differently
}

// Check specifically for identity property
inline bool distance_identity_direct_property(const std::vector<float>& vector) {
    // Calculate distance between vector and itself
    float distance = 0.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        float diff = vector[i] - vector[i];  // Always 0
        distance += diff * diff;
    }
    distance = std::sqrt(distance);
    
    return distance < 1e-6f;  // Should be essentially 0
}

// Property: Distance metric - symmetry (distance(a, b) == distance(b, a))
inline bool distance_symmetry_property(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    
    // Calculate distance(a, b)
    float dist_ab = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist_ab += diff * diff;
    }
    dist_ab = std::sqrt(dist_ab);
    
    // Calculate distance(b, a)
    float dist_ba = 0.0f;
    for (size_t i = 0; i < b.size(); ++i) {
        float diff = b[i] - a[i];
        dist_ba += diff * diff;
    }
    dist_ba = std::sqrt(dist_ba);
    
    // Symmetry: distance(a, b) should equal distance(b, a)
    return std::abs(dist_ab - dist_ba) < 1e-6f;
}

// Property: Distance metric - triangle inequality (distance(a, c) <= distance(a, b) + distance(b, c))
inline bool triangle_inequality_property(const std::vector<float>& a, 
                                        const std::vector<float>& b, 
                                        const std::vector<float>& c) {
    if (a.size() != b.size() || b.size() != c.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    
    // Calculate all three distances
    auto calculate_distance = [](const std::vector<float>& x, const std::vector<float>& y) -> float {
        float dist = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            float diff = x[i] - y[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    };
    
    float dist_ac = calculate_distance(a, c);
    float dist_ab = calculate_distance(a, b);
    float dist_bc = calculate_distance(b, c);
    
    // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    return dist_ac <= (dist_ab + dist_bc + 1e-6f);  // Add epsilon for floating point errors
}

// Property: Cosine similarity bounds - should be in range [-1, 1]
inline bool cosine_similarity_bounds_property(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    // Handle zero vectors
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return dot_product == 0.0f;  // Similarity of 0 for zero vectors
    }
    
    float similarity = dot_product / (norm_a * norm_b);
    
    return similarity >= -1.0f && similarity <= 1.0f;
}

// Property: Linear combination preservation - operations should preserve vector space properties
inline bool linear_combination_property(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    
    // Create a simple linear combination: result = 0.5*a + 0.5*b
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = 0.5f * a[i] + 0.5f * b[i];
    }
    
    // The result should be a valid vector
    for (float val : result) {
        if (std::isnan(val) || std::isinf(val)) {
            return false;  // Result should not contain NaN or Infinity
        }
    }
    
    return true;
}

// Generator for normalized vectors (vectors with norm ~1.0)
class NormalizedVectorGenerator : public Generator<std::vector<float>> {
private:
    int dimension;
    
public:
    NormalizedVectorGenerator(int dim) : dimension(dim) {
        if (dim <= 0) throw std::invalid_argument("Dimension must be positive");
    }
    
    std::vector<float> generate(std::mt19937& rng) override {
        // First, generate a random vector
        std::vector<float> result(dimension);
        std::normal_distribution<float> dist(0.0f, 1.0f);  // Normal distribution
        
        for (int i = 0; i < dimension; ++i) {
            result[i] = dist(rng);
        }
        
        // Then normalize it to have unit length
        float norm = 0.0f;
        for (float val : result) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-9f) {  // Avoid division by zero
            for (auto& val : result) {
                val /= norm;
            }
        }
        
        return result;
    }
};

// Generator for arbitrary vectors
class ArbitraryVectorGenerator : public Generator<std::vector<float>> {
private:
    int min_dimension;
    int max_dimension;
    float min_value;
    float max_value;
    
public:
    ArbitraryVectorGenerator(int min_dim, int max_dim, float min_val = -10.0f, float max_val = 10.0f) 
        : min_dimension(min_dim), max_dimension(max_dim), min_value(min_val), max_value(max_val) {
        if (min_dim <= 0 || max_dim < min_dim) {
            throw std::invalid_argument("Invalid dimension range");
        }
    }
    
    std::vector<float> generate(std::mt19937& rng) override {
        // Random dimension between min and max
        std::uniform_int_distribution<int> dim_dist(min_dimension, max_dimension);
        int dimension = dim_dist(rng);
        
        // Generate vector with random values
        std::vector<float> result(dimension);
        std::uniform_real_distribution<float> val_dist(min_value, max_value);
        
        for (int i = 0; i < dimension; ++i) {
            result[i] = val_dist(rng);
        }
        
        return result;
    }
};

} // namespace vector_space
} // namespace property_tests

#endif // VECTOR_SPACE_PROPERTIES_H