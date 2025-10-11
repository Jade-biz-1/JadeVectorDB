#include "vector.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace jadevectordb {

// Implementation file for Vector data structure
// Currently, the Vector struct is primarily data-focused with inline methods
// Additional utility methods can be added here as needed

// Example utility: calculate cosine similarity between two vectors
float cosineSimilarity(const Vector& v1, const Vector& v2) {
    if (v1.values.size() != v2.values.size()) {
        throw std::invalid_argument("Vectors must have the same dimension for similarity calculation");
    }
    
    double dotProduct = 0.0;
    double magnitude1 = 0.0;
    double magnitude2 = 0.0;
    
    for (size_t i = 0; i < v1.values.size(); ++i) {
        dotProduct += v1.values[i] * v2.values[i];
        magnitude1 += v1.values[i] * v1.values[i];
        magnitude2 += v2.values[i] * v2.values[i];
    }
    
    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);
    
    if (magnitude1 == 0.0 || magnitude2 == 0.0) {
        return 0.0f; // If one vector is zero vector, similarity is 0
    }
    
    return static_cast<float>(dotProduct / (magnitude1 * magnitude2));
}

// Example utility: calculate Euclidean distance between two vectors
float euclideanDistance(const Vector& v1, const Vector& v2) {
    if (v1.values.size() != v2.values.size()) {
        throw std::invalid_argument("Vectors must have the same dimension for distance calculation");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.values.size(); ++i) {
        double diff = v1.values[i] - v2.values[i];
        sum += diff * diff;
    }
    
    return static_cast<float>(std::sqrt(sum));
}

} // namespace jadevectordb