#include "search_utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace jadevectordb {

// Optimized similarity calculations using SIMD when available
float SearchUtils::cosine_similarity_optimized(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        return 0.0f; // Vectors of different dimensions are orthogonal
    }
    
    double dot_product = 0.0;
    double magnitude_v1 = 0.0;
    double magnitude_v2 = 0.0;
    
    // Use a single loop to calculate all values
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        magnitude_v1 += v1[i] * v1[i];
        magnitude_v2 += v2[i] * v2[i];
    }
    
    magnitude_v1 = std::sqrt(magnitude_v1);
    magnitude_v2 = std::sqrt(magnitude_v2);
    
    if (magnitude_v1 == 0.0 || magnitude_v2 == 0.0) {
        return 0.0f; // If one vector is zero vector, similarity is 0
    }
    
    return static_cast<float>(dot_product / (magnitude_v1 * magnitude_v2));
}

float SearchUtils::euclidean_distance_optimized(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        // For vectors of different dimensions, pad the smaller one with zeros
        size_t max_size = std::max(v1.size(), v2.size());
        double sum = 0.0;
        
        for (size_t i = 0; i < max_size; ++i) {
            float val1 = (i < v1.size()) ? v1[i] : 0.0f;
            float val2 = (i < v2.size()) ? v2[i] : 0.0f;
            float diff = val1 - val2;
            sum += diff * diff;
        }
        
        return static_cast<float>(std::sqrt(sum));
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    
    return static_cast<float>(std::sqrt(sum));
}

float SearchUtils::dot_product_optimized(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        // For vectors of different dimensions, pad the smaller one with zeros
        size_t max_size = std::max(v1.size(), v2.size());
        double result = 0.0;
        
        for (size_t i = 0; i < max_size; ++i) {
            float val1 = (i < v1.size()) ? v1[i] : 0.0f;
            float val2 = (i < v2.size()) ? v2[i] : 0.0f;
            result += val1 * val2;
        }
        
        return static_cast<float>(result);
    }
    
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    
    return static_cast<float>(result);
}

float SearchUtils::cosine_similarity_with_early_termination(
    const std::vector<float>& v1, 
    const std::vector<float>& v2, 
    float threshold) {
    
    if (v1.size() != v2.size()) {
        return 0.0f;
    }
    
    double dot_product = 0.0;
    double magnitude_v1 = 0.0;
    double magnitude_v2 = 0.0;
    
    // Calculate magnitudes first
    for (size_t i = 0; i < v1.size(); ++i) {
        magnitude_v1 += v1[i] * v1[i];
        magnitude_v2 += v2[i] * v2[i];
    }
    
    magnitude_v1 = std::sqrt(magnitude_v1);
    magnitude_v2 = std::sqrt(magnitude_v2);
    
    if (magnitude_v1 == 0.0 || magnitude_v2 == 0.0) {
        return 0.0f;
    }
    
    // Calculate dot product with early termination if threshold is exceeded
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        
        // Early termination: if current dot_product divided by magnitudes exceeds threshold
        // we continue, otherwise we could potentially abort (though not done here for correctness)
    }
    
    float similarity = static_cast<float>(dot_product / (magnitude_v1 * magnitude_v2));
    
    return similarity;
}

// KNN search implementation
std::vector<KnnSearch::SearchResult> KnnSearch::knn_search(
    const Vector& query_vector,
    const std::vector<Vector>& candidate_vectors,
    size_t k,
    Algorithm algorithm,
    float threshold) {
    
    std::vector<SearchResult> all_results;
    all_results.reserve(candidate_vectors.size());
    
    // Calculate similarities for all candidate vectors
    for (size_t i = 0; i < candidate_vectors.size(); ++i) {
        float similarity = SearchUtils::cosine_similarity_optimized(
            query_vector.values, 
            candidate_vectors[i].values
        );
        
        // Apply threshold
        if (similarity >= threshold) {
            SearchResult result;
            result.vector_id = candidate_vectors[i].id;
            result.similarity_score = similarity;
            result.original_index = i;
            all_results.push_back(result);
        }
    }
    
    // Apply top-k selection based on algorithm
    std::vector<SearchResult> top_k_results;
    
    if (algorithm == Algorithm::PARTIAL_SORT) {
        // Use partial sort for top-k
        if (k >= all_results.size()) {
            std::sort(all_results.begin(), all_results.end(), 
                     [](const SearchResult& a, const SearchResult& b) {
                         return a.similarity_score > b.similarity_score;  // Descending order
                     });
            return all_results;
        }
        
        std::partial_sort(all_results.begin(), 
                         all_results.begin() + std::min(k, all_results.size()), 
                         all_results.end(),
                         [](const SearchResult& a, const SearchResult& b) {
                             return a.similarity_score > b.similarity_score;  // Descending order
                         });
        all_results.resize(std::min(k, all_results.size()));
        return all_results;
    } 
    else if (algorithm == Algorithm::HEAP) {
        // Use heap for top-k (min-heap to keep largest k elements)
        if (k >= all_results.size()) {
            std::sort(all_results.begin(), all_results.end(), 
                     [](const SearchResult& a, const SearchResult& b) {
                         return a.similarity_score > b.similarity_score;  // Descending order
                     });
            return all_results;
        }
        
        // Create a min-heap to keep track of the k best results
        auto comp = [](const SearchResult& a, const SearchResult& b) {
            return a.similarity_score < b.similarity_score;  // Min-heap based on similarity
        };
        
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(comp)> min_heap(comp);
        
        for (const auto& result : all_results) {
            if (min_heap.size() < k) {
                min_heap.push(result);
            } else if (result.similarity_score > min_heap.top().similarity_score) {
                min_heap.pop();
                min_heap.push(result);
            }
        }
        
        // Extract results from the heap
        top_k_results.reserve(min_heap.size());
        while (!min_heap.empty()) {
            top_k_results.push_back(min_heap.top());
            min_heap.pop();
        }
        
        // Reverse to get descending order (since min_heap.pop() gives smallest first)
        std::reverse(top_k_results.begin(), top_k_results.end());
        return top_k_results;
    }
    else {  // LINEAR algorithm
        // For small datasets, linear scan might be sufficient
        std::sort(all_results.begin(), all_results.end(), 
                 [](const SearchResult& a, const SearchResult& b) {
                     return a.similarity_score > b.similarity_score;  // Descending order
                 });
        if (k < all_results.size()) {
            all_results.resize(k);
        }
        return all_results;
    }
}

bool KnnSearch::validate_search_quality(
    const std::vector<SearchResult>& results,
    const Vector& query_vector,
    const std::vector<Vector>& all_vectors,
    float tolerance) {
    
    // This function validates the search quality by comparing top results
    // against a brute-force search of all vectors
    
    if (results.empty()) {
        return true; // Empty results are valid
    }
    
    // Perform a full brute-force search to get the true top results
    std::vector<SearchResult> brute_force_results;
    brute_force_results.reserve(all_vectors.size());
    
    for (size_t i = 0; i < all_vectors.size(); ++i) {
        float similarity = SearchUtils::cosine_similarity_optimized(
            query_vector.values, 
            all_vectors[i].values
        );
        
        SearchResult result;
        result.vector_id = all_vectors[i].id;
        result.similarity_score = similarity;
        result.original_index = i;
        brute_force_results.push_back(result);
    }
    
    // Sort the brute force results
    std::sort(brute_force_results.begin(), brute_force_results.end(),
             [](const SearchResult& a, const SearchResult& b) {
                 return a.similarity_score > b.similarity_score;
             });
    
    // Compare the top results from both methods
    size_t comparison_size = std::min(results.size(), brute_force_results.size());
    
    for (size_t i = 0; i < comparison_size; ++i) {
        if (results[i].vector_id != brute_force_results[i].vector_id) {
            // IDs don't match, check if the similarity scores are close enough
            if (std::abs(results[i].similarity_score - brute_force_results[i].similarity_score) > tolerance) {
                return false; // Significant discrepancy in results
            }
        } else if (std::abs(results[i].similarity_score - brute_force_results[i].similarity_score) > tolerance) {
            return false; // Same vector but different similarity score
        }
    }
    
    return true;
}

} // namespace jadevectordb