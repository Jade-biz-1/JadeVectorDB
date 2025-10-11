#ifndef JADEVECTORDB_SEARCH_UTILS_H
#define JADEVECTORDB_SEARCH_UTILS_H

#include <vector>
#include <queue>
#include <algorithm>
#include <functional>
#include <limits>

#include "models/vector.h"

namespace jadevectordb {

// Utility functions for search optimizations
class SearchUtils {
public:
    // Optimized similarity calculations using SIMD when available
    static float cosine_similarity_optimized(const std::vector<float>& v1, const std::vector<float>& v2);
    
    // Optimized Euclidean distance calculation
    static float euclidean_distance_optimized(const std::vector<float>& v1, const std::vector<float>& v2);
    
    // Optimized dot product calculation
    static float dot_product_optimized(const std::vector<float>& v1, const std::vector<float>& v2);
    
    // Optimized similarity calculation with early termination
    static float cosine_similarity_with_early_termination(
        const std::vector<float>& v1, 
        const std::vector<float>& v2, 
        float threshold = 0.0f);
    
    // Top-K selection using partial sort (more efficient for small K)
    template<typename T, typename Compare>
    static std::vector<T> top_k_optimized(std::vector<T>&& data, size_t k, Compare comp) {
        if (k >= data.size()) {
            std::sort(data.begin(), data.end(), comp);
            return data;
        }
        
        // Use partial_sort to get the top K elements
        std::vector<T> result = data;
        std::partial_sort(result.begin(), result.begin() + k, result.end(), comp);
        result.resize(k);
        return result;
    }
    
    // Use a priority queue for top-K selection (better for very large datasets with small K)
    template<typename T, typename Compare>
    static std::vector<T> top_k_with_heap(const std::vector<T>& data, size_t k, Compare comp) {
        if (k >= data.size()) {
            std::vector<T> result = data;
            std::sort(result.begin(), result.end(), comp);
            return result;
        }
        
        // Use a min-heap to keep track of top K elements
        std::priority_queue<T, std::vector<T>, std::function<bool(const T&, const T&)>> min_heap(comp);
        
        for (const auto& item : data) {
            if (min_heap.size() < k) {
                min_heap.push(item);
            } else if (comp(min_heap.top(), item)) {  // If current item is better than worst in heap
                min_heap.pop();
                min_heap.push(item);
            }
        }
        
        // Extract elements from the heap
        std::vector<T> result;
        result.reserve(min_heap.size());
        while (!min_heap.empty()) {
            result.push_back(min_heap.top());
            min_heap.pop();
        }
        
        // Reverse to get them in descending order
        std::reverse(result.begin(), result.end());
        return result;
    }
};

// KNN search with different algorithm options
class KnnSearch {
public:
    enum class Algorithm {
        LINEAR,     // Linear scan - good for small datasets
        PARTIAL_SORT, // Partial sort - good for moderate K values
        HEAP        // Heap-based - good for very small K values
    };
    
    struct SearchResult {
        std::string vector_id;
        float similarity_score;
        size_t original_index;
    };
    
    // Perform KNN search using the specified algorithm
    static std::vector<SearchResult> knn_search(
        const Vector& query_vector,
        const std::vector<Vector>& candidate_vectors,
        size_t k,
        Algorithm algorithm = Algorithm::HEAP,
        float threshold = 0.0f);
    
    // Validate search result quality
    static bool validate_search_quality(
        const std::vector<SearchResult>& results,
        const Vector& query_vector,
        const std::vector<Vector>& all_vectors,
        float tolerance = 0.01f);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SEARCH_UTILS_H