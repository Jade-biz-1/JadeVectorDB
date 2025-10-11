#ifndef JADEVECTORDB_SIMILARITY_SEARCH_SERVICE_H
#define JADEVECTORDB_SIMILARITY_SEARCH_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>

#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "search_utils.h"

namespace jadevectordb {

// Structure to represent a search result
struct SearchResult {
    std::string vector_id;
    float similarity_score;
    Vector vector_data;  // Optional: include vector data in results
    
    SearchResult(const std::string& id, float score) : vector_id(id), similarity_score(score) {}
    SearchResult(const std::string& id, float score, const Vector& data) 
        : vector_id(id), similarity_score(score), vector_data(data) {}
};

// Structure to represent search parameters
struct SearchParams {
    int top_k = 10;  // Number of nearest neighbors to return
    float threshold = 0.0f;  // Minimum similarity threshold
    bool include_vector_data = false;  // Whether to include vector data in results
    bool include_metadata = false;  // Whether to include metadata in results
    std::vector<std::string> filter_tags;  // Tags to filter by
    std::string filter_owner;  // Owner to filter by
    std::string filter_category;  // Category to filter by
    float filter_min_score = 0.0f;  // Minimum score filter
    float filter_max_score = 1.0f;  // Maximum score filter
};

class SimilaritySearchService {
private:
    std::unique_ptr<VectorStorageService> vector_storage_;
    std::shared_ptr<logging::Logger> logger_;
    
    // Metrics for performance monitoring
    std::shared_ptr<Counter> search_requests_counter_;
    std::shared_ptr<Histogram> search_latency_histogram_;
    std::shared_ptr<Counter> search_results_counter_;
    std::shared_ptr<Gauge> active_searches_gauge_;

public:
    explicit SimilaritySearchService(std::unique_ptr<VectorStorageService> vector_storage = nullptr);
    ~SimilaritySearchService() = default;

    // Initialize the service
    Result<void> initialize();

    // Perform similarity search using cosine similarity
    Result<std::vector<SearchResult>> similarity_search(
        const std::string& database_id,
        const Vector& query_vector,
        const SearchParams& params = SearchParams()) const;

    // Perform similarity search using Euclidean distance
    Result<std::vector<SearchResult>> euclidean_search(
        const std::string& database_id,
        const Vector& query_vector,
        const SearchParams& params = SearchParams()) const;

    // Perform similarity search using dot product
    Result<std::vector<SearchResult>> dot_product_search(
        const std::string& database_id,
        const Vector& query_vector,
        const SearchParams& params = SearchParams()) const;

    // Get all available search algorithms
    std::vector<std::string> get_available_algorithms() const;

    // Validate search parameters
    Result<void> validate_search_params(const SearchParams& params) const;

private:
    // Core similarity calculation methods
    float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) const;
    float euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2) const;
    float dot_product(const std::vector<float>& v1, const std::vector<float>& v2) const;
    
    // Filter vectors based on metadata
    std::vector<Vector> apply_metadata_filters(const std::vector<Vector>& vectors, 
                                             const SearchParams& params) const;
    
    // Sort and limit results
    std::vector<SearchResult> sort_and_limit_results(std::vector<SearchResult>&& results, 
                                                   const SearchParams& params, 
                                                   bool ascending = false) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SIMILARITY_SEARCH_SERVICE_H