#ifndef JADEVECTORDB_SIMILARITY_SEARCH_SERVICE_H
#define JADEVECTORDB_SIMILARITY_SEARCH_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <algorithm>
#include <type_traits>

// Core C++20 features
#include <concepts>
#include <iterator>
#include <ranges>

// Range and iterator concepts
namespace ranges = std::ranges;
namespace concepts = std::ranges;

// Import core ranges functionality
using std::ranges::begin;
using std::ranges::end;
using std::ranges::size;
using std::ranges::empty;
using std::ranges::iterator_t;
using std::ranges::range_value_t;

// Iterator concepts
using std::input_iterator;
using std::forward_iterator;
using std::bidirectional_iterator;
using std::random_access_iterator;
using std::contiguous_iterator;

// Range concepts
using std::ranges::range;
using std::ranges::input_range;
using std::ranges::forward_range;
using std::ranges::bidirectional_range;
using std::ranges::random_access_range;
using std::ranges::contiguous_range;
using std::ranges::sized_range;
using std::ranges::common_range;
using std::ranges::viewable_range;

// Range adaptors and views
using namespace std::ranges::views;

// Iterator utilities 
using std::indirectly_readable;
using std::indirectly_writable;
using std::indirect_result_t;
using std::indirect_unary_predicate;
using std::indirect_binary_predicate;
using std::projected;

// Common concepts
using std::equality_comparable;
using std::totally_ordered;

// Functional
using std::identity;

#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "search_utils.h"
#include "metadata_filter.h"
#include "lib/metrics.h"
#include "lib/vector_operations.h"
#include "query_optimizer.h"

namespace jadevectordb {

// Structure to represent a search result
struct SearchResult {
    std::string vector_id;
    float similarity_score;
    Vector vector_data;  // Optional: include vector data in results

    SearchResult() : similarity_score(0.0f) {}
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

    // Vector operations for similarity computations (with CPU/GPU switching)
    std::shared_ptr<vector_ops::IVectorOperations> vector_ops_;

    // Metrics for performance monitoring
    std::shared_ptr<Counter> search_requests_counter_;
    std::shared_ptr<Histogram> search_latency_histogram_;
    std::shared_ptr<Counter> search_results_counter_;
    std::shared_ptr<Gauge> active_searches_gauge_;

    // Filtered search specific metrics
    std::shared_ptr<Counter> filtered_search_requests_counter_;
    std::shared_ptr<Histogram> filtered_search_latency_histogram_;
    std::shared_ptr<Counter> filtered_search_results_counter_;
    std::shared_ptr<Gauge> active_filtered_searches_gauge_;
    std::shared_ptr<Histogram> filter_application_time_histogram_;
    std::shared_ptr<Counter> filter_cache_hits_counter_;
    std::shared_ptr<Counter> filter_cache_misses_counter_;

    // Metadata filter service
    std::unique_ptr<MetadataFilter> metadata_filter_;
    
    // Query optimizer for cost-based optimization
    std::unique_ptr<QueryOptimizer> query_optimizer_;

public:
    // Test accessor methods
    VectorStorageService* get_vector_storage_for_testing() { return vector_storage_.get(); }
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
    
    // Get query optimizer instance
    QueryOptimizer* get_query_optimizer() { return query_optimizer_.get(); }

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

    // Advanced filtering with ComplexFilter
    Result<std::vector<Vector>> apply_advanced_filters(
        const ComplexFilter& filter,
        const std::vector<Vector>& vectors) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SIMILARITY_SEARCH_SERVICE_H