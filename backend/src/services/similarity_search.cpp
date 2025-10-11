#include "similarity_search.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "lib/metrics.h"
#include <algorithm>
#include <numeric>
#include <chrono>

namespace jadevectordb {

SimilaritySearchService::SimilaritySearchService(std::unique_ptr<VectorStorageService> vector_storage)
    : vector_storage_(std::move(vector_storage)) {
    
    if (!vector_storage_) {
        // If no vector storage service is provided, create a default one
        // with a default database layer
        vector_storage_ = std::make_unique<VectorStorageService>();
    }
    
    logger_ = logging::LoggerManager::get_logger("SimilaritySearchService");
}

Result<void> SimilaritySearchService::initialize() {
    auto result = vector_storage_->initialize();
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to initialize vector storage service: " << 
                 ErrorHandler::format_error(result.error()));
        return result;
    }
    
    // Initialize metrics
    auto metrics_registry = MetricsManager::get_registry();
    search_requests_counter_ = metrics_registry->register_counter(
        "search_requests_total", 
        "Total number of search requests"
    );
    search_latency_histogram_ = metrics_registry->register_histogram(
        "search_request_duration_seconds",
        "Time spent processing search requests",
        {0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0}
    );
    
    LOG_INFO(logger_, "SimilaritySearchService initialized successfully");
    return {};
}

Result<std::vector<SearchResult>> SimilaritySearchService::similarity_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    // Validate search parameters
    auto validation_result = validate_search_params(params);
    if (!validation_result.has_value()) {
        return std::vector<SearchResult>{}; // Return empty results on validation failure
    }
    
    // Record start time for metrics
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get all vectors from the database
    // Note: In a real implementation, this would use an index for efficiency
    // For now, we're doing a linear scan which is inefficient for large datasets
    auto all_vectors_result = vector_storage_->retrieve_vectors(database_id, {});
    if (!all_vectors_result.has_value()) {
        LOG_ERROR(logger_, "Failed to retrieve vectors from database: " << database_id);
        return std::vector<SearchResult>{};
    }
    
    auto all_vectors = all_vectors_result.value();
    
    // Apply metadata filters if specified
    if (!params.filter_tags.empty() || !params.filter_owner.empty() || 
        !params.filter_category.empty() || 
        params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
        all_vectors = apply_metadata_filters(all_vectors, params);
    }
    
    // Calculate cosine similarity for each vector
    std::vector<SearchResult> results;
    results.reserve(all_vectors.size());
    
    for (const auto& vector : all_vectors) {
        float similarity = cosine_similarity(query_vector.values, vector.values);
        
        // Apply threshold filter
        if (similarity >= params.threshold) {
            results.emplace_back(vector.id, similarity);
            
            // Optionally include vector data and metadata
            if (params.include_vector_data) {
                results.back().vector_data = vector;
            }
        }
    }
    
    // Sort and limit results
    results = sort_and_limit_results(std::move(results), params, false); // descending order for similarity
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    if (search_requests_counter_) {
        search_requests_counter_->increment();
    }
    if (search_latency_histogram_) {
        search_latency_histogram_->observe(duration);
    }
    
    LOG_DEBUG(logger_, "Similarity search completed: found " << results.size() << " results in " << duration << " seconds");
    
    return results;
}

Result<std::vector<SearchResult>> SimilaritySearchService::euclidean_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    // Validate search parameters
    auto validation_result = validate_search_params(params);
    if (!validation_result.has_value()) {
        return std::vector<SearchResult>{}; // Return empty results on validation failure
    }
    
    // Record start time for metrics
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get all vectors from the database
    auto all_vectors_result = vector_storage_->retrieve_vectors(database_id, {});
    if (!all_vectors_result.has_value()) {
        LOG_ERROR(logger_, "Failed to retrieve vectors from database: " << database_id);
        return std::vector<SearchResult>{};
    }
    
    auto all_vectors = all_vectors_result.value();
    
    // Apply metadata filters if specified
    if (!params.filter_tags.empty() || !params.filter_owner.empty() || 
        !params.filter_category.empty() || 
        params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
        all_vectors = apply_metadata_filters(all_vectors, params);
    }
    
    // Calculate Euclidean distance for each vector
    std::vector<SearchResult> results;
    results.reserve(all_vectors.size());
    
    for (const auto& vector : all_vectors) {
        float distance = euclidean_distance(query_vector.values, vector.values);
        
        // For Euclidean distance, smaller values mean more similar
        // But we'll store the inverse (1/(1+distance)) for similarity score consistency
        float similarity = 1.0f / (1.0f + distance);
        
        // Apply threshold filter
        if (similarity >= params.threshold) {
            results.emplace_back(vector.id, similarity);
            
            // Optionally include vector data and metadata
            if (params.include_vector_data) {
                results.back().vector_data = vector;
            }
        }
    }
    
    // Sort and limit results (ascending order of distance = descending order of similarity)
    results = sort_and_limit_results(std::move(results), params, false); // descending order for similarity
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    if (search_requests_counter_) {
        search_requests_counter_->increment();
    }
    if (search_latency_histogram_) {
        search_latency_histogram_->observe(duration);
    }
    
    LOG_DEBUG(logger_, "Euclidean search completed: found " << results.size() << " results in " << duration << " seconds");
    
    return results;
}

Result<std::vector<SearchResult>> SimilaritySearchService::dot_product_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    // Validate search parameters
    auto validation_result = validate_search_params(params);
    if (!validation_result.has_value()) {
        return std::vector<SearchResult>{}; // Return empty results on validation failure
    }
    
    // Record start time for metrics
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get all vectors from the database
    auto all_vectors_result = vector_storage_->retrieve_vectors(database_id, {});
    if (!all_vectors_result.has_value()) {
        LOG_ERROR(logger_, "Failed to retrieve vectors from database: " << database_id);
        return std::vector<SearchResult>{};
    }
    
    auto all_vectors = all_vectors_result.value();
    
    // Apply metadata filters if specified
    if (!params.filter_tags.empty() || !params.filter_owner.empty() || 
        !params.filter_category.empty() || 
        params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
        all_vectors = apply_metadata_filters(all_vectors, params);
    }
    
    // Calculate dot product for each vector
    std::vector<SearchResult> results;
    results.reserve(all_vectors.size());
    
    for (const auto& vector : all_vectors) {
        float dot_prod = dot_product(query_vector.values, vector.values);
        
        // Apply threshold filter
        if (dot_prod >= params.threshold) {
            results.emplace_back(vector.id, dot_prod);
            
            // Optionally include vector data and metadata
            if (params.include_vector_data) {
                results.back().vector_data = vector;
            }
        }
    }
    
    // Sort and limit results
    results = sort_and_limit_results(std::move(results), params, false); // descending order for dot product
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    if (search_requests_counter_) {
        search_requests_counter_->increment();
    }
    if (search_latency_histogram_) {
        search_latency_histogram_->observe(duration);
    }
    
    LOG_DEBUG(logger_, "Dot product search completed: found " << results.size() << " results in " << duration << " seconds");
    
    return results;
}

std::vector<std::string> SimilaritySearchService::get_available_algorithms() const {
    return {"cosine_similarity", "euclidean_distance", "dot_product"};
}

Result<void> SimilaritySearchService::validate_search_params(const SearchParams& params) const {
    if (params.top_k <= 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "top_k must be positive");
    }
    
    if (params.threshold < 0.0f || params.threshold > 1.0f) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "threshold must be between 0.0 and 1.0");
    }
    
    if (params.filter_min_score < 0.0f || params.filter_min_score > 1.0f) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "filter_min_score must be between 0.0 and 1.0");
    }
    
    if (params.filter_max_score < 0.0f || params.filter_max_score > 1.0f) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "filter_max_score must be between 0.0 and 1.0");
    }
    
    if (params.filter_min_score > params.filter_max_score) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "filter_min_score cannot be greater than filter_max_score");
    }
    
    return {};
}

float SimilaritySearchService::cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) const {
    if (v1.size() != v2.size()) {
        return 0.0f; // Vectors of different dimensions are orthogonal
    }
    
    double dot_product = 0.0;
    double magnitude_v1 = 0.0;
    double magnitude_v2 = 0.0;
    
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

float SimilaritySearchService::euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
    if (v1.size() != v2.size()) {
        // For vectors of different dimensions, pad the smaller one with zeros
        size_t max_size = std::max(v1.size(), v2.size());
        double sum = 0.0;
        
        for (size_t i = 0; i < max_size; ++i) {
            float val1 = (i < v1.size()) ? v1[i] : 0.0f;
            float val2 = (i < v2.size()) ? v2[i] : 0.0f;
            sum += (val1 - val2) * (val1 - val2);
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

float SimilaritySearchService::dot_product(const std::vector<float>& v1, const std::vector<float>& v2) const {
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

std::vector<Vector> SimilaritySearchService::apply_metadata_filters(
    const std::vector<Vector>& vectors, 
    const SearchParams& params) const {
    
    std::vector<Vector> filtered_vectors;
    filtered_vectors.reserve(vectors.size());
    
    for (const auto& vector : vectors) {
        bool include = true;
        
        // Filter by tags
        if (!params.filter_tags.empty()) {
            bool has_tag = false;
            for (const auto& required_tag : params.filter_tags) {
                for (const auto& vector_tag : vector.metadata.tags) {
                    if (required_tag == vector_tag) {
                        has_tag = true;
                        break;
                    }
                }
                if (has_tag) break;
            }
            if (!has_tag) {
                include = false;
            }
        }
        
        // Filter by owner
        if (include && !params.filter_owner.empty()) {
            if (vector.metadata.owner != params.filter_owner) {
                include = false;
            }
        }
        
        // Filter by category
        if (include && !params.filter_category.empty()) {
            if (vector.metadata.category != params.filter_category) {
                include = false;
            }
        }
        
        // Filter by score
        if (include && 
            (vector.metadata.score < params.filter_min_score || 
             vector.metadata.score > params.filter_max_score)) {
            include = false;
        }
        
        if (include) {
            filtered_vectors.push_back(vector);
        }
    }
    
    return filtered_vectors;
}

std::vector<SearchResult> SimilaritySearchService::sort_and_limit_results(
    std::vector<SearchResult>&& results, 
    const SearchParams& params, 
    bool ascending) const {
    
    // Sort results by similarity score
    if (ascending) {
        std::sort(results.begin(), results.end(), 
                 [](const SearchResult& a, const SearchResult& b) {
                     return a.similarity_score < b.similarity_score;
                 });
    } else {
        std::sort(results.begin(), results.end(), 
                 [](const SearchResult& a, const SearchResult& b) {
                     return a.similarity_score > b.similarity_score;
                 });
    }
    
    // Limit to top_k results
    if (results.size() > static_cast<size_t>(params.top_k)) {
        results.resize(params.top_k);
    }
    
    return results;
}

} // namespace jadevectordb