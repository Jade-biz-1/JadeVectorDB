#include "similarity_search.h"
#include "search_utils.h"
#include "lib/metrics.h"
#include <algorithm>
#include <execution>
#include <numeric>
#include <cmath>

namespace jadevectordb {

SimilaritySearchService::SimilaritySearchService(std::unique_ptr<VectorStorageService> vector_storage)
    : vector_storage_(std::move(vector_storage)),
      logger_(logging::LoggerManager::get_logger("SimilaritySearchService")),
      metadata_filter_(std::make_unique<MetadataFilter>()) {
    
    // Initialize vector operations with CPU as default (will try GPU if available)
    vector_ops_ = vector_ops::VectorOperationsFactory::create_operations(hardware::DeviceType::CPU);
}

Result<void> SimilaritySearchService::initialize() {
    if (!vector_storage_) {
        RETURN_ERROR(ErrorCode::INITIALIZE_ERROR, "Vector storage service not provided");
    }
    
    // Initialize metrics
    auto metrics_registry = MetricsManager::get_registry();
    if (metrics_registry) {
        search_requests_counter_ = metrics_registry->register_counter(
            "similarity_search_requests_total", 
            "Total number of similarity search requests");
        
        search_latency_histogram_ = metrics_registry->register_histogram(
            "similarity_search_duration_seconds",
            "Histogram of similarity search request durations",
            {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0});
        
        search_results_counter_ = metrics_registry->register_counter(
            "similarity_search_results_total",
            "Total number of results returned by similarity searches");
        
        active_searches_gauge_ = metrics_registry->register_gauge(
            "similarity_search_active_requests",
            "Number of currently active similarity search requests");
        
        filtered_search_requests_counter_ = metrics_registry->register_counter(
            "filtered_similarity_search_requests_total",
            "Total number of filtered similarity search requests");
        
        filtered_search_latency_histogram_ = metrics_registry->register_histogram(
            "filtered_similarity_search_duration_seconds",
            "Histogram of filtered similarity search request durations",
            {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0});
        
        filtered_search_results_counter_ = metrics_registry->register_counter(
            "filtered_similarity_search_results_total",
            "Total number of results returned by filtered similarity searches");
        
        active_filtered_searches_gauge_ = metrics_registry->register_gauge(
            "filtered_similarity_search_active_requests", 
            "Number of currently active filtered similarity search requests");
        
        filter_application_time_histogram_ = metrics_registry->register_histogram(
            "filter_application_duration_seconds",
            "Histogram of time spent applying filters",
            {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05});
        
        filter_cache_hits_counter_ = metrics_registry->register_counter(
            "filter_cache_hits_total",
            "Total number of filter cache hits");
        
        filter_cache_misses_counter_ = metrics_registry->register_counter(
            "filter_cache_misses_total",
            "Total number of filter cache misses");
    }
    
    LOG_INFO(logger_, "SimilaritySearchService initialized");
    return {};
}

Result<std::vector<SearchResult>> SimilaritySearchService::similarity_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    // Validate parameters
    auto validation_result = validate_search_params(params);
    if (!validation_result.has_value()) {
        return tl::make_unexpected(validation_result.error());
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    if (active_searches_gauge_) {
        active_searches_gauge_->increment();
    }
    if (search_requests_counter_) {
        search_requests_counter_->increment();
    }
    
    try {
        // Retrieve all vectors from the database
        auto all_vectors_result = vector_storage_->get_all_vector_ids(database_id);
        if (!all_vectors_result.has_value()) {
            if (active_searches_gauge_) {
                active_searches_gauge_->decrement();
            }
            return tl::make_unexpected(all_vectors_result.error());
        }
        
        const auto& all_vector_ids = all_vectors_result.value();
        
        // Retrieve vectors in batches to manage memory
        std::vector<SearchResult> results;
        const size_t batch_size = 1000; // Configurable batch size
        
        for (size_t i = 0; i < all_vector_ids.size(); i += batch_size) {
            size_t end_idx = std::min(i + batch_size, all_vector_ids.size());
            std::vector<std::string> batch_ids(all_vector_ids.begin() + i, all_vector_ids.begin() + end_idx);
            
            auto batch_result = vector_storage_->retrieve_vectors(database_id, batch_ids);
            if (!batch_result.has_value()) {
                if (active_searches_gauge_) {
                    active_searches_gauge_->decrement();
                }
                return tl::make_unexpected(batch_result.error());
            }
            
            const auto& batch_vectors = batch_result.value();
            
            // Apply metadata filters if specified in params
            std::vector<Vector> filtered_batch = batch_vectors;
            if (!params.filter_tags.empty() || !params.filter_owner.empty() || 
                !params.filter_category.empty() || 
                params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
                
                auto filter_start = std::chrono::high_resolution_clock::now();
                filtered_batch = apply_metadata_filters(batch_vectors, params);
                auto filter_end = std::chrono::high_resolution_clock::now();
                
                if (filter_application_time_histogram_) {
                    auto duration = std::chrono::duration<double>(filter_end - filter_start).count();
                    filter_application_time_histogram_->observe(duration);
                }
            }
            
            // Calculate cosine similarity for each vector in the batch
            for (const auto& vec : filtered_batch) {
                if (vec.values.size() != query_vector.values.size()) {
                    LOG_WARN(logger_, "Vector dimension mismatch, skipping vector: " + vec.id);
                    continue;
                }
                
                // Use the vector operations abstraction (could be CPU or GPU based)
                float similarity = vector_ops_->cosine_similarity(query_vector.values, vec.values);
                
                // Apply threshold filter
                if (similarity >= params.threshold) {
                    SearchResult result(vec.id, similarity);
                    
                    // Include vector data if requested
                    if (params.include_vector_data) {
                        result.vector_data = vec;
                    }
                    
                    results.emplace_back(std::move(result));
                }
            }
        }
        
        // Sort and limit results
        auto final_results = sort_and_limit_results(std::move(results), params);
        
        if (search_results_counter_) {
            search_results_counter_->add(static_cast<double>(final_results.size()));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        if (active_searches_gauge_) {
            active_searches_gauge_->decrement();
        }
        if (search_latency_histogram_) {
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
            search_latency_histogram_->observe(duration);
        }
        
        return std::move(final_results);
        
    } catch (const std::exception& e) {
        if (active_searches_gauge_) {
            active_searches_gauge_->decrement();
        }
        LOG_ERROR(logger_, "Exception during similarity search: " + std::string(e.what()));
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::SIMILARITY_SEARCH_FAILED,
                                                            "Exception during similarity search"));
    }
}

Result<std::vector<SearchResult>> SimilaritySearchService::euclidean_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    // Validate parameters
    auto validation_result = validate_search_params(params);
    if (!validation_result.has_value()) {
        return tl::make_unexpected(validation_result.error());
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    if (active_searches_gauge_) {
        active_searches_gauge_->increment();
    }
    if (search_requests_counter_) {
        search_requests_counter_->increment();
    }
    
    try {
        // Retrieve all vectors from the database
        auto all_vectors_result = vector_storage_->get_all_vector_ids(database_id);
        if (!all_vectors_result.has_value()) {
            if (active_searches_gauge_) {
                active_searches_gauge_->decrement();
            }
            return tl::make_unexpected(all_vectors_result.error());
        }
        
        const auto& all_vector_ids = all_vectors_result.value();
        
        // Retrieve vectors in batches to manage memory
        std::vector<SearchResult> results;
        const size_t batch_size = 1000; // Configurable batch size
        
        for (size_t i = 0; i < all_vector_ids.size(); i += batch_size) {
            size_t end_idx = std::min(i + batch_size, all_vector_ids.size());
            std::vector<std::string> batch_ids(all_vector_ids.begin() + i, all_vector_ids.begin() + end_idx);
            
            auto batch_result = vector_storage_->retrieve_vectors(database_id, batch_ids);
            if (!batch_result.has_value()) {
                if (active_searches_gauge_) {
                    active_searches_gauge_->decrement();
                }
                return tl::make_unexpected(batch_result.error());
            }
            
            const auto& batch_vectors = batch_result.value();
            
            // Apply metadata filters if specified in params
            std::vector<Vector> filtered_batch = batch_vectors;
            if (!params.filter_tags.empty() || !params.filter_owner.empty() || 
                !params.filter_category.empty() || 
                params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
                
                auto filter_start = std::chrono::high_resolution_clock::now();
                filtered_batch = apply_metadata_filters(batch_vectors, params);
                auto filter_end = std::chrono::high_resolution_clock::now();
                
                if (filter_application_time_histogram_) {
                    auto duration = std::chrono::duration<double>(filter_end - filter_start).count();
                    filter_application_time_histogram_->observe(duration);
                }
            }
            
            // Calculate Euclidean distance for each vector in the batch
            for (const auto& vec : filtered_batch) {
                if (vec.values.size() != query_vector.values.size()) {
                    LOG_WARN(logger_, "Vector dimension mismatch, skipping vector: " + vec.id);
                    continue;
                }
                
                // Use the vector operations abstraction (could be CPU or GPU based)
                float distance = vector_ops_->euclidean_distance(query_vector.values, vec.values);
                
                // Convert distance to similarity (1 / (1 + distance)) to maintain consistency
                float similarity = 1.0f / (1.0f + distance);
                
                // Apply threshold filter
                if (similarity >= params.threshold) {
                    SearchResult result(vec.id, similarity);
                    
                    // Include vector data if requested
                    if (params.include_vector_data) {
                        result.vector_data = vec;
                    }
                    
                    results.emplace_back(std::move(result));
                }
            }
        }
        
        // Sort and limit results (for Euclidean, we want ascending order - closest first)
        auto final_results = sort_and_limit_results(std::move(results), params, true);
        
        if (search_results_counter_) {
            search_results_counter_->add(static_cast<double>(final_results.size()));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        if (active_searches_gauge_) {
            active_searches_gauge_->decrement();
        }
        if (search_latency_histogram_) {
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
            search_latency_histogram_->observe(duration);
        }
        
        return std::move(final_results);
        
    } catch (const std::exception& e) {
        if (active_searches_gauge_) {
            active_searches_gauge_->decrement();
        }
        LOG_ERROR(logger_, "Exception during Euclidean search: " + std::string(e.what()));
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::SIMILARITY_SEARCH_FAILED,
                                                            "Exception during Euclidean search"));
    }
}

Result<std::vector<SearchResult>> SimilaritySearchService::dot_product_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    // Validate parameters
    auto validation_result = validate_search_params(params);
    if (!validation_result.has_value()) {
        return tl::make_unexpected(validation_result.error());
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    if (active_searches_gauge_) {
        active_searches_gauge_->increment();
    }
    if (search_requests_counter_) {
        search_requests_counter_->increment();
    }
    
    try {
        // Retrieve all vectors from the database
        auto all_vectors_result = vector_storage_->get_all_vector_ids(database_id);
        if (!all_vectors_result.has_value()) {
            if (active_searches_gauge_) {
                active_searches_gauge_->decrement();
            }
            return tl::make_unexpected(all_vectors_result.error());
        }
        
        const auto& all_vector_ids = all_vectors_result.value();
        
        // Retrieve vectors in batches to manage memory
        std::vector<SearchResult> results;
        const size_t batch_size = 1000; // Configurable batch size
        
        for (size_t i = 0; i < all_vector_ids.size(); i += batch_size) {
            size_t end_idx = std::min(i + batch_size, all_vector_ids.size());
            std::vector<std::string> batch_ids(all_vector_ids.begin() + i, all_vector_ids.begin() + end_idx);
            
            auto batch_result = vector_storage_->retrieve_vectors(database_id, batch_ids);
            if (!batch_result.has_value()) {
                if (active_searches_gauge_) {
                    active_searches_gauge_->decrement();
                }
                return tl::make_unexpected(batch_result.error());
            }
            
            const auto& batch_vectors = batch_result.value();
            
            // Apply metadata filters if specified in params
            std::vector<Vector> filtered_batch = batch_vectors;
            if (!params.filter_tags.empty() || !params.filter_owner.empty() || 
                !params.filter_category.empty() || 
                params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
                
                auto filter_start = std::chrono::high_resolution_clock::now();
                filtered_batch = apply_metadata_filters(batch_vectors, params);
                auto filter_end = std::chrono::high_resolution_clock::now();
                
                if (filter_application_time_histogram_) {
                    auto duration = std::chrono::duration<double>(filter_end - filter_start).count();
                    filter_application_time_histogram_->observe(duration);
                }
            }
            
            // Calculate dot product for each vector in the batch
            for (const auto& vec : filtered_batch) {
                if (vec.values.size() != query_vector.values.size()) {
                    LOG_WARN(logger_, "Vector dimension mismatch, skipping vector: " + vec.id);
                    continue;
                }
                
                // Use the vector operations abstraction (could be CPU or GPU based)
                float dot_product = vector_ops_->dot_product(query_vector.values, vec.values);
                
                // Apply threshold filter
                if (dot_product >= params.threshold) {
                    SearchResult result(vec.id, dot_product);
                    
                    // Include vector data if requested
                    if (params.include_vector_data) {
                        result.vector_data = vec;
                    }
                    
                    results.emplace_back(std::move(result));
                }
            }
        }
        
        // Sort and limit results
        auto final_results = sort_and_limit_results(std::move(results), params);
        
        if (search_results_counter_) {
            search_results_counter_->add(static_cast<double>(final_results.size()));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        if (active_searches_gauge_) {
            active_searches_gauge_->decrement();
        }
        if (search_latency_histogram_) {
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
            search_latency_histogram_->observe(duration);
        }
        
        return std::move(final_results);
        
    } catch (const std::exception& e) {
        if (active_searches_gauge_) {
            active_searches_gauge_->decrement();
        }
        LOG_ERROR(logger_, "Exception during dot product search: " + std::string(e.what()));
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::SIMILARITY_SEARCH_FAILED,
                                                            "Exception during dot product search"));
    }
}

std::vector<std::string> SimilaritySearchService::get_available_algorithms() const {
    return {"cosine", "euclidean", "dot_product"};
}

Result<void> SimilaritySearchService::validate_search_params(const SearchParams& params) const {
    if (params.top_k <= 0) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "top_k must be positive"));
    }
    
    if (params.threshold < 0.0f || params.threshold > 1.0f) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "threshold must be between 0 and 1"));
    }
    
    if (params.filter_min_score < 0.0f || params.filter_min_score > 1.0f) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "filter_min_score must be between 0 and 1"));
    }
    
    if (params.filter_max_score < 0.0f || params.filter_max_score > 1.0f) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "filter_max_score must be between 0 and 1"));
    }
    
    if (params.filter_min_score > params.filter_max_score) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT,
                                      "filter_min_score cannot be greater than filter_max_score"));
    }
    
    return {};
}

std::vector<Vector> SimilaritySearchService::apply_metadata_filters(
    const std::vector<Vector>& vectors, 
    const SearchParams& params) const {
    
    std::vector<Vector> filtered;
    filtered.reserve(vectors.size()); // Reserve space to potentially reduce allocations
    
    for (const auto& vec : vectors) {
        bool include = true;
        
        // Tag filter
        if (!params.filter_tags.empty()) {
            bool has_tag = false;
            for (const auto& tag : params.filter_tags) {
                if (std::find(vec.metadata.tags.begin(), vec.metadata.tags.end(), tag) != vec.metadata.tags.end()) {
                    has_tag = true;
                    break;
                }
            }
            if (!has_tag) {
                include = false;
            }
        }

        // Owner filter
        if (include && !params.filter_owner.empty()) {
            if (vec.metadata.owner.empty() || vec.metadata.owner != params.filter_owner) {
                include = false;
            }
        }

        // Category filter
        if (include && !params.filter_category.empty()) {
            if (vec.metadata.category.empty() || vec.metadata.category != params.filter_category) {
                include = false;
            }
        }

        // Score filter (using the score field in metadata)
        if (include) {
            if (vec.metadata.score < params.filter_min_score || vec.metadata.score > params.filter_max_score) {
                include = false;
            }
        }
        
        if (include) {
            filtered.push_back(vec);
        }
    }
    
    return filtered;
}

std::vector<SearchResult> SimilaritySearchService::sort_and_limit_results(
    std::vector<SearchResult>&& results, 
    const SearchParams& params, 
    bool ascending) const {
    
    // Sort by similarity score
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
    
    // Apply top-k limit
    if (static_cast<int>(results.size()) > params.top_k) {
        results.resize(params.top_k);
    }
    
    return results;
}

// Deprecated methods that were in the original header but implemented here
float SimilaritySearchService::cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) const {
    return vector_ops_->cosine_similarity(v1, v2);
}

float SimilaritySearchService::euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
    return vector_ops_->euclidean_distance(v1, v2);
}

float SimilaritySearchService::dot_product(const std::vector<float>& v1, const std::vector<float>& v2) const {
    return vector_ops_->dot_product(v1, v2);
}

} // namespace jadevectordb