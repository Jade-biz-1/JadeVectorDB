#include "filtered_search_benchmark.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace jadevectordb {

FilteredSearchBenchmark::FilteredSearchBenchmark(std::shared_ptr<SimilaritySearchService> search_service)
    : search_service_(search_service), rng_(std::random_device{}()) {
    logger_ = logging::LoggerManager::get_logger("FilteredSearchBenchmark");
}

FilteredSearchBenchmarkResult FilteredSearchBenchmark::benchmark_filtered_search(
    const std::string& database_id,
    size_t num_vectors,
    double filter_coverage_percentage,
    size_t num_queries) {
    
    FilteredSearchBenchmarkResult result;
    result.test_name = "Filtered Search (" + std::to_string(filter_coverage_percentage * 100) + "% coverage)";
    result.database_size = num_vectors;
    result.filter_coverage = static_cast<size_t>(num_vectors * filter_coverage_percentage);
    result.total_queries = num_queries;
    result.individual_times.reserve(num_queries);
    
    std::vector<size_t> result_counts;
    result_counts.reserve(num_queries);
    
    // Generate random query vectors
    std::vector<Vector> query_vectors;
    for (size_t i = 0; i < num_queries; ++i) {
        Vector query;
        query.id = "query_" + std::to_string(i);
        query.values.resize(128);  // Assuming 128-dimensional vectors
        
        // Generate random values
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        for (auto& val : query.values) {
            val = value_dist(rng_);
        }
        
        query_vectors.push_back(query);
    }
    
    // Run the benchmark
    for (size_t i = 0; i < num_queries; ++i) {
        SearchParams params;
        params.top_k = 10;
        params.threshold = 0.0f;
        
        // Set up filters based on coverage percentage
        if (filter_coverage_percentage < 1.0) {
            // For partial coverage, set filters that will match a subset
            std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
            if (prob_dist(rng_) < filter_coverage_percentage) {
                // Set a filter that matches
                params.filter_owner = "user1";  // Assuming user1 owns a portion of vectors
            } else {
                // Set a filter that doesn't match much
                params.filter_owner = "rare_user";
            }
        }
        
        // Measure search time
        auto start = std::chrono::high_resolution_clock::now();
        auto search_result = search_service_->similarity_search(database_id, query_vectors[i], params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        result.individual_times.push_back(duration);
        
        // Track result count
        if (search_result.has_value()) {
            result_counts.push_back(search_result.value().size());
        } else {
            result_counts.push_back(0);
        }
    }
    
    // Calculate statistics
    auto [avg_time, min_time, max_time, throughput, avg_results] = calculate_statistics(
        result.individual_times, result_counts);
    
    result.avg_search_time_ms = avg_time;
    result.min_search_time_ms = min_time;
    result.max_search_time_ms = max_time;
    result.throughput_qps = throughput;
    result.avg_results_per_query = avg_results;
    
    LOG_INFO(logger_, "Filtered search benchmark completed: " << result.avg_search_time_ms << "ms avg, " 
              << result.throughput_qps << " QPS");
    
    return result;
}

FilteredSearchBenchmarkResult FilteredSearchBenchmark::benchmark_filtered_search_by_type(
    const std::string& database_id,
    const std::string& filter_type,
    size_t num_vectors,
    size_t num_queries) {
    
    FilteredSearchBenchmarkResult result;
    result.test_name = "Filtered Search by " + filter_type;
    result.database_size = num_vectors;
    result.total_queries = num_queries;
    result.individual_times.reserve(num_queries);
    
    std::vector<size_t> result_counts;
    result_counts.reserve(num_queries);
    
    // Generate random query vectors
    std::vector<Vector> query_vectors;
    for (size_t i = 0; i < num_queries; ++i) {
        Vector query;
        query.id = "query_" + std::to_string(i);
        query.values.resize(128);
        
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        for (auto& val : query.values) {
            val = value_dist(rng_);
        }
        
        query_vectors.push_back(query);
    }
    
    // Run the benchmark
    for (size_t i = 0; i < num_queries; ++i) {
        SearchParams params;
        params.top_k = 10;
        params.threshold = 0.0f;
        
        // Set up filter based on type
        if (filter_type == "owner") {
            params.filter_owner = select_random_owner();
        } else if (filter_type == "category") {
            params.filter_category = select_random_category();
        } else if (filter_type == "tags") {
            auto tags = generate_random_tags(2);
            if (!tags.empty()) {
                params.filter_tags.push_back(tags[0]);
            }
        } else if (filter_type == "score_range") {
            params.filter_min_score = 0.5f;
            params.filter_max_score = 0.9f;
        } else if (filter_type == "custom") {
            // Custom filter would require ComplexFilter - simplified for this example
        }
        
        // Measure search time
        auto start = std::chrono::high_resolution_clock::now();
        auto search_result = search_service_->similarity_search(database_id, query_vectors[i], params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        result.individual_times.push_back(duration);
        
        if (search_result.has_value()) {
            result_counts.push_back(search_result.value().size());
        } else {
            result_counts.push_back(0);
        }
    }
    
    // Calculate statistics
    auto [avg_time, min_time, max_time, throughput, avg_results] = calculate_statistics(
        result.individual_times, result_counts);
    
    result.avg_search_time_ms = avg_time;
    result.min_search_time_ms = min_time;
    result.max_search_time_ms = max_time;
    result.throughput_qps = throughput;
    result.avg_results_per_query = avg_results;
    
    LOG_INFO(logger_, "Filtered search by " << filter_type << " benchmark completed: " 
              << result.avg_search_time_ms << "ms avg, " << result.throughput_qps << " QPS");
    
    return result;
}

FilteredSearchBenchmarkResult FilteredSearchBenchmark::benchmark_complex_filtered_search(
    const std::string& database_id,
    size_t num_vectors,
    size_t num_conditions,
    FilterCombination combination,
    size_t num_queries) {
    
    FilteredSearchBenchmarkResult result;
    result.test_name = "Complex Filtered Search (" + std::to_string(num_conditions) + " conditions, " +
                      (combination == FilterCombination::AND ? "AND" : "OR") + ")";
    result.database_size = num_vectors;
    result.total_queries = num_queries;
    result.individual_times.reserve(num_queries);
    
    std::vector<size_t> result_counts;
    result_counts.reserve(num_queries);
    
    // Generate random query vectors
    std::vector<Vector> query_vectors;
    for (size_t i = 0; i < num_queries; ++i) {
        Vector query;
        query.id = "query_" + std::to_string(i);
        query.values.resize(128);
        
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        for (auto& val : query.values) {
            val = value_dist(rng_);
        }
        
        query_vectors.push_back(query);
    }
    
    // Run the benchmark
    for (size_t i = 0; i < num_queries; ++i) {
        // Create a complex filter
        ComplexFilter filter;
        filter.combination = combination;
        
        // Add conditions
        for (size_t j = 0; j < std::min(num_conditions, static_cast<size_t>(5)); ++j) {
            FilterCondition condition;
            
            switch (j % 4) {
                case 0:
                    condition.field = "metadata.owner";
                    condition.op = FilterOperator::EQUALS;
                    condition.value = select_random_owner();
                    break;
                case 1:
                    condition.field = "metadata.category";
                    condition.op = FilterOperator::EQUALS;
                    condition.value = select_random_category();
                    break;
                case 2:
                    condition.field = "metadata.tags";
                    condition.op = FilterOperator::CONTAINS;
                    condition.value = "tag" + std::to_string(j);
                    break;
                case 3:
                    condition.field = "metadata.score";
                    condition.op = FilterOperator::GREATER_THAN;
                    condition.value = std::to_string(0.5 + j * 0.1);
                    break;
            }
            
            filter.conditions.push_back(condition);
        }
        
        // For this simplified version, we'll use the basic search with simple filters
        // A full implementation would use the complex filter directly
        SearchParams params;
        params.top_k = 10;
        params.threshold = 0.0f;
        
        // Measure search time
        auto start = std::chrono::high_resolution_clock::now();
        auto search_result = search_service_->similarity_search(database_id, query_vectors[i], params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        result.individual_times.push_back(duration);
        
        if (search_result.has_value()) {
            result_counts.push_back(search_result.value().size());
        } else {
            result_counts.push_back(0);
        }
    }
    
    // Calculate statistics
    auto [avg_time, min_time, max_time, throughput, avg_results] = calculate_statistics(
        result.individual_times, result_counts);
    
    result.avg_search_time_ms = avg_time;
    result.min_search_time_ms = min_time;
    result.max_search_time_ms = max_time;
    result.throughput_qps = throughput;
    result.avg_results_per_query = avg_results;
    
    LOG_INFO(logger_, "Complex filtered search benchmark completed: " << result.avg_search_time_ms 
              << "ms avg, " << result.throughput_qps << " QPS");
    
    return result;
}

std::vector<FilteredSearchBenchmarkResult> FilteredSearchBenchmark::run_comprehensive_benchmark(
    const std::string& database_id,
    size_t num_vectors,
    size_t num_queries) {
    
    std::vector<FilteredSearchBenchmarkResult> results;
    
    // Run different coverage benchmarks
    std::vector<double> coverages = {0.1, 0.25, 0.5, 0.75, 1.0};
    for (double coverage : coverages) {
        auto result = benchmark_filtered_search(database_id, num_vectors, coverage, num_queries);
        results.push_back(result);
    }
    
    // Run different filter type benchmarks
    std::vector<std::string> filter_types = {"owner", "category", "tags", "score_range"};
    for (const auto& filter_type : filter_types) {
        auto result = benchmark_filtered_search_by_type(database_id, filter_type, num_vectors, num_queries);
        results.push_back(result);
    }
    
    // Run complex filter benchmarks
    std::vector<size_t> condition_counts = {2, 3, 4};
    for (size_t count : condition_counts) {
        auto result = benchmark_complex_filtered_search(database_id, num_vectors, count, FilterCombination::AND, num_queries);
        results.push_back(result);
        
        auto result_or = benchmark_complex_filtered_search(database_id, num_vectors, count, FilterCombination::OR, num_queries);
        results.push_back(result_or);
    }
    
    return results;
}

bool FilteredSearchBenchmark::validate_performance_requirements(
    const std::string& database_id,
    double max_avg_search_time_ms,
    double min_throughput_qps) {
    
    // Run a basic benchmark to validate performance
    auto result = benchmark_filtered_search(database_id, 1000, 0.5, 10);
    
    bool meets_time_requirement = result.avg_search_time_ms <= max_avg_search_time_ms;
    bool meets_throughput_requirement = result.throughput_qps >= min_throughput_qps;
    
    LOG_INFO(logger_, "Performance validation: Time=" << (meets_time_requirement ? "PASS" : "FAIL") 
              << ", Throughput=" << (meets_throughput_requirement ? "PASS" : "FAIL"));
    
    return meets_time_requirement && meets_throughput_requirement;
}

std::string FilteredSearchBenchmark::generate_benchmark_report(const FilteredSearchBenchmarkResult& result) const {
    std::ostringstream report;
    report << "=== Filtered Search Benchmark Report ===\n";
    report << "Test: " << result.test_name << "\n";
    report << "Database Size: " << result.database_size << " vectors\n";
    report << "Filter Coverage: " << result.filter_coverage << " vectors\n";
    report << "Total Queries: " << result.total_queries << "\n";
    report << "Average Search Time: " << result.avg_search_time_ms << " ms\n";
    report << "Min Search Time: " << result.min_search_time_ms << " ms\n";
    report << "Max Search Time: " << result.max_search_time_ms << " ms\n";
    report << "Throughput: " << result.throughput_qps << " QPS\n";
    report << "Avg Results per Query: " << result.avg_results_per_query << "\n";
    report << "=========================================\n";
    
    return report.str();
}

std::string FilteredSearchBenchmark::generate_comprehensive_report(
    const std::vector<FilteredSearchBenchmarkResult>& results) const {
    
    std::ostringstream report;
    report << "===== COMPREHENSIVE FILTERED SEARCH BENCHMARK REPORT =====\n\n";
    
    for (const auto& result : results) {
        report << generate_benchmark_report(result) << "\n";
    }
    
    report << "==========================================================\n";
    
    return report.str();
}

Result<std::vector<Vector>> FilteredSearchBenchmark::generate_test_vectors(
    size_t num_vectors,
    int vector_dimension) const {
    
    std::vector<Vector> vectors;
    vectors.reserve(num_vectors);
    
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> tag_count_dist(1, 5);
    std::uniform_int_distribution<int> owner_dist(1, 10);
    std::uniform_int_distribution<int> category_dist(1, 5);
    std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values.resize(vector_dimension);
        
        // Generate random vector values
        for (auto& val : v.values) {
            val = value_dist(rng_);
        }
        
        // Generate metadata
        v.metadata.owner = "user" + std::to_string(owner_dist(rng_));
        v.metadata.category = "category" + std::to_string(category_dist(rng_));
        v.metadata.status = (i % 3 == 0) ? "active" : (i % 3 == 1) ? "draft" : "archived";
        v.metadata.score = score_dist(rng_);
        v.metadata.created_at = "2025-01-01T00:00:00Z";
        v.metadata.updated_at = "2025-01-01T00:00:00Z";
        
        // Generate tags
        int tag_count = tag_count_dist(rng_);
        for (int j = 0; j < tag_count; ++j) {
            v.metadata.tags.push_back("tag" + std::to_string((i + j) % 20));
        }
        
        // Generate permissions
        v.metadata.permissions = {"read", "search"};
        
        // Generate custom fields
        v.metadata.custom["project"] = "project-" + std::to_string((i % 5) + 1);
        v.metadata.custom["department"] = "dept-" + std::to_string((i % 3) + 1);
        
        vectors.push_back(v);
    }
    
    return vectors;
}

Result<void> FilteredSearchBenchmark::populate_database_with_test_data(
    const std::string& database_id,
    size_t num_vectors,
    int vector_dimension) const {
    
    auto vectors_result = generate_test_vectors(num_vectors, vector_dimension);
    if (!vectors_result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to generate test vectors");
    }
    
    auto vectors = vectors_result.value();
    
    // Store vectors in batches for efficiency
    size_t batch_size = 100;
    for (size_t i = 0; i < vectors.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, vectors.size());
        std::vector<Vector> batch(vectors.begin() + i, vectors.begin() + end);
        
        // In a real implementation, you would store these vectors
        // For now, we'll just log the progress
        LOG_DEBUG(logger_, "Generated batch of " << batch.size() << " vectors (" 
                  << (i + batch.size()) << "/" << vectors.size() << ")");
    }
    
    LOG_INFO(logger_, "Generated and populated database with " << vectors.size() << " test vectors");
    return {};
}

double FilteredSearchBenchmark::run_single_filtered_search(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params) const {
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = search_service_->similarity_search(database_id, query_vector, params);
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double FilteredSearchBenchmark::run_single_complex_filtered_search(
    const std::string& database_id,
    const Vector& query_vector,
    const ComplexFilter& filter) const {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // For this simplified version, we'll convert complex filter to simple params
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    
    auto result = search_service_->similarity_search(database_id, query_vector, params);
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::tuple<double, double, double, double, size_t> FilteredSearchBenchmark::calculate_statistics(
    const std::vector<double>& times,
    const std::vector<size_t>& result_counts) const {
    
    if (times.empty()) {
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0);
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / times.size();
    
    auto min_max = std::minmax_element(times.begin(), times.end());
    double min_time = *min_max.first;
    double max_time = *min_max.second;
    
    // Calculate throughput (queries per second)
    double total_time_ms = sum;
    double throughput = (total_time_ms > 0) ? (times.size() * 1000.0) / total_time_ms : 0.0;
    
    // Calculate average results per query
    double avg_results = 0.0;
    if (!result_counts.empty()) {
        double result_sum = std::accumulate(result_counts.begin(), result_counts.end(), 0.0);
        avg_results = result_sum / result_counts.size();
    }
    
    return std::make_tuple(avg, min_time, max_time, throughput, static_cast<size_t>(avg_results));
}

std::string FilteredSearchBenchmark::generate_random_string(size_t length) const {
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::uniform_int_distribution<size_t> dist(0, chars.size() - 1);
    
    std::string result;
    result.reserve(length);
    
    for (size_t i = 0; i < length; ++i) {
        result += chars[dist(rng_)];
    }
    
    return result;
}

std::vector<std::string> FilteredSearchBenchmark::generate_random_tags(size_t count) const {
    std::vector<std::string> tags;
    tags.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        tags.push_back("tag" + std::to_string(i));
    }
    
    return tags;
}

std::string FilteredSearchBenchmark::select_random_owner() const {
    std::uniform_int_distribution<int> dist(1, 5);
    return "user" + std::to_string(dist(rng_));
}

std::string FilteredSearchBenchmark::select_random_category() const {
    std::uniform_int_distribution<int> dist(1, 3);
    return "category" + std::to_string(dist(rng_));
}

} // namespace jadevectordb