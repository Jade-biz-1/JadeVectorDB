#ifndef JADEVECTORDB_FILTERED_SEARCH_BENCHMARK_H
#define JADEVECTORDB_FILTERED_SEARCH_BENCHMARK_H

#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <random>

#include "services/similarity_search.h"
#include "services/metadata_filter.h"
#include "models/vector.h"
#include "lib/logging.h"

namespace jadevectordb {

struct FilteredSearchBenchmarkResult {
    std::string test_name;
    size_t database_size;
    size_t filter_coverage;
    double avg_search_time_ms;
    double min_search_time_ms;
    double max_search_time_ms;
    double throughput_qps;
    size_t total_queries;
    size_t avg_results_per_query;
    std::vector<double> individual_times;
};

class FilteredSearchBenchmark {
private:
    std::shared_ptr<SimilaritySearchService> search_service_;
    std::shared_ptr<logging::Logger> logger_;
    std::mt19937 rng_;

public:
    explicit FilteredSearchBenchmark(std::shared_ptr<SimilaritySearchService> search_service);
    ~FilteredSearchBenchmark() = default;
    
    // Benchmark filtered search with different filter coverage percentages
    FilteredSearchBenchmarkResult benchmark_filtered_search(
        const std::string& database_id,
        size_t num_vectors,
        double filter_coverage_percentage,  // 0.0 to 1.0
        size_t num_queries = 100);
    
    // Benchmark filtered search with different filter types
    FilteredSearchBenchmarkResult benchmark_filtered_search_by_type(
        const std::string& database_id,
        const std::string& filter_type,  // "owner", "category", "tags", "score_range", "custom"
        size_t num_vectors,
        size_t num_queries = 100);
    
    // Benchmark filtered search with complex filter combinations
    FilteredSearchBenchmarkResult benchmark_complex_filtered_search(
        const std::string& database_id,
        size_t num_vectors,
        size_t num_conditions,
        FilterCombination combination,
        size_t num_queries = 100);
    
    // Run comprehensive benchmark suite
    std::vector<FilteredSearchBenchmarkResult> run_comprehensive_benchmark(
        const std::string& database_id,
        size_t num_vectors = 10000,
        size_t num_queries = 100);
    
    // Performance validation
    bool validate_performance_requirements(
        const std::string& database_id,
        double max_avg_search_time_ms = 100.0,
        double min_throughput_qps = 10.0);
    
    // Generate benchmark report
    std::string generate_benchmark_report(const FilteredSearchBenchmarkResult& result) const;
    std::string generate_comprehensive_report(const std::vector<FilteredSearchBenchmarkResult>& results) const;
    
    // Create test data for benchmarking
    Result<std::vector<Vector>> generate_test_vectors(
        size_t num_vectors,
        int vector_dimension = 128) const;
    
    Result<void> populate_database_with_test_data(
        const std::string& database_id,
        size_t num_vectors,
        int vector_dimension = 128) const;

private:
    // Helper methods
    double run_single_filtered_search(
        const std::string& database_id,
        const Vector& query_vector,
        const SearchParams& params) const;
    
    double run_single_complex_filtered_search(
        const std::string& database_id,
        const Vector& query_vector,
        const ComplexFilter& filter) const;
    
    std::tuple<double, double, double, double, size_t> calculate_statistics(
        const std::vector<double>& times,
        const std::vector<size_t>& result_counts) const;
    
    std::string generate_random_string(size_t length) const;
    std::vector<std::string> generate_random_tags(size_t count) const;
    std::string select_random_owner() const;
    std::string select_random_category() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_FILTERED_SEARCH_BENCHMARK_H