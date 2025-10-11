#ifndef JADEVECTORDB_SEARCH_BENCHMARK_H
#define JADEVECTORDB_SEARCH_BENCHMARK_H

#include <chrono>
#include <vector>
#include <string>
#include <memory>

#include "models/vector.h"
#include "services/similarity_search.h"

namespace jadevectordb {

struct BenchmarkResult {
    std::string algorithm_name;
    double avg_search_time_ms;
    double min_search_time_ms;
    double max_search_time_ms;
    double throughput_qps;  // Queries per second
    double accuracy;        // Accuracy compared to ground truth
    size_t total_queries;
    std::vector<double> individual_times;  // Individual query times
};

class SearchBenchmark {
private:
    std::shared_ptr<SimilaritySearchService> search_service_;
    std::vector<Vector> test_vectors_;
    std::vector<Vector> query_vectors_;

public:
    explicit SearchBenchmark(std::shared_ptr<SimilaritySearchService> search_service);
    ~SearchBenchmark() = default;
    
    // Add test vectors to the benchmark
    void add_test_vectors(const std::vector<Vector>& vectors);
    
    // Add query vectors for testing
    void add_query_vectors(const std::vector<Vector>& queries);
    
    // Run a benchmark for a specific algorithm
    BenchmarkResult run_benchmark(
        const std::string& database_id,
        const std::string& algorithm,  // "cosine_similarity", "euclidean_distance", "dot_product"
        const SearchParams& params,
        size_t num_iterations = 100);
    
    // Run comprehensive benchmark comparing multiple algorithms
    std::vector<BenchmarkResult> run_comprehensive_benchmark(
        const std::string& database_id,
        const std::vector<std::string>& algorithms,
        const SearchParams& params,
        size_t num_iterations = 100);
    
    // Run performance validation: verify search meets performance requirements
    bool validate_performance_requirements(
        const std::string& database_id,
        double max_avg_search_time_ms = 50.0,  // 50ms for datasets up to 10M vectors
        double min_accuracy = 0.95);            // 95% accuracy
    
    // Run quality validation: ensure search results are accurate
    bool validate_quality_requirements(
        const std::string& database_id,
        const SearchParams& params,
        double min_accuracy = 0.95);
    
    // Get detailed performance metrics
    std::string get_performance_report(const BenchmarkResult& result) const;
    
private:
    // Helper function to run a single query and measure time
    double run_single_query(
        const std::string& database_id,
        const Vector& query_vector,
        const SearchParams& params,
        const std::string& algorithm);
    
    // Calculate statistics from individual times
    std::tuple<double, double, double, double> calculate_statistics(
        const std::vector<double>& times) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SEARCH_BENCHMARK_H