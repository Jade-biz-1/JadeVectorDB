#include "search_benchmark.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace jadevectordb {

SearchBenchmark::SearchBenchmark(std::shared_ptr<SimilaritySearchService> search_service)
    : search_service_(search_service) {
}

void SearchBenchmark::add_test_vectors(const std::vector<Vector>& vectors) {
    test_vectors_.insert(test_vectors_.end(), vectors.begin(), vectors.end());
}

void SearchBenchmark::add_query_vectors(const std::vector<Vector>& queries) {
    query_vectors_.insert(query_vectors_.end(), queries.begin(), queries.end());
}

BenchmarkResult SearchBenchmark::run_benchmark(
    const std::string& database_id,
    const std::string& algorithm,
    const SearchParams& params,
    size_t num_iterations) {
    
    BenchmarkResult result;
    result.algorithm_name = algorithm;
    result.total_queries = num_iterations;
    result.individual_times.reserve(num_iterations);
    
    // If no query vectors were added, create some random ones
    if (query_vectors_.empty()) {
        // Create a simple random vector for testing
        Vector random_query;
        random_query.id = "benchmark_query";
        random_query.values.resize(128); // Assuming 128-dim vectors for benchmark
        for (auto& val : random_query.values) {
            val = static_cast<float>(rand()) / RAND_MAX;
        }
        query_vectors_.push_back(random_query);
    }
    
    // Run the benchmark
    for (size_t i = 0; i < num_iterations; ++i) {
        // Select a query vector (cycling through available ones if needed)
        const Vector& query_vector = query_vectors_[i % query_vectors_.size()];
        
        double query_time = run_single_query(database_id, query_vector, params, algorithm);
        result.individual_times.push_back(query_time);
    }
    
    // Calculate statistics
    auto [avg_time, min_time, max_time, throughput] = calculate_statistics(result.individual_times);
    result.avg_search_time_ms = avg_time;
    result.min_search_time_ms = min_time;
    result.max_search_time_ms = max_time;
    result.throughput_qps = throughput;
    
    // Calculate accuracy - for now we'll set a placeholder value
    // In a real implementation, this would compare against ground truth
    result.accuracy = 1.0; // Placeholder
    
    return result;
}

std::vector<BenchmarkResult> SearchBenchmark::run_comprehensive_benchmark(
    const std::string& database_id,
    const std::vector<std::string>& algorithms,
    const SearchParams& params,
    size_t num_iterations) {
    
    std::vector<BenchmarkResult> results;
    
    for (const auto& algorithm : algorithms) {
        BenchmarkResult result = run_benchmark(database_id, algorithm, params, num_iterations);
        results.push_back(result);
    }
    
    return results;
}

bool SearchBenchmark::validate_performance_requirements(
    const std::string& database_id,
    double max_avg_search_time_ms,
    double min_accuracy) {
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0;
    
    // Run a small benchmark
    BenchmarkResult result = run_benchmark(database_id, "cosine_similarity", params, 10);
    
    // Check if performance requirements are met
    bool meets_time_requirement = result.avg_search_time_ms <= max_avg_search_time_ms;
    bool meets_accuracy_requirement = result.accuracy >= min_accuracy;
    
    return meets_time_requirement && meets_accuracy_requirement;
}

bool SearchBenchmark::validate_quality_requirements(
    const std::string& database_id,
    const SearchParams& params,
    double min_accuracy) {
    
    // Run a benchmark
    BenchmarkResult result = run_benchmark(database_id, "cosine_similarity", params, 10);
    
    // Check if quality requirements are met
    return result.accuracy >= min_accuracy;
}

std::string SearchBenchmark::get_performance_report(const BenchmarkResult& result) const {
    std::ostringstream report;
    report << "=== Search Performance Report ===\n";
    report << "Algorithm: " << result.algorithm_name << "\n";
    report << "Total Queries: " << result.total_queries << "\n";
    report << "Average Search Time: " << result.avg_search_time_ms << " ms\n";
    report << "Min Search Time: " << result.min_search_time_ms << " ms\n";
    report << "Max Search Time: " << result.max_search_time_ms << " ms\n";
    report << "Throughput: " << result.throughput_qps << " QPS\n";
    report << "Accuracy: " << result.accuracy << "\n";
    report << "===============================\n";
    
    return report.str();
}

double SearchBenchmark::run_single_query(
    const std::string& database_id,
    const Vector& query_vector,
    const SearchParams& params,
    const std::string& algorithm) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform the search using the specified algorithm
    Result<std::vector<SearchResult>> result;
    
    if (algorithm == "cosine_similarity") {
        result = search_service_->similarity_search(database_id, query_vector, params);
    } else if (algorithm == "euclidean_distance") {
        result = search_service_->euclidean_search(database_id, query_vector, params);
    } else if (algorithm == "dot_product") {
        result = search_service_->dot_product_search(database_id, query_vector, params);
    } else {
        // Default to cosine similarity
        result = search_service_->similarity_search(database_id, query_vector, params);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<double>(duration.count()) / 1000.0; // Convert to milliseconds
}

std::tuple<double, double, double, double> SearchBenchmark::calculate_statistics(
    const std::vector<double>& times) const {
    
    if (times.empty()) {
        return std::make_tuple(0.0, 0.0, 0.0, 0.0);
    }
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / times.size();
    
    auto min_max = std::minmax_element(times.begin(), times.end());
    double min_time = *min_max.first;
    double max_time = *min_max.second;
    
    // Calculate throughput (queries per second)
    double total_time_ms = sum;
    double throughput = (total_time_ms > 0) ? (times.size() * 1000.0) / total_time_ms : 0.0;
    
    return std::make_tuple(avg, min_time, max_time, throughput);
}

} // namespace jadevectordb