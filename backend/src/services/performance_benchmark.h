#ifndef JADEVECTORDB_PERFORMANCE_BENCHMARK_H
#define JADEVECTORDB_PERFORMANCE_BENCHMARK_H

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "models/vector.h"
#include "models/database.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <functional>
#include <thread>
#include <atomic>

namespace jadevectordb {

// Types of benchmark operations
enum class BenchmarkOperation {
    INSERT,
    SEARCH,
    UPDATE,
    DELETE,
    BATCH_INSERT,
    BATCH_SEARCH,
    SIMILARITY_SEARCH,
    RANGE_QUERY,
    AGGREGATION
};

// Configuration for a benchmark run
struct BenchmarkConfig {
    std::string benchmark_name;
    BenchmarkOperation operation_type;
    int num_operations;               // Total number of operations to perform
    int num_concurrent_threads;      // Number of concurrent threads
    int batch_size;                  // Size of batch operations (if applicable)
    std::chrono::milliseconds warmup_period;  // Warmup period before timing
    std::chrono::milliseconds cooldown_period; // Cooldown after benchmark
    bool measure_latency;            // Whether to measure individual operation latency
    bool measure_throughput;         // Whether to measure operations per second
    bool measure_memory_usage;       // Whether to measure memory usage
    bool measure_cpu_usage;          // Whether to measure CPU usage
    std::string output_format;       // Output format: "json", "csv", "text"
    std::string output_path;         // Path to save benchmark results
    bool include_percentiles;        // Whether to include percentile calculations (50th, 95th, 99th)
    int dimensions;                  // Vector dimensions for synthetic data
    int num_vectors;                 // Number of vectors in dataset for search operations
    
    BenchmarkConfig() : num_operations(1000), num_concurrent_threads(1), 
                        batch_size(10), warmup_period(std::chrono::milliseconds(1000)),
                        cooldown_period(std::chrono::milliseconds(500)),
                        measure_latency(true), measure_throughput(true),
                        measure_memory_usage(false), measure_cpu_usage(false),
                        output_format("json"), include_percentiles(true),
                        dimensions(128), num_vectors(10000) {}
};

// Results of a single benchmark operation
struct BenchmarkOperationResult {
    std::chrono::nanoseconds latency_ns;
    bool success;
    std::string error_message;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    
    BenchmarkOperationResult() : success(false) {}
    BenchmarkOperationResult(const std::chrono::nanoseconds& lat, bool succ = true, 
                           const std::string& err = "")
        : latency_ns(lat), success(succ), error_message(err),
          start_time(std::chrono::system_clock::now()),
          end_time(std::chrono::system_clock::now()) {}
};

// Results of a benchmark run
struct BenchmarkResult {
    std::string benchmark_name;
    BenchmarkOperation operation_type;
    int total_operations;
    int successful_operations;
    int failed_operations;
    std::chrono::nanoseconds total_time_ns;
    std::chrono::nanoseconds avg_latency_ns;
    std::chrono::nanoseconds min_latency_ns;
    std::chrono::nanoseconds max_latency_ns;
    double operations_per_second;
    std::vector<double> latency_percentiles; // 50th, 95th, 99th percentiles
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::unordered_map<std::string, std::string> additional_metrics; // Additional custom metrics
    
    BenchmarkResult() : total_operations(0), successful_operations(0), failed_operations(0),
                        operations_per_second(0.0) {}
};

/**
 * @brief Performance Benchmarking Framework for Vector Database Operations
 * 
 * This service provides benchmarking capabilities to measure the performance 
 * of various vector database operations including insertion, search, update,
 * and deletion under different load patterns and configurations.
 */
class PerformanceBenchmark {
private:
    std::shared_ptr<logging::Logger> logger_;
    BenchmarkConfig default_config_;
    mutable std::mutex config_mutex_;
    std::atomic<bool> benchmark_running_;
    
public:
    explicit PerformanceBenchmark();
    ~PerformanceBenchmark() = default;
    
    // Run a performance benchmark
    Result<BenchmarkResult> run_benchmark(const BenchmarkConfig& config);
    
    // Run a benchmark with a custom operation function
    Result<BenchmarkResult> run_custom_benchmark(
        const BenchmarkConfig& config,
        std::function<Result<BenchmarkOperationResult>()> operation_func);
    
    // Run a vector insertion benchmark
    Result<BenchmarkResult> run_insert_benchmark(
        const BenchmarkConfig& config,
        const std::vector<Vector>& vectors);
    
    // Run a similarity search benchmark
    Result<BenchmarkResult> run_search_benchmark(
        const BenchmarkConfig& config,
        const std::vector<Vector>& query_vectors);
    
    // Run a batch operation benchmark
    Result<BenchmarkResult> run_batch_benchmark(
        const BenchmarkConfig& config,
        const std::vector<std::vector<Vector>>& batch_operations);
    
    // Run multiple benchmarks in sequence
    Result<std::vector<BenchmarkResult>> run_multiple_benchmarks(
        const std::vector<BenchmarkConfig>& configs);
    
    // Generate a report from benchmark results
    Result<std::string> generate_report(const std::vector<BenchmarkResult>& results) const;
    
    // Save benchmark results to file
    Result<bool> save_results(const std::vector<BenchmarkResult>& results, 
                            const std::string& file_path) const;
    
    // Calculate percentiles from latency measurements
    std::vector<double> calculate_percentiles(const std::vector<std::chrono::nanoseconds>& latencies) const;
    
    // Get the default benchmark configuration
    BenchmarkConfig get_default_config() const;
    
    // Update the default benchmark configuration
    void set_default_config(const BenchmarkConfig& config);

private:
    // Prepare for benchmark (e.g., warmup operations)
    Result<bool> prepare_benchmark(const BenchmarkConfig& config);
    
    // Execute the benchmark with multiple threads
    Result<BenchmarkResult> execute_benchmark_multi_threaded(
        const BenchmarkConfig& config,
        std::function<Result<BenchmarkOperationResult>()> operation_func);
    
    // Execute the benchmark with a single thread
    Result<BenchmarkResult> execute_benchmark_single_threaded(
        const BenchmarkConfig& config,
        std::function<Result<BenchmarkOperationResult>()> operation_func);
    
    // Generate synthetic vector data for benchmarking
    std::vector<Vector> generate_synthetic_vectors(int count, int dimensions, 
                                                 const std::string& database_id) const;
    
    // Process benchmark results to calculate metrics
    BenchmarkResult process_results(const std::vector<BenchmarkOperationResult>& operation_results,
                                  const BenchmarkConfig& config,
                                  const std::chrono::system_clock::time_point& start_time,
                                  const std::chrono::system_clock::time_point& end_time) const;
    
    // Format results based on configured output format
    std::string format_results(const BenchmarkResult& result, 
                             const std::string& format) const;
    
    // Validate benchmark configuration
    bool validate_config(const BenchmarkConfig& config) const;
    
    // Get system metrics during benchmark
    std::unordered_map<std::string, std::string> get_system_metrics() const;
    
    // Generate a summary of the benchmark run
    std::string generate_summary(const BenchmarkResult& result) const;
    
    // Calculate operations per second
    double calculate_ops_per_second(int operations, 
                                  const std::chrono::nanoseconds& duration) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_PERFORMANCE_BENCHMARK_H