#include "performance_benchmark.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread>
#include <future>
#include <chrono>

namespace jadevectordb {

PerformanceBenchmark::PerformanceBenchmark() : benchmark_running_(false) {
    logger_ = logging::LoggerManager::get_logger("PerformanceBenchmark");
    
    // Set up default configuration
    default_config_.warmup_period = std::chrono::milliseconds(1000);
    default_config_.cooldown_period = std::chrono::milliseconds(500);
    default_config_.num_operations = 1000;
    default_config_.num_concurrent_threads = 1;
    default_config_.batch_size = 10;
    default_config_.dimensions = 128;
    default_config_.num_vectors = 10000;
}

Result<BenchmarkResult> PerformanceBenchmark::run_benchmark(const BenchmarkConfig& config) {
    try {
        if (!validate_config(config)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid benchmark configuration");
        }
        
        LOG_INFO(logger_, "Starting benchmark: " + config.benchmark_name);
        
        // Prepare for benchmark (warmup)
        auto prepare_result = prepare_benchmark(config);
        if (!prepare_result.has_value()) {
            LOG_ERROR(logger_, "Failed to prepare benchmark: " + 
                     ErrorHandler::format_error(prepare_result.error()));
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, ErrorHandler::format_error(prepare_result.error()));
        }
        
        // Define a simple operation based on the operation type
        std::function<Result<BenchmarkOperationResult>()> operation_func;
        
        switch (config.operation_type) {
            case BenchmarkOperation::INSERT:
            case BenchmarkOperation::SEARCH:
            case BenchmarkOperation::UPDATE:
            case BenchmarkOperation::DELETE:
            case BenchmarkOperation::SIMILARITY_SEARCH:
                // For these operations, we'll generate synthetic data
                operation_func = [this, config]() -> Result<BenchmarkOperationResult> {
                    // Simulate the operation with a small sleep to mimic work
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    std::this_thread::sleep_for(std::chrono::microseconds(100)); // Simulate work
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                    
                    return BenchmarkOperationResult(latency, true);
                };
                break;
                
            case BenchmarkOperation::BATCH_INSERT:
            case BenchmarkOperation::BATCH_SEARCH:
                // Batch operations
                operation_func = [this, config]() -> Result<BenchmarkOperationResult> {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    // Simulate batch work
                    for (int i = 0; i < config.batch_size; ++i) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50)); // Simulate work per item
                    }
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                    
                    return BenchmarkOperationResult(latency, true);
                };
                break;
                
            default:
                // For other operations, use simple timing
                operation_func = [this]() -> Result<BenchmarkOperationResult> {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    // Minimal work
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                    
                    return BenchmarkOperationResult(latency, true);
                };
                break;
        }
        
        Result<BenchmarkResult> result;
        if (config.num_concurrent_threads > 1) {
            result = execute_benchmark_multi_threaded(config, operation_func);
        } else {
            result = execute_benchmark_single_threaded(config, operation_func);
        }
        
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Benchmark execution failed: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Completed benchmark: " + config.benchmark_name);
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run benchmark: " + std::string(e.what()));
    }
}

Result<BenchmarkResult> PerformanceBenchmark::run_custom_benchmark(
    const BenchmarkConfig& config,
    std::function<Result<BenchmarkOperationResult>()> operation_func) {
    try {
        if (!validate_config(config)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid benchmark configuration");
        }
        
        LOG_INFO(logger_, "Starting custom benchmark: " + config.benchmark_name);
        
        // Prepare for benchmark (warmup)
        auto prepare_result = prepare_benchmark(config);
        if (!prepare_result.has_value()) {
            LOG_ERROR(logger_, "Failed to prepare custom benchmark: " + 
                     ErrorHandler::format_error(prepare_result.error()));
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, ErrorHandler::format_error(prepare_result.error()));
        }
        
        Result<BenchmarkResult> result;
        if (config.num_concurrent_threads > 1) {
            result = execute_benchmark_multi_threaded(config, operation_func);
        } else {
            result = execute_benchmark_single_threaded(config, operation_func);
        }
        
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Custom benchmark execution failed: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Completed custom benchmark: " + config.benchmark_name);
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_custom_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run custom benchmark: " + std::string(e.what()));
    }
}

Result<BenchmarkResult> PerformanceBenchmark::run_insert_benchmark(
    const BenchmarkConfig& config,
    const std::vector<Vector>& vectors) {
    try {
        if (!validate_config(config)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid benchmark configuration");
        }
        
        LOG_INFO(logger_, "Starting insert benchmark with " + std::to_string(vectors.size()) + " vectors");
        
        // Create a function that performs an insert operation
        size_t vector_index = 0;
        std::mutex vector_mutex;
        
        auto operation_func = [&vectors, &vector_index, &vector_mutex]() -> Result<BenchmarkOperationResult> {
            std::lock_guard<std::mutex> lock(vector_mutex);
            
            if (vector_index >= vectors.size()) {
                vector_index = 0; // Loop back if needed
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate vector insertion (in a real implementation, this would call the actual insert API)
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            vector_index++;
            return BenchmarkOperationResult(latency, true);
        };
        
        return run_custom_benchmark(config, operation_func);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_insert_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run insert benchmark: " + std::string(e.what()));
    }
}

Result<BenchmarkResult> PerformanceBenchmark::run_search_benchmark(
    const BenchmarkConfig& config,
    const std::vector<Vector>& query_vectors) {
    try {
        if (!validate_config(config)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid benchmark configuration");
        }
        
        LOG_INFO(logger_, "Starting search benchmark with " + std::to_string(query_vectors.size()) + " queries");
        
        // Create a function that performs a search operation
        size_t query_index = 0;
        std::mutex query_mutex;
        
        auto operation_func = [&query_vectors, &query_index, &query_mutex]() -> Result<BenchmarkOperationResult> {
            std::lock_guard<std::mutex> lock(query_mutex);
            
            if (query_index >= query_vectors.size()) {
                query_index = 0; // Loop back if needed
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate vector search (in a real implementation, this would call the actual search API)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            query_index++;
            return BenchmarkOperationResult(latency, true);
        };
        
        return run_custom_benchmark(config, operation_func);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_search_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run search benchmark: " + std::string(e.what()));
    }
}

Result<BenchmarkResult> PerformanceBenchmark::run_batch_benchmark(
    const BenchmarkConfig& config,
    const std::vector<std::vector<Vector>>& batch_operations) {
    try {
        if (!validate_config(config)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid benchmark configuration");
        }
        
        LOG_INFO(logger_, "Starting batch benchmark with " + std::to_string(batch_operations.size()) + " batches");
        
        // Create a function that performs a batch operation
        size_t batch_index = 0;
        std::mutex batch_mutex;
        
        auto operation_func = [&batch_operations, &batch_index, &batch_mutex]() -> Result<BenchmarkOperationResult> {
            std::lock_guard<std::mutex> lock(batch_mutex);
            
            if (batch_index >= batch_operations.size()) {
                batch_index = 0; // Loop back if needed
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate batch operation (in a real implementation, this would call the actual batch API)
            std::this_thread::sleep_for(std::chrono::microseconds(500 + (batch_operations[batch_index].size() * 10)));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            batch_index++;
            return BenchmarkOperationResult(latency, true);
        };
        
        return run_custom_benchmark(config, operation_func);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_batch_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run batch benchmark: " + std::string(e.what()));
    }
}

Result<std::vector<BenchmarkResult>> PerformanceBenchmark::run_multiple_benchmarks(
    const std::vector<BenchmarkConfig>& configs) {
    try {
        LOG_INFO(logger_, "Starting " + std::to_string(configs.size()) + " benchmarks");
        
        std::vector<BenchmarkResult> results;
        
        for (const auto& config : configs) {
            LOG_DEBUG(logger_, "Running benchmark: " + config.benchmark_name);
            
            auto result = run_benchmark(config);
            if (result.has_value()) {
                results.push_back(result.value());
                LOG_DEBUG(logger_, "Completed benchmark: " + config.benchmark_name);
            } else {
                LOG_WARN(logger_, "Benchmark failed: " + config.benchmark_name + 
                        " - " + ErrorHandler::format_error(result.error()));
                // Continue with other benchmarks even if one fails
            }
        }
        
        LOG_INFO(logger_, "Completed " + std::to_string(results.size()) + " benchmarks");
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_multiple_benchmarks: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run multiple benchmarks: " + std::string(e.what()));
    }
}

Result<std::string> PerformanceBenchmark::generate_report(const std::vector<BenchmarkResult>& results) const {
    try {
        std::ostringstream report;
        
        report << "Performance Benchmark Report\n";
        report << "==========================\n";
        report << "Generated on: " 
               << std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count() 
               << "\n\n";
        
        for (const auto& result : results) {
            report << generate_summary(result) << "\n";
        }
        
        LOG_DEBUG(logger_, "Generated performance benchmark report");
        return report.str();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in generate_report: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to generate report: " + std::string(e.what()));
    }
}

Result<bool> PerformanceBenchmark::save_results(const std::vector<BenchmarkResult>& results, 
                                               const std::string& file_path) const {
    try {
        std::ofstream output_file(file_path);
        if (!output_file.is_open()) {
            RETURN_ERROR(ErrorCode::IO_ERROR, "Failed to open output file: " + file_path);
        }
        
        // Generate the report and save to file
        auto report_result = generate_report(results);
        if (!report_result.has_value()) {
            return report_result;
        }
        
        output_file << report_result.value();
        output_file.close();
        
        LOG_INFO(logger_, "Benchmark results saved to: " + file_path);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in save_results: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to save results: " + std::string(e.what()));
    }
}

std::vector<double> PerformanceBenchmark::calculate_percentiles(const std::vector<std::chrono::nanoseconds>& latencies) const {
    if (latencies.empty()) {
        return {0.0, 0.0, 0.0}; // 50th, 95th, 99th percentiles
    }
    
    // Create a copy and sort the latencies
    std::vector<double> sorted_latencies;
    sorted_latencies.reserve(latencies.size());
    
    for (const auto& latency : latencies) {
        sorted_latencies.push_back(static_cast<double>(latency.count()));
    }
    
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    
    // Calculate percentiles: 50th (median), 95th, 99th
    std::vector<double> percentiles;
    
    // 50th percentile (median)
    size_t p50_idx = static_cast<size_t>(sorted_latencies.size() * 0.5);
    if (p50_idx >= sorted_latencies.size()) p50_idx = sorted_latencies.size() - 1;
    percentiles.push_back(sorted_latencies[p50_idx]);
    
    // 95th percentile
    size_t p95_idx = static_cast<size_t>(sorted_latencies.size() * 0.95);
    if (p95_idx >= sorted_latencies.size()) p95_idx = sorted_latencies.size() - 1;
    percentiles.push_back(sorted_latencies[p95_idx]);
    
    // 99th percentile
    size_t p99_idx = static_cast<size_t>(sorted_latencies.size() * 0.99);
    if (p99_idx >= sorted_latencies.size()) p99_idx = sorted_latencies.size() - 1;
    percentiles.push_back(sorted_latencies[p99_idx]);
    
    return percentiles;
}

BenchmarkConfig PerformanceBenchmark::get_default_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return default_config_;
}

void PerformanceBenchmark::set_default_config(const BenchmarkConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    default_config_ = config;
}

// Private methods

Result<bool> PerformanceBenchmark::prepare_benchmark(const BenchmarkConfig& config) {
    try {
        LOG_DEBUG(logger_, "Preparing benchmark: " + config.benchmark_name + 
                 ", warmup period: " + std::to_string(config.warmup_period.count()) + "ms");
        
        // Perform warmup operations if specified
        if (config.warmup_period.count() > 0) {
            auto warmup_start = std::chrono::steady_clock::now();
            
            // Perform minimal warmup work
            while (std::chrono::steady_clock::now() - warmup_start < config.warmup_period) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        LOG_DEBUG(logger_, "Benchmark preparation completed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in prepare_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to prepare benchmark: " + std::string(e.what()));
    }
}

Result<BenchmarkResult> PerformanceBenchmark::execute_benchmark_multi_threaded(
    const BenchmarkConfig& config,
    std::function<Result<BenchmarkOperationResult>()> operation_func) {
    try {
        LOG_DEBUG(logger_, "Executing benchmark with " + std::to_string(config.num_concurrent_threads) + " threads");
        
        auto start_time = std::chrono::system_clock::now();
        
        // Distribute operations across threads
        int ops_per_thread = config.num_operations / config.num_concurrent_threads;
        int remaining_ops = config.num_operations % config.num_concurrent_threads;
        
        std::vector<std::future<std::vector<BenchmarkOperationResult>>> futures;
        
        benchmark_running_ = true;
        
        // Launch threads
        for (int i = 0; i < config.num_concurrent_threads; ++i) {
            int ops_for_this_thread = ops_per_thread;
            if (i < remaining_ops) {
                ops_for_this_thread++;  // Distribute remainder among first threads
            }
            
            futures.push_back(std::async(std::launch::async, [this, ops_for_this_thread, operation_func]() {
                std::vector<BenchmarkOperationResult> thread_results;
                thread_results.reserve(ops_for_this_thread);
                
                for (int j = 0; j < ops_for_this_thread && benchmark_running_; ++j) {
                    auto result = operation_func();
                    if (result.has_value()) {
                        thread_results.push_back(result.value());
                    } else {
                        // Log error but continue
                        LOG_WARN(logger_, "Operation failed during benchmark: " + 
                                ErrorHandler::format_error(result.error()));
                        thread_results.push_back(BenchmarkOperationResult(std::chrono::nanoseconds(0), false, 
                                                                        ErrorHandler::format_error(result.error())));
                    }
                }
                
                return thread_results;
            }));
        }
        
        // Collect results from all threads
        std::vector<BenchmarkOperationResult> all_results;
        for (auto& future : futures) {
            auto thread_results = future.get();
            all_results.insert(all_results.end(), 
                             std::make_move_iterator(thread_results.begin()),
                             std::make_move_iterator(thread_results.end()));
        }
        
        benchmark_running_ = false;
        
        auto end_time = std::chrono::system_clock::now();
        
        // Process all results to create the final benchmark result
        return process_results(all_results, config, start_time, end_time);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in execute_benchmark_multi_threaded: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed multi-threaded execution: " + std::string(e.what()));
    }
}

Result<BenchmarkResult> PerformanceBenchmark::execute_benchmark_single_threaded(
    const BenchmarkConfig& config,
    std::function<Result<BenchmarkOperationResult>()> operation_func) {
    try {
        LOG_DEBUG(logger_, "Executing benchmark single-threaded with " + std::to_string(config.num_operations) + " operations");
        
        auto start_time = std::chrono::system_clock::now();
        
        std::vector<BenchmarkOperationResult> results;
        results.reserve(config.num_operations);
        
        benchmark_running_ = true;
        
        // Execute operations sequentially
        for (int i = 0; i < config.num_operations && benchmark_running_; ++i) {
            auto result = operation_func();
            if (result.has_value()) {
                results.push_back(result.value());
            } else {
                // Log error but continue
                LOG_WARN(logger_, "Operation failed during benchmark: " + 
                        ErrorHandler::format_error(result.error()));
                results.push_back(BenchmarkOperationResult(std::chrono::nanoseconds(0), false, 
                                                         ErrorHandler::format_error(result.error())));
            }
        }
        
        benchmark_running_ = false;
        
        auto end_time = std::chrono::system_clock::now();
        
        // Process results to create the final benchmark result
        return process_results(results, config, start_time, end_time);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in execute_benchmark_single_threaded: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed single-threaded execution: " + std::string(e.what()));
    }
}

std::vector<Vector> PerformanceBenchmark::generate_synthetic_vectors(int count, int dimensions, 
                                                                  const std::string& database_id) const {
    std::vector<Vector> vectors;
    vectors.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < count; ++i) {
        Vector vec;
        vec.id = "synthetic_vec_" + std::to_string(i);
        vec.databaseId = database_id;
        
        // Generate random vector values
        vec.values.reserve(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            vec.values.push_back(dis(gen));
        }
        
        vectors.push_back(vec);
    }
    
    return vectors;
}

BenchmarkResult PerformanceBenchmark::process_results(
    const std::vector<BenchmarkOperationResult>& operation_results,
    const BenchmarkConfig& config,
    const std::chrono::system_clock::time_point& start_time,
    const std::chrono::system_clock::time_point& end_time) const {
    
    BenchmarkResult result;
    result.benchmark_name = config.benchmark_name;
    result.operation_type = config.operation_type;
    result.total_operations = static_cast<int>(operation_results.size());
    result.start_time = start_time;
    result.end_time = end_time;
    
    if (operation_results.empty()) {
        return result;
    }
    
    // Calculate derived metrics
    std::vector<std::chrono::nanoseconds> latencies;
    latencies.reserve(operation_results.size());
    
    int successful_ops = 0;
    std::chrono::nanoseconds total_latency(0);
    std::chrono::nanoseconds min_latency = operation_results[0].latency_ns;
    std::chrono::nanoseconds max_latency(0);
    
    for (const auto& op_result : operation_results) {
        if (op_result.success) {
            successful_ops++;
            latencies.push_back(op_result.latency_ns);
            total_latency += op_result.latency_ns;
            
            if (op_result.latency_ns < min_latency) {
                min_latency = op_result.latency_ns;
            }
            if (op_result.latency_ns > max_latency) {
                max_latency = op_result.latency_ns;
            }
        }
    }
    
    result.successful_operations = successful_ops;
    result.failed_operations = result.total_operations - successful_ops;
    
    if (!latencies.empty()) {
        result.avg_latency_ns = total_latency / latencies.size();
        result.min_latency_ns = min_latency;
        result.max_latency_ns = max_latency;
        
        // Calculate percentiles if requested
        if (config.include_percentiles) {
            result.latency_percentiles = calculate_percentiles(latencies);
        }
    }
    
    // Calculate total execution time and operations per second
    result.total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    result.operations_per_second = calculate_ops_per_second(successful_ops, result.total_time_ns);
    
    // Add system metrics if requested
    if (config.measure_memory_usage || config.measure_cpu_usage) {
        result.additional_metrics = get_system_metrics();
    }
    
    return result;
}

std::string PerformanceBenchmark::format_results(const BenchmarkResult& result, 
                                                const std::string& format) const {
    if (format == "json") {
        std::ostringstream json_stream;
        json_stream << "{"
                   << "\"benchmark_name\":\"" << result.benchmark_name << "\","
                   << "\"operation_type\":" << static_cast<int>(result.operation_type) << ","
                   << "\"total_operations\":" << result.total_operations << ","
                   << "\"successful_operations\":" << result.successful_operations << ","
                   << "\"failed_operations\":" << result.failed_operations << ","
                   << "\"total_time_ns\":" << result.total_time_ns.count() << ","
                   << "\"avg_latency_ns\":" << result.avg_latency_ns.count() << ","
                   << "\"min_latency_ns\":" << result.min_latency_ns.count() << ","
                   << "\"max_latency_ns\":" << result.max_latency_ns.count() << ","
                   << "\"operations_per_second\":" << result.operations_per_second;
        
        if (!result.latency_percentiles.empty()) {
            json_stream << ",\"latency_percentiles\":[";
            for (size_t i = 0; i < result.latency_percentiles.size(); ++i) {
                if (i > 0) json_stream << ",";
                json_stream << result.latency_percentiles[i];
            }
            json_stream << "]";
        }
        
        json_stream << "}";
        return json_stream.str();
    } else if (format == "csv") {
        std::ostringstream csv_stream;
        csv_stream << "Benchmark,Operation,Total,Successful,Failed,TotalTimeNs,AvgLatencyNs,"
                   << "MinLatencyNs,MaxLatencyNs,OpsPerSecond\n";
        csv_stream << result.benchmark_name << ","
                   << static_cast<int>(result.operation_type) << ","
                   << result.total_operations << ","
                   << result.successful_operations << ","
                   << result.failed_operations << ","
                   << result.total_time_ns.count() << ","
                   << result.avg_latency_ns.count() << ","
                   << result.min_latency_ns.count() << ","
                   << result.max_latency_ns.count() << ","
                   << result.operations_per_second;
        return csv_stream.str();
    } else {
        // Default text format
        std::ostringstream text_stream;
        text_stream << "Benchmark: " << result.benchmark_name << "\n";
        text_stream << "Operation Type: " << static_cast<int>(result.operation_type) << "\n";
        text_stream << "Total Operations: " << result.total_operations << "\n";
        text_stream << "Successful: " << result.successful_operations << "\n";
        text_stream << "Failed: " << result.failed_operations << "\n";
        text_stream << "Total Time (ns): " << result.total_time_ns.count() << "\n";
        text_stream << "Average Latency (ns): " << result.avg_latency_ns.count() << "\n";
        text_stream << "Min Latency (ns): " << result.min_latency_ns.count() << "\n";
        text_stream << "Max Latency (ns): " << result.max_latency_ns.count() << "\n";
        text_stream << "Operations/Second: " << result.operations_per_second << "\n";
        
        if (!result.latency_percentiles.empty() && result.latency_percentiles.size() >= 3) {
            text_stream << "Latency Percentiles - 50th: " << result.latency_percentiles[0] << "ns, "
                       << "95th: " << result.latency_percentiles[1] << "ns, "
                       << "99th: " << result.latency_percentiles[2] << "ns\n";
        }
        
        return text_stream.str();
    }
}

bool PerformanceBenchmark::validate_config(const BenchmarkConfig& config) const {
    // Basic validation
    if (config.num_operations <= 0) {
        LOG_ERROR(logger_, "Invalid number of operations: " + std::to_string(config.num_operations));
        return false;
    }
    
    if (config.num_concurrent_threads <= 0) {
        LOG_ERROR(logger_, "Invalid number of concurrent threads: " + std::to_string(config.num_concurrent_threads));
        return false;
    }
    
    if (config.batch_size <= 0 && (config.operation_type == BenchmarkOperation::BATCH_INSERT ||
                                   config.operation_type == BenchmarkOperation::BATCH_SEARCH)) {
        LOG_ERROR(logger_, "Invalid batch size for batch operation: " + std::to_string(config.batch_size));
        return false;
    }
    
    if (config.dimensions <= 0) {
        LOG_ERROR(logger_, "Invalid vector dimensions: " + std::to_string(config.dimensions));
        return false;
    }
    
    if (config.output_format != "json" && config.output_format != "csv" && config.output_format != "text") {
        LOG_ERROR(logger_, "Invalid output format: " + config.output_format);
        return false;
    }
    
    return true;
}

std::unordered_map<std::string, std::string> PerformanceBenchmark::get_system_metrics() const {
    // Placeholder implementation
    // In a real implementation, this would gather actual system metrics
    std::unordered_map<std::string, std::string> metrics;
    metrics["memory_usage_mb"] = "0"; // Placeholder
    metrics["cpu_usage_percent"] = "0.0"; // Placeholder
    metrics["disk_io"] = "0"; // Placeholder
    
    return metrics;
}

std::string PerformanceBenchmark::generate_summary(const BenchmarkResult& result) const {
    std::ostringstream summary;
    summary << "Benchmark: " << result.benchmark_name << "\n";
    summary << "  Operations: " << result.successful_operations << "/" << result.total_operations << " succeeded\n";
    summary << "  Avg Latency: " << (result.avg_latency_ns.count() / 1000000.0) << " ms\n";
    summary << "  Throughput: " << result.operations_per_second << " ops/sec\n";
    summary << "  Min/Max Latency: " << (result.min_latency_ns.count() / 1000000.0) << "/" 
            << (result.max_latency_ns.count() / 1000000.0) << " ms\n";
    
    if (!result.latency_percentiles.empty() && result.latency_percentiles.size() >= 3) {
        summary << "  50th/95th/99th Latency Percentiles: " 
                << (result.latency_percentiles[0] / 1000000.0) << "/"
                << (result.latency_percentiles[1] / 1000000.0) << "/"
                << (result.latency_percentiles[2] / 1000000.0) << " ms\n";
    }
    
    return summary.str();
}

double PerformanceBenchmark::calculate_ops_per_second(int operations, 
                                                     const std::chrono::nanoseconds& duration) const {
    if (duration.count() <= 0) {
        return 0.0;
    }
    
    double duration_seconds = static_cast<double>(duration.count()) / 1000000000.0;
    return static_cast<double>(operations) / duration_seconds;
}

} // namespace jadevectordb