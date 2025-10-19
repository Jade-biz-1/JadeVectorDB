#include "workload_balancer.h"
#include <algorithm>
#include <chrono>

namespace jadevectordb {
namespace workload {

// SimpleThresholdStrategy implementation
SimpleThresholdStrategy::SimpleThresholdStrategy(size_t threshold) 
    : threshold_vectors_(threshold) {
}

int SimpleThresholdStrategy::get_gpu_work_percentage(const WorkloadMetrics& metrics, 
                                                   size_t vector_count, 
                                                   size_t vector_dimension) const {
    // If GPU is not available, assign 0% to GPU
    if (!metrics.gpu_available) {
        return 0;
    }
    
    // If vector count is below threshold, use CPU
    if (vector_count < threshold_vectors_) {
        return 0;
    }
    
    // For larger workloads, use GPU
    return 100;  // For this simple strategy, use 100% GPU when threshold is met
}

// PerformanceBasedStrategy implementation
int PerformanceBasedStrategy::get_gpu_work_percentage(const WorkloadMetrics& metrics, 
                                                    size_t vector_count, 
                                                    size_t vector_dimension) const {
    // If no historical metrics, use a conservative approach
    if (historical_metrics_.empty()) {
        // Use GPU if it's available and vector count is substantial
        if (metrics.gpu_available && vector_count > 500) {
            return 75; // Start with 75% on GPU
        }
        return 0; // Use CPU otherwise
    }
    
    // If GPU isn't available, use CPU
    if (!metrics.gpu_available) {
        return 0;
    }
    
    // Calculate average performance from historical data
    size_t total_cpu_time = 0, total_gpu_time = 0;
    size_t valid_entries = 0;
    
    for (const auto& hist_metric : historical_metrics_) {
        if (hist_metric.cpu_processing_time_ms > 0 && hist_metric.gpu_processing_time_ms > 0) {
            total_cpu_time += hist_metric.cpu_processing_time_ms;
            total_gpu_time += hist_metric.gpu_processing_time_ms;
            valid_entries++;
        }
    }
    
    if (valid_entries > 0) {
        double avg_cpu_time = static_cast<double>(total_cpu_time) / valid_entries;
        double avg_gpu_time = static_cast<double>(total_gpu_time) / valid_entries;
        
        // If GPU is faster, use it more; if CPU is faster, use CPU more
        if (avg_gpu_time < avg_cpu_time) {
            // GPU is faster, increase GPU usage (up to 100%)
            double ratio = avg_cpu_time / avg_gpu_time;
            int percentage = static_cast<int>(std::min(100.0, 50.0 * ratio));
            return std::min(100, percentage);
        } else {
            // CPU is faster, use less GPU
            double ratio = avg_gpu_time / avg_cpu_time;
            int percentage = static_cast<int>(100.0 / ratio / 2); // Use CPU bias
            return std::max(0, std::min(100, percentage));
        }
    }
    
    // Default to 50% if no valid historical data
    return 50;
}

void PerformanceBasedStrategy::add_historical_metrics(const WorkloadMetrics& metrics) {
    historical_metrics_.push_back(metrics);
    
    // Keep only the most recent 100 metrics to prevent unbounded growth
    if (historical_metrics_.size() > 100) {
        historical_metrics_.erase(historical_metrics_.begin(), 
                                 historical_metrics_.begin() + 50); // Keep 50 most recent
    }
}

// WorkloadBalancer implementation
WorkloadBalancer::WorkloadBalancer(std::shared_ptr<hardware::IDevice> cpu_device,
                                 std::shared_ptr<hardware::IDevice> gpu_device,
                                 std::shared_ptr<IWorkloadDistributionStrategy> strategy)
    : cpu_device_(cpu_device), gpu_device_(gpu_device), strategy_(strategy) {
}

std::vector<float> WorkloadBalancer::process_vectors(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<float>& query,
    std::function<float(const std::vector<float>&, const std::vector<float>&)> similarity_func) {
    
    if (vectors.empty()) {
        return std::vector<float>();
    }
    
    // Prepare workload metrics
    WorkloadMetrics metrics;
    metrics.num_vectors_processed = vectors.size();
    metrics.gpu_available = gpu_device_ && gpu_device_->is_available();
    metrics.cpu_memory_used_bytes = cpu_device_ ? cpu_device_->get_memory_size() : 0;
    metrics.gpu_memory_used_bytes = gpu_device_ ? gpu_device_->get_memory_size() : 0;
    
    // Determine how much work should go to GPU using the strategy
    int gpu_percentage = strategy_->get_gpu_work_percentage(
        metrics, vectors.size(), 
        vectors.empty() ? 0 : vectors[0].size());
    
    if (gpu_percentage > 0 && gpu_device_ && gpu_device_->is_available()) {
        // Split workload based on strategy
        size_t gpu_count = (vectors.size() * gpu_percentage) / 100;
        size_t cpu_count = vectors.size() - gpu_count;
        
        std::vector<float> results;
        results.reserve(vectors.size());
        
        // Process GPU portion (for now, just use CPU implementation since we don't have actual GPU kernels)
        // In a real implementation, this would call GPU-optimized kernels
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < gpu_count; ++i) {
            results.push_back(similarity_func(query, vectors[i]));
        }
        auto gpu_end_time = std::chrono::high_resolution_clock::now();
        
        // Process CPU portion
        for (size_t i = gpu_count; i < vectors.size(); ++i) {
            results.push_back(similarity_func(query, vectors[i]));
        }
        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        
        // Update metrics
        metrics.gpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end_time - start_time).count();
        metrics.cpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - gpu_end_time).count();
        
        update_metrics(metrics);
        
        return results;
    } else {
        // Pure CPU implementation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<float> results;
        results.reserve(vectors.size());
        
        for (const auto& vec : vectors) {
            results.push_back(similarity_func(query, vec));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        WorkloadMetrics metrics;
        metrics.cpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        metrics.gpu_processing_time_ms = 0;
        metrics.num_vectors_processed = vectors.size();
        metrics.gpu_available = false;
        
        update_metrics(metrics);
        return results;
    }
}

std::vector<float> WorkloadBalancer::process_batch_operation(
    std::function<float(const std::vector<float>&, const std::vector<float>&)> operation_func,
    const std::vector<std::vector<float>>& vectors_a,
    const std::vector<std::vector<float>>& vectors_b) {
    
    if (vectors_a.empty() || vectors_b.empty()) {
        return std::vector<float>();
    }
    
    // For batch operations, we need to determine how to distribute work
    // This example assumes vectors_a and vectors_b are aligned for pairwise operations
    size_t min_size = std::min(vectors_a.size(), vectors_b.size());
    
    WorkloadMetrics metrics;
    metrics.num_vectors_processed = min_size;
    metrics.gpu_available = gpu_device_ && gpu_device_->is_available();
    metrics.cpu_memory_used_bytes = cpu_device_ ? cpu_device_->get_memory_size() : 0;
    metrics.gpu_memory_used_bytes = gpu_device_ ? gpu_device_->get_memory_size() : 0;
    
    // Determine how much work should go to GPU using the strategy
    int gpu_percentage = strategy_->get_gpu_work_percentage(
        metrics, min_size, 
        min_size > 0 ? vectors_a[0].size() : 0);
    
    if (gpu_percentage > 0 && gpu_device_ && gpu_device_->is_available()) {
        // Split workload based on strategy
        size_t gpu_count = (min_size * gpu_percentage) / 100;
        size_t cpu_count = min_size - gpu_count;
        
        std::vector<float> results;
        results.reserve(min_size);
        
        // Process GPU portion (for now, just use CPU implementation)
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < gpu_count; ++i) {
            results.push_back(operation_func(vectors_a[i], vectors_b[i]));
        }
        auto gpu_end_time = std::chrono::high_resolution_clock::now();
        
        // Process CPU portion
        for (size_t i = gpu_count; i < min_size; ++i) {
            results.push_back(operation_func(vectors_a[i], vectors_b[i]));
        }
        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        
        // Update metrics
        metrics.gpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end_time - start_time).count();
        metrics.cpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - gpu_end_time).count();
        
        update_metrics(metrics);
        
        return results;
    } else {
        // Pure CPU implementation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<float> results;
        results.reserve(min_size);
        
        for (size_t i = 0; i < min_size; ++i) {
            results.push_back(operation_func(vectors_a[i], vectors_b[i]));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        WorkloadMetrics metrics;
        metrics.cpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        metrics.gpu_processing_time_ms = 0;
        metrics.num_vectors_processed = min_size;
        metrics.gpu_available = false;
        
        update_metrics(metrics);
        return results;
    }
}

float WorkloadBalancer::process_single_operation(
    std::function<float(const std::vector<float>&, const std::vector<float>&)> operation_func,
    const std::vector<float>& vector_a,
    const std::vector<float>& vector_b) {
    
    // For single operations, decide based on vector size and complexity
    WorkloadMetrics metrics;
    metrics.num_vectors_processed = 1;
    metrics.gpu_available = gpu_device_ && gpu_device_->is_available();
    
    // For single operations, it's often not worth the overhead of GPU for small vectors
    if (vector_a.size() < 1000 || !metrics.gpu_available) {
        // Use CPU for small vectors or when GPU not available
        auto start_time = std::chrono::high_resolution_clock::now();
        float result = operation_func(vector_a, vector_b);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        metrics.cpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        metrics.gpu_processing_time_ms = 0;
        
        update_metrics(metrics);
        return result;
    } else {
        // Use GPU for larger vectors
        auto start_time = std::chrono::high_resolution_clock::now();
        float result = operation_func(vector_a, vector_b);  // Placeholder - would use GPU in real impl
        auto end_time = std::chrono::high_resolution_clock::now();
        
        metrics.gpu_processing_time_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        metrics.cpu_processing_time_ms = 0;
        
        update_metrics(metrics);
        return result;
    }
}

void WorkloadBalancer::update_metrics(const WorkloadMetrics& metrics) {
    recent_metrics_.push_back(metrics);
    
    // Keep only recent metrics to prevent unbounded growth
    if (recent_metrics_.size() > 50) {
        recent_metrics_.erase(recent_metrics_.begin(), recent_metrics_.begin() + 10);
    }
}

std::shared_ptr<IWorkloadDistributionStrategy> WorkloadBalancer::get_strategy() const {
    return strategy_;
}

void WorkloadBalancer::set_strategy(std::shared_ptr<IWorkloadDistributionStrategy> strategy) {
    strategy_ = strategy;
}

} // namespace workload
} // namespace jadevectordb