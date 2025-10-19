#ifndef JADEVECTORDB_WORKLOAD_BALANCER_H
#define JADEVECTORDB_WORKLOAD_BALANCER_H

#include "gpu_acceleration.h"
#include <vector>
#include <memory>
#include <functional>

namespace jadevectordb {
namespace workload {

    struct WorkloadMetrics {
        size_t cpu_processing_time_ms;
        size_t gpu_processing_time_ms;
        size_t cpu_memory_used_bytes;
        size_t gpu_memory_used_bytes;
        size_t num_vectors_processed;
        bool gpu_available;
    };

    /**
     * @brief Interface for workload distribution strategies
     * 
     * Different strategies for distributing work between CPU and GPU
     */
    class IWorkloadDistributionStrategy {
    public:
        virtual ~IWorkloadDistributionStrategy() = default;
        
        /**
         * @brief Determine what percentage of work should go to GPU
         * @param metrics Current system metrics
         * @param vector_count Number of vectors to process
         * @param vector_dimension Dimension of each vector
         * @return Percentage of work (0-100) to assign to GPU
         */
        virtual int get_gpu_work_percentage(const WorkloadMetrics& metrics, 
                                          size_t vector_count, 
                                          size_t vector_dimension) const = 0;
    };

    /**
     * @brief Simple strategy that switches to GPU when available and input is large enough
     */
    class SimpleThresholdStrategy : public IWorkloadDistributionStrategy {
    private:
        size_t threshold_vectors_;  // Minimum number of vectors to use GPU
        
    public:
        explicit SimpleThresholdStrategy(size_t threshold = 1000);
        int get_gpu_work_percentage(const WorkloadMetrics& metrics, 
                                  size_t vector_count, 
                                  size_t vector_dimension) const override;
    };

    /**
     * @brief Performance-based strategy that learns from past performance
     */
    class PerformanceBasedStrategy : public IWorkloadDistributionStrategy {
    private:
        std::vector<WorkloadMetrics> historical_metrics_;
        
    public:
        int get_gpu_work_percentage(const WorkloadMetrics& metrics, 
                                  size_t vector_count, 
                                  size_t vector_dimension) const override;
        
        void add_historical_metrics(const WorkloadMetrics& metrics);
    };

    /**
     * @brief Manages workload distribution between CPU and GPU
     * 
     * This class determines how to distribute computational work between
     * CPU and GPU based on various factors like vector size, available resources, etc.
     */
    class WorkloadBalancer {
    private:
        std::shared_ptr<hardware::IDevice> cpu_device_;
        std::shared_ptr<hardware::IDevice> gpu_device_;
        std::shared_ptr<IWorkloadDistributionStrategy> strategy_;
        std::vector<WorkloadMetrics> recent_metrics_;
        
    public:
        WorkloadBalancer(std::shared_ptr<hardware::IDevice> cpu_device,
                        std::shared_ptr<hardware::IDevice> gpu_device,
                        std::shared_ptr<IWorkloadDistributionStrategy> strategy);
        
        /**
         * @brief Process vectors using optimal CPU/GPU distribution
         * @param vectors Input vectors to process
         * @param query Query vector for similarity computation
         * @param similarity_func Function to compute similarity
         * @return Vector of similarity scores
         */
        std::vector<float> process_vectors(
            const std::vector<std::vector<float>>& vectors,
            const std::vector<float>& query,
            std::function<float(const std::vector<float>&, const std::vector<float>&)> similarity_func);
        
        /**
         * @brief Process a batch of vector operations using hybrid approach
         * @param operation_func Function to perform the operation
         * @param vectors_a First set of vectors
         * @param vectors_b Second set of vectors (for operations like similarity)
         * @return Vector of results
         */
        std::vector<float> process_batch_operation(
            std::function<float(const std::vector<float>&, const std::vector<float>&)> operation_func,
            const std::vector<std::vector<float>>& vectors_a,
            const std::vector<std::vector<float>>& vectors_b);
        
        /**
         * @brief Process a single vector operation using optimal device
         * @param operation_func Function to perform the operation
         * @param vector_a First vector
         * @param vector_b Second vector
         * @return Result of the operation
         */
        float process_single_operation(
            std::function<float(const std::vector<float>&, const std::vector<float>&)> operation_func,
            const std::vector<float>& vector_a,
            const std::vector<float>& vector_b);
        
        /**
         * @brief Update workload metrics for future decisions
         * @param metrics Performance metrics from recent operations
         */
        void update_metrics(const WorkloadMetrics& metrics);
        
        /**
         * @brief Get current strategy
         */
        std::shared_ptr<IWorkloadDistributionStrategy> get_strategy() const;
        
        /**
         * @brief Set new strategy
         */
        void set_strategy(std::shared_ptr<IWorkloadDistributionStrategy> strategy);
    };

} // namespace workload
} // namespace jadevectordb

#endif // JADEVECTORDB_WORKLOAD_BALANCER_H