#pragma once

#include "analytics_engine.h"
#include "query_logger.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace jadedb {
namespace analytics {

/**
 * @brief Configuration for batch processing jobs
 */
struct BatchProcessorConfig {
    // Aggregation settings
    bool enable_hourly_aggregation = true;
    int hourly_aggregation_minute = 5;  // Run at :05 of every hour

    // Cleanup settings
    bool enable_daily_cleanup = true;
    int daily_cleanup_hour = 2;  // Run at 2 AM
    int retention_days = 30;  // Keep logs for 30 days

    // Performance settings
    int check_interval_seconds = 60;  // Check schedule every minute

    BatchProcessorConfig() = default;
};

/**
 * @brief Job execution result
 */
struct JobResult {
    std::string job_name;
    bool success;
    std::string error_message;
    int64_t start_time;
    int64_t end_time;
    int64_t duration_ms;
    std::string details;  // Job-specific details

    JobResult() : success(false), start_time(0), end_time(0), duration_ms(0) {}
};

/**
 * @brief Statistics for batch processor
 */
struct BatchProcessorStats {
    size_t total_jobs_run = 0;
    size_t successful_jobs = 0;
    size_t failed_jobs = 0;
    int64_t last_aggregation_time = 0;
    int64_t last_cleanup_time = 0;
    double avg_aggregation_duration_ms = 0.0;
    double avg_cleanup_duration_ms = 0.0;
};

/**
 * @brief Batch processor for periodic analytics jobs
 *
 * Runs scheduled background jobs for:
 * - Hourly aggregation: Computes and stores statistics
 * - Daily cleanup: Purges old logs based on retention policy
 */
class BatchProcessor {
public:
    /**
     * @brief Construct batch processor
     * @param database_id Database identifier
     * @param analytics_engine Analytics engine instance
     * @param config Batch processor configuration
     */
    BatchProcessor(
        const std::string& database_id,
        std::shared_ptr<AnalyticsEngine> analytics_engine,
        const BatchProcessorConfig& config = BatchProcessorConfig()
    );

    ~BatchProcessor();

    // Non-copyable
    BatchProcessor(const BatchProcessor&) = delete;
    BatchProcessor& operator=(const BatchProcessor&) = delete;

    /**
     * @brief Start batch processor
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> start();

    /**
     * @brief Stop batch processor
     */
    void stop();

    /**
     * @brief Check if processor is running
     */
    bool is_running() const;

    /**
     * @brief Run hourly aggregation job immediately (for testing)
     * @return Result<JobResult> Job result
     */
    jadevectordb::Result<JobResult> run_aggregation_now();

    /**
     * @brief Run daily cleanup job immediately (for testing)
     * @return Result<JobResult> Job result
     */
    jadevectordb::Result<JobResult> run_cleanup_now();

    /**
     * @brief Get processor statistics
     */
    BatchProcessorStats get_statistics() const;

    /**
     * @brief Register a custom job
     * @param job_name Job identifier
     * @param interval_seconds How often to run (0 = run once)
     * @param job_function Function to execute
     */
    void register_job(
        const std::string& job_name,
        int interval_seconds,
        std::function<JobResult()> job_function
    );

private:
    std::string database_id_;
    std::shared_ptr<AnalyticsEngine> analytics_engine_;
    BatchProcessorConfig config_;

    // Scheduler thread
    std::thread scheduler_thread_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cv_;

    // Statistics
    mutable std::mutex stats_mutex_;
    BatchProcessorStats stats_;

    // Logger
    std::shared_ptr<jadevectordb::logging::Logger> logger_;

    // Custom jobs
    struct CustomJob {
        std::string name;
        int interval_seconds;
        int64_t last_run;
        std::function<JobResult()> function;
    };
    std::vector<CustomJob> custom_jobs_;

    /**
     * @brief Scheduler thread function
     */
    void scheduler_thread_func();

    /**
     * @brief Run hourly aggregation
     */
    JobResult run_hourly_aggregation();

    /**
     * @brief Run daily cleanup
     */
    JobResult run_daily_cleanup();

    /**
     * @brief Check if it's time to run hourly aggregation
     */
    bool should_run_hourly_aggregation() const;

    /**
     * @brief Check if it's time to run daily cleanup
     */
    bool should_run_daily_cleanup() const;

    /**
     * @brief Update statistics after job completion
     */
    void update_statistics(const JobResult& result);

    /**
     * @brief Get current time as Unix timestamp in milliseconds
     */
    static int64_t get_current_time_ms();

    /**
     * @brief Get current hour and minute
     */
    static std::pair<int, int> get_current_hour_minute();
};

} // namespace analytics
} // namespace jadedb
