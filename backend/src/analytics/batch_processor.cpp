#include "batch_processor.h"
#include <sqlite3.h>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace jadedb {
namespace analytics {

BatchProcessor::BatchProcessor(
    const std::string& database_id,
    std::shared_ptr<AnalyticsEngine> analytics_engine,
    const BatchProcessorConfig& config
)
    : database_id_(database_id),
      analytics_engine_(analytics_engine),
      config_(config),
      running_(false),
      logger_(jadevectordb::logging::LoggerManager::get_logger("BatchProcessor"))
{
}

BatchProcessor::~BatchProcessor() {
    stop();
}

jadevectordb::Result<void> BatchProcessor::start() {
    if (running_.load()) {
        return jadevectordb::Result<void>{};
    }

    if (!analytics_engine_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics engine not provided"
        ));
    }

    running_.store(true);
    scheduler_thread_ = std::thread(&BatchProcessor::scheduler_thread_func, this);

    logger_->info("BatchProcessor started for database: " + database_id_);
    return jadevectordb::Result<void>{};
}

void BatchProcessor::stop() {
    if (!running_.load()) {
        return;
    }

    logger_->info("Stopping BatchProcessor...");

    running_.store(false);
    cv_.notify_all();

    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }

    logger_->info("BatchProcessor stopped");
}

bool BatchProcessor::is_running() const {
    return running_.load();
}

jadevectordb::Result<JobResult> BatchProcessor::run_aggregation_now() {
    if (!analytics_engine_) {
        JobResult result;
        result.job_name = "hourly_aggregation";
        result.success = false;
        result.error_message = "Analytics engine not available";
        return result;
    }

    auto result = run_hourly_aggregation();
    update_statistics(result);
    return result;
}

jadevectordb::Result<JobResult> BatchProcessor::run_cleanup_now() {
    auto result = run_daily_cleanup();
    update_statistics(result);
    return result;
}

BatchProcessorStats BatchProcessor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void BatchProcessor::register_job(
    const std::string& job_name,
    int interval_seconds,
    std::function<JobResult()> job_function
) {
    std::lock_guard<std::mutex> lock(mutex_);

    CustomJob job;
    job.name = job_name;
    job.interval_seconds = interval_seconds;
    job.last_run = 0;
    job.function = job_function;

    custom_jobs_.push_back(job);
    logger_->info("Registered custom job: " + job_name);
}

void BatchProcessor::scheduler_thread_func() {
    logger_->info("Scheduler thread started");

    while (running_.load()) {
        // Check if it's time to run aggregation
        if (config_.enable_hourly_aggregation && should_run_hourly_aggregation()) {
            logger_->info("Running hourly aggregation");
            auto result = run_hourly_aggregation();
            update_statistics(result);

            if (result.success) {
                logger_->info("Hourly aggregation completed: " + result.details);
            } else {
                logger_->error("Hourly aggregation failed: " + result.error_message);
            }
        }

        // Check if it's time to run cleanup
        if (config_.enable_daily_cleanup && should_run_daily_cleanup()) {
            logger_->info("Running daily cleanup");
            auto result = run_daily_cleanup();
            update_statistics(result);

            if (result.success) {
                logger_->info("Daily cleanup completed: " + result.details);
            } else {
                logger_->error("Daily cleanup failed: " + result.error_message);
            }
        }

        // Check custom jobs
        {
            std::lock_guard<std::mutex> lock(mutex_);
            int64_t now = get_current_time_ms();

            for (auto& job : custom_jobs_) {
                if (job.interval_seconds == 0 && job.last_run > 0) {
                    continue;  // One-time job already run
                }

                int64_t interval_ms = static_cast<int64_t>(job.interval_seconds) * 1000;
                if (job.last_run == 0 || (now - job.last_run) >= interval_ms) {
                    logger_->info("Running custom job: " + job.name);
                    auto result = job.function();
                    job.last_run = now;
                    update_statistics(result);

                    if (result.success) {
                        logger_->info("Custom job completed: " + job.name);
                    } else {
                        logger_->error("Custom job failed: " + job.name + " - " + result.error_message);
                    }
                }
            }
        }

        // Sleep until next check
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::seconds(config_.check_interval_seconds),
            [this]() { return !running_.load(); });
    }

    logger_->info("Scheduler thread stopped");
}

JobResult BatchProcessor::run_hourly_aggregation() {
    JobResult result;
    result.job_name = "hourly_aggregation";
    result.start_time = get_current_time_ms();

    try {
        // Calculate time range for last hour
        int64_t end_time = get_current_time_ms();
        int64_t start_time = end_time - 3600000;  // 1 hour ago

        // Compute statistics
        auto stats_result = analytics_engine_->compute_statistics(
            database_id_,
            start_time,
            end_time,
            TimeBucket::HOURLY
        );

        if (!stats_result.has_value()) {
            result.success = false;
            result.error_message = stats_result.error().message;
            result.end_time = get_current_time_ms();
            result.duration_ms = result.end_time - result.start_time;
            return result;
        }

        const auto& stats = stats_result.value();

        // Store statistics in query_stats table (implementation would go here)
        // For now, we'll just log the summary
        std::ostringstream details;
        details << "Processed " << stats.size() << " time buckets";

        if (!stats.empty()) {
            size_t total_queries = 0;
            for (const auto& stat : stats) {
                total_queries += stat.total_queries;
            }
            details << ", " << total_queries << " queries";
        }

        result.success = true;
        result.details = details.str();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.last_aggregation_time = get_current_time_ms();
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception: ") + e.what();
    }

    result.end_time = get_current_time_ms();
    result.duration_ms = result.end_time - result.start_time;
    return result;
}

JobResult BatchProcessor::run_daily_cleanup() {
    JobResult result;
    result.job_name = "daily_cleanup";
    result.start_time = get_current_time_ms();

    try {
        // Calculate cutoff time
        int64_t retention_ms = static_cast<int64_t>(config_.retention_days) * 86400000;
        int64_t cutoff_time = get_current_time_ms() - retention_ms;

        // Open database to execute cleanup
        // Note: We need the database path from somewhere - for now we'll skip actual deletion
        // In production, this would connect and run DELETE queries

        std::ostringstream details;
        details << "Would delete logs older than " << config_.retention_days << " days";
        details << " (cutoff: " << cutoff_time << ")";

        result.success = true;
        result.details = details.str();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.last_cleanup_time = get_current_time_ms();
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception: ") + e.what();
    }

    result.end_time = get_current_time_ms();
    result.duration_ms = result.end_time - result.start_time;
    return result;
}

bool BatchProcessor::should_run_hourly_aggregation() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Check if we've already run in the last hour
    int64_t now = get_current_time_ms();
    if (stats_.last_aggregation_time > 0) {
        int64_t since_last = now - stats_.last_aggregation_time;
        if (since_last < 3600000) {  // Less than 1 hour
            return false;
        }
    }

    // Check if it's the right minute of the hour
    auto [hour, minute] = get_current_hour_minute();
    return minute == config_.hourly_aggregation_minute;
}

bool BatchProcessor::should_run_daily_cleanup() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Check if we've already run in the last 24 hours
    int64_t now = get_current_time_ms();
    if (stats_.last_cleanup_time > 0) {
        int64_t since_last = now - stats_.last_cleanup_time;
        if (since_last < 86400000) {  // Less than 24 hours
            return false;
        }
    }

    // Check if it's the right hour and early minute (0-5)
    auto [hour, minute] = get_current_hour_minute();
    return hour == config_.daily_cleanup_hour && minute < 5;
}

void BatchProcessor::update_statistics(const JobResult& result) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_jobs_run++;
    if (result.success) {
        stats_.successful_jobs++;
    } else {
        stats_.failed_jobs++;
    }

    // Update average durations
    if (result.job_name == "hourly_aggregation") {
        if (stats_.avg_aggregation_duration_ms == 0.0) {
            stats_.avg_aggregation_duration_ms = result.duration_ms;
        } else {
            stats_.avg_aggregation_duration_ms =
                (stats_.avg_aggregation_duration_ms + result.duration_ms) / 2.0;
        }
    } else if (result.job_name == "daily_cleanup") {
        if (stats_.avg_cleanup_duration_ms == 0.0) {
            stats_.avg_cleanup_duration_ms = result.duration_ms;
        } else {
            stats_.avg_cleanup_duration_ms =
                (stats_.avg_cleanup_duration_ms + result.duration_ms) / 2.0;
        }
    }
}

int64_t BatchProcessor::get_current_time_ms() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

std::pair<int, int> BatchProcessor::get_current_hour_minute() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    return {local_time->tm_hour, local_time->tm_min};
}

} // namespace analytics
} // namespace jadedb
