#include <gtest/gtest.h>
#include "../src/analytics/batch_processor.h"
#include "../src/analytics/query_logger.h"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace jadedb::analytics;

class BatchProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_path_ = "/tmp/test_batch_processor.db";
        // Clean up any existing test database
        std::filesystem::remove(test_db_path_);

        // Create analytics engine
        analytics_engine_ = std::make_shared<AnalyticsEngine>(test_db_path_);
        ASSERT_TRUE(analytics_engine_->initialize().has_value());

        // Populate test data
        populate_test_data();
    }

    void TearDown() override {
        analytics_engine_.reset();
        // Clean up test database
        std::filesystem::remove(test_db_path_);
        std::filesystem::remove(test_db_path_ + "-shm");
        std::filesystem::remove(test_db_path_ + "-wal");
    }

    void populate_test_data() {
        QueryLoggerConfig config;
        config.database_path = test_db_path_;
        config.enable_async = false;

        auto logger = std::make_unique<QueryLogger>("test_db", config);
        ASSERT_TRUE(logger->initialize().has_value());

        int64_t base_time = QueryLogger::get_current_timestamp_ms() - 3600000;

        for (int i = 0; i < 10; i++) {
            QueryLogEntry entry;
            entry.query_id = QueryLogger::generate_query_id();
            entry.database_id = "test_db";
            entry.query_text = "test query " + std::to_string(i);
            entry.query_type = "vector";
            entry.retrieval_time_ms = 50;
            entry.total_time_ms = 100;
            entry.num_results = 5;
            entry.avg_similarity_score = 0.85;
            entry.timestamp = base_time + (i * 360000);
            logger->log_query_sync(entry);
        }
    }

    std::string test_db_path_;
    std::shared_ptr<AnalyticsEngine> analytics_engine_;
};

// Test 1: Initialization
TEST_F(BatchProcessorTest, Initialization) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    auto result = processor.start();

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(processor.is_running());

    processor.stop();
    EXPECT_FALSE(processor.is_running());
}

// Test 2: Run aggregation immediately
TEST_F(BatchProcessorTest, RunAggregationNow) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    auto result = processor.run_aggregation_now();

    ASSERT_TRUE(result.has_value());
    const auto& job_result = result.value();

    EXPECT_EQ(job_result.job_name, "hourly_aggregation");
    EXPECT_TRUE(job_result.success);
    EXPECT_GE(job_result.duration_ms, 0);  // Can be 0 if very fast
    EXPECT_FALSE(job_result.details.empty());

    processor.stop();
}

// Test 3: Run cleanup immediately
TEST_F(BatchProcessorTest, RunCleanupNow) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    auto result = processor.run_cleanup_now();

    ASSERT_TRUE(result.has_value());
    const auto& job_result = result.value();

    EXPECT_EQ(job_result.job_name, "daily_cleanup");
    EXPECT_TRUE(job_result.success);
    EXPECT_GE(job_result.duration_ms, 0);  // Can be 0 if very fast
    EXPECT_FALSE(job_result.details.empty());

    processor.stop();
}

// Test 4: Statistics tracking
TEST_F(BatchProcessorTest, StatisticsTracking) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    auto stats1 = processor.get_statistics();
    EXPECT_EQ(stats1.total_jobs_run, 0);

    processor.run_aggregation_now();
    processor.run_cleanup_now();

    auto stats2 = processor.get_statistics();
    EXPECT_EQ(stats2.total_jobs_run, 2);
    EXPECT_EQ(stats2.successful_jobs, 2);
    EXPECT_EQ(stats2.failed_jobs, 0);
    EXPECT_GE(stats2.avg_aggregation_duration_ms, 0.0);  // Can be 0 if very fast
    EXPECT_GE(stats2.avg_cleanup_duration_ms, 0.0);  // Can be 0 if very fast

    processor.stop();
}

// Test 5: Multiple aggregations
TEST_F(BatchProcessorTest, MultipleAggregations) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    for (int i = 0; i < 3; i++) {
        auto result = processor.run_aggregation_now();
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result.value().success);
    }

    auto stats = processor.get_statistics();
    EXPECT_EQ(stats.total_jobs_run, 3);
    EXPECT_EQ(stats.successful_jobs, 3);

    processor.stop();
}

// Test 6: Custom job registration
TEST_F(BatchProcessorTest, CustomJobRegistration) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;
    config.check_interval_seconds = 1;  // Check every second

    BatchProcessor processor("test_db", analytics_engine_, config);

    std::atomic<int> custom_job_count{0};

    // Register a custom job that runs every 2 seconds
    processor.register_job("test_job", 2, [&custom_job_count]() {
        JobResult result;
        result.job_name = "test_job";
        result.success = true;
        result.start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        result.end_time = result.start_time;
        result.duration_ms = 0;
        custom_job_count++;
        return result;
    });

    ASSERT_TRUE(processor.start().has_value());

    // Wait for job to run at least once
    std::this_thread::sleep_for(std::chrono::seconds(3));

    processor.stop();

    EXPECT_GT(custom_job_count.load(), 0);
}

// Test 7: One-time custom job
TEST_F(BatchProcessorTest, OneTimeCustomJob) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;
    config.check_interval_seconds = 1;

    BatchProcessor processor("test_db", analytics_engine_, config);

    std::atomic<int> custom_job_count{0};

    // Register a one-time job (interval = 0)
    processor.register_job("one_time_job", 0, [&custom_job_count]() {
        JobResult result;
        result.job_name = "one_time_job";
        result.success = true;
        result.start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        result.end_time = result.start_time;
        result.duration_ms = 0;
        custom_job_count++;
        return result;
    });

    ASSERT_TRUE(processor.start().has_value());

    // Wait for job to potentially run multiple times
    std::this_thread::sleep_for(std::chrono::seconds(3));

    processor.stop();

    // Should only run once
    EXPECT_EQ(custom_job_count.load(), 1);
}

// Test 8: Aggregation with no data
TEST_F(BatchProcessorTest, AggregationWithNoData) {
    // Create a new database with no data
    std::string empty_db = "/tmp/test_empty_batch.db";
    std::filesystem::remove(empty_db);

    // Create logger to initialize schema
    QueryLoggerConfig qconfig;
    qconfig.database_path = empty_db;
    qconfig.enable_async = false;
    auto logger = std::make_unique<QueryLogger>("empty_db", qconfig);
    ASSERT_TRUE(logger->initialize().has_value());
    logger.reset();  // Close logger

    auto empty_engine = std::make_shared<AnalyticsEngine>(empty_db);
    ASSERT_TRUE(empty_engine->initialize().has_value());

    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("empty_db", empty_engine, config);
    ASSERT_TRUE(processor.start().has_value());

    auto result = processor.run_aggregation_now();

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().success);

    processor.stop();

    // Cleanup
    std::filesystem::remove(empty_db);
    std::filesystem::remove(empty_db + "-shm");
    std::filesystem::remove(empty_db + "-wal");
}

// Test 9: Stop without start
TEST_F(BatchProcessorTest, StopWithoutStart) {
    BatchProcessorConfig config;
    BatchProcessor processor("test_db", analytics_engine_, config);

    // Should not crash
    processor.stop();
    EXPECT_FALSE(processor.is_running());
}

// Test 10: Multiple start/stop cycles
TEST_F(BatchProcessorTest, MultipleStartStopCycles) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);

    for (int i = 0; i < 3; i++) {
        ASSERT_TRUE(processor.start().has_value());
        EXPECT_TRUE(processor.is_running());

        processor.stop();
        EXPECT_FALSE(processor.is_running());
    }
}

// Test 11: Job timing
TEST_F(BatchProcessorTest, JobTiming) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    auto result = processor.run_aggregation_now();

    ASSERT_TRUE(result.has_value());
    const auto& job_result = result.value();

    EXPECT_GT(job_result.start_time, 0);
    EXPECT_GT(job_result.end_time, 0);
    EXPECT_GE(job_result.end_time, job_result.start_time);
    EXPECT_EQ(job_result.duration_ms, job_result.end_time - job_result.start_time);

    processor.stop();
}

// Test 12: Statistics after failures
TEST_F(BatchProcessorTest, StatisticsAfterFailures) {
    // Create processor with null analytics engine to force failures
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", nullptr, config);

    // Start should fail
    auto start_result = processor.start();
    EXPECT_FALSE(start_result.has_value());
}

// Test 13: Cleanup retention days
TEST_F(BatchProcessorTest, CleanupRetentionDays) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;
    config.retention_days = 7;  // 7 days

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    auto result = processor.run_cleanup_now();

    ASSERT_TRUE(result.has_value());
    const auto& job_result = result.value();

    EXPECT_TRUE(job_result.success);
    EXPECT_TRUE(job_result.details.find("7 days") != std::string::npos);

    processor.stop();
}

// Test 14: Concurrent operations
TEST_F(BatchProcessorTest, ConcurrentOperations) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    // Run aggregation and cleanup concurrently
    std::thread t1([&processor]() {
        processor.run_aggregation_now();
    });

    std::thread t2([&processor]() {
        processor.run_cleanup_now();
    });

    t1.join();
    t2.join();

    auto stats = processor.get_statistics();
    EXPECT_EQ(stats.total_jobs_run, 2);

    processor.stop();
}

// Test 15: Last run timestamps
TEST_F(BatchProcessorTest, LastRunTimestamps) {
    BatchProcessorConfig config;
    config.enable_hourly_aggregation = false;
    config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine_, config);
    ASSERT_TRUE(processor.start().has_value());

    auto stats1 = processor.get_statistics();
    EXPECT_EQ(stats1.last_aggregation_time, 0);
    EXPECT_EQ(stats1.last_cleanup_time, 0);

    processor.run_aggregation_now();
    auto stats2 = processor.get_statistics();
    EXPECT_GT(stats2.last_aggregation_time, 0);

    processor.run_cleanup_now();
    auto stats3 = processor.get_statistics();
    EXPECT_GT(stats3.last_cleanup_time, 0);

    processor.stop();
}
