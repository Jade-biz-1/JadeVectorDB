#include <gtest/gtest.h>
#include "../src/analytics/analytics_engine.h"
#include "../src/analytics/query_logger.h"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace jadedb::analytics;

class AnalyticsEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_path_ = "/tmp/test_analytics_engine.db";
        // Clean up any existing test database
        std::filesystem::remove(test_db_path_);

        // Create and populate test database
        QueryLoggerConfig config;
        config.database_path = test_db_path_;
        config.enable_async = false;

        logger_ = std::make_unique<QueryLogger>("test_db", config);
        ASSERT_TRUE(logger_->initialize().has_value());

        // Insert test data
        populate_test_data();
    }

    void TearDown() override {
        logger_.reset();
        // Clean up test database
        std::filesystem::remove(test_db_path_);
        std::filesystem::remove(test_db_path_ + "-shm");
        std::filesystem::remove(test_db_path_ + "-wal");
    }

    void populate_test_data() {
        int64_t base_time = QueryLogger::get_current_timestamp_ms() - 3600000;  // 1 hour ago

        // Add successful queries
        for (int i = 0; i < 10; i++) {
            QueryLogEntry entry;
            entry.query_id = QueryLogger::generate_query_id();
            entry.database_id = "test_db";
            entry.query_text = "search query " + std::to_string(i % 3);  // Create patterns
            entry.query_type = "vector";
            entry.retrieval_time_ms = 50 + (i * 10);
            entry.total_time_ms = 100 + (i * 20);
            entry.num_results = 5;
            entry.avg_similarity_score = 0.85;
            entry.min_similarity_score = 0.70;
            entry.max_similarity_score = 0.95;
            entry.user_id = "user" + std::to_string(i % 3);
            entry.session_id = "session" + std::to_string(i % 2);
            entry.timestamp = base_time + (i * 360000);  // 6 minutes apart
            entry.has_error = false;
            logger_->log_query_sync(entry);
        }

        // Add failed queries
        for (int i = 0; i < 2; i++) {
            QueryLogEntry entry;
            entry.query_id = QueryLogger::generate_query_id();
            entry.database_id = "test_db";
            entry.query_text = "failed query";
            entry.query_type = "vector";
            entry.timestamp = base_time + (i * 600000);
            entry.has_error = true;
            entry.error_message = "Test error";
            logger_->log_query_sync(entry);
        }

        // Add zero-result queries
        for (int i = 0; i < 3; i++) {
            QueryLogEntry entry;
            entry.query_id = QueryLogger::generate_query_id();
            entry.database_id = "test_db";
            entry.query_text = "no results query";
            entry.query_type = "vector";
            entry.retrieval_time_ms = 30;
            entry.total_time_ms = 50;
            entry.num_results = 0;
            entry.timestamp = base_time + (i * 900000);
            entry.has_error = false;
            logger_->log_query_sync(entry);
        }

        // Add slow queries
        for (int i = 0; i < 2; i++) {
            QueryLogEntry entry;
            entry.query_id = QueryLogger::generate_query_id();
            entry.database_id = "test_db";
            entry.query_text = "slow query";
            entry.query_type = "vector";
            entry.retrieval_time_ms = 1000;
            entry.total_time_ms = 2000;
            entry.num_results = 3;
            entry.avg_similarity_score = 0.75;
            entry.timestamp = base_time + (i * 1200000);
            entry.has_error = false;
            logger_->log_query_sync(entry);
        }
    }

    std::string test_db_path_;
    std::unique_ptr<QueryLogger> logger_;
};

// Test 1: Initialization
TEST_F(AnalyticsEngineTest, Initialization) {
    AnalyticsEngine engine(test_db_path_);
    auto result = engine.initialize();

    ASSERT_TRUE(result.has_value());
}

// Test 2: Compute statistics
TEST_F(AnalyticsEngineTest, ComputeStatistics) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;  // 2 hours ago
    int64_t end = now;

    auto result = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);

    ASSERT_TRUE(result.has_value());
    const auto& stats = result.value();

    EXPECT_FALSE(stats.empty());
    if (!stats.empty()) {
        size_t total_queries = 0;
        for (const auto& stat : stats) {
            total_queries += stat.total_queries;
            EXPECT_EQ(stat.database_id, "test_db");
        }
        EXPECT_EQ(total_queries, 17);  // 10 + 2 + 3 + 2
    }
}

// Test 3: Statistics with failed queries
TEST_F(AnalyticsEngineTest, StatisticsWithErrors) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);

    ASSERT_TRUE(result.has_value());
    const auto& stats = result.value();

    size_t total_failed = 0;
    size_t total_successful = 0;
    for (const auto& stat : stats) {
        total_failed += stat.failed_queries;
        total_successful += stat.successful_queries;
    }

    EXPECT_EQ(total_failed, 2);
    EXPECT_EQ(total_successful, 15);  // 10 + 3 + 2
}

// Test 4: Identify patterns
TEST_F(AnalyticsEngineTest, IdentifyPatterns) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.identify_patterns("test_db", start, end, 2, 10);

    ASSERT_TRUE(result.has_value());
    const auto& patterns = result.value();

    EXPECT_FALSE(patterns.empty());
    // Should find patterns for "search query" variants and other repeated queries
    EXPECT_GE(patterns.size(), 1);
}

// Test 5: Detect slow queries
TEST_F(AnalyticsEngineTest, DetectSlowQueries) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.detect_slow_queries("test_db", start, end, 1000, 10);

    ASSERT_TRUE(result.has_value());
    const auto& slow_queries = result.value();

    EXPECT_EQ(slow_queries.size(), 2);  // We inserted 2 slow queries
    for (const auto& query : slow_queries) {
        EXPECT_GE(query.total_time_ms, 1000);
    }
}

// Test 6: Analyze zero-result queries
TEST_F(AnalyticsEngineTest, AnalyzeZeroResults) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.analyze_zero_results("test_db", start, end, 1, 10);

    ASSERT_TRUE(result.has_value());
    const auto& zero_queries = result.value();

    EXPECT_EQ(zero_queries.size(), 1);  // Should group the 3 identical queries into 1
    if (!zero_queries.empty()) {
        EXPECT_EQ(zero_queries[0].occurrence_count, 3);
    }
}

// Test 7: Detect trending queries
TEST_F(AnalyticsEngineTest, DetectTrending) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t current_start = now - 1800000;  // Last 30 minutes
    int64_t current_end = now;

    auto result = engine.detect_trending("test_db", current_start, current_end, TimeBucket::HOURLY, 0.0, 10);

    ASSERT_TRUE(result.has_value());
    // Trending detection may or may not find trends depending on time distribution
    // Just verify it doesn't crash
}

// Test 8: Generate insights
TEST_F(AnalyticsEngineTest, GenerateInsights) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.generate_insights("test_db", start, end);

    ASSERT_TRUE(result.has_value());
    const auto& insights = result.value();

    // Verify insights structure
    EXPECT_FALSE(insights.top_patterns.empty());
    EXPECT_FALSE(insights.slow_queries.empty());
    EXPECT_FALSE(insights.zero_result_queries.empty());
    EXPECT_GT(insights.overall_success_rate, 0.0);
    EXPECT_LE(insights.overall_success_rate, 100.0);
}

// Test 9: Normalize query text
TEST_F(AnalyticsEngineTest, NormalizeQueryText) {
    EXPECT_EQ(AnalyticsEngine::normalize_query_text("The quick brown fox"), "quick brown fox");
    EXPECT_EQ(AnalyticsEngine::normalize_query_text("UPPERCASE QUERY"), "uppercase query");
    EXPECT_EQ(AnalyticsEngine::normalize_query_text("Query with, punctuation!"), "query punctuation");
    EXPECT_EQ(AnalyticsEngine::normalize_query_text("  extra   spaces  "), "extra spaces");
    EXPECT_EQ(AnalyticsEngine::normalize_query_text("a the and or"), "");  // All stop words
}

// Test 10: Empty time range
TEST_F(AnalyticsEngineTest, EmptyTimeRange) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now + 3600000;  // 1 hour in future
    int64_t end = now + 7200000;    // 2 hours in future

    auto result = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);

    ASSERT_TRUE(result.has_value());
    const auto& stats = result.value();
    EXPECT_TRUE(stats.empty());  // No data in future
}

// Test 11: Different time buckets
TEST_F(AnalyticsEngineTest, DifferentTimeBuckets) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    // Test hourly
    auto hourly = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);
    ASSERT_TRUE(hourly.has_value());

    // Test daily
    auto daily = engine.compute_statistics("test_db", start, end, TimeBucket::DAILY);
    ASSERT_TRUE(daily.has_value());

    // Hourly should have more buckets than daily for same period
    EXPECT_GE(hourly.value().size(), daily.value().size());
}

// Test 12: Pattern minimum count filter
TEST_F(AnalyticsEngineTest, PatternMinimumCount) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    // With min_count = 1
    auto result1 = engine.identify_patterns("test_db", start, end, 1, 100);
    ASSERT_TRUE(result1.has_value());

    // With min_count = 5
    auto result2 = engine.identify_patterns("test_db", start, end, 5, 100);
    ASSERT_TRUE(result2.has_value());

    // Higher min_count should return fewer or equal patterns
    EXPECT_LE(result2.value().size(), result1.value().size());
}

// Test 13: Latency percentiles
TEST_F(AnalyticsEngineTest, LatencyPercentiles) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);

    ASSERT_TRUE(result.has_value());
    const auto& stats = result.value();

    for (const auto& stat : stats) {
        if (stat.total_queries > 0) {
            // Percentiles should be ordered
            EXPECT_LE(stat.p50_latency_ms, stat.p95_latency_ms);
            EXPECT_LE(stat.p95_latency_ms, stat.p99_latency_ms);
            EXPECT_LE(stat.p99_latency_ms, stat.max_latency_ms);
        }
    }
}

// Test 14: Query type breakdown
TEST_F(AnalyticsEngineTest, QueryTypeBreakdown) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    auto result = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);

    ASSERT_TRUE(result.has_value());
    const auto& stats = result.value();

    size_t total_vector = 0;
    for (const auto& stat : stats) {
        total_vector += stat.vector_queries;
    }

    EXPECT_GT(total_vector, 0);  // We inserted vector queries
}

// Test 15: Concurrent access
TEST_F(AnalyticsEngineTest, ConcurrentAccess) {
    AnalyticsEngine engine(test_db_path_);
    ASSERT_TRUE(engine.initialize().has_value());

    int64_t now = QueryLogger::get_current_timestamp_ms();
    int64_t start = now - 7200000;
    int64_t end = now;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < 3; i++) {
        threads.emplace_back([&engine, start, end, &success_count]() {
            auto result = engine.compute_statistics("test_db", start, end, TimeBucket::HOURLY);
            if (result.has_value()) {
                success_count++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count, 3);  // All threads should succeed
}
