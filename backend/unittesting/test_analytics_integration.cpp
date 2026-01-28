/**
 * Integration Tests for Query Analytics System (T16.22)
 *
 * Tests end-to-end analytics flow:
 * - Query logging → Storage → Analysis → Insights
 * - Full pipeline validation
 * - Performance benchmarks
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include "../src/analytics/query_logger.h"
#include "../src/analytics/analytics_engine.h"
#include "../src/analytics/batch_processor.h"
#include "../src/analytics/query_analytics_manager.h"

using namespace jadevectordb::analytics;
namespace fs = std::filesystem;

class AnalyticsIntegrationTest : public ::testing::Test {
protected:
    std::string test_db_path;

    void SetUp() override {
        test_db_path = "/tmp/test_analytics_integration.db";
        // Clean up any existing test database
        if (fs::exists(test_db_path)) {
            fs::remove(test_db_path);
        }
    }

    void TearDown() override {
        // Clean up test database
        if (fs::exists(test_db_path)) {
            fs::remove(test_db_path);
        }
    }
};

// Test 1: End-to-End Analytics Flow
TEST_F(AnalyticsIntegrationTest, EndToEndAnalyticsFlow) {
    // Create logger
    QueryLoggerConfig config;
    config.db_path = test_db_path;
    config.batch_size = 10;
    config.flush_interval_ms = 100;

    QueryLogger logger(config);
    ASSERT_TRUE(logger.initialize());

    // Log some queries
    std::vector<std::string> queries = {
        "search for machine learning tutorials",
        "find python programming examples",
        "search for machine learning algorithms",
        "query database optimization tips",
        "search for machine learning models"
    };

    for (size_t i = 0; i < queries.size(); ++i) {
        QueryLogEntry entry;
        entry.query_id = "q" + std::to_string(i);
        entry.database_id = "test_db";
        entry.query_text = queries[i];
        entry.query_type = "vector";
        entry.retrieval_time_ms = 10 + (i * 5);
        entry.total_time_ms = 15 + (i * 5);
        entry.num_results = 10 - i;
        entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        logger.log_query(entry);
    }

    // Flush to ensure all queries are written
    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Create analytics engine
    AnalyticsEngine engine(test_db_path);
    ASSERT_TRUE(engine.initialize());

    // Compute statistics
    int64_t start_time = 0;
    int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();

    auto stats = engine.compute_statistics(start_time, end_time, TimeBucket::HOURLY);
    EXPECT_GT(stats.size(), 0);

    // Identify patterns
    auto patterns = engine.identify_patterns(2);
    EXPECT_GT(patterns.size(), 0);

    // Check for "machine learning" pattern
    bool found_ml_pattern = false;
    for (const auto& pattern : patterns) {
        if (pattern.normalized_text.find("machine learning") != std::string::npos) {
            found_ml_pattern = true;
            EXPECT_GE(pattern.count, 3); // Should appear 3 times
        }
    }
    EXPECT_TRUE(found_ml_pattern);

    // Generate insights
    auto insights = engine.generate_insights(start_time, end_time);
    EXPECT_GT(insights.summary.total_queries, 0);
    EXPECT_EQ(insights.summary.total_queries, queries.size());

    logger.shutdown();
}

// Test 2: Query Analytics Manager Integration
TEST_F(AnalyticsIntegrationTest, QueryAnalyticsManagerIntegration) {
    QueryLoggerConfig logger_config;
    logger_config.db_path = test_db_path;

    QueryAnalyticsManager manager(logger_config);
    ASSERT_TRUE(manager.initialize());

    // Log different types of queries
    std::vector<double> query_vector = {1.0, 2.0, 3.0};

    // Vector search
    manager.log_vector_search(
        "test_db",
        query_vector,
        {{0.95, "doc1", {}}, {0.85, "doc2", {}}},
        50,
        "user1",
        "session1"
    );

    // Hybrid search
    manager.log_hybrid_search(
        "test_db",
        "test query",
        query_vector,
        {{0.95, "doc1", {}}},
        75,
        0.7,
        "RRF",
        "user1",
        "session1"
    );

    // Re-ranking
    manager.log_reranking(
        "test_db",
        "test query",
        {{0.95, "doc1", {}}},
        25,
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "user1",
        "session1"
    );

    // Error logging
    manager.log_error(
        "test_db",
        "invalid query",
        "vector",
        "Invalid vector dimension",
        "user1",
        "session1"
    );

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Get statistics
    auto stats = manager.get_statistics();
    EXPECT_GT(stats.total_logged, 0);

    manager.shutdown();
}

// Test 3: Batch Processor Integration
TEST_F(AnalyticsIntegrationTest, BatchProcessorIntegration) {
    // Create logger and log some queries
    QueryLoggerConfig logger_config;
    logger_config.db_path = test_db_path;

    QueryLogger logger(logger_config);
    ASSERT_TRUE(logger.initialize());

    // Log queries with timestamps spread over time
    auto now = std::chrono::system_clock::now();
    for (int i = 0; i < 20; ++i) {
        QueryLogEntry entry;
        entry.query_id = "batch_q" + std::to_string(i);
        entry.database_id = "test_db";
        entry.query_text = "test query " + std::to_string(i);
        entry.query_type = "vector";
        entry.retrieval_time_ms = 10;
        entry.total_time_ms = 15;
        entry.num_results = 10;

        // Vary timestamps
        auto time_offset = std::chrono::hours(i);
        entry.timestamp = (now - time_offset).time_since_epoch().count();

        logger.log_query(entry);
    }

    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    logger.shutdown();

    // Create batch processor
    BatchProcessorConfig bp_config;
    bp_config.enable_hourly_aggregation = false; // Disable scheduled jobs
    bp_config.enable_daily_cleanup = false;

    BatchProcessor processor(test_db_path, bp_config);
    ASSERT_TRUE(processor.initialize());

    // Run aggregation manually
    processor.run_aggregation_now();

    // Run cleanup with short retention (should delete old entries)
    processor.run_cleanup_now(1); // 1 day retention

    // Get statistics
    auto stats = processor.get_statistics();
    EXPECT_GT(stats.total_jobs, 0);
    EXPECT_EQ(stats.failed_jobs, 0);

    processor.shutdown();
}

// Test 4: Performance - Logging Overhead
TEST_F(AnalyticsIntegrationTest, PerformanceLoggingOverhead) {
    QueryLoggerConfig config;
    config.db_path = test_db_path;
    config.batch_size = 100;
    config.flush_interval_ms = 1000;

    QueryLogger logger(config);
    ASSERT_TRUE(logger.initialize());

    const int num_queries = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_queries; ++i) {
        QueryLogEntry entry;
        entry.query_id = "perf_q" + std::to_string(i);
        entry.database_id = "test_db";
        entry.query_text = "performance test query";
        entry.query_type = "vector";
        entry.retrieval_time_ms = 10;
        entry.total_time_ms = 15;
        entry.num_results = 10;
        entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        logger.log_query(entry);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double avg_per_query = static_cast<double>(duration_ms) / num_queries;

    std::cout << "Logging Performance:" << std::endl;
    std::cout << "  Total time: " << duration_ms << "ms" << std::endl;
    std::cout << "  Queries logged: " << num_queries << std::endl;
    std::cout << "  Avg per query: " << avg_per_query << "ms" << std::endl;

    // Target: <1ms per query
    EXPECT_LT(avg_per_query, 1.0) << "Logging overhead should be <1ms per query";

    logger.flush();
    logger.shutdown();
}

// Test 5: Performance - Analytics Query Speed
TEST_F(AnalyticsIntegrationTest, PerformanceAnalyticsQueries) {
    // Create and populate database
    QueryLoggerConfig logger_config;
    logger_config.db_path = test_db_path;

    QueryLogger logger(logger_config);
    ASSERT_TRUE(logger.initialize());

    // Log 1000 queries
    for (int i = 0; i < 1000; ++i) {
        QueryLogEntry entry;
        entry.query_id = "aq" + std::to_string(i);
        entry.database_id = "test_db";
        entry.query_text = "query " + std::to_string(i % 100); // Create patterns
        entry.query_type = (i % 3 == 0) ? "vector" : (i % 3 == 1) ? "hybrid" : "bm25";
        entry.retrieval_time_ms = 10 + (i % 100);
        entry.total_time_ms = 15 + (i % 100);
        entry.num_results = i % 20;
        entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        logger.log_query(entry);
    }

    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    logger.shutdown();

    // Create analytics engine
    AnalyticsEngine engine(test_db_path);
    ASSERT_TRUE(engine.initialize());

    int64_t start_time = 0;
    int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();

    // Test statistics computation speed
    auto stats_start = std::chrono::high_resolution_clock::now();
    auto stats = engine.compute_statistics(start_time, end_time, TimeBucket::HOURLY);
    auto stats_end = std::chrono::high_resolution_clock::now();
    auto stats_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stats_end - stats_start).count();

    std::cout << "Statistics computation: " << stats_duration << "ms" << std::endl;
    EXPECT_LT(stats_duration, 500) << "Statistics should compute in <500ms";

    // Test pattern identification speed
    auto patterns_start = std::chrono::high_resolution_clock::now();
    auto patterns = engine.identify_patterns(2);
    auto patterns_end = std::chrono::high_resolution_clock::now();
    auto patterns_duration = std::chrono::duration_cast<std::chrono::milliseconds>(patterns_end - patterns_start).count();

    std::cout << "Pattern identification: " << patterns_duration << "ms" << std::endl;
    EXPECT_LT(patterns_duration, 500) << "Pattern identification should complete in <500ms";

    // Test insights generation speed
    auto insights_start = std::chrono::high_resolution_clock::now();
    auto insights = engine.generate_insights(start_time, end_time);
    auto insights_end = std::chrono::high_resolution_clock::now();
    auto insights_duration = std::chrono::duration_cast<std::chrono::milliseconds>(insights_end - insights_start).count();

    std::cout << "Insights generation: " << insights_duration << "ms" << std::endl;
    EXPECT_LT(insights_duration, 500) << "Insights should generate in <500ms";
}

// Test 6: Concurrent Access
TEST_F(AnalyticsIntegrationTest, ConcurrentLogging) {
    QueryLoggerConfig config;
    config.db_path = test_db_path;
    config.batch_size = 50;

    QueryLogger logger(config);
    ASSERT_TRUE(logger.initialize());

    const int num_threads = 5;
    const int queries_per_thread = 100;

    std::vector<std::thread> threads;

    auto worker = [&logger](int thread_id) {
        for (int i = 0; i < 100; ++i) {
            QueryLogEntry entry;
            entry.query_id = "t" + std::to_string(thread_id) + "_q" + std::to_string(i);
            entry.database_id = "test_db";
            entry.query_text = "concurrent query from thread " + std::to_string(thread_id);
            entry.query_type = "vector";
            entry.retrieval_time_ms = 10;
            entry.total_time_ms = 15;
            entry.num_results = 10;
            entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

            logger.log_query(entry);
        }
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto stats = logger.get_statistics();

    std::cout << "Concurrent Logging:" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Total queries: " << num_threads * queries_per_thread << std::endl;
    std::cout << "  Time: " << duration_ms << "ms" << std::endl;
    std::cout << "  Logged: " << stats.total_logged << std::endl;
    std::cout << "  Dropped: " << stats.total_dropped << std::endl;

    // All queries should be logged (no drops)
    EXPECT_EQ(stats.total_dropped, 0);
    EXPECT_GE(stats.total_logged, num_threads * queries_per_thread);

    logger.shutdown();
}

// Test 7: Data Persistence Across Restarts
TEST_F(AnalyticsIntegrationTest, DataPersistence) {
    const std::string query_text = "persistent test query";

    // First session: Log queries
    {
        QueryLoggerConfig config;
        config.db_path = test_db_path;

        QueryLogger logger(config);
        ASSERT_TRUE(logger.initialize());

        for (int i = 0; i < 10; ++i) {
            QueryLogEntry entry;
            entry.query_id = "persist_q" + std::to_string(i);
            entry.database_id = "test_db";
            entry.query_text = query_text;
            entry.query_type = "vector";
            entry.retrieval_time_ms = 10;
            entry.total_time_ms = 15;
            entry.num_results = 10;
            entry.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

            logger.log_query(entry);
        }

        logger.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        logger.shutdown();
    }

    // Second session: Verify data persisted
    {
        AnalyticsEngine engine(test_db_path);
        ASSERT_TRUE(engine.initialize());

        int64_t start_time = 0;
        int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();

        auto patterns = engine.identify_patterns(1);
        EXPECT_GT(patterns.size(), 0);

        bool found = false;
        for (const auto& pattern : patterns) {
            if (pattern.normalized_text.find("persistent test query") != std::string::npos) {
                found = true;
                EXPECT_EQ(pattern.count, 10);
            }
        }

        EXPECT_TRUE(found) << "Persisted queries should be retrievable after restart";
    }
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
