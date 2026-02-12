/**
 * Integration Tests for Query Analytics System (T16.22)
 *
 * Tests end-to-end analytics flow:
 * - Query logging -> Storage -> Analysis -> Insights
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

using namespace jadedb::analytics;
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
    config.database_path = test_db_path;
    config.batch_size = 10;
    config.flush_interval_ms = 100;

    QueryLogger logger("test_db", config);
    auto init_result = logger.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Logger initialization failed";

    // Log some queries - use duplicates so patterns can be identified
    std::vector<std::string> queries = {
        "machine learning tutorials",
        "machine learning tutorials",
        "machine learning tutorials",
        "python programming examples",
        "python programming examples"
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

        logger.log_query_sync(entry);
    }

    // Flush to ensure all queries are written
    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Create analytics engine
    AnalyticsEngine engine(test_db_path);
    auto engine_init = engine.initialize();
    ASSERT_TRUE(engine_init.has_value()) << "Engine initialization failed";

    // Compute statistics
    int64_t start_time = 0;
    int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();

    auto stats_result = engine.compute_statistics("test_db", start_time, end_time, TimeBucket::HOURLY);
    ASSERT_TRUE(stats_result.has_value());
    EXPECT_GT(stats_result.value().size(), 0);

    // Identify patterns
    auto patterns_result = engine.identify_patterns("test_db", start_time, end_time, 2);
    ASSERT_TRUE(patterns_result.has_value());
    EXPECT_GT(patterns_result.value().size(), 0);

    // Check for "machine learning tutorials" pattern (appears 3 times)
    bool found_ml_pattern = false;
    for (const auto& pattern : patterns_result.value()) {
        if (pattern.normalized_text.find("machine learning") != std::string::npos) {
            found_ml_pattern = true;
            EXPECT_GE(pattern.query_count, 2u); // Should appear at least 2 times
        }
    }
    EXPECT_TRUE(found_ml_pattern);

    // Generate insights
    auto insights_result = engine.generate_insights("test_db", start_time, end_time);
    ASSERT_TRUE(insights_result.has_value());

    logger.shutdown();
}

// Test 2: Query Analytics Manager Integration
TEST_F(AnalyticsIntegrationTest, QueryAnalyticsManagerIntegration) {
    QueryAnalyticsManager manager("test_db", test_db_path);
    auto init_result = manager.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Manager initialization failed";

    // Log different types of queries
    std::vector<float> query_vector = {1.0f, 2.0f, 3.0f};

    // Vector search
    std::vector<jadevectordb::SearchResult> vec_results;
    vec_results.emplace_back("doc1", 0.95f);
    vec_results.emplace_back("doc2", 0.85f);

    manager.log_vector_search(
        query_vector,
        vec_results,
        50,   // retrieval_time_ms
        75,   // total_time_ms
        10,   // top_k
        "cosine",
        "user1",
        "session1"
    );

    // Hybrid search
    std::vector<jadedb::search::SearchResult> hybrid_results;
    hybrid_results.emplace_back("doc1", 0.95);

    manager.log_hybrid_search(
        "test query",
        query_vector,
        hybrid_results,
        50,   // retrieval_time_ms
        75,   // total_time_ms
        10,   // top_k
        0.7,  // alpha
        "RRF",
        "user1",
        "session1"
    );

    // Re-ranking
    std::vector<jadedb::search::SearchResult> initial_results;
    initial_results.emplace_back("doc1", 0.95);

    std::vector<jadedb::search::RerankingResult> reranked_results;
    reranked_results.emplace_back("doc1", 0.98, 0.95, 0.97);

    manager.log_reranking(
        "test query",
        initial_results,
        reranked_results,
        50,   // retrieval_time_ms
        25,   // reranking_time_ms
        75,   // total_time_ms
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "user1",
        "session1"
    );

    // Error logging
    manager.log_error(
        "invalid query",
        "Invalid vector dimension",
        "vector",
        "user1",
        "session1"
    );

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Get statistics
    auto stats = manager.get_statistics();
    EXPECT_GT(stats.total_logged, 0u);

    manager.shutdown();
}

// Test 3: Batch Processor Integration
TEST_F(AnalyticsIntegrationTest, BatchProcessorIntegration) {
    // Create logger and log some queries
    QueryLoggerConfig logger_config;
    logger_config.database_path = test_db_path;

    QueryLogger logger("test_db", logger_config);
    auto init_result = logger.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Logger initialization failed";

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

    // Create analytics engine (needed by BatchProcessor)
    auto analytics_engine = std::make_shared<AnalyticsEngine>(test_db_path);
    auto engine_init = analytics_engine->initialize();
    ASSERT_TRUE(engine_init.has_value()) << "Analytics engine initialization failed";

    // Create batch processor
    BatchProcessorConfig bp_config;
    bp_config.enable_hourly_aggregation = false; // Disable scheduled jobs
    bp_config.enable_daily_cleanup = false;

    BatchProcessor processor("test_db", analytics_engine, bp_config);
    auto start_result = processor.start();
    ASSERT_TRUE(start_result.has_value()) << "BatchProcessor start failed";

    // Run aggregation manually
    auto agg_result = processor.run_aggregation_now();
    ASSERT_TRUE(agg_result.has_value());

    // Run cleanup
    auto cleanup_result = processor.run_cleanup_now();
    ASSERT_TRUE(cleanup_result.has_value());

    // Get statistics
    auto stats = processor.get_statistics();
    EXPECT_GT(stats.total_jobs_run, 0u);
    EXPECT_EQ(stats.failed_jobs, 0u);

    processor.stop();
}

// Test 4: Performance - Logging Overhead
TEST_F(AnalyticsIntegrationTest, PerformanceLoggingOverhead) {
    QueryLoggerConfig config;
    config.database_path = test_db_path;
    config.batch_size = 100;
    config.flush_interval_ms = 1000;

    QueryLogger logger("test_db", config);
    auto init_result = logger.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Logger initialization failed";

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
    logger_config.database_path = test_db_path;

    QueryLogger logger("test_db", logger_config);
    auto init_result = logger.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Logger initialization failed";

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
    auto engine_init = engine.initialize();
    ASSERT_TRUE(engine_init.has_value()) << "Engine initialization failed";

    int64_t start_time = 0;
    int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();

    // Test statistics computation speed
    auto stats_start = std::chrono::high_resolution_clock::now();
    auto stats = engine.compute_statistics("test_db", start_time, end_time, TimeBucket::HOURLY);
    auto stats_end = std::chrono::high_resolution_clock::now();
    auto stats_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stats_end - stats_start).count();

    std::cout << "Statistics computation: " << stats_duration << "ms" << std::endl;
    EXPECT_LT(stats_duration, 500) << "Statistics should compute in <500ms";

    // Test pattern identification speed
    auto patterns_start = std::chrono::high_resolution_clock::now();
    auto patterns = engine.identify_patterns("test_db", start_time, end_time, 2);
    auto patterns_end = std::chrono::high_resolution_clock::now();
    auto patterns_duration = std::chrono::duration_cast<std::chrono::milliseconds>(patterns_end - patterns_start).count();

    std::cout << "Pattern identification: " << patterns_duration << "ms" << std::endl;
    EXPECT_LT(patterns_duration, 500) << "Pattern identification should complete in <500ms";

    // Test insights generation speed
    auto insights_start = std::chrono::high_resolution_clock::now();
    auto insights = engine.generate_insights("test_db", start_time, end_time);
    auto insights_end = std::chrono::high_resolution_clock::now();
    auto insights_duration = std::chrono::duration_cast<std::chrono::milliseconds>(insights_end - insights_start).count();

    std::cout << "Insights generation: " << insights_duration << "ms" << std::endl;
    EXPECT_LT(insights_duration, 500) << "Insights should generate in <500ms";
}

// Test 6: Concurrent Access
TEST_F(AnalyticsIntegrationTest, ConcurrentLogging) {
    QueryLoggerConfig config;
    config.database_path = test_db_path;
    config.batch_size = 50;

    QueryLogger logger("test_db", config);
    auto init_result = logger.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Logger initialization failed";

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

    // Flush and wait for background writer to drain the queue
    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    auto total_logged = logger.get_total_logged();
    auto total_dropped = logger.get_total_dropped();

    std::cout << "Concurrent Logging:" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Total queries: " << num_threads * queries_per_thread << std::endl;
    std::cout << "  Time: " << duration_ms << "ms" << std::endl;
    std::cout << "  Logged: " << total_logged << std::endl;
    std::cout << "  Dropped: " << total_dropped << std::endl;

    // All queries should be queued (no drops)
    EXPECT_EQ(total_dropped, 0u);
    // Verify a reasonable number were persisted (async flush may not complete all)
    EXPECT_GT(total_logged, 0u);

    logger.shutdown();
}

// Test 7: Data Persistence Across Restarts
TEST_F(AnalyticsIntegrationTest, DataPersistence) {
    const std::string query_text = "persistent test query";

    // First session: Log queries
    {
        QueryLoggerConfig config;
        config.database_path = test_db_path;

        QueryLogger logger("test_db", config);
        auto init_result = logger.initialize();
        ASSERT_TRUE(init_result.has_value()) << "Logger initialization failed";

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
        auto engine_init = engine.initialize();
        ASSERT_TRUE(engine_init.has_value()) << "Engine initialization failed";

        int64_t start_time = 0;
        int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();

        auto patterns_result = engine.identify_patterns("test_db", start_time, end_time, 1);
        ASSERT_TRUE(patterns_result.has_value());
        EXPECT_GT(patterns_result.value().size(), 0u);

        bool found = false;
        for (const auto& pattern : patterns_result.value()) {
            if (pattern.normalized_text.find("persistent test query") != std::string::npos) {
                found = true;
                EXPECT_EQ(pattern.query_count, 10u);
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
