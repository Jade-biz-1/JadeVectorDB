#include <gtest/gtest.h>
#include "../src/analytics/query_logger.h"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace jadedb::analytics;

class QueryLoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_path_ = "/tmp/test_query_logger.db";
        // Clean up any existing test database
        std::filesystem::remove(test_db_path_);
    }

    void TearDown() override {
        // Clean up test database
        std::filesystem::remove(test_db_path_);
        std::filesystem::remove(test_db_path_ + "-shm");
        std::filesystem::remove(test_db_path_ + "-wal");
    }

    QueryLogEntry create_test_entry() {
        QueryLogEntry entry;
        entry.query_id = QueryLogger::generate_query_id();
        entry.database_id = "test_db";
        entry.query_text = "test query";
        entry.query_type = "vector";
        entry.retrieval_time_ms = 10;
        entry.total_time_ms = 15;
        entry.num_results = 5;
        entry.avg_similarity_score = 0.85;
        entry.min_similarity_score = 0.70;
        entry.max_similarity_score = 0.95;
        entry.user_id = "user123";
        entry.session_id = "session456";
        entry.client_ip = "127.0.0.1";
        entry.timestamp = QueryLogger::get_current_timestamp_ms();
        entry.top_k = 10;
        entry.vector_metric = "cosine";
        entry.has_error = false;
        return entry;
    }

    std::string test_db_path_;
};

// Test 1: Initialization
TEST_F(QueryLoggerTest, InitializationSuccess) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = false;

    QueryLogger logger("test_db", config);
    auto result = logger.initialize();

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(logger.is_ready());
}

// Test 2: Synchronous logging
TEST_F(QueryLoggerTest, SynchronousLogging) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = false;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    auto entry = create_test_entry();
    auto result = logger.log_query_sync(entry);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(logger.get_total_logged(), 1);
}

// Test 3: Asynchronous logging
TEST_F(QueryLoggerTest, AsynchronousLogging) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;
    config.batch_size = 10;
    config.flush_interval_ms = 1000;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    // Log multiple entries
    for (int i = 0; i < 5; i++) {
        auto entry = create_test_entry();
        entry.query_text = "query " + std::to_string(i);
        auto result = logger.log_query(entry);
        EXPECT_TRUE(result.has_value());
    }

    EXPECT_EQ(logger.get_queue_size(), 5);

    // Flush and wait
    logger.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(logger.get_queue_size(), 0);
    EXPECT_EQ(logger.get_total_logged(), 5);
}

// Test 4: Batch writes
TEST_F(QueryLoggerTest, BatchWrites) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;
    config.batch_size = 5;
    config.flush_interval_ms = 100;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    // Log exactly batch_size entries
    for (int i = 0; i < 5; i++) {
        auto entry = create_test_entry();
        logger.log_query(entry);
    }

    // Wait for background thread to process
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(logger.get_total_logged(), 5);
}

// Test 5: Queue size limit
TEST_F(QueryLoggerTest, QueueSizeLimit) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;
    config.max_queue_size = 10;
    config.flush_interval_ms = 10000;  // Very long interval

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    // Try to exceed queue limit
    int success = 0;
    int failed = 0;
    for (int i = 0; i < 20; i++) {
        auto entry = create_test_entry();
        auto result = logger.log_query(entry);
        if (result.has_value()) {
            success++;
        } else {
            failed++;
        }
    }

    EXPECT_EQ(success, 10);
    EXPECT_EQ(failed, 10);
    EXPECT_EQ(logger.get_total_dropped(), 10);
}

// Test 6: Flush functionality
TEST_F(QueryLoggerTest, FlushFunctionality) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;
    config.flush_interval_ms = 60000;  // Very long interval

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    // Log entries
    for (int i = 0; i < 3; i++) {
        auto entry = create_test_entry();
        logger.log_query(entry);
    }

    EXPECT_EQ(logger.get_queue_size(), 3);

    // Explicit flush
    auto result = logger.flush();
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(logger.get_queue_size(), 0);
    EXPECT_EQ(logger.get_total_logged(), 3);
}

// Test 7: Hybrid search entry
TEST_F(QueryLoggerTest, HybridSearchEntry) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = false;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    auto entry = create_test_entry();
    entry.query_type = "hybrid";
    entry.hybrid_alpha = 0.7;
    entry.fusion_method = "rrf";

    auto result = logger.log_query_sync(entry);
    ASSERT_TRUE(result.has_value());
}

// Test 8: Reranking entry
TEST_F(QueryLoggerTest, RerankingEntry) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = false;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    auto entry = create_test_entry();
    entry.query_type = "rerank";
    entry.used_reranking = true;
    entry.reranking_model = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    entry.reranking_time_ms = 50;

    auto result = logger.log_query_sync(entry);
    ASSERT_TRUE(result.has_value());
}

// Test 9: Error logging
TEST_F(QueryLoggerTest, ErrorLogging) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = false;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    auto entry = create_test_entry();
    entry.has_error = true;
    entry.error_message = "Test error message";

    auto result = logger.log_query_sync(entry);
    ASSERT_TRUE(result.has_value());
}

// Test 10: Multiple loggers (different databases)
TEST_F(QueryLoggerTest, MultipleLoggers) {
    std::string db1 = "/tmp/test_logger1.db";
    std::string db2 = "/tmp/test_logger2.db";

    QueryLoggerConfig config1;
    config1.database_path = db1;
    config1.enable_async = false;

    QueryLoggerConfig config2;
    config2.database_path = db2;
    config2.enable_async = false;

    QueryLogger logger1("db1", config1);
    QueryLogger logger2("db2", config2);

    ASSERT_TRUE(logger1.initialize().has_value());
    ASSERT_TRUE(logger2.initialize().has_value());

    auto entry1 = create_test_entry();
    entry1.database_id = "db1";
    logger1.log_query_sync(entry1);

    auto entry2 = create_test_entry();
    entry2.database_id = "db2";
    logger2.log_query_sync(entry2);

    EXPECT_EQ(logger1.get_total_logged(), 1);
    EXPECT_EQ(logger2.get_total_logged(), 1);

    // Cleanup
    std::filesystem::remove(db1);
    std::filesystem::remove(db2);
}

// Test 11: Query ID generation uniqueness
TEST_F(QueryLoggerTest, QueryIdUniqueness) {
    std::set<std::string> ids;
    for (int i = 0; i < 1000; i++) {
        auto id = QueryLogger::generate_query_id();
        EXPECT_TRUE(ids.find(id) == ids.end()) << "Duplicate ID generated: " << id;
        ids.insert(id);
    }
}

// Test 12: Timestamp generation
TEST_F(QueryLoggerTest, TimestampGeneration) {
    auto ts1 = QueryLogger::get_current_timestamp_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto ts2 = QueryLogger::get_current_timestamp_ms();

    EXPECT_GT(ts2, ts1);
    EXPECT_GE(ts2 - ts1, 10);
}

// Test 13: Shutdown and restart
TEST_F(QueryLoggerTest, ShutdownAndRestart) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    auto entry = create_test_entry();
    logger.log_query(entry);
    logger.flush();

    logger.shutdown();
    EXPECT_FALSE(logger.is_ready());

    // Reinitialize
    auto result = logger.initialize();
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(logger.is_ready());
}

// Test 14: Performance benchmark (async)
TEST_F(QueryLoggerTest, PerformanceBenchmarkAsync) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;
    config.batch_size = 100;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    const int num_entries = 1000;
    auto entry = create_test_entry();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_entries; i++) {
        logger.log_query(entry);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = static_cast<double>(duration.count()) / num_entries;

    // Should be < 1000 microseconds (1ms) per entry
    EXPECT_LT(avg_us, 1000.0) << "Average logging time: " << avg_us << " us";

    logger.flush();

    // Wait for all entries to be processed (with timeout)
    auto wait_start = std::chrono::steady_clock::now();
    while (logger.get_total_logged() < num_entries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto elapsed = std::chrono::steady_clock::now() - wait_start;
        if (elapsed > std::chrono::seconds(5)) {
            break;  // Timeout after 5 seconds
        }
    }

    EXPECT_EQ(logger.get_total_logged(), num_entries);
}

// Test 15: Concurrent logging
TEST_F(QueryLoggerTest, ConcurrentLogging) {
    QueryLoggerConfig config;
    config.database_path = test_db_path_;
    config.enable_async = true;
    config.max_queue_size = 1000;

    QueryLogger logger("test_db", config);
    ASSERT_TRUE(logger.initialize().has_value());

    const int num_threads = 4;
    const int entries_per_thread = 25;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&logger, entries_per_thread]() {
            for (int i = 0; i < entries_per_thread; i++) {
                auto entry = QueryLogEntry();
                entry.query_id = QueryLogger::generate_query_id();
                entry.database_id = "test_db";
                entry.query_text = "concurrent query";
                entry.query_type = "vector";
                entry.timestamp = QueryLogger::get_current_timestamp_ms();
                logger.log_query(entry);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    logger.flush();

    // Wait for all entries to be processed (with timeout)
    auto wait_start = std::chrono::steady_clock::now();
    const int expected_count = num_threads * entries_per_thread;
    while (logger.get_total_logged() < expected_count) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto elapsed = std::chrono::steady_clock::now() - wait_start;
        if (elapsed > std::chrono::seconds(5)) {
            break;  // Timeout after 5 seconds
        }
    }

    EXPECT_EQ(logger.get_total_logged(), num_threads * entries_per_thread);
}
