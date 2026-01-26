#include <gtest/gtest.h>
#include "../src/analytics/query_analytics_manager.h"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace jadedb::analytics;
using namespace jadedb::search;

class QueryAnalyticsManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_path_ = "/tmp/test_analytics_manager.db";
        // Clean up any existing test database
        std::filesystem::remove(test_db_path_);
    }

    void TearDown() override {
        // Clean up test database
        std::filesystem::remove(test_db_path_);
        std::filesystem::remove(test_db_path_ + "-shm");
        std::filesystem::remove(test_db_path_ + "-wal");
    }

    std::string test_db_path_;
};

// Test 1: Initialization
TEST_F(QueryAnalyticsManagerTest, Initialization) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    auto result = manager.initialize();

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(manager.is_ready());
}

// Test 2: Log vector search
TEST_F(QueryAnalyticsManagerTest, LogVectorSearch) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadevectordb::SearchResult> results = {
        {"doc1", 0.95f},
        {"doc2", 0.85f},
        {"doc3", 0.75f}
    };

    auto query_id = manager.log_vector_search(
        query_vector,
        results,
        10, // retrieval_time_ms
        15, // total_time_ms
        10, // top_k
        "cosine",
        "user1",
        "session1",
        "127.0.0.1"
    );

    ASSERT_TRUE(query_id.has_value());
    EXPECT_FALSE(query_id.value().empty());

    // Wait for async write
    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 1);
}

// Test 3: Log hybrid search
TEST_F(QueryAnalyticsManagerTest, LogHybridSearch) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadedb::search::SearchResult> results = {
        {"doc1", 0.95},
        {"doc2", 0.85}
    };

    auto query_id = manager.log_hybrid_search(
        "test query",
        query_vector,
        results,
        20, // retrieval_time_ms
        25, // total_time_ms
        10, // top_k
        0.7, // alpha
        "rrf",
        "user1",
        "session1",
        "127.0.0.1"
    );

    ASSERT_TRUE(query_id.has_value());
    EXPECT_FALSE(query_id.value().empty());

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 1);
}

// Test 4: Log re-ranking
TEST_F(QueryAnalyticsManagerTest, LogReranking) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::vector<jadedb::search::SearchResult> initial_results = {
        {"doc1", 0.85},
        {"doc2", 0.75}
    };

    std::vector<jadedb::search::RerankingResult> reranked_results;
    RerankingResult r1("doc2", 0.92, 0.75, 0.85);
    RerankingResult r2("doc1", 0.88, 0.85, 0.87);
    reranked_results.push_back(r1);
    reranked_results.push_back(r2);

    auto query_id = manager.log_reranking(
        "test query",
        initial_results,
        reranked_results,
        15, // retrieval_time_ms
        50, // reranking_time_ms
        65, // total_time_ms
        "cross-encoder/ms-marco",
        "user1",
        "session1",
        "127.0.0.1"
    );

    ASSERT_TRUE(query_id.has_value());
    EXPECT_FALSE(query_id.value().empty());

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 1);
}

// Test 5: Log error
TEST_F(QueryAnalyticsManagerTest, LogError) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    auto query_id = manager.log_error(
        "invalid query",
        "Vector dimension mismatch",
        "vector",
        "user1",
        "session1",
        "127.0.0.1"
    );

    ASSERT_TRUE(query_id.has_value());
    EXPECT_FALSE(query_id.value().empty());

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 1);
}

// Test 6: Multiple queries
TEST_F(QueryAnalyticsManagerTest, MultipleQueries) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadevectordb::SearchResult> results = {{"doc1", 0.95f}};

    for (int i = 0; i < 10; i++) {
        auto query_id = manager.log_vector_search(
            query_vector,
            results,
            10, 15, 10, "cosine"
        );
        ASSERT_TRUE(query_id.has_value());
    }

    manager.flush();

    // Wait for async write
    auto wait_start = std::chrono::steady_clock::now();
    while (manager.get_statistics().total_logged < 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto elapsed = std::chrono::steady_clock::now() - wait_start;
        if (elapsed > std::chrono::seconds(5)) {
            break;
        }
    }

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 10);
}

// Test 7: Statistics tracking
TEST_F(QueryAnalyticsManagerTest, StatisticsTracking) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    auto stats1 = manager.get_statistics();
    EXPECT_EQ(stats1.total_logged, 0);
    EXPECT_EQ(stats1.total_dropped, 0);

    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadevectordb::SearchResult> results = {{"doc1", 0.95f}};

    manager.log_vector_search(query_vector, results, 10, 15, 10, "cosine");
    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats2 = manager.get_statistics();
    EXPECT_EQ(stats2.total_logged, 1);
}

// Test 8: Empty results
TEST_F(QueryAnalyticsManagerTest, EmptyResults) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadevectordb::SearchResult> empty_results;

    auto query_id = manager.log_vector_search(
        query_vector,
        empty_results,
        10, 15, 10, "cosine"
    );

    ASSERT_TRUE(query_id.has_value());

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 1);
}

// Test 9: Shutdown and restart
TEST_F(QueryAnalyticsManagerTest, ShutdownAndRestart) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadevectordb::SearchResult> results = {{"doc1", 0.95f}};

    manager.log_vector_search(query_vector, results, 10, 15, 10, "cosine");
    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    manager.shutdown();
    EXPECT_FALSE(manager.is_ready());

    // Reinitialize
    auto result = manager.initialize();
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(manager.is_ready());
}

// Test 10: Large query text
TEST_F(QueryAnalyticsManagerTest, LargeQueryText) {
    QueryAnalyticsManager manager("test_db", test_db_path_);
    ASSERT_TRUE(manager.initialize().has_value());

    std::string large_query(10000, 'x');  // 10KB query
    std::vector<float> query_vector = {0.1f, 0.2f, 0.3f};
    std::vector<jadedb::search::SearchResult> results = {{"doc1", 0.95}};

    auto query_id = manager.log_hybrid_search(
        large_query,
        query_vector,
        results,
        20, 25, 10, 0.7, "rrf"
    );

    ASSERT_TRUE(query_id.has_value());

    manager.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = manager.get_statistics();
    EXPECT_EQ(stats.total_logged, 1);
}
