#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>

#include "services/search/reranking_service.h"
#include "services/search/score_fusion.h"

using namespace jadedb::search;

// Test fixture for RerankingService tests
class RerankingServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test echo server script
        test_script_path_ = "/tmp/test_rerank_server.py";
        create_test_rerank_script();

        // Configure reranking service
        database_id_ = "test_db";

        config_.model_name = "test-model";
        config_.batch_size = 32;
        config_.score_threshold = 0.0;
        config_.combine_scores = true;
        config_.rerank_weight = 0.7;

        // Setup test documents
        doc_ids_ = {"doc1", "doc2", "doc3", "doc4", "doc5"};
        documents_ = {
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
            "computer vision applications",
            "data science and analytics"
        };
        original_scores_ = {0.9, 0.85, 0.8, 0.75, 0.7};

        // Setup search results
        for (size_t i = 0; i < doc_ids_.size(); i++) {
            SearchResult result;
            result.doc_id = doc_ids_[i];
            result.score = original_scores_[i];
            search_results_.push_back(result);
        }

        // Setup document texts map
        for (size_t i = 0; i < doc_ids_.size(); i++) {
            document_texts_[doc_ids_[i]] = documents_[i];
        }
    }

    void TearDown() override {
        std::remove(test_script_path_.c_str());
    }

    void create_test_rerank_script() {
        std::ofstream script(test_script_path_);
        script << R"(#!/usr/bin/env python3
import sys
import json

# Send ready signal
print(json.dumps({"status": "ready"}))
sys.stdout.flush()

# Request loop
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break

        request = json.loads(line)

        # Handle ping
        if request.get("command") == "ping":
            response = {"status": "pong"}
        # Handle reranking
        elif "query" in request and "documents" in request:
            docs = request["documents"]
            # Generate mock scores (descending from 0.95)
            scores = [0.95 - i * 0.05 for i in range(len(docs))]
            response = {"scores": scores, "latency_ms": 50}
        else:
            response = {"error": "Unknown request"}

        print(json.dumps(response))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()
        break
)";
        script.close();
        chmod(test_script_path_.c_str(), 0755);
    }

    std::string test_script_path_;
    std::string database_id_;
    RerankingConfig config_;
    std::vector<std::string> doc_ids_;
    std::vector<std::string> documents_;
    std::vector<double> original_scores_;
    std::vector<SearchResult> search_results_;
    std::unordered_map<std::string, std::string> document_texts_;
};

// Test basic initialization
TEST_F(RerankingServiceTest, Initialization) {
    // Override script path for testing
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);

    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value()) << "Service should initialize successfully";

    EXPECT_TRUE(service.is_ready()) << "Service should be ready after initialization";

    service.shutdown();
}

// Test reranking with valid documents
TEST_F(RerankingServiceTest, BasicReranking) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    // Perform reranking
    auto result = service.rerank_batch(
        "test query",
        doc_ids_,
        documents_,
        original_scores_
    );

    ASSERT_TRUE(result.has_value()) << "Reranking should succeed: " <<
        (result.has_value() ? "" : result.error().message);

    auto results = result.value();
    EXPECT_EQ(results.size(), doc_ids_.size()) << "Should return all documents";

    // Verify results are sorted by combined score (descending)
    for (size_t i = 1; i < results.size(); i++) {
        EXPECT_GE(results[i-1].combined_score, results[i].combined_score)
            << "Results should be sorted descending";
    }

    // Verify score combination
    for (const auto& r : results) {
        EXPECT_GT(r.rerank_score, 0.0) << "Rerank score should be positive";
        EXPECT_GT(r.original_score, 0.0) << "Original score should be positive";
        EXPECT_GT(r.combined_score, 0.0) << "Combined score should be positive";

        // Combined score should be weighted average
        double expected_combined = config_.rerank_weight * r.rerank_score +
                                   (1.0 - config_.rerank_weight) * r.original_score;
        EXPECT_NEAR(r.combined_score, expected_combined, 0.001)
            << "Combined score should match weighted formula";
    }

    service.shutdown();
}

// Test reranking via SearchResult interface
TEST_F(RerankingServiceTest, RerankSearchResults) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    // Perform reranking using SearchResult interface
    auto result = service.rerank(
        "test query",
        search_results_,
        document_texts_
    );

    ASSERT_TRUE(result.has_value()) << "Reranking should succeed";

    auto results = result.value();
    EXPECT_EQ(results.size(), search_results_.size());

    service.shutdown();
}

// Test empty documents
TEST_F(RerankingServiceTest, EmptyDocuments) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    std::vector<std::string> empty_ids;
    std::vector<std::string> empty_docs;
    std::vector<double> empty_scores;

    auto result = service.rerank_batch("query", empty_ids, empty_docs, empty_scores);
    ASSERT_TRUE(result.has_value()) << "Should handle empty documents";

    auto results = result.value();
    EXPECT_EQ(results.size(), 0) << "Should return empty results";

    service.shutdown();
}

// Test single document
TEST_F(RerankingServiceTest, SingleDocument) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    std::vector<std::string> single_id = {"doc1"};
    std::vector<std::string> single_doc = {"single document"};
    std::vector<double> single_score = {0.9};

    auto result = service.rerank_batch("query", single_id, single_doc, single_score);
    ASSERT_TRUE(result.has_value());

    auto results = result.value();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].doc_id, "doc1");
    // Single document gets perfect rerank score
    EXPECT_NEAR(results[0].rerank_score, 1.0, 0.001);

    service.shutdown();
}

// Test score threshold filtering
TEST_F(RerankingServiceTest, ScoreThreshold) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingConfig threshold_config = config_;
    threshold_config.score_threshold = 0.8;  // High threshold

    RerankingService service(database_id_, threshold_config);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    auto result = service.rerank_batch("test query", doc_ids_, documents_, original_scores_);
    ASSERT_TRUE(result.has_value());

    auto results = result.value();
    // Some results should be filtered by threshold
    EXPECT_LT(results.size(), doc_ids_.size()) << "Threshold should filter some results";

    // All remaining results should be above threshold
    for (const auto& r : results) {
        EXPECT_GE(r.combined_score, threshold_config.score_threshold);
    }

    service.shutdown();
}

// Test without score combination
TEST_F(RerankingServiceTest, NoScoreCombination) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingConfig no_combine_config = config_;
    no_combine_config.combine_scores = false;

    RerankingService service(database_id_, no_combine_config);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    auto result = service.rerank_batch("test query", doc_ids_, documents_, original_scores_);
    ASSERT_TRUE(result.has_value());

    auto results = result.value();
    // When not combining, combined_score should equal rerank_score
    for (const auto& r : results) {
        EXPECT_NEAR(r.combined_score, r.rerank_score, 0.001)
            << "Combined score should equal rerank score when combination disabled";
    }

    service.shutdown();
}

// Test service not initialized error
TEST_F(RerankingServiceTest, NotInitialized) {
    RerankingService service(database_id_, config_);

    // Try to rerank without initialization
    auto result = service.rerank_batch("query", doc_ids_, documents_, original_scores_);
    EXPECT_FALSE(result.has_value()) << "Should fail when not initialized";

    if (!result.has_value()) {
        EXPECT_EQ(result.error().code, jadevectordb::ErrorCode::SERVICE_UNAVAILABLE);
    }
}

// Test size mismatch errors
TEST_F(RerankingServiceTest, SizeMismatch) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    // doc_ids and documents size mismatch
    std::vector<std::string> short_docs = {"doc1", "doc2"};
    auto result1 = service.rerank_batch("query", doc_ids_, short_docs, original_scores_);
    EXPECT_FALSE(result1.has_value()) << "Should fail with size mismatch";

    // original_scores size mismatch
    std::vector<double> short_scores = {0.9, 0.8};
    auto result2 = service.rerank_batch("query", doc_ids_, documents_, short_scores);
    EXPECT_FALSE(result2.has_value()) << "Should fail with scores size mismatch";

    service.shutdown();
}

// Test statistics tracking
TEST_F(RerankingServiceTest, Statistics) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    // Get initial stats
    auto stats_before = service.get_statistics();
    EXPECT_EQ(stats_before.total_requests, 0);

    // Perform reranking
    auto result = service.rerank_batch("query", doc_ids_, documents_, original_scores_);
    ASSERT_TRUE(result.has_value());

    // Check updated stats
    auto stats_after = service.get_statistics();
    EXPECT_EQ(stats_after.total_requests, 1);
    EXPECT_EQ(stats_after.total_documents_reranked, doc_ids_.size());
    EXPECT_GT(stats_after.avg_latency_ms, 0.0);

    // Reset stats
    service.reset_statistics();
    auto stats_reset = service.get_statistics();
    EXPECT_EQ(stats_reset.total_requests, 0);

    service.shutdown();
}

// Test configuration update
TEST_F(RerankingServiceTest, ConfigUpdate) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    // Update config
    RerankingConfig new_config = config_;
    new_config.rerank_weight = 0.5;
    new_config.score_threshold = 0.5;

    service.set_config(new_config);

    auto updated_config = service.get_config();
    EXPECT_NEAR(updated_config.rerank_weight, 0.5, 0.001);
    EXPECT_NEAR(updated_config.score_threshold, 0.5, 0.001);

    service.shutdown();
}

// Test concurrent reranking requests
TEST_F(RerankingServiceTest, ConcurrentRequests) {
    setenv("RERANKING_SCRIPT_PATH", test_script_path_.c_str(), 1);

    RerankingService service(database_id_, config_);
    auto init_result = service.initialize();
    ASSERT_TRUE(init_result.has_value());

    const int num_threads = 3;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([this, &service, &success_count]() {
            auto result = service.rerank_batch(
                "concurrent query",
                doc_ids_,
                documents_,
                original_scores_
            );
            if (result.has_value()) {
                success_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads) << "All concurrent requests should succeed";

    service.shutdown();
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
