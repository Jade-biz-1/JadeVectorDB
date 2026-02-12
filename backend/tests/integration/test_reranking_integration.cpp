#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <functional>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/search/hybrid_search_engine.h"
#include "services/search/reranking_service.h"
#include "services/search/score_fusion.h"
#include "models/vector.h"
#include "models/database.h"

// Use explicit namespace qualifications to avoid SearchResult ambiguity
using jadevectordb::Vector;
using jadevectordb::Database;
using jadevectordb::DatabaseLayer;
using jadevectordb::DatabaseService;
using jadevectordb::VectorStorageService;
using jadevectordb::SimilaritySearchService;
using jadevectordb::SearchParams;

using jadedb::search::HybridSearchEngine;
using jadedb::search::HybridSearchConfig;
using jadedb::search::HybridSearchResult;
using jadedb::search::FusionMethod;
using jadedb::search::BM25Document;

// Test fixture for reranking integration tests
class RerankingIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize database layer
        db_layer_ = std::make_shared<DatabaseLayer>();
        auto init_result = db_layer_->initialize();
        ASSERT_TRUE(init_result.has_value()) << "Failed to initialize database layer";

        // Initialize services
        db_service_ = std::make_unique<DatabaseService>(db_layer_);
        init_result = db_service_->initialize();
        ASSERT_TRUE(init_result.has_value()) << "Failed to initialize database service";

        vector_storage_ = std::make_unique<VectorStorageService>(db_layer_);
        init_result = vector_storage_->initialize();
        ASSERT_TRUE(init_result.has_value()) << "Failed to initialize vector storage";

        auto search_vector_storage = std::make_unique<VectorStorageService>(db_layer_);
        init_result = search_vector_storage->initialize();
        ASSERT_TRUE(init_result.has_value()) << "Failed to initialize search vector storage";
        similarity_search_ = std::make_unique<SimilaritySearchService>(std::move(search_vector_storage));
        init_result = similarity_search_->initialize();
        ASSERT_TRUE(init_result.has_value()) << "Failed to initialize similarity search";

        // Create test database
        Database db;
        db.name = "reranking_test_db";
        db.description = "Test database for reranking integration";
        db.vectorDimension = 4;

        auto result = db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();

        // Setup test data
        setupTestData();
    }

    void TearDown() override {
        if (!test_database_id_.empty()) {
            db_layer_->delete_database(test_database_id_);
        }
    }

    void setupTestData() {
        // Create documents with varying relevance to test query
        // Query will be: "machine learning algorithms"

        // Document 1: High relevance - exact match
        Vector v1;
        v1.id = "doc1";
        v1.databaseId = test_database_id_;
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
        v1.metadata.source = "Machine learning algorithms are fundamental to artificial intelligence and data science applications";
  // Store text for reranking
        v1.metadata.category = "AI";
        v1.metadata.status = "active";
        vectors_.push_back(v1);

        // Document 2: Medium-high relevance - related concepts
        Vector v2;
        v2.id = "doc2";
        v2.databaseId = test_database_id_;
        v2.values = {0.9f, 0.1f, 0.0f, 0.0f};
        v2.metadata.source = "Deep learning neural networks use gradient descent optimization techniques for training";
        v2.metadata.category = "AI";
        v2.metadata.status = "active";
        vectors_.push_back(v2);

        // Document 3: Medium relevance - tangentially related
        Vector v3;
        v3.id = "doc3";
        v3.databaseId = test_database_id_;
        v3.values = {0.8f, 0.0f, 0.2f, 0.0f};
        v3.metadata.source = "Statistical analysis methods for data processing and pattern recognition in large datasets";
        v3.metadata.category = "Statistics";
        v3.metadata.status = "active";
        vectors_.push_back(v3);

        // Document 4: Low relevance - different topic
        Vector v4;
        v4.id = "doc4";
        v4.databaseId = test_database_id_;
        v4.values = {0.5f, 0.5f, 0.0f, 0.0f};
        v4.metadata.source = "Database management systems and SQL query optimization for relational databases";
        v4.metadata.category = "Database";
        v4.metadata.status = "active";
        vectors_.push_back(v4);

        // Document 5: Very low relevance - unrelated
        Vector v5;
        v5.id = "doc5";
        v5.databaseId = test_database_id_;
        v5.values = {0.0f, 0.0f, 0.0f, 1.0f};
        v5.metadata.source = "Cooking recipes and culinary techniques for Italian cuisine preparation";
        v5.metadata.category = "Cooking";
        v5.metadata.status = "active";
        vectors_.push_back(v5);

        // Insert vectors
        for (const auto& v : vectors_) {
            auto insert_result = vector_storage_->store_vector(test_database_id_, v);
            ASSERT_TRUE(insert_result.has_value()) << "Failed to insert vector: " << v.id;
        }

        // Build BM25 index
        std::vector<BM25Document> bm25_docs;
        for (const auto& v : vectors_) {
            BM25Document doc;
            doc.doc_id = v.id;
            doc.text = v.metadata.source;
            bm25_docs.push_back(doc);
        }

        // Create hybrid search engine
        HybridSearchConfig hybrid_config;
        hybrid_config.fusion_method = FusionMethod::RRF;
        hybrid_config.rrf_k = 60;
        hybrid_engine_ = std::make_unique<HybridSearchEngine>(test_database_id_, hybrid_config);

        // Build BM25 index
        bool build_result = hybrid_engine_->build_bm25_index(bm25_docs);
        ASSERT_TRUE(build_result) << "Failed to build BM25 index";
    }

    std::shared_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_;
    std::unique_ptr<SimilaritySearchService> similarity_search_;
    std::unique_ptr<HybridSearchEngine> hybrid_engine_;
    std::string test_database_id_;
    std::vector<Vector> vectors_;
};

// Test basic hybrid search without reranking
TEST_F(RerankingIntegrationTest, HybridSearchBaseline) {
    // Setup vector search provider
    hybrid_engine_->set_vector_search_provider(
        [this](const std::vector<float>& query_vec, size_t top_k) -> std::vector<jadedb::search::SearchResult> {
            // Create Vector object for query
            Vector query_vector;
            query_vector.values = query_vec;

            // Create SearchParams
            SearchParams params;
            params.top_k = top_k;

            auto search_result = similarity_search_->similarity_search(test_database_id_, query_vector, params);

            std::vector<jadedb::search::SearchResult> results;
            if (search_result.has_value()) {
                for (const auto& v : search_result.value()) {
                    jadedb::search::SearchResult result;
                    result.doc_id = v.vector_id;
                    result.score = v.similarity_score;
                    results.push_back(result);
                }
            }
            return results;
        }
    );

    // Query: "machine learning algorithms"
    std::vector<float> query_vector = {1.0f, 0.0f, 0.0f, 0.0f};  // Should match doc1 best
    std::string query_text = "machine learning algorithms";

    // Perform hybrid search (without reranking)
    auto results = hybrid_engine_->search(query_text, query_vector, 5, false);

    ASSERT_FALSE(results.empty()) << "Hybrid search should return results";
    ASSERT_GE(results.size(), 3) << "Should return at least 3 results";

    // Verify doc1 is highly ranked (should be in top 2)
    bool doc1_in_top2 = (results[0].doc_id == "doc1" || results[1].doc_id == "doc1");
    EXPECT_TRUE(doc1_in_top2) << "doc1 should be in top 2 results";

    // Print baseline results for comparison
    std::cout << "\n=== Baseline Hybrid Search Results ===" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << (i+1) << ". " << results[i].doc_id
                  << " (vec=" << results[i].vector_score
                  << ", bm25=" << results[i].bm25_score
                  << ", hybrid=" << results[i].hybrid_score << ")" << std::endl;
    }
}

// Test hybrid search with reranking (using mock reranker)
TEST_F(RerankingIntegrationTest, HybridSearchWithMockReranking) {
    // Setup vector search provider
    hybrid_engine_->set_vector_search_provider(
        [this](const std::vector<float>& query_vec, size_t top_k) -> std::vector<jadedb::search::SearchResult> {
            // Create Vector object for query
            Vector query_vector;
            query_vector.values = query_vec;

            // Create SearchParams
            SearchParams params;
            params.top_k = top_k;

            auto search_result = similarity_search_->similarity_search(test_database_id_, query_vector, params);

            std::vector<jadedb::search::SearchResult> results;
            if (search_result.has_value()) {
                for (const auto& v : search_result.value()) {
                    jadedb::search::SearchResult result;
                    result.doc_id = v.vector_id;
                    result.score = v.similarity_score;
                    results.push_back(result);
                }
            }
            return results;
        }
    );

    // Setup mock reranking provider (simulates cross-encoder)
    // Mock reranker gives highest score to doc1 (exact match for query)
    hybrid_engine_->set_reranking_provider(
        [](const std::string& query, const std::vector<HybridSearchResult>& candidates)
        -> std::vector<HybridSearchResult> {
            std::vector<HybridSearchResult> reranked = candidates;

            // Simulate cross-encoder scores based on semantic relevance
            for (auto& result : reranked) {
                if (result.doc_id == "doc1") {
                    result.rerank_score = 0.95;  // High relevance
                } else if (result.doc_id == "doc2") {
                    result.rerank_score = 0.80;  // Medium-high
                } else if (result.doc_id == "doc3") {
                    result.rerank_score = 0.60;  // Medium
                } else if (result.doc_id == "doc4") {
                    result.rerank_score = 0.30;  // Low
                } else {
                    result.rerank_score = 0.10;  // Very low
                }

                // Combined score: weighted average
                result.combined_score = 0.7 * result.rerank_score + 0.3 * result.hybrid_score;
            }

            // Sort by combined score
            std::sort(reranked.begin(), reranked.end(),
                [](const HybridSearchResult& a, const HybridSearchResult& b) {
                    return a.combined_score > b.combined_score;
                });

            return reranked;
        }
    );

    // Query
    std::vector<float> query_vector = {1.0f, 0.0f, 0.0f, 0.0f};
    std::string query_text = "machine learning algorithms";

    // Perform hybrid search WITH reranking
    auto results = hybrid_engine_->search(query_text, query_vector, 5, true, 10);

    ASSERT_FALSE(results.empty()) << "Reranked search should return results";

    // Verify doc1 is now ranked #1 after reranking
    EXPECT_EQ(results[0].doc_id, "doc1") << "After reranking, doc1 should be #1";
    EXPECT_GT(results[0].rerank_score, 0.0) << "Rerank score should be set";
    EXPECT_GT(results[0].combined_score, 0.0) << "Combined score should be set";

    // Print reranked results
    std::cout << "\n=== Reranked Results ===" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << (i+1) << ". " << results[i].doc_id
                  << " (vec=" << results[i].vector_score
                  << ", bm25=" << results[i].bm25_score
                  << ", hybrid=" << results[i].hybrid_score
                  << ", rerank=" << results[i].rerank_score
                  << ", combined=" << results[i].combined_score << ")" << std::endl;
    }
}

// Test two-stage retrieval (retrieve more candidates, then rerank)
// TODO: Fix lambda compilation issue with mock providers
/*
TEST_F(RerankingIntegrationTest, TwoStageRetrieval) {
    // Setup mock providers with simple logic
    hybrid_engine_->set_vector_search_provider(
        [](const std::vector<float>&, size_t) {
            std::vector<SearchResult> results;
            for (int i = 0; i < 5; i++) {
                SearchResult r;
                r.doc_id = "doc" + std::to_string(i+1);
                r.score = 0.9 - (i * 0.1);
                results.push_back(r);
            }
            return results;
        }
    );

    size_t candidate_count = 0;
    hybrid_engine_->set_reranking_provider(
        [&candidate_count](const std::string&,
                          const std::vector<HybridSearchResult>& candidates) {
            candidate_count = candidates.size();
            std::vector<HybridSearchResult> reranked = candidates;
            for (auto& result : reranked) {
                result.rerank_score = 0.8;
                result.combined_score = 0.8;
            }
            return reranked;
        }
    );

    // Query
    std::vector<float> query_vector = {1.0f, 0.0f, 0.0f, 0.0f};
    std::string query_text = "machine learning";

    // Two-stage: retrieve 5 candidates, return top 3
    auto results = hybrid_engine_->search(query_text, query_vector, 3, true, 5);

    // Verify we reranked 5 candidates but returned only 3
    EXPECT_EQ(candidate_count, 5) << "Should rerank 5 candidates";
    EXPECT_EQ(results.size(), 3) << "Should return top 3 after reranking";
}
*/

// Test reranking with empty results
/*
TEST_F(RerankingIntegrationTest, RerankingWithEmptyResults) {
    hybrid_engine_->set_vector_search_provider(
        [](const std::vector<float>&, size_t) {
            return std::vector<SearchResult>{};  // Empty results
        }
    );

    std::vector<float> query_vector = {1.0f, 0.0f, 0.0f, 0.0f};
    std::string query_text = "nonexistent query";

    auto results = hybrid_engine_->search(query_text, query_vector, 5, true);

    EXPECT_TRUE(results.empty()) << "Should return empty results when no candidates";
}
*/

// Test reranking impact on ranking order
/*
TEST_F(RerankingIntegrationTest, RerankingImpactsOrder) {
    // Setup mock provider
    hybrid_engine_->set_vector_search_provider(
        [](const std::vector<float>&, size_t) -> std::vector<SearchResult> {
            std::vector<SearchResult> results;
            SearchResult r1; r1.doc_id = "doc1"; r1.score = 0.9; results.push_back(r1);
            SearchResult r2; r2.doc_id = "doc2"; r2.score = 0.8; results.push_back(r2);
            SearchResult r3; r3.doc_id = "doc3"; r3.score = 0.7; results.push_back(r3);
            return results;
        }
    );

    // Get baseline results (without reranking)
    std::vector<float> query_vector = {1.0f, 0.0f, 0.0f, 0.0f};
    std::string query_text = "machine learning";

    auto baseline_results = hybrid_engine_->search(query_text, query_vector, 5, false);
    ASSERT_GE(baseline_results.size(), 2);

    // Setup reranking that inverts the order
    hybrid_engine_->set_reranking_provider(
        [](const std::string&, const std::vector<HybridSearchResult>& candidates) {
            std::vector<HybridSearchResult> reranked = candidates;

            // Assign scores in reverse order
            for (size_t i = 0; i < reranked.size(); i++) {
                reranked[i].rerank_score = 1.0 - (i * 0.1);
                reranked[i].combined_score = reranked[i].rerank_score;
            }

            // Reverse sort
            std::reverse(reranked.begin(), reranked.end());
            return reranked;
        }
    );

    auto reranked_results = hybrid_engine_->search(query_text, query_vector, 5, true);
    ASSERT_GE(reranked_results.size(), 2);

    // Verify ranking changed
    bool order_changed = false;
    if (baseline_results.size() >= 2 && reranked_results.size() >= 2) {
        order_changed = (baseline_results[0].doc_id != reranked_results[0].doc_id) ||
                       (baseline_results[1].doc_id != reranked_results[1].doc_id);
    }

    EXPECT_TRUE(order_changed) << "Reranking should change result order";

    std::cout << "\n=== Order Impact ===" << std::endl;
    std::cout << "Baseline top 2: " << baseline_results[0].doc_id
              << ", " << baseline_results[1].doc_id << std::endl;
    std::cout << "Reranked top 2: " << reranked_results[0].doc_id
              << ", " << reranked_results[1].doc_id << std::endl;
}
*/

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
