#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/search/hybrid_search_engine.h"
#include "models/vector.h"
#include "models/database.h"

using namespace jadedb::search;
using namespace jadevectordb;

// Test fixture for hybrid search integration tests
class HybridSearchIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize database layer
        db_layer_ = std::make_shared<DatabaseLayer>();
        db_layer_->initialize();

        // Initialize services
        db_service_ = std::make_unique<DatabaseService>(db_layer_);
        db_service_->initialize();

        vector_storage_ = std::make_unique<VectorStorageService>(db_layer_);
        vector_storage_->initialize();

        auto search_vector_storage = std::make_unique<VectorStorageService>(db_layer_);
        search_vector_storage->initialize();
        similarity_search_ = std::make_unique<SimilaritySearchService>(std::move(search_vector_storage));
        similarity_search_->initialize();

        // Create test database
        Database db;
        db.name = "hybrid_test_db";
        db.description = "Test database for hybrid search";
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
        // Create vectors with associated text content
        // Theme: AI/ML documents with different semantic and keyword relevance

        // Document 1: Strong match for "machine learning" (keyword + semantic)
        Vector v1;
        v1.id = "doc_ml_1";
        v1.databaseId = test_database_id_;
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};  // Unique position
        v1.metadata.source = "Machine learning algorithms for data analysis and pattern recognition";
        v1.metadata.category = "AI";
        v1.metadata.tags = {"machine_learning", "algorithms"};
        v1.metadata.status = "active";
        vectors_.push_back(v1);

        // Document 2: Strong semantic match but different keywords
        Vector v2;
        v2.id = "doc_ml_2";
        v2.databaseId = test_database_id_;
        v2.values = {0.95f, 0.05f, 0.0f, 0.0f};  // Very close to v1
        v2.metadata.source = "Supervised and unsupervised learning techniques";
        v2.metadata.category = "AI";
        v2.metadata.tags = {"learning", "supervised"};
        v2.metadata.status = "active";
        vectors_.push_back(v2);

        // Document 3: Strong keyword match for "machine learning"
        Vector v3;
        v3.id = "doc_ml_3";
        v3.databaseId = test_database_id_;
        v3.values = {0.0f, 1.0f, 0.0f, 0.0f};  // Different semantic space
        v3.metadata.source = "Machine learning revolutionizes modern computing with machine intelligence";
        v3.metadata.category = "AI";
        v3.metadata.tags = {"machine_learning", "computing"};
        v3.metadata.status = "active";
        vectors_.push_back(v3);

        // Document 4: Related but different topic (deep learning)
        Vector v4;
        v4.id = "doc_dl_1";
        v4.databaseId = test_database_id_;
        v4.values = {0.7f, 0.3f, 0.0f, 0.0f};  // Somewhat similar to v1
        v4.metadata.source = "Deep learning neural networks for image recognition";
        v4.metadata.category = "AI";
        v4.metadata.tags = {"deep_learning", "neural_networks"};
        v4.metadata.status = "active";
        vectors_.push_back(v4);

        // Document 5: Different topic (NLP)
        Vector v5;
        v5.id = "doc_nlp_1";
        v5.databaseId = test_database_id_;
        v5.values = {0.0f, 0.0f, 1.0f, 0.0f};  // Different semantic space
        v5.metadata.source = "Natural language processing for text understanding";
        v5.metadata.category = "AI";
        v5.metadata.tags = {"nlp", "text"};
        v5.metadata.status = "active";
        vectors_.push_back(v5);

        // Store vectors
        for (const auto& vec : vectors_) {
            auto result = vector_storage_->store_vector(test_database_id_, vec);
            ASSERT_TRUE(result.has_value()) << "Failed to store vector: " << vec.id;
        }

        // Build BM25 documents from vector metadata
        for (const auto& vec : vectors_) {
            BM25Document doc;
            doc.doc_id = vec.id;
            doc.text = vec.metadata.source;
            bm25_documents_.push_back(doc);
        }
    }

    std::shared_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_;
    std::unique_ptr<SimilaritySearchService> similarity_search_;
    std::string test_database_id_;
    std::vector<Vector> vectors_;
    std::vector<BM25Document> bm25_documents_;
};

// Test BM25-only search
TEST_F(HybridSearchIntegrationTest, BM25OnlySearch) {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::RRF;

    HybridSearchEngine engine(test_database_id_, config);

    // Build BM25 index
    bool success = engine.build_bm25_index(bm25_documents_);
    ASSERT_TRUE(success) << "BM25 index build should succeed";
    ASSERT_TRUE(engine.is_bm25_index_ready()) << "BM25 index should be ready";

    // Search for "machine learning"
    auto results = engine.search_bm25_only("machine learning", 5);

    ASSERT_GT(results.size(), 0) << "Should return results";

    // Documents containing "machine learning" should rank higher
    // doc_ml_1 and doc_ml_3 both contain "machine learning"
    bool found_ml_1 = false;
    bool found_ml_3 = false;

    for (const auto& result : results) {
        if (result.doc_id == "doc_ml_1") {
            found_ml_1 = true;
            EXPECT_GT(result.bm25_score, 0.0) << "BM25 score should be positive";
        }
        if (result.doc_id == "doc_ml_3") {
            found_ml_3 = true;
            EXPECT_GT(result.bm25_score, 0.0) << "BM25 score should be positive";
        }
    }

    EXPECT_TRUE(found_ml_1 || found_ml_3) << "Should find at least one document with 'machine learning'";
}

// Test vector similarity search baseline (through similarity search service)
TEST_F(HybridSearchIntegrationTest, VectorSimilarityBaseline) {
    // Query vector similar to v1 (doc_ml_1)
    Vector query_vec;
    query_vec.values = {0.99f, 0.01f, 0.0f, 0.0f};

    SearchParams params;
    params.top_k = 5;
    params.threshold = 0.0;

    auto result = similarity_search_->similarity_search(test_database_id_, query_vec, params);

    ASSERT_TRUE(result.has_value()) << "Vector search should succeed";
    ASSERT_GT(result.value().size(), 0) << "Should return vector search results";

    // doc_ml_1 should be top result (or very close) since query is most similar to it
    bool found_ml_1_high = false;
    for (size_t i = 0; i < std::min(size_t(3), result.value().size()); i++) {
        if (result.value()[i].vector_id == "doc_ml_1") {
            found_ml_1_high = true;
            break;
        }
    }

    EXPECT_TRUE(found_ml_1_high) << "doc_ml_1 should be in top 3 vector results";
}

// Test hybrid search with RRF fusion
TEST_F(HybridSearchIntegrationTest, HybridSearchRRF) {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::RRF;
    config.rrf_k = 60;
    config.vector_candidates = 100;
    config.bm25_candidates = 100;

    HybridSearchEngine engine(test_database_id_, config);

    // Build BM25 index
    bool success = engine.build_bm25_index(bm25_documents_);
    ASSERT_TRUE(success);

    // Set up vector search provider
    engine.set_vector_search_provider(
        [this](const std::vector<float>& query_vector, size_t top_k) -> std::vector<jadedb::search::SearchResult> {
            Vector vec;
            vec.values = query_vector;

            SearchParams params;
            params.top_k = top_k;
            params.threshold = 0.0;

            auto result = similarity_search_->similarity_search(test_database_id_, vec, params);

            std::vector<jadedb::search::SearchResult> search_results;
            if (result.has_value()) {
                for (const auto& sr : result.value()) {
                    jadedb::search::SearchResult res;
                    res.doc_id = sr.vector_id;
                    res.score = sr.similarity_score;
                    search_results.push_back(res);
                }
            }

            return search_results;
        }
    );

    // Hybrid query: vector similar to doc_ml_1 + text "machine learning"
    std::vector<float> query_vector = {0.98f, 0.02f, 0.0f, 0.0f};
    std::string query_text = "machine learning";

    auto results = engine.search(query_text, query_vector, 5);

    ASSERT_GT(results.size(), 0) << "Should return hybrid search results";

    // Verify results have both scores
    for (const auto& result : results) {
        EXPECT_GT(result.hybrid_score, 0.0) << "Hybrid score should be positive";
        // At least one of the component scores should be positive
        EXPECT_TRUE(result.vector_score > 0.0 || result.bm25_score > 0.0)
            << "At least one component score should be positive";
    }

    // doc_ml_1 should rank highly (strong in both vector and keyword match)
    bool found_ml_1 = false;
    for (const auto& result : results) {
        if (result.doc_id == "doc_ml_1") {
            found_ml_1 = true;
            EXPECT_GT(result.vector_score, 0.0) << "Should have vector score";
            EXPECT_GT(result.bm25_score, 0.0) << "Should have BM25 score";
            break;
        }
    }

    EXPECT_TRUE(found_ml_1) << "doc_ml_1 should be in hybrid results";
}

// Test hybrid search with weighted linear fusion
TEST_F(HybridSearchIntegrationTest, HybridSearchLinearFusion) {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::LINEAR;
    config.alpha = 0.7;  // 70% vector, 30% BM25

    HybridSearchEngine engine(test_database_id_, config);

    // Build BM25 index
    bool success = engine.build_bm25_index(bm25_documents_);
    ASSERT_TRUE(success);

    // Set up vector search provider
    engine.set_vector_search_provider(
        [this](const std::vector<float>& query_vector, size_t top_k) -> std::vector<jadedb::search::SearchResult> {
            Vector vec;
            vec.values = query_vector;

            SearchParams params;
            params.top_k = top_k;
            params.threshold = 0.0;

            auto result = similarity_search_->similarity_search(test_database_id_, vec, params);

            std::vector<jadedb::search::SearchResult> search_results;
            if (result.has_value()) {
                for (const auto& sr : result.value()) {
                    jadedb::search::SearchResult res;
                    res.doc_id = sr.vector_id;
                    res.score = sr.similarity_score;
                    search_results.push_back(res);
                }
            }

            return search_results;
        }
    );

    // Hybrid search
    std::vector<float> query_vector = {0.97f, 0.03f, 0.0f, 0.0f};
    std::string query_text = "machine learning algorithms";

    auto results = engine.search(query_text, query_vector, 5);

    ASSERT_GT(results.size(), 0) << "Should return results with linear fusion";

    // Verify hybrid scores are computed
    for (const auto& result : results) {
        EXPECT_GT(result.hybrid_score, 0.0) << "Hybrid score should be positive";
    }
}

// Test that hybrid search combines both signals effectively
TEST_F(HybridSearchIntegrationTest, HybridSearchCombinesBothSignals) {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::RRF;

    HybridSearchEngine engine(test_database_id_, config);

    // Build BM25 index
    ASSERT_TRUE(engine.build_bm25_index(bm25_documents_));

    // Set up vector search provider
    engine.set_vector_search_provider(
        [this](const std::vector<float>& query_vector, size_t top_k) -> std::vector<jadedb::search::SearchResult> {
            Vector vec;
            vec.values = query_vector;

            SearchParams params;
            params.top_k = top_k;
            params.threshold = 0.0;

            auto result = similarity_search_->similarity_search(test_database_id_, vec, params);

            std::vector<jadedb::search::SearchResult> search_results;
            if (result.has_value()) {
                for (const auto& sr : result.value()) {
                    jadedb::search::SearchResult res;
                    res.doc_id = sr.vector_id;
                    res.score = sr.similarity_score;
                    search_results.push_back(res);
                }
            }

            return search_results;
        }
    );

    // Query: vector close to doc_ml_2, but keywords match doc_ml_1 and doc_ml_3
    std::vector<float> query_vector = {0.95f, 0.05f, 0.0f, 0.0f};  // Close to doc_ml_2
    std::string query_text = "machine learning";  // Matches doc_ml_1 and doc_ml_3

    auto results = engine.search(query_text, query_vector, 5);

    ASSERT_GT(results.size(), 0);

    // The hybrid search should consider both:
    // - doc_ml_2 has high vector similarity
    // - doc_ml_1 has both good vector similarity AND keyword match
    // - doc_ml_3 has keyword match but different vector

    // At least one of these should be in top results
    bool found_relevant_doc = false;
    for (size_t i = 0; i < std::min(size_t(3), results.size()); i++) {
        if (results[i].doc_id == "doc_ml_1" ||
            results[i].doc_id == "doc_ml_2" ||
            results[i].doc_id == "doc_ml_3") {
            found_relevant_doc = true;
            break;
        }
    }

    EXPECT_TRUE(found_relevant_doc) << "Top results should include relevant documents";
}

// Test different alpha values for linear fusion
TEST_F(HybridSearchIntegrationTest, DifferentAlphaValues) {
    HybridSearchEngine engine(test_database_id_, HybridSearchConfig());

    // Build BM25 index
    ASSERT_TRUE(engine.build_bm25_index(bm25_documents_));

    // Set up vector search provider
    engine.set_vector_search_provider(
        [this](const std::vector<float>& query_vector, size_t top_k) -> std::vector<jadedb::search::SearchResult> {
            Vector vec;
            vec.values = query_vector;
            SearchParams params;
            params.top_k = top_k;
            params.threshold = 0.0;

            auto result = similarity_search_->similarity_search(test_database_id_, vec, params);

            std::vector<jadedb::search::SearchResult> search_results;
            if (result.has_value()) {
                for (const auto& sr : result.value()) {
                    jadedb::search::SearchResult res;
                    res.doc_id = sr.vector_id;
                    res.score = sr.similarity_score;
                    search_results.push_back(res);
                }
            }
            return search_results;
        }
    );

    std::vector<float> query_vector = {1.0f, 0.0f, 0.0f, 0.0f};
    std::string query_text = "machine learning";

    // Test with different alpha values
    std::vector<double> alphas = {0.2, 0.5, 0.8};

    for (double alpha : alphas) {
        HybridSearchConfig config;
        config.fusion_method = FusionMethod::LINEAR;
        config.alpha = alpha;

        engine.set_config(config);

        auto results = engine.search(query_text, query_vector, 5);

        ASSERT_GT(results.size(), 0) << "Should return results for alpha=" << alpha;

        // Verify all results have positive hybrid scores
        for (const auto& result : results) {
            EXPECT_GT(result.hybrid_score, 0.0) << "Hybrid score should be positive for alpha=" << alpha;
        }
    }
}

// Test config getter
TEST_F(HybridSearchIntegrationTest, ConfigManagement) {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::RRF;
    config.rrf_k = 75;
    config.alpha = 0.6;

    HybridSearchEngine engine(test_database_id_, config);

    auto retrieved_config = engine.get_config();

    EXPECT_EQ(retrieved_config.fusion_method, FusionMethod::RRF);
    EXPECT_EQ(retrieved_config.rrf_k, 75);
    EXPECT_DOUBLE_EQ(retrieved_config.alpha, 0.6);
}

// Test empty query handling
TEST_F(HybridSearchIntegrationTest, EmptyQueryHandling) {
    HybridSearchEngine engine(test_database_id_, HybridSearchConfig());

    ASSERT_TRUE(engine.build_bm25_index(bm25_documents_));

    // Empty text query should not crash
    auto results = engine.search_bm25_only("", 5);
    // Empty query returns no results or all results with zero scores
    // Either is acceptable
}
