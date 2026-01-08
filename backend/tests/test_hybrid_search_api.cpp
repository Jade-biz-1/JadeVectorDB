#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/search/hybrid_search_engine.h"
#include "services/search/bm25_scorer.h"
#include "api/rest/rest_api.h"
#include "models/vector.h"
#include "models/database.h"

namespace jadevectordb {

// Test fixture for hybrid search API tests
class HybridSearchAPITest : public ::testing::Test {
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

        // Create a test database
        Database db;
        db.name = "hybrid_search_test_db";
        db.description = "Test database for hybrid search API";
        db.vectorDimension = 4;

        auto result = db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();

        // Add test vectors with text content
        addTestVectors();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            db_layer_->delete_database(test_database_id_);
        }
    }

    void addTestVectors() {
        // Vector 1: Machine learning document
        Vector v1;
        v1.id = "doc_ml_1";
        v1.databaseId = test_database_id_;
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
        v1.metadata.source = "Machine learning algorithms for data analysis";
        v1.metadata.tags = {"machine_learning", "algorithms"};
        v1.metadata.category = "AI";
        v1.metadata.owner = "researcher1";
        v1.metadata.status = "active";

        // Vector 2: Deep learning document
        Vector v2;
        v2.id = "doc_dl_1";
        v2.databaseId = test_database_id_;
        v2.values = {0.9f, 0.1f, 0.0f, 0.0f};
        v2.metadata.source = "Deep learning neural networks for image recognition";
        v2.metadata.tags = {"deep_learning", "neural_networks"};
        v2.metadata.category = "AI";
        v2.metadata.owner = "researcher2";
        v2.metadata.status = "active";

        // Vector 3: Natural language processing
        Vector v3;
        v3.id = "doc_nlp_1";
        v3.databaseId = test_database_id_;
        v3.values = {0.0f, 1.0f, 0.0f, 0.0f};
        v3.metadata.source = "Natural language processing with machine learning";
        v3.metadata.tags = {"nlp", "machine_learning"};
        v3.metadata.category = "AI";
        v3.metadata.owner = "researcher1";
        v3.metadata.status = "active";

        // Vector 4: Data science
        Vector v4;
        v4.id = "doc_ds_1";
        v4.databaseId = test_database_id_;
        v4.values = {0.5f, 0.5f, 0.5f, 0.5f};
        v4.metadata.source = "Data science techniques and statistical analysis";
        v4.metadata.tags = {"data_science", "statistics"};
        v4.metadata.category = "Analytics";
        v4.metadata.owner = "data_scientist";
        v4.metadata.status = "active";

        vector_storage_->store_vector(test_database_id_, v1);
        vector_storage_->store_vector(test_database_id_, v2);
        vector_storage_->store_vector(test_database_id_, v3);
        vector_storage_->store_vector(test_database_id_, v4);
    }

    std::shared_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_;
    std::unique_ptr<SimilaritySearchService> similarity_search_;
    std::string test_database_id_;
};

// Test BM25-only search
TEST_F(HybridSearchAPITest, BM25OnlySearch) {
    jadedb::search::HybridSearchConfig config;
    config.fusion_method = jadedb::search::FusionMethod::RRF;

    auto engine = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);

    // Build BM25 index from test documents
    std::vector<jadedb::search::BM25Document> documents;

    jadedb::search::BM25Document doc1;
    doc1.doc_id = "doc_ml_1";
    doc1.text = "Machine learning algorithms for data analysis";
    documents.push_back(doc1);

    jadedb::search::BM25Document doc2;
    doc2.doc_id = "doc_dl_1";
    doc2.text = "Deep learning neural networks for image recognition";
    documents.push_back(doc2);

    jadedb::search::BM25Document doc3;
    doc3.doc_id = "doc_nlp_1";
    doc3.text = "Natural language processing with machine learning";
    documents.push_back(doc3);

    jadedb::search::BM25Document doc4;
    doc4.doc_id = "doc_ds_1";
    doc4.text = "Data science techniques and statistical analysis";
    documents.push_back(doc4);

    ASSERT_TRUE(engine->build_bm25_index(documents));

    // Search for "machine learning"
    auto results = engine->search_bm25_only("machine learning", 3);

    ASSERT_GT(results.size(), 0);

    // Should find documents containing "machine learning"
    bool found_ml_doc = false;
    bool found_nlp_doc = false;

    for (const auto& result : results) {
        if (result.doc_id == "doc_ml_1") found_ml_doc = true;
        if (result.doc_id == "doc_nlp_1") found_nlp_doc = true;
    }

    EXPECT_TRUE(found_ml_doc) << "Should find ML document";
    EXPECT_TRUE(found_nlp_doc) << "Should find NLP document (contains 'machine learning')";
}

// Test hybrid search with both vector and text
TEST_F(HybridSearchAPITest, HybridSearchWithBothQueries) {
    jadedb::search::HybridSearchConfig config;
    config.fusion_method = jadedb::search::FusionMethod::RRF;
    config.rrf_k = 60;

    auto engine = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);

    // Build BM25 index
    std::vector<jadedb::search::BM25Document> documents;

    jadedb::search::BM25Document doc1;
    doc1.doc_id = "doc_ml_1";
    doc1.text = "Machine learning algorithms for data analysis";
    documents.push_back(doc1);

    jadedb::search::BM25Document doc2;
    doc2.doc_id = "doc_dl_1";
    doc2.text = "Deep learning neural networks for image recognition";
    documents.push_back(doc2);

    jadedb::search::BM25Document doc3;
    doc3.doc_id = "doc_nlp_1";
    doc3.text = "Natural language processing with machine learning";
    documents.push_back(doc3);

    jadedb::search::BM25Document doc4;
    doc4.doc_id = "doc_ds_1";
    doc4.text = "Data science techniques and statistical analysis";
    documents.push_back(doc4);

    ASSERT_TRUE(engine->build_bm25_index(documents));

    // Set vector search provider
    engine->set_vector_search_provider(
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

    // Perform hybrid search
    std::vector<float> query_vector = {0.95f, 0.05f, 0.0f, 0.0f}; // Similar to doc_ml_1 and doc_dl_1
    std::string query_text = "machine learning";

    auto results = engine->search(query_text, query_vector, 3);

    ASSERT_GT(results.size(), 0);

    // Verify results have both vector and BM25 scores
    for (const auto& result : results) {
        EXPECT_GT(result.hybrid_score, 0.0) << "Hybrid score should be positive";
    }
}

// Test RRF fusion method
TEST_F(HybridSearchAPITest, RRFFusionMethod) {
    jadedb::search::HybridSearchConfig config;
    config.fusion_method = jadedb::search::FusionMethod::RRF;
    config.rrf_k = 60;

    auto engine = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);

    EXPECT_EQ(engine->get_config().fusion_method, jadedb::search::FusionMethod::RRF);
    EXPECT_EQ(engine->get_config().rrf_k, 60);
}

// Test weighted linear fusion method
TEST_F(HybridSearchAPITest, WeightedLinearFusionMethod) {
    jadedb::search::HybridSearchConfig config;
    config.fusion_method = jadedb::search::FusionMethod::LINEAR;
    config.alpha = 0.7;

    auto engine = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);

    EXPECT_EQ(engine->get_config().fusion_method, jadedb::search::FusionMethod::LINEAR);
    EXPECT_EQ(engine->get_config().alpha, 0.7);
}

// Test config update
TEST_F(HybridSearchAPITest, ConfigUpdate) {
    jadedb::search::HybridSearchConfig config;
    config.fusion_method = jadedb::search::FusionMethod::RRF;

    auto engine = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);

    // Update config
    jadedb::search::HybridSearchConfig new_config;
    new_config.fusion_method = jadedb::search::FusionMethod::LINEAR;
    new_config.alpha = 0.5;

    engine->set_config(new_config);

    EXPECT_EQ(engine->get_config().fusion_method, jadedb::search::FusionMethod::LINEAR);
    EXPECT_EQ(engine->get_config().alpha, 0.5);
}

// Test BM25 index persistence
TEST_F(HybridSearchAPITest, BM25IndexPersistence) {
    jadedb::search::HybridSearchConfig config;

    auto engine = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);

    // Build BM25 index
    std::vector<jadedb::search::BM25Document> documents;

    jadedb::search::BM25Document doc1;
    doc1.doc_id = "doc_ml_1";
    doc1.text = "Machine learning algorithms for data analysis";
    documents.push_back(doc1);

    ASSERT_TRUE(engine->build_bm25_index(documents));
    EXPECT_TRUE(engine->is_bm25_index_ready());

    // Test persistence (save and load)
    std::string persistence_path = "/tmp/test_bm25_index.db";
    ASSERT_TRUE(engine->save_bm25_index(persistence_path));

    // Create new engine and load index
    auto engine2 = std::make_shared<jadedb::search::HybridSearchEngine>(test_database_id_, config);
    ASSERT_TRUE(engine2->load_bm25_index(persistence_path));
    EXPECT_TRUE(engine2->is_bm25_index_ready());

    // Clean up
    std::remove(persistence_path.c_str());
}

} // namespace jadevectordb
