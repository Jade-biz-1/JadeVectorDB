#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;

// Test fixture for SimilaritySearchService - Integration Tests
class SimilaritySearchServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging
        logging::LoggerManager::initialize(logging::LogLevel::ERROR);
        
        // Create database layer
        auto db_layer = std::make_unique<DatabaseLayer>();
        auto db_init = db_layer->initialize();
        ASSERT_TRUE(db_init.has_value()) << "Failed to initialize database layer: " << db_init.error().message;
        
        // Create vector storage service
        auto vector_storage = std::make_unique<VectorStorageService>(std::move(db_layer));
        auto vs_init = vector_storage->initialize();
        ASSERT_TRUE(vs_init.has_value()) << "Failed to initialize vector storage: " << vs_init.error().message;
        
        // Create similarity search service
        similarity_search_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage));
        auto ss_init = similarity_search_->initialize();
        ASSERT_TRUE(ss_init.has_value()) << "Failed to initialize similarity search: " << ss_init.error().message;
        
        // Create test database
        Database test_db;
        test_db.name = "test_db_sim";
        test_db.vectorDimension = 128;
        test_db.indexType = "FLAT";
        test_db.description = "Test database for similarity search";
        
        auto db_result = similarity_search_->get_vector_storage_for_testing()->get_db_layer_for_testing()->create_database(test_db);
        ASSERT_TRUE(db_result.has_value()) << "Failed to create test database: " << db_result.error().message;
        test_db_id_ = db_result.value();
    }
    
    void TearDown() override {
        // Clean up test database
        if (!test_db_id_.empty()) {
            similarity_search_->get_vector_storage_for_testing()->get_db_layer_for_testing()->delete_database(test_db_id_);
        }
        
        // Clean up
        similarity_search_.reset();
        
        logging::LoggerManager::shutdown();
    }
    
    // Helper to create a test vector
    Vector create_test_vector(const std::string& id, const std::vector<float>& values) {
        Vector v;
        v.id = id;
        v.values = values;
        v.databaseId = test_db_id_;
        v.metadata.status = "active";
        return v;
    }
    
    std::string test_db_id_;
    std::unique_ptr<SimilaritySearchService> similarity_search_;
};

// Test that the service initializes correctly
TEST_F(SimilaritySearchServiceTest, InitializeService) {
    EXPECT_NE(similarity_search_, nullptr);
}

// Test basic vector search with cosine similarity
TEST_F(SimilaritySearchServiceTest, BasicCosineSimilaritySearch) {
    // Store test vectors - create vectors close to query and far from query
    std::vector<float> base_data(128, 0.0f);
    base_data[0] = 1.0f;
    
    Vector v1 = create_test_vector("vec1", base_data);
    
    std::vector<float> similar_data = base_data;
    similar_data[0] = 0.95f;
    similar_data[1] = 0.05f;
    Vector v2 = create_test_vector("vec2", similar_data);
    
    std::vector<float> diff_data(128, 0.0f);
    diff_data[10] = 1.0f;
    Vector v3 = create_test_vector("vec3", diff_data);
    
    // Store vectors
    auto store1 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v1);
    auto store2 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v2);
    auto store3 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v3);
    
    ASSERT_TRUE(store1.has_value()) << "Failed to store vector 1: " << store1.error().message;
    ASSERT_TRUE(store2.has_value()) << "Failed to store vector 2: " << store2.error().message;
    ASSERT_TRUE(store3.has_value()) << "Failed to store vector 3: " << store3.error().message;
    
    // Create query vector matching base_data
    Vector query = create_test_vector("query", base_data);
    
    // Search for similar vectors
    SearchParams params;
    params.top_k = 2;
    params.threshold = 0.0f;
    
    auto search_result = similarity_search_->similarity_search(test_db_id_, query, params);
    ASSERT_TRUE(search_result.has_value()) << "Search failed: " << search_result.error().message;
    
    auto results = search_result.value();
    ASSERT_GE(results.size(), 1);
    
    // First result should be vec1 (identical vector)
    EXPECT_EQ(results[0].vector_id, "vec1");
}

// Test Euclidean distance search
TEST_F(SimilaritySearchServiceTest, EuclideanDistanceSearch) {
    // Create vectors at known distances
    std::vector<float> origin(128, 0.0f);
    Vector v1 = create_test_vector("origin", origin);
    
    std::vector<float> close(128, 0.0f);
    close[0] = 1.0f;  // Distance = 1.0
    Vector v2 = create_test_vector("close", close);
    
    std::vector<float> far(128, 0.0f);
    far[0] = 10.0f;  // Distance = 10.0
    Vector v3 = create_test_vector("far", far);
    
    // Store vectors
    auto store1 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v1);
    auto store2 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v2);
    auto store3 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v3);
    
    ASSERT_TRUE(store1.has_value());
    ASSERT_TRUE(store2.has_value());
    ASSERT_TRUE(store3.has_value());
    
    // Create query vector
    Vector query = create_test_vector("query", origin);
    
    // Search using Euclidean distance
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    
    auto search_result = similarity_search_->euclidean_search(test_db_id_, query, params);
    ASSERT_TRUE(search_result.has_value());
    
    auto results = search_result.value();
    ASSERT_GE(results.size(), 1);
    
    // Closest result should be origin (distance = 0)
    EXPECT_EQ(results[0].vector_id, "origin");
}

// Test dot product similarity
TEST_F(SimilaritySearchServiceTest, DotProductSearch) {
    // Create vectors with known dot products
    std::vector<float> query_data(128, 0.0f);
    query_data[0] = 1.0f;
    query_data[1] = 1.0f;
    
    std::vector<float> high_dot(128, 0.0f);
    high_dot[0] = 2.0f;
    high_dot[1] = 2.0f;  // Dot product = 4.0
    Vector v1 = create_test_vector("high", high_dot);
    
    std::vector<float> low_dot(128, 0.0f);
    low_dot[0] = 0.5f;
    low_dot[1] = 0.5f;  // Dot product = 1.0
    Vector v2 = create_test_vector("low", low_dot);
    
    // Store vectors
    auto store1 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v1);
    auto store2 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v2);
    
    ASSERT_TRUE(store1.has_value());
    ASSERT_TRUE(store2.has_value());
    
    // Create query vector
    Vector query = create_test_vector("query", query_data);
    
    // Search using dot product
    SearchParams params;
    params.top_k = 2;
    params.threshold = 0.0f;
    
    auto search_result = similarity_search_->dot_product_search(test_db_id_, query, params);
    ASSERT_TRUE(search_result.has_value());
    
    auto results = search_result.value();
    ASSERT_GE(results.size(), 1);
    
    // Higher dot product should rank first
    EXPECT_EQ(results[0].vector_id, "high");
}

// Test search with metadata filters
TEST_F(SimilaritySearchServiceTest, SearchWithMetadataFilters) {
    // Create vectors with metadata
    Vector v1 = create_test_vector("v1", std::vector<float>(128, 1.0f));
    v1.metadata.category = "A";
    v1.metadata.custom["score"] = 10;
    
    Vector v2 = create_test_vector("v2", std::vector<float>(128, 1.0f));
    v2.metadata.category = "B";
    v2.metadata.custom["score"] = 20;
    
    Vector v3 = create_test_vector("v3", std::vector<float>(128, 1.0f));
    v3.metadata.category = "A";
    v3.metadata.custom["score"] = 30;
    
    // Store vectors
    auto store1 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v1);
    auto store2 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v2);
    auto store3 = similarity_search_->get_vector_storage_for_testing()->store_vector(test_db_id_, v3);
    
    ASSERT_TRUE(store1.has_value());
    ASSERT_TRUE(store2.has_value());
    ASSERT_TRUE(store3.has_value());
    
    // Create query vector
    Vector query = create_test_vector("query", std::vector<float>(128, 1.0f));
    
    // Search with metadata filter
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.filter_category = "A";
    
    auto search_result = similarity_search_->similarity_search(test_db_id_, query, params);
    ASSERT_TRUE(search_result.has_value());
    
    auto results = search_result.value();
    
    // Should only return vectors with category "A"
    for (const auto& result : results) {
        EXPECT_TRUE(result.vector_id == "v1" || result.vector_id == "v3");
    }
}

// Test empty database search
TEST_F(SimilaritySearchServiceTest, SearchEmptyDatabase) {
    Vector query = create_test_vector("query", std::vector<float>(128, 1.0f));
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    
    auto search_result = similarity_search_->similarity_search(test_db_id_, query, params);
    
    // Should return empty results, not an error
    if (search_result.has_value()) {
        EXPECT_EQ(search_result.value().size(), 0);
    }
}
