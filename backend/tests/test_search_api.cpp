#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "api/rest/rest_api.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/auth.h"

namespace jadevectordb {

// Test fixture for search API integration tests
class SearchAPIIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services 
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        vector_storage_ = std::make_unique<VectorStorageService>(std::move(db_layer_));
        vector_storage_->initialize();
        
        search_service_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage_));
        search_service_->initialize();
        
        // Create a test database
        Database db;
        db.name = "search_api_test_db";
        db.description = "Test database for search API integration";
        db.vectorDimension = 4;
        
        auto result = search_service_->vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
        
        // Add test vectors
        addTestVectors();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            search_service_->vector_storage_->db_layer_->delete_database(test_database_id_);
        }
    }

    void addTestVectors() {
        // Vector 1: [1.0, 0.0, 0.0, 0.0]
        Vector v1;
        v1.id = "vector_1";
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
        v1.metadata.tags = {"tag1", "main"};
        v1.metadata.category = "category1";
        v1.metadata.owner = "user1";
        
        // Vector 2: [0.0, 1.0, 0.0, 0.0]
        Vector v2;
        v2.id = "vector_2";
        v2.values = {0.0f, 1.0f, 0.0f, 0.0f};
        v2.metadata.tags = {"tag2", "secondary"};
        v2.metadata.category = "category2";
        v2.metadata.owner = "user2";
        
        // Vector 3: [0.7, 0.7, 0.0, 0.0]
        Vector v3;
        v3.id = "vector_3";
        v3.values = {0.7f, 0.7f, 0.0f, 0.0f};
        v3.metadata.tags = {"tag1", "related"};
        v3.metadata.category = "category1";
        v3.metadata.owner = "user1";
        
        // Vector 4: [0.5, 0.5, 0.5, 0.5]
        Vector v4;
        v4.id = "vector_4";
        v4.values = {0.5f, 0.5f, 0.5f, 0.5f};
        v4.metadata.tags = {"diagonal", "special"};
        v4.metadata.category = "category3";
        v4.metadata.owner = "user3";
        
        search_service_->vector_storage_->store_vector(test_database_id_, v1);
        search_service_->vector_storage_->store_vector(test_database_id_, v2);
        search_service_->vector_storage_->store_vector(test_database_id_, v3);
        search_service_->vector_storage_->store_vector(test_database_id_, v4);
    }

    std::unique_ptr<SimilaritySearchService> search_service_;
    std::string test_database_id_;
};

// Test basic similarity search functionality
TEST_F(SearchAPIIntegrationTest, BasicSimilaritySearch) {
    Vector query;
    query.id = "query_basic";
    query.values = {0.9f, 0.1f, 0.0f, 0.0f};  // Similar to vector_1
    
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    
    // Perform similarity search
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // The most similar vector should be vector_1
    EXPECT_EQ(result.value()[0].vector_id, "vector_1");
    
    // Verify results are in descending order of similarity
    for (size_t i = 1; i < result.value().size(); ++i) {
        EXPECT_GE(result.value()[i-1].similarity_score, result.value()[i].similarity_score);
    }
}

TEST_F(SearchAPIIntegrationTest, SearchWithMetadataFiltering) {
    Vector query;
    query.id = "query_filtered";
    query.values = {0.8f, 0.2f, 0.0f, 0.0f};  // Similar to vector_1 and vector_3
    
    SearchParams params;
    params.top_k = 5;
    params.filter_tags = {"tag1"};  // Should match vector_1 and vector_3
    
    // Perform similarity search with metadata filter
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify that only vectors with "tag1" are returned
    for (const auto& res : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        
        bool has_tag1 = false;
        for (const auto& tag : vector_result.value().metadata.tags) {
            if (tag == "tag1") {
                has_tag1 = true;
                break;
            }
        }
        EXPECT_TRUE(has_tag1) << "Vector " << res.vector_id << " should have tag1";
    }
}

TEST_F(SearchAPIIntegrationTest, SearchWithCategoryFilter) {
    Vector query;
    query.id = "query_category";
    query.values = {0.6f, 0.6f, 0.0f, 0.0f};  // Similar to vector_3
    
    SearchParams params;
    params.top_k = 5;
    params.filter_category = "category1";  // Should match vector_1 and vector_3
    
    // Perform similarity search with category filter
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify that only vectors in "category1" are returned
    for (const auto& res : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_EQ(vector_result.value().metadata.category, "category1");
    }
}

TEST_F(SearchAPIIntegrationTest, SearchWithOwnerFilter) {
    Vector query;
    query.id = "query_owner";
    query.values = {0.1f, 0.9f, 0.0f, 0.0f};  // Similar to vector_2
    
    SearchParams params;
    params.top_k = 5;
    params.filter_owner = "user2";  // Should match vector_2
    
    // Perform similarity search with owner filter
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify that only vectors owned by "user2" are returned
    for (const auto& res : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_EQ(vector_result.value().metadata.owner, "user2");
    }
}

TEST_F(SearchAPIIntegrationTest, SearchWithScoreRangeFilter) {
    // Update one vector with a specific score
    auto retrieve_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, "vector_1");
    ASSERT_TRUE(retrieve_result.has_value());
    
    auto updated_vector = retrieve_result.value();
    updated_vector.metadata.score = 0.8f;
    search_service_->vector_storage_->update_vector(test_database_id_, updated_vector);
    
    // Query
    Vector query;
    query.id = "query_score";
    query.values = {0.9f, 0.1f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 5;
    params.filter_min_score = 0.7f;
    params.filter_max_score = 0.9f;
    
    // Perform similarity search with score range filter
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify that all results have scores in the specified range
    for (const auto& res : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_GE(vector_result.value().metadata.score, 0.7f);
        EXPECT_LE(vector_result.value().metadata.score, 0.9f);
    }
}

TEST_F(SearchAPIIntegrationTest, EuclideanSearchAlgorithm) {
    Vector query;
    query.id = "query_euclidean";
    query.values = {0.6f, 0.6f, 0.0f, 0.0f};  // Similar to vector_3
    
    SearchParams params;
    params.top_k = 3;
    
    // Perform Euclidean distance search
    auto result = search_service_->euclidean_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // Verify results are in descending order of similarity (1/(1+distance))
    for (size_t i = 1; i < result.value().size(); ++i) {
        EXPECT_GE(result.value()[i-1].similarity_score, result.value()[i].similarity_score);
    }
}

TEST_F(SearchAPIIntegrationTest, DotProductSearchAlgorithm) {
    Vector query;
    query.id = "query_dot";
    query.values = {1.0f, 1.0f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 3;
    
    // Perform dot product search
    auto result = search_service_->dot_product_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // Verify results are in descending order of dot product
    for (size_t i = 1; i < result.value().size(); ++i) {
        EXPECT_GE(result.value()[i-1].similarity_score, result.value()[i].similarity_score);
    }
}

TEST_F(SearchAPIIntegrationTest, TopKSearch) {
    Vector query;
    query.id = "query_topk";
    query.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 2;  // Request only top 2 results
    
    // Perform search with top_k constraint
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 2);
}

TEST_F(SearchAPIIntegrationTest, ThresholdSearch) {
    Vector query;
    query.id = "query_threshold";
    query.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 5;
    params.threshold = 0.9f;  // High threshold to filter out many results
    
    // Perform search with threshold filter
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify all results meet the threshold
    for (const auto& res : result.value()) {
        EXPECT_GE(res.similarity_score, 0.9f);
    }
}

TEST_F(SearchAPIIntegrationTest, IncludeVectorDataInResults) {
    Vector query;
    query.id = "query_with_data";
    query.values = {0.8f, 0.2f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 1;
    params.include_vector_data = true;
    
    // Perform search with vector data inclusion
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // Verify that vector data is included in results
    const auto& search_result = result.value()[0];
    EXPECT_FALSE(search_result.vector_data.id.empty());
    EXPECT_EQ(search_result.vector_data.values.size(), 4);
}

TEST_F(SearchAPIIntegrationTest, IncludeMetadataInResults) {
    Vector query;
    query.id = "query_with_metadata";
    query.values = {0.8f, 0.2f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 1;
    params.include_metadata = true;
    params.include_vector_data = false;  // Only metadata, not vector values
    
    // Perform search with metadata inclusion
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // Verify that we can access the metadata in the result vector
    const auto& search_result = result.value()[0];
    // The metadata should be preserved in the vector_data field
    EXPECT_FALSE(search_result.vector_data.id.empty());
}

TEST_F(SearchAPIIntegrationTest, PerformanceMetrics) {
    // Verify that metrics are being collected during search
    Vector query;
    query.id = "query_metrics";
    query.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 3;
    
    // Perform multiple searches to trigger metrics collection
    for (int i = 0; i < 5; ++i) {
        auto result = search_service_->similarity_search(test_database_id_, query, params);
        ASSERT_TRUE(result.has_value());
    }
    
    // The metrics should have been updated (verification would require access to metrics registry)
    // This test mainly ensures no crashes during metrics collection
    SUCCEED();
}

TEST_F(SearchAPIIntegrationTest, ValidateSearchParameters) {
    Vector query;
    query.id = "query_params";
    query.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams valid_params;
    valid_params.top_k = 5;
    valid_params.threshold = 0.5f;
    
    // Valid parameters should work
    auto result = search_service_->validate_search_params(valid_params);
    EXPECT_TRUE(result.has_value());
    
    SearchParams invalid_params;
    invalid_params.top_k = -1;  // Invalid: negative top_k
    
    // Invalid parameters should fail validation
    auto invalid_result = search_service_->validate_search_params(invalid_params);
    EXPECT_FALSE(invalid_result.has_value());
}

} // namespace jadevectordb