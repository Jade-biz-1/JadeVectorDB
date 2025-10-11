#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <cmath>

#include "services/similarity_search.h"
#include "models/vector.h"
#include "models/database.h"
#include "services/database_layer.h"

namespace jadevectordb {

// Test fixture for similarity search functionality
class SimilaritySearchTest : public ::testing::Test {
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
        db.name = "similarity_test_db";
        db.description = "Test database for similarity search";
        db.vectorDimension = 4;
        
        auto result = search_service_->vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
        
        // Add test vectors that we'll use for similarity testing
        addTestVectors();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            search_service_->vector_storage_->db_layer_->delete_database(test_database_id_);
        }
    }

    void addTestVectors() {
        // Vector 1: [1.0, 0.0, 0.0, 0.0] - unit vector along x-axis
        Vector v1;
        v1.id = "v1";
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
        search_service_->vector_storage_->store_vector(test_database_id_, v1);
        
        // Vector 2: [0.0, 1.0, 0.0, 0.0] - unit vector along y-axis (orthogonal to v1)
        Vector v2;
        v2.id = "v2";
        v2.values = {0.0f, 1.0f, 0.0f, 0.0f};
        search_service_->vector_storage_->store_vector(test_database_id_, v2);
        
        // Vector 3: [0.7, 0.7, 0.0, 0.0] - similar to v1 (45-degree angle to x-axis)
        Vector v3;
        v3.id = "v3";
        v3.values = {0.7f, 0.7f, 0.0f, 0.0f};
        search_service_->vector_storage_->store_vector(test_database_id_, v3);
        
        // Vector 4: [0.5, 0.5, 0.5, 0.5] - diagonal vector
        Vector v4;
        v4.id = "v4";
        v4.values = {0.5f, 0.5f, 0.5f, 0.5f};
        search_service_->vector_storage_->store_vector(test_database_id_, v4);
    }

    std::unique_ptr<SimilaritySearchService> search_service_;
    std::string test_database_id_;
};

TEST_F(SimilaritySearchTest, CosineSimilarityCalculation) {
    // Create a query vector similar to v1
    Vector query;
    query.id = "query_v1";
    query.values = {0.9f, 0.1f, 0.0f, 0.0f};  // Close to v1 = [1, 0, 0, 0]
    
    SearchParams params;
    params.top_k = 4;
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // The most similar vector to [0.9, 0.1, 0.0, 0.0] should be v1 = [1.0, 0.0, 0.0, 0.0]
    EXPECT_EQ(result.value()[0].vector_id, "v1");
    
    // Verify the similarity score makes sense
    // Cosine similarity between [0.9, 0.1, 0.0, 0.0] and [1.0, 0.0, 0.0, 0.0] 
    // should be close to 1 since they're very similar directions
    float expected_similarity = (0.9f * 1.0f + 0.1f * 0.0f + 0.0f * 0.0f + 0.0f * 0.0f) /
                               (std::sqrt(0.9f*0.9f + 0.1f*0.1f) * std::sqrt(1.0f*1.0f));
    EXPECT_NEAR(result.value()[0].similarity_score, expected_similarity, 0.01f);
}

TEST_F(SimilaritySearchTest, OrthogonalVectorsHaveLowCosineSimilarity) {
    // Test with v1 and v2 which are orthogonal [1,0,0,0] and [0,1,0,0]
    Vector query;
    query.id = "query_v1";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 4;
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Find the result for v2 (orthogonal vector)
    for (const auto& res : result.value()) {
        if (res.vector_id == "v2") {
            // Cosine similarity between orthogonal vectors should be close to 0
            EXPECT_LT(res.similarity_score, 0.1f);
            break;
        }
    }
}

TEST_F(SimilaritySearchTest, EuclideanDistanceSearch) {
    // Create a query vector
    Vector query;
    query.id = "query_euc";
    query.values = {0.6f, 0.6f, 0.0f, 0.0f};  // Close to v3 = [0.7, 0.7, 0.0, 0.0]
    
    SearchParams params;
    params.top_k = 4;
    
    auto result = search_service_->euclidean_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // The closest vector to [0.6, 0.6, 0.0, 0.0] should be v3 = [0.7, 0.7, 0.0, 0.0]
    EXPECT_EQ(result.value()[0].vector_id, "v3");
}

TEST_F(SimilaritySearchTest, DotProductSearch) {
    // Create a query vector
    Vector query;
    query.id = "query_dot";
    query.values = {1.0f, 1.0f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 4;
    
    auto result = search_service_->dot_product_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // The highest dot product with [1, 1, 0, 0] should be with v3 = [0.7, 0.7, 0, 0] 
    // (0.7*1 + 0.7*1 = 1.4) or v4 = [0.5, 0.5, 0.5, 0.5] (0.5*1 + 0.5*1 = 1.0)
    // v3 should come first as it has the higher dot product
    EXPECT_EQ(result.value()[0].vector_id, "v3");
    
    // Verify the dot product value
    float expected_dot_product = 1.0f * 0.7f + 1.0f * 0.7f + 0.0f * 0.0f + 0.0f * 0.0f;
    EXPECT_FLOAT_EQ(result.value()[0].similarity_score, expected_dot_product);
}

TEST_F(SimilaritySearchTest, SearchWithThreshold) {
    Vector query;
    query.id = "query_thresh";
    query.values = {0.1f, 0.1f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.5f;  // Set a high threshold
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // All results should meet the threshold
    for (const auto& res : result.value()) {
        EXPECT_GE(res.similarity_score, 0.5f);
    }
}

TEST_F(SimilaritySearchTest, SearchResultLimit) {
    Vector query;
    query.id = "query_limit";
    query.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 2;  // Limit to 2 results
    params.threshold = 0.0f;  // No threshold filter
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Should have at most 2 results
    EXPECT_LE(result.value().size(), 2);
}

TEST_F(SimilaritySearchTest, ValidateSearchParams) {
    SearchParams valid_params;
    valid_params.top_k = 10;
    valid_params.threshold = 0.5f;
    
    // Valid parameters should pass validation
    auto valid_result = search_service_->validate_search_params(valid_params);
    EXPECT_TRUE(valid_result.has_value());
    
    SearchParams invalid_params;
    invalid_params.top_k = -1;  // Invalid: negative top_k
    
    // Invalid parameters should fail validation
    auto invalid_result = search_service_->validate_search_params(invalid_params);
    EXPECT_FALSE(invalid_result.has_value());
}

TEST_F(SimilaritySearchTest, GetAvailableAlgorithms) {
    auto algorithms = search_service_->get_available_algorithms();
    
    // Should include the expected algorithms
    EXPECT_THAT(algorithms, ::testing::Contains("cosine_similarity"));
    EXPECT_THAT(algorithms, ::testing::Contains("euclidean_distance"));
    EXPECT_THAT(algorithms, ::testing::Contains("dot_product"));
}

} // namespace jadevectordb