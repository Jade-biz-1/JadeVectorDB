#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <cmath>

#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;
using ::testing::Eq;
using ::testing::ByRef;

// Test fixture for SimilaritySearchService
class SimilaritySearchServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test database configuration
        test_db_.databaseId = "test_db_123";
        test_db_.name = "test_database";
        test_db_.description = "Test database for similarity search unit tests";
        test_db_.vectorDimension = 4; // Small dimension for testing
        test_db_.indexType = "HNSW";
        
        // Create vector storage service
        vector_storage_service_ = std::make_unique<VectorStorageService>();
        
        // Initialize the vector storage service
        auto init_result = vector_storage_service_->initialize();
        ASSERT_TRUE(init_result.has_value());
        
        // Create similarity search service
        similarity_search_service_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage_service_));
        
        // Initialize the similarity search service
        auto search_init_result = similarity_search_service_->initialize();
        ASSERT_TRUE(search_init_result.has_value());
        
        // Store some test vectors for searching
        store_test_vectors();
    }
    
    void TearDown() override {
        // Clean up
        similarity_search_service_.reset();
    }
    
    void store_test_vectors() {
        // Create test vectors with known relationships
        std::vector<Vector> test_vectors = {
            // Vector A - reference vector
            Vector{"vector_A", {1.0f, 0.0f, 0.0f, 0.0f}},
            // Vector B - very similar to A
            Vector{"vector_B", {0.9f, 0.1f, 0.0f, 0.0f}},
            // Vector C - somewhat similar to A
            Vector{"vector_C", {0.7f, 0.3f, 0.0f, 0.0f}},
            // Vector D - less similar to A
            Vector{"vector_D", {0.5f, 0.5f, 0.0f, 0.0f}},
            // Vector E - quite different from A
            Vector{"vector_E", {0.0f, 1.0f, 0.0f, 0.0f}},
            // Vector F - very different from A (orthogonal)
            Vector{"vector_F", {0.0f, 0.0f, 1.0f, 0.0f}}
        };
        
        // Store all test vectors
        for (const auto& v : test_vectors) {
            auto store_result = similarity_search_service_->vector_storage_->store_vector(test_db_.databaseId, v);
            ASSERT_TRUE(store_result.has_value()) << "Failed to store vector: " << v.id;
        }
    }
    
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
    Database test_db_;
};

// Test that the service initializes correctly
TEST_F(SimilaritySearchServiceTest, InitializeService) {
    // Service should already be initialized in SetUp
    EXPECT_NE(similarity_search_service_, nullptr);
}

// Test cosine similarity calculation
TEST_F(SimilaritySearchServiceTest, CosineSimilarityCalculation) {
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v3 = {0.0f, 1.0f, 0.0f, 0.0f};
    
    // Identical vectors should have cosine similarity of 1.0
    float similarity_identical = similarity_search_service_->cosine_similarity(v1, v2);
    EXPECT_FLOAT_EQ(similarity_identical, 1.0f);
    
    // Orthogonal vectors should have cosine similarity of 0.0
    float similarity_orthogonal = similarity_search_service_->cosine_similarity(v1, v3);
    EXPECT_NEAR(similarity_orthogonal, 0.0f, 0.0001f);
    
    // Test with similar but not identical vectors
    std::vector<float> v4 = {0.9f, 0.1f, 0.0f, 0.0f};
    float similarity_similar = similarity_search_service_->cosine_similarity(v1, v4);
    EXPECT_GT(similarity_similar, 0.9f);
    EXPECT_LT(similarity_similar, 1.0f);
}

// Test Euclidean distance calculation
TEST_F(SimilaritySearchServiceTest, EuclideanDistanceCalculation) {
    std::vector<float> v1 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {3.0f, 4.0f, 0.0f, 0.0f}; // Distance should be 5 (3-4-5 triangle)
    
    float distance = similarity_search_service_->euclidean_distance(v1, v2);
    EXPECT_NEAR(distance, 5.0f, 0.0001f);
    
    // Distance from a point to itself should be 0
    distance = similarity_search_service_->euclidean_distance(v1, v1);
    EXPECT_FLOAT_EQ(distance, 0.0f);
}

// Test dot product calculation
TEST_F(SimilaritySearchServiceTest, DotProductCalculation) {
    std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 0.0f};
    std::vector<float> v2 = {4.0f, 5.0f, 6.0f, 0.0f};
    
    float dot_product = similarity_search_service_->dot_product(v1, v2);
    // Expected: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(dot_product, 32.0f);
    
    // Dot product of perpendicular vectors should be 0
    std::vector<float> v3 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v4 = {0.0f, 1.0f, 0.0f, 0.0f};
    dot_product = similarity_search_service_->dot_product(v3, v4);
    EXPECT_FLOAT_EQ(dot_product, 0.0f);
}

// Test basic similarity search with cosine similarity
TEST_F(SimilaritySearchServiceTest, BasicSimilaritySearch) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f}; // Same as vector_A
    
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    auto result = similarity_search_service_->similarity_search(test_db_.databaseId, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3); // Should return top 3 results
    
    // Results should be ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // First result should be vector_A (identical to query)
    EXPECT_EQ(search_results[0].vector_id, "vector_A");
    EXPECT_FLOAT_EQ(search_results[0].similarity_score, 1.0f);
    
    // Second result should be vector_B (very similar to query)
    EXPECT_EQ(search_results[1].vector_id, "vector_B");
    EXPECT_GT(search_results[1].similarity_score, 0.9f);
}

// Test similarity search with threshold filtering
TEST_F(SimilaritySearchServiceTest, SimilaritySearchWithThreshold) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f}; // Same as vector_A
    
    SearchParams params;
    params.top_k = 10; // Get all results
    params.threshold = 0.8f; // Only results with similarity >= 0.8
    params.include_vector_data = false;
    
    auto result = similarity_search_service_->similarity_search(test_db_.databaseId, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    
    // All results should meet the threshold
    for (const auto& res : search_results) {
        EXPECT_GE(res.similarity_score, 0.8f);
    }
    
    // We should have fewer results than without threshold
    EXPECT_LT(search_results.size(), 6); // Less than all 6 test vectors
}

// Test Euclidean search
TEST_F(SimilaritySearchServiceTest, EuclideanSearch) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {0.0f, 0.0f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    auto result = similarity_search_service_->euclidean_search(test_db_.databaseId, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3); // Should return top 3 results
    
    // Results should be ordered by similarity (descending)
    // For Euclidean distance, smaller distances mean higher similarity
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
}

// Test dot product search
TEST_F(SimilaritySearchServiceTest, DotProductSearch) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 1.0f, 1.0f, 1.0f};
    
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    auto result = similarity_search_service_->dot_product_search(test_db_.databaseId, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3); // Should return top 3 results
    
    // Results should be ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
}

// Test search parameter validation
TEST_F(SimilaritySearchServiceTest, ValidateSearchParams) {
    // Valid parameters
    SearchParams valid_params;
    valid_params.top_k = 10;
    valid_params.threshold = 0.5f;
    
    auto result = similarity_search_service_->validate_search_params(valid_params);
    EXPECT_TRUE(result.has_value());
    
    // Invalid top_k (negative)
    SearchParams invalid_params1 = valid_params;
    invalid_params1.top_k = -1;
    
    result = similarity_search_service_->validate_search_params(invalid_params1);
    EXPECT_FALSE(result.has_value());
    
    // Invalid threshold (too high)
    SearchParams invalid_params2 = valid_params;
    invalid_params2.threshold = 1.5f;
    
    result = similarity_search_service_->validate_search_params(invalid_params2);
    EXPECT_FALSE(result.has_value());
    
    // Invalid threshold (negative)
    SearchParams invalid_params3 = valid_params;
    invalid_params3.threshold = -0.5f;
    
    result = similarity_search_service_->validate_search_params(invalid_params3);
    EXPECT_FALSE(result.has_value());
}

// Test search with vector data inclusion
TEST_F(SimilaritySearchServiceTest, SimilaritySearchWithVectorData) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 2;
    params.threshold = 0.0f;
    params.include_vector_data = true; // Include vector data in results
    
    auto result = similarity_search_service_->similarity_search(test_db_.databaseId, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 2);
    
    // All results should include vector data
    for (const auto& res : search_results) {
        EXPECT_FALSE(res.vector_data.id.empty());
        EXPECT_GT(res.vector_data.values.size(), 0);
    }
}

// Test KNN search
TEST_F(SimilaritySearchServiceTest, KnnSearch) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 4;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    
    auto result = similarity_search_service_->similarity_search(test_db_.databaseId, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 4); // Should return exactly 4 results (K=4)
    
    // Results should be ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}