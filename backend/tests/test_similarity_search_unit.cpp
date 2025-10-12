#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Mock class for VectorStorageService to use in unit tests
class MockVectorStorageService : public VectorStorageService {
public:
    MOCK_METHOD(Result<void>, initialize, (), (override));
    MOCK_METHOD(Result<void>, store_vector, (const std::string& database_id, const Vector& vector), (override));
    MOCK_METHOD(Result<Vector>, retrieve_vector, (const std::string& database_id, const std::string& vector_id), (const, override));
    MOCK_METHOD(Result<std::vector<Vector>>, retrieve_vectors, (const std::string& database_id, const std::vector<std::string>& vector_ids), (const, override));
    MOCK_METHOD(Result<void>, update_vector, (const std::string& database_id, const Vector& vector), (override));
    MOCK_METHOD(Result<void>, delete_vector, (const std::string& database_id, const std::string& vector_id), (override));
    MOCK_METHOD(Result<void>, batch_store_vectors, (const std::string& database_id, const std::vector<Vector>& vectors), (override));
    MOCK_METHOD(Result<void>, batch_delete_vectors, (const std::string& database_id, const std::vector<std::string>& vector_ids), (override));
    MOCK_METHOD(Result<bool>, vector_exists, (const std::string& database_id, const std::string& vector_id), (const, override));
    MOCK_METHOD(Result<size_t>, get_vector_count, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<std::vector<std::string>>, get_all_vector_ids, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<void>, validate_vector, (const std::string& database_id, const Vector& vector), (const, override));
};

// Test fixture for SimilaritySearchService
class SimilaritySearchServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock vector storage service
        mock_vector_storage_ = std::make_unique<MockVectorStorageService>();
        
        // Create similarity search service with mock vector storage
        similarity_search_service_ = std::make_unique<SimilaritySearchService>(std::move(mock_vector_storage_));
        
        // Initialize the service
        auto init_result = similarity_search_service_->initialize();
        ASSERT_TRUE(init_result.has_value());
    }
    
    void TearDown() override {
        // Clean up
        similarity_search_service_.reset();
        mock_vector_storage_.reset();
    }
    
    std::unique_ptr<MockVectorStorageService> mock_vector_storage_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
};

// Test that the service initializes correctly
TEST_F(SimilaritySearchServiceTest, InitializeService) {
    // Service should already be initialized in SetUp
    EXPECT_NE(similarity_search_service_, nullptr);
}

// Test cosine similarity calculation with known vectors
TEST_F(SimilaritySearchServiceTest, CosineSimilarityWithKnownVectors) {
    // Test with identical vectors (should have cosine similarity of 1.0)
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {1.0f, 0.0f, 0.0f, 0.0f};
    
    float similarity = similarity_search_service_->cosine_similarity(v1, v2);
    EXPECT_FLOAT_EQ(similarity, 1.0f);
    
    // Test with orthogonal vectors (should have cosine similarity of 0.0)
    std::vector<float> v3 = {0.0f, 1.0f, 0.0f, 0.0f};
    
    similarity = similarity_search_service_->cosine_similarity(v1, v3);
    EXPECT_NEAR(similarity, 0.0f, 0.0001f);
    
    // Test with opposite vectors (should have cosine similarity of -1.0)
    std::vector<float> v4 = {-1.0f, 0.0f, 0.0f, 0.0f};
    
    similarity = similarity_search_service_->cosine_similarity(v1, v4);
    EXPECT_NEAR(similarity, -1.0f, 0.0001f);
    
    // Test with similar but not identical vectors
    std::vector<float> v5 = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<float> v6 = {0.8f, 0.2f, 0.0f, 0.0f};
    
    similarity = similarity_search_service_->cosine_similarity(v5, v6);
    EXPECT_GT(similarity, 0.9f);
    EXPECT_LT(similarity, 1.0f);
}

// Test Euclidean distance calculation with known vectors
TEST_F(SimilaritySearchServiceTest, EuclideanDistanceWithKnownVectors) {
    // Test with identical vectors (should have Euclidean distance of 0.0)
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {1.0f, 0.0f, 0.0f, 0.0f};
    
    float distance = similarity_search_service_->euclidean_distance(v1, v2);
    EXPECT_FLOAT_EQ(distance, 0.0f);
    
    // Test with vectors forming a 3-4-5 triangle (should have Euclidean distance of 5.0)
    std::vector<float> v3 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v4 = {3.0f, 4.0f, 0.0f, 0.0f};
    
    distance = similarity_search_service_->euclidean_distance(v3, v4);
    EXPECT_NEAR(distance, 5.0f, 0.0001f);
    
    // Test with vectors with different dimensions (should pad with zeros)
    std::vector<float> v5 = {1.0f, 2.0f, 3.0f};
    std::vector<float> v6 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    distance = similarity_search_service_->euclidean_distance(v5, v6);
    // Distance should be sqrt((1-1)^2 + (2-2)^2 + (3-3)^2 + (0-4)^2 + (0-5)^2) = sqrt(0 + 0 + 0 + 16 + 25) = sqrt(41)
    EXPECT_NEAR(distance, std::sqrt(41.0f), 0.0001f);
}

// Test dot product calculation with known vectors
TEST_F(SimilaritySearchServiceTest, DotProductWithKnownVectors) {
    // Test with identical unit vectors (should have dot product of 1.0)
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {1.0f, 0.0f, 0.0f, 0.0f};
    
    float dot_product = similarity_search_service_->dot_product(v1, v2);
    EXPECT_FLOAT_EQ(dot_product, 1.0f);
    
    // Test with orthogonal vectors (should have dot product of 0.0)
    std::vector<float> v3 = {0.0f, 1.0f, 0.0f, 0.0f};
    
    dot_product = similarity_search_service_->dot_product(v1, v3);
    EXPECT_FLOAT_EQ(dot_product, 0.0f);
    
    // Test with opposite unit vectors (should have dot product of -1.0)
    std::vector<float> v4 = {-1.0f, 0.0f, 0.0f, 0.0f};
    
    dot_product = similarity_search_service_->dot_product(v1, v4);
    EXPECT_FLOAT_EQ(dot_product, -1.0f);
    
    // Test with arbitrary vectors
    std::vector<float> v5 = {1.0f, 2.0f, 3.0f, 0.0f};
    std::vector<float> v6 = {4.0f, 5.0f, 6.0f, 0.0f};
    
    dot_product = similarity_search_service_->dot_product(v5, v6);
    // Expected: (1*4) + (2*5) + (3*6) + (0*0) = 4 + 10 + 18 + 0 = 32
    EXPECT_FLOAT_EQ(dot_product, 32.0f);
}

// Test similarity search with cosine similarity
TEST_F(SimilaritySearchServiceTest, SimilaritySearchWithCosine) {
    // Create test vectors
    std::vector<Vector> test_vectors;
    
    // Vector A - reference vector
    Vector v1;
    v1.id = "vector_A";
    v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
    test_vectors.push_back(v1);
    
    // Vector B - very similar to A
    Vector v2;
    v2.id = "vector_B";
    v2.values = {0.9f, 0.1f, 0.0f, 0.0f};
    test_vectors.push_back(v2);
    
    // Vector C - somewhat similar to A
    Vector v3;
    v3.id = "vector_C";
    v3.values = {0.7f, 0.3f, 0.0f, 0.0f};
    test_vectors.push_back(v3);
    
    // Vector D - less similar to A
    Vector v4;
    v4.id = "vector_D";
    v4.values = {0.5f, 0.5f, 0.0f, 0.0f};
    test_vectors.push_back(v4);
    
    // Vector E - quite different from A
    Vector v5;
    v5.id = "vector_E";
    v5.values = {0.0f, 1.0f, 0.0f, 0.0f};
    test_vectors.push_back(v5);
    
    std::string database_id = "test_db";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_vector_storage_, retrieve_vectors(database_id, _))
        .WillOnce(Return(Result<std::vector<Vector>>{test_vectors}));
    
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 3;  // Get top 3 results
    params.threshold = 0.0f;  // No threshold filtering
    params.include_vector_data = false;  // Don't include vector data in results
    params.include_metadata = false;  // Don't include metadata in results
    
    // Perform similarity search
    auto result = similarity_search_service_->similarity_search(database_id, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3);  // Should return exactly 3 results (top_k = 3)
    
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
    
    // Third result should be vector_C (somewhat similar to query)
    EXPECT_EQ(search_results[2].vector_id, "vector_C");
    EXPECT_GT(search_results[2].similarity_score, 0.7f);
}

// Test Euclidean search
TEST_F(SimilaritySearchServiceTest, EuclideanSearch) {
    // Create test vectors
    std::vector<Vector> test_vectors;
    
    // Vector A - reference vector
    Vector v1;
    v1.id = "vector_A";
    v1.values = {0.0f, 0.0f, 0.0f, 0.0f};
    test_vectors.push_back(v1);
    
    // Vector B - close to A
    Vector v2;
    v2.id = "vector_B";
    v2.values = {1.0f, 0.0f, 0.0f, 0.0f};
    test_vectors.push_back(v2);
    
    // Vector C - further from A
    Vector v3;
    v3.id = "vector_C";
    v3.values = {3.0f, 4.0f, 0.0f, 0.0f};
    test_vectors.push_back(v3);
    
    std::string database_id = "test_db";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_vector_storage_, retrieve_vectors(database_id, _))
        .WillOnce(Return(Result<std::vector<Vector>>{test_vectors}));
    
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Perform Euclidean search
    auto result = similarity_search_service_->euclidean_search(database_id, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3);
    
    // Results should be ordered by similarity (descending)
    // For Euclidean distance, smaller distances mean higher similarity
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // First result should be vector_A (identical to query, distance = 0)
    EXPECT_EQ(search_results[0].vector_id, "vector_A");
    EXPECT_FLOAT_EQ(search_results[0].similarity_score, 1.0f);  // 1/(1+0) = 1
    
    // Second result should be vector_B (distance = 1)
    EXPECT_EQ(search_results[1].vector_id, "vector_B");
    EXPECT_FLOAT_EQ(search_results[1].similarity_score, 0.5f);  // 1/(1+1) = 0.5
    
    // Third result should be vector_C (distance = 5)
    EXPECT_EQ(search_results[2].vector_id, "vector_C");
    EXPECT_FLOAT_EQ(search_results[2].similarity_score, 1.0f/6.0f);  // 1/(1+5) = 1/6
}

// Test dot product search
TEST_F(SimilaritySearchServiceTest, DotProductSearch) {
    // Create test vectors
    std::vector<Vector> test_vectors;
    
    // Vector A - reference vector
    Vector v1;
    v1.id = "vector_A";
    v1.values = {1.0f, 1.0f, 1.0f, 1.0f};
    test_vectors.push_back(v1);
    
    // Vector B - positive dot product with A
    Vector v2;
    v2.id = "vector_B";
    v2.values = {1.0f, 1.0f, 0.0f, 0.0f};
    test_vectors.push_back(v2);
    
    // Vector C - negative dot product with A
    Vector v3;
    v3.id = "vector_C";
    v3.values = {-1.0f, -1.0f, 0.0f, 0.0f};
    test_vectors.push_back(v3);
    
    std::string database_id = "test_db";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_vector_storage_, retrieve_vectors(database_id, _))
        .WillOnce(Return(Result<std::vector<Vector>>{test_vectors}));
    
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 1.0f, 1.0f, 1.0f};
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Perform dot product search
    auto result = similarity_search_service_->dot_product_search(database_id, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3);
    
    // Results should be ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // First result should be vector_A (dot product = 4)
    EXPECT_EQ(search_results[0].vector_id, "vector_A");
    EXPECT_FLOAT_EQ(search_results[0].similarity_score, 4.0f);  // 1.0*1.0 + 1.0*1.0 + 1.0*1.0 + 1.0*1.0 = 4.0
    
    // Second result should be vector_B (dot product = 2)
    EXPECT_EQ(search_results[1].vector_id, "vector_B");
    EXPECT_FLOAT_EQ(search_results[1].similarity_score, 2.0f);  // 1.0*1.0 + 1.0*1.0 + 1.0*0.0 + 1.0*0.0 = 2.0
    
    // Third result should be vector_C (dot product = -2)
    EXPECT_EQ(search_results[2].vector_id, "vector_C");
    EXPECT_FLOAT_EQ(search_results[2].similarity_score, -2.0f);  // 1.0*(-1.0) + 1.0*(-1.0) + 1.0*0.0 + 1.0*0.0 = -2.0
}

// Test search parameter validation
TEST_F(SimilaritySearchServiceTest, ValidateSearchParams) {
    // Test valid search parameters
    SearchParams valid_params;
    valid_params.top_k = 10;
    valid_params.threshold = 0.5f;
    valid_params.include_vector_data = false;
    valid_params.include_metadata = false;
    
    auto result = similarity_search_service_->validate_search_params(valid_params);
    EXPECT_TRUE(result.has_value());
    
    // Test invalid top_k (negative)
    SearchParams invalid_params1 = valid_params;
    invalid_params1.top_k = -1;
    
    result = similarity_search_service_->validate_search_params(invalid_params1);
    EXPECT_FALSE(result.has_value());
    
    // Test invalid threshold (negative)
    SearchParams invalid_params2 = valid_params;
    invalid_params2.threshold = -0.5f;
    
    result = similarity_search_service_->validate_search_params(invalid_params2);
    EXPECT_FALSE(result.has_value());
    
    // Test invalid threshold (greater than 1.0)
    SearchParams invalid_params3 = valid_params;
    invalid_params3.threshold = 1.5f;
    
    result = similarity_search_service_->validate_search_params(invalid_params3);
    EXPECT_FALSE(result.has_value());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}