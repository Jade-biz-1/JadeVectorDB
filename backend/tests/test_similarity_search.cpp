#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "models/vector.h"
#include "models/database.h"
#include "services/search_utils.h"
#include "services/search_benchmark.h"

namespace jadevectordb {

// Test fixture for similarity search unit tests
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
        db.name = "search_test_db";
        db.description = "Test database for similarity search";
        db.vectorDimension = 4;
        
        auto result = search_service_->vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
        
        // Add some test vectors
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
        
        // Vector 5: [0.1, 0.2, 0.3, 0.4] - another test vector
        Vector v5;
        v5.id = "v5";
        v5.values = {0.1f, 0.2f, 0.3f, 0.4f};
        search_service_->vector_storage_->store_vector(test_database_id_, v5);
    }

    std::unique_ptr<SimilaritySearchService> search_service_;
    std::string test_database_id_;
};

// Test basic similarity search
TEST_F(SimilaritySearchTest, CosineSimilaritySearch) {
    // Create a query vector similar to v1
    Vector query;
    query.id = "query_v1";
    query.values = {0.9f, 0.1f, 0.0f, 0.0f};  // Close to v1 = [1, 0, 0, 0]
    
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // The most similar vector to [0.9, 0.1, 0.0, 0.0] should be v1 = [1.0, 0.0, 0.0, 0.0]
    EXPECT_EQ(result.value()[0].vector_id, "v1");
    
    // Verify the similarity score makes sense
    float expected_similarity = (0.9f * 1.0f + 0.1f * 0.0f + 0.0f * 0.0f + 0.0f * 0.0f) /
                               (std::sqrt(0.9f*0.9f + 0.1f*0.1f) * std::sqrt(1.0f*1.0f));
    EXPECT_NEAR(result.value()[0].similarity_score, expected_similarity, 0.01f);
}

TEST_F(SimilaritySearchTest, EuclideanDistanceSearch) {
    // Create a query vector
    Vector query;
    query.id = "query_euc";
    query.values = {0.6f, 0.6f, 0.0f, 0.0f};  // Close to v3 = [0.7, 0.7, 0.0, 0.0]
    
    SearchParams params;
    params.top_k = 3;
    
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
    params.top_k = 3;
    
    auto result = search_service_->dot_product_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // The highest dot product with [1, 1, 0, 0] should be with v3 = [0.7, 0.7, 0, 0] 
    // (0.7*1 + 0.7*1 = 1.4) or v4 = [0.5, 0.5, 0.5, 0.5] (0.5*1 + 0.5*1 = 1.0)
    EXPECT_EQ(result.value()[0].vector_id, "v3");
    
    // Verify the dot product value
    float expected_dot_product = 1.0f * 0.7f + 1.0f * 0.7f + 0.0f * 0.0f + 0.0f * 0.0f;
    EXPECT_FLOAT_EQ(result.value()[0].similarity_score, expected_dot_product);
}

TEST_F(SimilaritySearchTest, KNearestNeighbors) {
    Vector query;
    query.id = "query_knn";
    query.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 2;  // Get top 2 results
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 2);
    
    // Results should be sorted by similarity score (descending)
    auto results = result.value();
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity_score, results[i].similarity_score);
    }
}

TEST_F(SimilaritySearchTest, ThresholdFiltering) {
    Vector query;
    query.id = "query_thresh";
    query.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.8f;  // Set a high threshold
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // All results should meet the threshold
    for (const auto& res : result.value()) {
        EXPECT_GE(res.similarity_score, 0.8f);
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

TEST_F(SimilaritySearchTest, IncludeVectorData) {
    Vector query;
    query.id = "query_with_data";
    query.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 1;
    params.include_vector_data = true;
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // Verify that vector data is included in the result
    const auto& search_result = result.value()[0];
    EXPECT_FALSE(search_result.vector_data.id.empty());
    EXPECT_EQ(search_result.vector_data.values.size(), 4);
}

TEST_F(SimilaritySearchTest, IncludeMetadata) {
    Vector query;
    query.id = "query_with_metadata";
    query.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 1;
    params.include_metadata = true;
    params.include_vector_data = false;  // Only metadata, not vector values
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1);
    
    // Verify that vector data structure is present with metadata
    const auto& search_result = result.value()[0];
    EXPECT_FALSE(search_result.vector_data.id.empty());
}

TEST_F(SimilaritySearchTest, SearchWithMetadataFilters) {
    // First update some vectors with metadata
    Vector v1;
    v1.id = "v1";
    v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
    v1.metadata.tags = {"tag1", "tag2"};
    v1.metadata.owner = "user1";
    v1.metadata.category = "category1";
    
    Vector v2;
    v2.id = "v2";
    v2.values = {0.0f, 1.0f, 0.0f, 0.0f};
    v2.metadata.tags = {"tag3"};
    v2.metadata.owner = "user2";
    v2.metadata.category = "category2";
    
    search_service_->vector_storage_->update_vector(test_database_id_, v1);
    search_service_->vector_storage_->update_vector(test_database_id_, v2);
    
    Vector query;
    query.id = "query_filtered";
    query.values = {0.9f, 0.1f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 5;
    params.filter_tags = {"tag1"};  // Should only match v1
    
    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value());
    
    // Should only return vectors matching the filter
    bool found_v1 = false;
    for (const auto& res : result.value()) {
        if (res.vector_id == "v1") {
            found_v1 = true;
        } else if (res.vector_id == "v2") {
            FAIL() << "Found v2 which should be filtered out";
        }
    }
    EXPECT_TRUE(found_v1);
}

// Test search utilities
TEST(SearchUtilsTest, CosineSimilarityOptimized) {
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f, 0.0f};
    
    float similarity = SearchUtils::cosine_similarity_optimized(v1, v2);
    EXPECT_NEAR(similarity, 0.0f, 0.001f);  // Orthogonal vectors have 0 cosine similarity
    
    std::vector<float> v3 = {1.0f, 0.0f, 0.0f};
    std::vector<float> v4 = {1.0f, 0.0f, 0.0f};
    
    similarity = SearchUtils::cosine_similarity_optimized(v3, v4);
    EXPECT_NEAR(similarity, 1.0f, 0.001f);  // Same vectors have 1 cosine similarity
}

TEST(SearchUtilsTest, EuclideanDistanceOptimized) {
    std::vector<float> v1 = {0.0f, 0.0f};
    std::vector<float> v2 = {3.0f, 4.0f};
    
    float distance = SearchUtils::euclidean_distance_optimized(v1, v2);
    EXPECT_NEAR(distance, 5.0f, 0.001f);  // 3-4-5 triangle
}

TEST(SearchUtilsTest, DotProductOptimized) {
    std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> v2 = {4.0f, 5.0f, 6.0f};
    
    float dot_product = SearchUtils::dot_product_optimized(v1, v2);
    EXPECT_FLOAT_EQ(dot_product, 32.0f);  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST(SearchUtilsTest, TopKWithHeap) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3};
    auto result = SearchUtils::top_k_with_heap(data, 3, 
        [](const int& a, const int& b) { return a < b; });
    
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 9);  // Highest
    EXPECT_EQ(result[1], 8);
    EXPECT_EQ(result[2], 5);  // Lowest of the top 3
}

// Test KNN search
TEST(KnnSearchTest, BasicKnnSearch) {
    // Create test vectors
    Vector v1; v1.id = "v1"; v1.values = {1.0f, 0.0f, 0.0f};
    Vector v2; v2.id = "v2"; v2.values = {0.0f, 1.0f, 0.0f};
    Vector v3; v3.id = "v3"; v3.values = {0.0f, 0.0f, 1.0f};
    
    std::vector<Vector> candidates = {v1, v2, v3};
    
    Vector query; query.values = {0.8f, 0.1f, 0.1f};
    
    auto results = KnnSearch::knn_search(query, candidates, 2, KnnSearch::Algorithm::HEAP);
    
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].vector_id, "v1");  // Most similar to [1,0,0]
}

// Test search benchmarking
TEST(BenchmarkTest, CanRunBenchmark) {
    // This test mainly ensures the benchmark can be constructed and run without errors
    auto mock_search_service = std::make_shared<SimilaritySearchService>();
    SearchBenchmark benchmark(mock_search_service);
    
    Vector test_vector;
    test_vector.id = "test";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    benchmark.add_test_vectors({test_vector});
    benchmark.add_query_vectors({test_vector});
    
    SearchParams params;
    params.top_k = 1;
    
    // Run a minimal benchmark
    auto result = benchmark.run_benchmark("fake_db", "cosine_similarity", params, 1);
    
    // Verify the result has expected properties
    EXPECT_EQ(result.algorithm_name, "cosine_similarity");
    EXPECT_EQ(result.total_queries, 1);
}

} // namespace jadevectordb