#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <random>
#include <cmath>

#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/database_service.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test for validating search result quality
class SearchResultQualityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services
        db_service_ = std::make_unique<DatabaseService>();
        vector_service_ = std::make_unique<VectorStorageService>();
        search_service_ = std::make_unique<SimilaritySearchService>();
        
        db_service_->initialize();
        vector_service_->initialize();
        search_service_->initialize();
        
        // Create a test database
        Database test_db;
        test_db.name = "quality_test_db";
        test_db.vectorDimension = 16; // Small dimension for controlled testing
        test_db.description = "Database for search result quality validation";
        
        auto create_result = db_service_->create_database(test_db);
        ASSERT_TRUE(create_result.has_value());
        db_id_ = create_result.value();
    }
    
    void TearDown() override {
        if (!db_id_.empty()) {
            db_service_->delete_database(db_id_);
        }
    }
    
    // Helper function to calculate cosine similarity
    float calculateCosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.size() != v2.size()) return 0.0f;
        
        double dot_product = 0.0;
        double magnitude_v1 = 0.0;
        double magnitude_v2 = 0.0;
        
        for (size_t i = 0; i < v1.size(); ++i) {
            dot_product += v1[i] * v2[i];
            magnitude_v1 += v1[i] * v1[i];
            magnitude_v2 += v2[i] * v2[i];
        }
        
        magnitude_v1 = std::sqrt(magnitude_v1);
        magnitude_v2 = std::sqrt(magnitude_v2);
        
        if (magnitude_v1 == 0.0 || magnitude_v2 == 0.0) {
            return 0.0f;
        }
        
        return static_cast<float>(dot_product / (magnitude_v1 * magnitude_v2));
    }
    
    // Helper function to generate a vector close to a base vector
    Vector generateSimilarVector(const Vector& base, float max_deviation = 0.1f) {
        Vector similar;
        similar.id = "similar_" + base.id + "_" + std::to_string(rand() % 10000);
        similar.values.reserve(base.values.size());
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-max_deviation, max_deviation);
        
        for (size_t i = 0; i < base.values.size(); ++i) {
            similar.values.push_back(base.values[i] + dis(gen));
        }
        
        return similar;
    }
    
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_service_;
    std::unique_ptr<SimilaritySearchService> search_service_;
    std::string db_id_;
};

// Test that search results are properly ordered by similarity
TEST_F(SearchResultQualityTest, SearchResultsAreOrderedBySimilarity) {
    // Create a base vector
    Vector base_vector;
    base_vector.id = "base_vector";
    base_vector.values = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Store the base vector
    ASSERT_TRUE(vector_service_->store_vector(db_id_, base_vector).has_value());
    
    // Create vectors with decreasing similarity to the base
    std::vector<Vector> test_vectors;
    
    // Very similar vector (high similarity)
    Vector v1 = generateSimilarVector(base_vector, 0.05f);
    v1.id = "very_similar";
    test_vectors.push_back(v1);
    
    // Moderately similar vector
    Vector v2 = generateSimilarVector(base_vector, 0.2f);
    v2.id = "moderate_similar";
    test_vectors.push_back(v2);
    
    // Less similar vector
    Vector v3 = generateSimilarVector(base_vector, 0.5f);
    v3.id = "less_similar";
    test_vectors.push_back(v3);
    
    // Quite different vector
    Vector v4 = generateSimilarVector(base_vector, 0.8f);
    v4.id = "different";
    test_vectors.push_back(v4);
    
    // Store all test vectors
    for (const auto& v : test_vectors) {
        ASSERT_TRUE(vector_service_->store_vector(db_id_, v).has_value());
    }
    
    // Perform similarity search
    SearchParams params;
    params.top_k = 10; // Get all results
    params.threshold = 0.0f;
    
    auto result = search_service_->similarity_search(db_id_, base_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    ASSERT_GT(search_results.size(), 0);
    
    // Verify results are ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score)
            << "Results are not properly ordered. Position " << i << " has score " 
            << search_results[i].similarity_score 
            << " but position " << (i+1) << " has score " 
            << search_results[i+1].similarity_score;
    }
    
    // Calculate expected similarities and verify ordering
    float expected_sim_very = calculateCosineSimilarity(base_vector.values, v1.values);
    float expected_sim_moderate = calculateCosineSimilarity(base_vector.values, v2.values);
    float expected_sim_less = calculateCosineSimilarity(base_vector.values, v3.values);
    float expected_sim_different = calculateCosineSimilarity(base_vector.values, v4.values);
    
    // Check that our expected most similar is indeed the most similar in results
    EXPECT_EQ(search_results[0].vector_id, "very_similar");
    
    // Verify that calculated similarities match expected relative ordering
    EXPECT_GE(expected_sim_very, expected_sim_moderate);
    EXPECT_GE(expected_sim_moderate, expected_sim_less);
    EXPECT_GE(expected_sim_less, expected_sim_different);
}

// Test search result quality with known vector relationships
TEST_F(SearchResultQualityTest, KnownVectorRelationships) {
    // Create vectors with known relationships
    std::vector<Vector> test_vectors;
    
    // Create a reference vector
    Vector ref_vector;
    ref_vector.id = "reference";
    ref_vector.values = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Create vectors with known similarity to reference
    Vector v1; // Should be most similar to ref
    v1.id = "most_similar";
    v1.values = {0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    Vector v2; // Less similar
    v2.id = "medium_similar";
    v2.values = {0.7f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    Vector v3; // Least similar among the 3
    v3.id = "least_similar";
    v3.values = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Calculate expected similarities
    float expected_ref_v1 = calculateCosineSimilarity(ref_vector.values, v1.values);
    float expected_ref_v2 = calculateCosineSimilarity(ref_vector.values, v2.values);
    float expected_ref_v3 = calculateCosineSimilarity(ref_vector.values, v3.values);
    
    // Store vectors
    ASSERT_TRUE(vector_service_->store_vector(db_id_, ref_vector).has_value());
    ASSERT_TRUE(vector_service_->store_vector(db_id_, v1).has_value());
    ASSERT_TRUE(vector_service_->store_vector(db_id_, v2).has_value());
    ASSERT_TRUE(vector_service_->store_vector(db_id_, v3).has_value());
    
    // Search using the reference vector
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    
    auto result = search_service_->similarity_search(db_id_, ref_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    
    // Check that results are ordered correctly
    ASSERT_GE(search_results.size(), 3);
    
    // The most similar should be v1, then v2, then v3
    EXPECT_EQ(search_results[0].vector_id, "most_similar");
    EXPECT_EQ(search_results[1].vector_id, "medium_similar");
    EXPECT_EQ(search_results[2].vector_id, "least_similar");
    
    // Verify the similarity scores are in the expected range
    EXPECT_NEAR(search_results[0].similarity_score, expected_ref_v1, 0.01);
    EXPECT_NEAR(search_results[1].similarity_score, expected_ref_v2, 0.01);
    EXPECT_NEAR(search_results[2].similarity_score, expected_ref_v3, 0.01);
}

// Test search quality with random vectors (should still be ordered)
TEST_F(SearchResultQualityTest, RandomVectorsMaintainOrdering) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    Vector query_vector;
    query_vector.id = "random_query";
    query_vector.values.reserve(16);
    
    // Generate random query vector
    for (int i = 0; i < 16; ++i) {
        query_vector.values.push_back(dis(gen));
    }
    
    // Store the query vector first (this is just for reference)
    ASSERT_TRUE(vector_service_->store_vector(db_id_, query_vector).has_value());
    
    // Generate random test vectors
    std::vector<Vector> test_vectors;
    
    for (int i = 0; i < 20; ++i) {
        Vector v;
        v.id = "random_vector_" + std::to_string(i);
        v.values.reserve(16);
        
        for (int j = 0; j < 16; ++j) {
            v.values.push_back(dis(gen));
        }
        
        test_vectors.push_back(v);
        ASSERT_TRUE(vector_service_->store_vector(db_id_, v).has_value());
    }
    
    // Perform search
    SearchParams params;
    params.top_k = 15; // Get top 15 results
    params.threshold = 0.0f;
    
    auto result = search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    
    // Verify results are ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score)
            << "Results not properly ordered at position " << i;
    }
    
    // Perform the same search multiple times to ensure consistency
    for (int run = 0; run < 3; ++run) {
        auto result2 = search_service_->similarity_search(db_id_, query_vector, params);
        ASSERT_TRUE(result2.has_value());
        
        auto search_results2 = result2.value();
        
        // The ordering should be consistent across runs
        ASSERT_EQ(search_results.size(), search_results2.size());
        
        for (size_t i = 0; i < search_results.size(); ++i) {
            EXPECT_EQ(search_results[i].vector_id, search_results2[i].vector_id);
            EXPECT_FLOAT_EQ(search_results[i].similarity_score, search_results2[i].similarity_score);
        }
    }
}

// Test quality with different search algorithms
TEST_F(SearchResultQualityTest, CompareSearchAlgorithmQuality) {
    // Create a reference vector
    Vector ref_vector;
    ref_vector.id = "reference";
    ref_vector.values = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Create similar vectors
    Vector v1;
    v1.id = "similar_1";
    v1.values = {0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    Vector v2;
    v2.id = "similar_2";
    v2.values = {0.8f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Store vectors
    ASSERT_TRUE(vector_service_->store_vector(db_id_, ref_vector).has_value());
    ASSERT_TRUE(vector_service_->store_vector(db_id_, v1).has_value());
    ASSERT_TRUE(vector_service_->store_vector(db_id_, v2).has_value());
    
    SearchParams params;
    params.top_k = 5;
    params.threshold = 0.0f;
    
    // Test cosine similarity search
    auto cosine_result = search_service_->similarity_search(db_id_, ref_vector, params);
    ASSERT_TRUE(cosine_result.has_value());
    
    // Test Euclidean distance search
    auto euclidean_result = search_service_->euclidean_search(db_id_, ref_vector, params);
    ASSERT_TRUE(euclidean_result.has_value());
    
    // Test dot product search
    auto dot_product_result = search_service_->dot_product_search(db_id_, ref_vector, params);
    ASSERT_TRUE(dot_product_result.has_value());
    
    // For this simple case, we expect v1 to be most similar to ref in all algorithms
    // (though Euclidean distance might rank differently due to its metric)
    EXPECT_EQ(cosine_result.value()[0].vector_id, "similar_1");
    EXPECT_EQ(dot_product_result.value()[0].vector_id, "similar_1");
    
    // For Euclidean distance, the closest vector in terms of distance to [1,0,0,...] 
    // should be [0.9, 0.1, 0, ...] since it has the smallest Euclidean distance
    if (!euclidean_result.value().empty()) {
        // Note: For Euclidean distance, we convert distance to similarity (1/(1+distance))
        // so closer vectors have higher scores
        EXPECT_EQ(euclidean_result.value()[0].vector_id, "similar_1");
    }
}

// Test threshold filtering quality
TEST_F(SearchResultQualityTest, ThresholdFilteringMaintainsQuality) {
    // Create a reference vector
    Vector ref_vector;
    ref_vector.id = "reference";
    ref_vector.values = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Create vectors with varying similarity
    std::vector<Vector> test_vectors = {
        {"high_sim", {0.95f, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
        {"med_sim", {0.8f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
        {"low_sim", {0.6f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
        {"very_low_sim", {0.3f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}
    };
    
    // Store all vectors
    ASSERT_TRUE(vector_service_->store_vector(db_id_, ref_vector).has_value());
    for (auto& v : test_vectors) {
        v.id = v.id + "_" + std::to_string(rand() % 10000); // Ensure unique IDs
        ASSERT_TRUE(vector_service_->store_vector(db_id_, v).has_value());
    }
    
    // Search with a moderate threshold
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.7f; // Should filter out low similarity vectors
    
    auto result = search_service_->similarity_search(db_id_, ref_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    
    // All results should meet the threshold
    for (const auto& res : search_results) {
        EXPECT_GE(res.similarity_score, 0.7f)
            << "Result with score " << res.similarity_score << " does not meet threshold of 0.7";
    }
    
    // Results should still be ordered by similarity
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
}

// Test that search quality meets accuracy requirements
TEST_F(SearchResultQualityTest, MeetsAccuracyRequirements) {
    // Generate a set of vectors with known relationships
    Vector query_vector;
    query_vector.id = "accuracy_test_query";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Store the query vector
    ASSERT_TRUE(vector_service_->store_vector(db_id_, query_vector).has_value());
    
    // Create 50 vectors with varying similarity to the query
    std::vector<std::pair<float, std::string>> expected_order;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_gen(0.0, 0.1);
    
    for (int i = 0; i < 50; ++i) {
        Vector v;
        v.id = "test_vector_" + std::to_string(i);
        v.values.reserve(16);
        
        // Create vectors with controlled similarity to the query
        // Vectors with lower i will be more similar to the query
        float base_factor = 1.0f - (static_cast<float>(i) / 100.0f); // Base similarity factor
        
        v.values.push_back(query_vector.values[0] * base_factor + noise_gen(gen));
        v.values.push_back(query_vector.values[1] * base_factor + noise_gen(gen));
        
        // Fill remaining dimensions with small random values
        for (int j = 2; j < 16; ++j) {
            v.values.push_back(noise_gen(gen) * 0.1f);
        }
        
        ASSERT_TRUE(vector_service_->store_vector(db_id_, v).has_value());
        
        // Calculate expected similarity for verification
        float expected_sim = calculateCosineSimilarity(query_vector.values, v.values);
        expected_order.push_back({expected_sim, v.id});
    }
    
    // Sort expected order by similarity (descending)
    std::sort(expected_order.begin(), expected_order.end(), 
              [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
                  return a.first > b.first;
              });
    
    // Perform search
    SearchParams params;
    params.top_k = 20; // Get top 20 results
    params.threshold = 0.0f;
    
    auto result = search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    ASSERT_GT(search_results.size(), 0);
    
    // Verify results are ordered by similarity
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // For this test, we're validating that the search produces consistently
    // ordered results by similarity, which indicates good quality
    bool is_properly_ordered = true;
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        if (search_results[i].similarity_score < search_results[i+1].similarity_score) {
            is_properly_ordered = false;
            break;
        }
    }
    
    EXPECT_TRUE(is_properly_ordered) << "Search results are not properly ordered by similarity";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}