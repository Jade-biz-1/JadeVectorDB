#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <cmath>

#include "api/rest/rest_api.h"
#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "services/database_service.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;

// Helper function to convert Database to DatabaseCreationParams
static DatabaseCreationParams to_creation_params(const Database& db) {
    DatabaseCreationParams params;
    params.name = db.name;
    params.description = db.description;
    params.vectorDimension = db.vectorDimension;
    params.indexType = db.indexType;
    params.indexParameters = db.indexParameters;
    params.sharding = db.sharding;
    params.replication = db.replication;
    params.embeddingModels = db.embeddingModels;
    params.metadataSchema = db.metadataSchema;
    if (db.retentionPolicy) {
        params.retentionPolicy = std::make_unique<Database::RetentionPolicy>(*db.retentionPolicy);
    }
    params.accessControl = db.accessControl;
    return params;
}

// Helper function to convert Database to DatabaseUpdateParams
static DatabaseUpdateParams to_update_params(const Database& db) {
    DatabaseUpdateParams params;
    params.name = db.name;
    params.description = db.description;
    params.vectorDimension = db.vectorDimension;
    params.indexType = db.indexType;
    std::unordered_map<std::string, std::string> idx_params;
    for (const auto& [k, v] : db.indexParameters) {
        idx_params[k] = v;
    }
    params.indexParameters = idx_params;
    params.sharding = db.sharding;
    params.replication = db.replication;
    params.embeddingModels = db.embeddingModels;
    std::unordered_map<std::string, std::string> meta_schema;
    for (const auto& [k, v] : db.metadataSchema) {
        meta_schema[k] = v;
    }
    params.metadataSchema = meta_schema;
    if (db.retentionPolicy) {
        params.retentionPolicy = std::make_unique<Database::RetentionPolicy>(*db.retentionPolicy);
    }
    params.accessControl = db.accessControl;
    return params;
}



using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Integration test for search API endpoints
class SearchApiIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create services for integration testing
        db_service_ = std::make_unique<DatabaseService>();
        vector_storage_service_ = std::make_unique<VectorStorageService>();
        similarity_search_service_ = std::make_unique<SimilaritySearchService>();
        
        // Initialize services
        db_service_->initialize();
        vector_storage_service_->initialize();
        similarity_search_service_->initialize();
        
        // Create a test database for integration testing
        Database test_db;
        test_db.name = "search_integration_test_db";
        test_db.vectorDimension = 4;
        test_db.description = "Test database for search integration testing";
        test_db.indexType = "HNSW";
        
        auto create_result = db_service_->create_database(to_creation_params(test_db));
        ASSERT_TRUE(create_result.has_value());
        db_id_ = create_result.value();
        
        // Store test vectors in the database
        store_test_vectors();
    }
    
    void TearDown() override {
        if (!db_id_.empty()) {
            auto delete_result = db_service_->delete_database(db_id_);
        }
    }
    
    void store_test_vectors() {
        // Create test vectors with known relationships for search testing
        std::vector<Vector> test_vectors;
        
        // Vector A - reference vector
        Vector v1;
        v1.id = "vector_A";
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
        v1.metadata.custom["category"] = "finance";
        v1.metadata.custom["score"] = 0.95f;
        v1.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
        test_vectors.push_back(v1);
        
        // Vector B - very similar to A
        Vector v2;
        v2.id = "vector_B";
        v2.values = {0.9f, 0.1f, 0.0f, 0.0f};
        v2.metadata.custom["category"] = "finance";
        v2.metadata.custom["score"] = 0.85f;
        v2.metadata.custom["tags"] = nlohmann::json::array({"investment", "banking"});
        test_vectors.push_back(v2);
        
        // Vector C - somewhat similar to A
        Vector v3;
        v3.id = "vector_C";
        v3.values = {0.7f, 0.3f, 0.0f, 0.0f};
        v3.metadata.custom["category"] = "technology";
        v3.metadata.custom["score"] = 0.75f;
        v3.metadata.custom["tags"] = nlohmann::json::array({"ai", "ml"});
        test_vectors.push_back(v3);
        
        // Vector D - less similar to A
        Vector v4;
        v4.id = "vector_D";
        v4.values = {0.5f, 0.5f, 0.0f, 0.0f};
        v4.metadata.custom["category"] = "healthcare";
        v4.metadata.custom["score"] = 0.65f;
        v4.metadata.custom["tags"] = nlohmann::json::array({"research", "clinical"});
        test_vectors.push_back(v4);
        
        // Vector E - quite different from A
        Vector v5;
        v5.id = "vector_E";
        v5.values = {0.0f, 1.0f, 0.0f, 0.0f};
        v5.metadata.custom["category"] = "technology";
        v5.metadata.custom["score"] = 0.55f;
        v5.metadata.custom["tags"] = nlohmann::json::array({"ai", "research"});
        test_vectors.push_back(v5);
        
        // Vector F - orthogonal to A
        Vector v6;
        v6.id = "vector_F";
        v6.values = {0.0f, 0.0f, 1.0f, 0.0f};
        v6.metadata.custom["category"] = "finance";
        v6.metadata.custom["score"] = 0.45f;
        v6.metadata.custom["tags"] = nlohmann::json::array({"trading", "cryptocurrency"});
        test_vectors.push_back(v6);
        
        // Store all test vectors
        for (const auto& v : test_vectors) {
            auto store_result = vector_storage_service_->store_vector(db_id_, v);
            ASSERT_TRUE(store_result.has_value()) << "Failed to store vector: " << v.id;
        }
    }
    
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
    std::string db_id_;
};

// Test similarity search endpoint integration
TEST_F(SearchApiIntegrationTest, SimilaritySearchEndpoint) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test cosine similarity search
    auto result = similarity_search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3);
    
    // Verify results are ordered by similarity (descending)
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

// Test Euclidean distance search endpoint integration
TEST_F(SearchApiIntegrationTest, EuclideanSearchEndpoint) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test Euclidean distance search
    auto result = similarity_search_service_->euclidean_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3);
    
    // Verify results are ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // For Euclidean distance, smaller distances mean higher similarity
    // vector_A should have distance 0 (similarity = 1.0)
    EXPECT_EQ(search_results[0].vector_id, "vector_A");
    EXPECT_FLOAT_EQ(search_results[0].similarity_score, 1.0f);
    
    // vector_B should have distance 0.1 (similarity = 1/(1+0.1) = 0.909...)
    EXPECT_EQ(search_results[1].vector_id, "vector_B");
    EXPECT_NEAR(search_results[1].similarity_score, 1.0f / 1.1f, 0.001f);
    
    // vector_C should have distance sqrt(0.7^2 + 0.3^2) = sqrt(0.49 + 0.09) = sqrt(0.58) = 0.761...
    // similarity = 1/(1+0.761) = 0.568...
    EXPECT_EQ(search_results[2].vector_id, "vector_C");
    EXPECT_NEAR(search_results[2].similarity_score, 1.0f / (1.0f + std::sqrt(0.58f)), 0.001f);
}

// Test dot product search endpoint integration
TEST_F(SearchApiIntegrationTest, DotProductSearchEndpoint) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 3;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test dot product search
    auto result = similarity_search_service_->dot_product_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 3);
    
    // Verify results are ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // For dot product, larger values mean higher similarity
    // vector_A should have dot product 1.0 (1*1 + 0*0 + 0*0 + 0*0)
    EXPECT_EQ(search_results[0].vector_id, "vector_A");
    EXPECT_FLOAT_EQ(search_results[0].similarity_score, 1.0f);
    
    // vector_B should have dot product 0.9 (1*0.9 + 0*0.1 + 0*0 + 0*0)
    EXPECT_EQ(search_results[1].vector_id, "vector_B");
    EXPECT_FLOAT_EQ(search_results[1].similarity_score, 0.9f);
    
    // vector_C should have dot product 0.7 (1*0.7 + 0*0.3 + 0*0 + 0*0)
    EXPECT_EQ(search_results[2].vector_id, "vector_C");
    EXPECT_FLOAT_EQ(search_results[2].similarity_score, 0.7f);
}

// Test search with threshold filtering
TEST_F(SearchApiIntegrationTest, SearchWithThresholdFiltering) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters with threshold
    SearchParams params;
    params.top_k = 10; // Get all results
    params.threshold = 0.8f; // Only results with similarity >= 0.8
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test cosine similarity search with threshold
    auto result = similarity_search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    
    // Should only return results with similarity >= 0.8
    EXPECT_EQ(search_results.size(), 2); // Only vector_A (1.0) and vector_B (0.9) meet threshold
    
    for (const auto& res : search_results) {
        EXPECT_GE(res.similarity_score, 0.8f);
    }
    
    // Results should still be ordered by similarity (descending)
    for (size_t i = 0; i < search_results.size() - 1; ++i) {
        EXPECT_GE(search_results[i].similarity_score, search_results[i+1].similarity_score);
    }
    
    // First result should be vector_A (similarity = 1.0)
    EXPECT_EQ(search_results[0].vector_id, "vector_A");
    EXPECT_FLOAT_EQ(search_results[0].similarity_score, 1.0f);
    
    // Second result should be vector_B (similarity = 0.9)
    EXPECT_EQ(search_results[1].vector_id, "vector_B");
    EXPECT_FLOAT_EQ(search_results[1].similarity_score, 0.9f);
}

// Test search with vector data inclusion
TEST_F(SearchApiIntegrationTest, SearchWithVectorDataInclusion) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters with vector data inclusion
    SearchParams params;
    params.top_k = 2;
    params.threshold = 0.0f;
    params.include_vector_data = true; // Include vector data in results
    params.include_metadata = false;
    
    // Test cosine similarity search with vector data inclusion
    auto result = similarity_search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 2);
    
    // All results should include vector data
    for (const auto& res : search_results) {
        EXPECT_FALSE(res.vector_data.id.empty());
        EXPECT_GT(res.vector_data.values.size(), 0);
    }
    
    // Check specific vector data
    EXPECT_EQ(search_results[0].vector_data.id, "vector_A");
    EXPECT_EQ(search_results[0].vector_data.values.size(), 4);
    EXPECT_FLOAT_EQ(search_results[0].vector_data.values[0], 1.0f);
    EXPECT_FLOAT_EQ(search_results[0].vector_data.values[1], 0.0f);
    EXPECT_FLOAT_EQ(search_results[0].vector_data.values[2], 0.0f);
    EXPECT_FLOAT_EQ(search_results[0].vector_data.values[3], 0.0f);
    
    EXPECT_EQ(search_results[1].vector_data.id, "vector_B");
    EXPECT_EQ(search_results[1].vector_data.values.size(), 4);
    EXPECT_FLOAT_EQ(search_results[1].vector_data.values[0], 0.9f);
    EXPECT_FLOAT_EQ(search_results[1].vector_data.values[1], 0.1f);
    EXPECT_FLOAT_EQ(search_results[1].vector_data.values[2], 0.0f);
    EXPECT_FLOAT_EQ(search_results[1].vector_data.values[3], 0.0f);
}

// Test search with metadata inclusion
TEST_F(SearchApiIntegrationTest, SearchWithMetadataInclusion) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters with metadata inclusion
    SearchParams params;
    params.top_k = 2;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = true; // Include metadata in results
    
    // Test cosine similarity search with metadata inclusion
    auto result = similarity_search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 2);
    
    // All results should include metadata
    for (const auto& res : search_results) {
        EXPECT_FALSE(res.vector_data.metadata.custom.empty());
    }
    
    // Check specific metadata
    EXPECT_EQ(search_results[0].vector_data.id, "vector_A");
    EXPECT_EQ(search_results[0].vector_data.metadata.custom["category"].get<std::string>(), "finance");
    EXPECT_FLOAT_EQ(search_results[0].vector_data.metadata.custom["score"].get<float>(), 0.95f);
    
    EXPECT_EQ(search_results[1].vector_data.id, "vector_B");
    EXPECT_EQ(search_results[1].vector_data.metadata.custom["category"].get<std::string>(), "finance");
    EXPECT_FLOAT_EQ(search_results[1].vector_data.metadata.custom["score"].get<float>(), 0.85f);
}

// Test K-nearest neighbor (KNN) search
TEST_F(SearchApiIntegrationTest, KnnSearch) {
    // Create query vector (same as vector_A)
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Set up search parameters for KNN
    SearchParams params;
    params.top_k = 4; // Get top 4 results (K=4)
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test KNN search
    auto result = similarity_search_service_->similarity_search(db_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    auto search_results = result.value();
    EXPECT_EQ(search_results.size(), 4); // Should return exactly 4 results (K=4)
    
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
    
    // Fourth result should be vector_D (less similar to query)
    EXPECT_EQ(search_results[3].vector_id, "vector_D");
    EXPECT_GT(search_results[3].similarity_score, 0.5f);
}

// Test search parameter validation
TEST_F(SearchApiIntegrationTest, ValidateSearchParams) {
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
    
    // Test invalid threshold (too high)
    SearchParams invalid_params2 = valid_params;
    invalid_params2.threshold = 1.5f;
    
    result = similarity_search_service_->validate_search_params(invalid_params2);
    EXPECT_FALSE(result.has_value());
    
    // Test invalid threshold (negative)
    SearchParams invalid_params3 = valid_params;
    invalid_params3.threshold = -0.5f;
    
    result = similarity_search_service_->validate_search_params(invalid_params3);
    EXPECT_FALSE(result.has_value());
}

// Test search algorithm availability
TEST_F(SearchApiIntegrationTest, GetAvailableAlgorithms) {
    auto algorithms = similarity_search_service_->get_available_algorithms();
    
    // Should include the basic algorithms
    EXPECT_THAT(algorithms, ::testing::Contains("cosine_similarity"));
    EXPECT_THAT(algorithms, ::testing::Contains("euclidean_distance"));
    EXPECT_THAT(algorithms, ::testing::Contains("dot_product"));
    
    // Should have exactly 3 algorithms
    EXPECT_EQ(algorithms.size(), 3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}