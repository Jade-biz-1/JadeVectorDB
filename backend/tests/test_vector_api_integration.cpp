#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>

#include "api/rest/rest_api.h"
#include "services/vector_storage.h"
#include "services/database_service.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"

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

using ::testing::Return;
using ::testing::_;

// Integration test for vector API endpoints
class VectorApiIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create services for integration testing
        db_service_ = std::make_unique<DatabaseService>();
        vector_service_ = std::make_unique<VectorStorageService>();
        
        // Initialize services
        db_service_->initialize();
        vector_service_->initialize();
        
        // Create a test database for integration testing
        Database test_db;
        test_db.name = "integration_test_db";
        test_db.vectorDimension = 4;
        test_db.description = "Test database for integration testing";
        
        auto create_result = db_service_->create_database(to_creation_params(test_db));
        ASSERT_TRUE(create_result.has_value());
        db_id_ = create_result.value();
    }
    
    void TearDown() override {
        if (!db_id_.empty()) {
            auto delete_result = db_service_->delete_database(db_id_);
        }
    }
    
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_service_;
    std::unique_ptr<RestApiService> rest_api_service_;
    std::string db_id_;
};

// Test vector storage and retrieval integration
TEST_F(VectorApiIntegrationTest, StoreAndRetrieveVector) {
    // Store test vectors in the database
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 0.1f, 0.2f, 0.3f};
    v1.metadata.custom["category"] = "finance";
    v1.metadata.custom["score"] = 0.95f;
    v1.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
    
    auto store_result = vector_service_->store_vector(db_id_, v1);
    ASSERT_TRUE(store_result.has_value());
    
    Vector v2;
    v2.id = "vector_2";
    v2.values = {0.1f, 1.0f, 0.2f, 0.8f};
    v2.metadata.custom["category"] = "technology";
    v2.metadata.custom["score"] = 0.75f;
    v2.metadata.custom["tags"] = nlohmann::json::array({"ai", "ml"});
    
    store_result = vector_service_->store_vector(db_id_, v2);
    ASSERT_TRUE(store_result.has_value());
    
    // Retrieve vectors by ID
    auto retrieve_result = vector_service_->retrieve_vector(db_id_, v1.id);
    ASSERT_TRUE(retrieve_result.has_value());
    
    Vector retrieved_v1 = retrieve_result.value();
    EXPECT_EQ(retrieved_v1.id, v1.id);
    EXPECT_EQ(retrieved_v1.values.size(), v1.values.size());
    for (size_t i = 0; i < v1.values.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_v1.values[i], v1.values[i]);
    }
    EXPECT_EQ(retrieved_v1.metadata.custom["category"].get<std::string>(), "finance");
    EXPECT_FLOAT_EQ(retrieved_v1.metadata.custom["score"].get<float>(), 0.95f);
    
    retrieve_result = vector_service_->retrieve_vector(db_id_, v2.id);
    ASSERT_TRUE(retrieve_result.has_value());
    
    Vector retrieved_v2 = retrieve_result.value();
    EXPECT_EQ(retrieved_v2.id, v2.id);
    EXPECT_EQ(retrieved_v2.values.size(), v2.values.size());
    for (size_t i = 0; i < v2.values.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_v2.values[i], v2.values[i]);
    }
    EXPECT_EQ(retrieved_v2.metadata.custom["category"].get<std::string>(), "technology");
    EXPECT_FLOAT_EQ(retrieved_v2.metadata.custom["score"].get<float>(), 0.75f);
}

// Test batch vector operations integration
TEST_F(VectorApiIntegrationTest, BatchVectorOperations) {
    // Create test vectors for batch operations
    std::vector<Vector> test_vectors;
    
    for (int i = 0; i < 5; ++i) {
        Vector v;
        v.id = "batch_vector_" + std::to_string(i);
        v.values = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2), static_cast<float>(i+3)};
        v.metadata.custom["category"] = (i % 2 == 0) ? "finance" : "technology";
        v.metadata.custom["score"] = 0.5f + (static_cast<float>(i) * 0.1f);
        v.metadata.custom["tags"] = nlohmann::json::array({"tag" + std::to_string(i)});
        test_vectors.push_back(v);
    }
    
    // Store all vectors in a batch
    auto store_result = vector_service_->batch_store_vectors(db_id_, test_vectors);
    ASSERT_TRUE(store_result.has_value());
    
    // Retrieve all vectors by ID
    std::vector<std::string> vector_ids;
    for (const auto& v : test_vectors) {
        vector_ids.push_back(v.id);
    }
    
    auto retrieve_result = vector_service_->retrieve_vectors(db_id_, vector_ids);
    ASSERT_TRUE(retrieve_result.has_value());
    
    std::vector<Vector> retrieved_vectors = retrieve_result.value();
    EXPECT_EQ(retrieved_vectors.size(), test_vectors.size());
    
    // Verify each vector was retrieved correctly
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        bool found = false;
        for (const auto& retrieved : retrieved_vectors) {
            if (retrieved.id == test_vectors[i].id) {
                EXPECT_EQ(retrieved.values.size(), test_vectors[i].values.size());
                for (size_t j = 0; j < test_vectors[i].values.size(); ++j) {
                    EXPECT_FLOAT_EQ(retrieved.values[j], test_vectors[i].values[j]);
                }
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Vector " << test_vectors[i].id << " was not found in retrieval";
    }
}

// Test vector update integration
TEST_F(VectorApiIntegrationTest, UpdateVector) {
    // Create and store a test vector
    Vector initial_vector;
    initial_vector.id = "update_test_vector";
    initial_vector.values = {1.0f, 0.1f, 0.2f, 0.3f};
    initial_vector.metadata.custom["category"] = "finance";
    initial_vector.metadata.custom["score"] = 0.95f;
    initial_vector.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
    
    auto store_result = vector_service_->store_vector(db_id_, initial_vector);
    ASSERT_TRUE(store_result.has_value());
    
    // Retrieve the vector to verify it was stored
    auto retrieve_result = vector_service_->retrieve_vector(db_id_, initial_vector.id);
    ASSERT_TRUE(retrieve_result.has_value());
    Vector retrieved_vector = retrieve_result.value();
    EXPECT_EQ(retrieved_vector.metadata.custom["category"].get<std::string>(), "finance");
    EXPECT_FLOAT_EQ(retrieved_vector.metadata.custom["score"].get<float>(), 0.95f);
    
    // Update the vector with new values
    Vector updated_vector = initial_vector;
    updated_vector.values = {0.9f, 0.2f, 0.1f, 0.4f};
    updated_vector.metadata.custom["category"] = "technology";
    updated_vector.metadata.custom["score"] = 0.85f;
    updated_vector.metadata.custom["tags"] = nlohmann::json::array({"ai", "ml"});
    
    auto update_result = vector_service_->update_vector(db_id_, updated_vector);
    ASSERT_TRUE(update_result.has_value());
    
    // Retrieve the vector again to verify it was updated
    retrieve_result = vector_service_->retrieve_vector(db_id_, initial_vector.id);
    ASSERT_TRUE(retrieve_result.has_value());
    retrieved_vector = retrieve_result.value();
    EXPECT_EQ(retrieved_vector.metadata.custom["category"].get<std::string>(), "technology");
    EXPECT_FLOAT_EQ(retrieved_vector.metadata.custom["score"].get<float>(), 0.85f);
    EXPECT_EQ(retrieved_vector.values.size(), updated_vector.values.size());
    for (size_t i = 0; i < updated_vector.values.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_vector.values[i], updated_vector.values[i]);
    }
}

// Test vector deletion integration
TEST_F(VectorApiIntegrationTest, DeleteVector) {
    // Create and store a test vector
    Vector test_vector;
    test_vector.id = "delete_test_vector";
    test_vector.values = {1.0f, 0.1f, 0.2f, 0.3f};
    test_vector.metadata.custom["category"] = "finance";
    test_vector.metadata.custom["score"] = 0.95f;
    test_vector.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
    
    auto store_result = vector_service_->store_vector(db_id_, test_vector);
    ASSERT_TRUE(store_result.has_value());
    
    // Verify the vector exists before deletion
    auto exists_result = vector_service_->vector_exists(db_id_, test_vector.id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
    
    // Delete the vector
    auto delete_result = vector_service_->delete_vector(db_id_, test_vector.id);
    ASSERT_TRUE(delete_result.has_value());
    
    // Verify the vector no longer exists
    exists_result = vector_service_->vector_exists(db_id_, test_vector.id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
    
    // Attempt to retrieve the deleted vector (should fail)
    auto retrieve_result = vector_service_->retrieve_vector(db_id_, test_vector.id);
    EXPECT_FALSE(retrieve_result.has_value());
}

// Test vector count and ID listing integration
TEST_F(VectorApiIntegrationTest, VectorCountAndIdListing) {
    // Initially, there should be 0 vectors
    auto count_result = vector_service_->get_vector_count(db_id_);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 0);
    
    // Get all vector IDs (should be empty)
    auto ids_result = vector_service_->get_all_vector_ids(db_id_);
    ASSERT_TRUE(ids_result.has_value());
    EXPECT_EQ(ids_result.value().size(), 0);
    
    // Store some test vectors
    std::vector<Vector> test_vectors;
    
    for (int i = 0; i < 3; ++i) {
        Vector v;
        v.id = "count_test_vector_" + std::to_string(i);
        v.values = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2), static_cast<float>(i+3)};
        v.metadata.custom["category"] = "test";
        v.metadata.custom["score"] = 0.5f + (static_cast<float>(i) * 0.1f);
        test_vectors.push_back(v);
    }
    
    for (const auto& v : test_vectors) {
        auto store_result = vector_service_->store_vector(db_id_, v);
        ASSERT_TRUE(store_result.has_value());
    }
    
    // Now there should be 3 vectors
    count_result = vector_service_->get_vector_count(db_id_);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 3);
    
    // Get all vector IDs (should have 3 IDs)
    ids_result = vector_service_->get_all_vector_ids(db_id_);
    ASSERT_TRUE(ids_result.has_value());
    EXPECT_EQ(ids_result.value().size(), 3);
    
    // Verify all expected IDs are present
    auto retrieved_ids = ids_result.value();
    for (const auto& v : test_vectors) {
        bool found = false;
        for (const auto& id : retrieved_ids) {
            if (id == v.id) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Vector ID " << v.id << " not found in retrieved IDs";
    }
}

// Test vector validation integration
TEST_F(VectorApiIntegrationTest, VectorValidation) {
    // Create a valid vector
    Vector valid_vector;
    valid_vector.id = "valid_vector";
    valid_vector.values = {1.0f, 0.1f, 0.2f, 0.3f};
    valid_vector.metadata.custom["category"] = "finance";
    valid_vector.metadata.custom["score"] = 0.95f;
    valid_vector.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
    
    // Validation should pass for a valid vector
    auto result = vector_service_->validate_vector(db_id_, valid_vector);
    EXPECT_TRUE(result.has_value());
    
    // Create an invalid vector (wrong dimension)
    Vector invalid_vector;
    invalid_vector.id = "invalid_vector";
    invalid_vector.values = {1.0f, 0.1f, 0.2f}; // Only 3 dimensions, but database expects 4
    invalid_vector.metadata.custom["category"] = "finance";
    invalid_vector.metadata.custom["score"] = 0.95f;
    invalid_vector.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
    
    // Validation should fail for an invalid vector
    result = vector_service_->validate_vector(db_id_, invalid_vector);
    EXPECT_FALSE(result.has_value());
    
    // Create an invalid vector (empty ID)
    Vector invalid_vector2;
    invalid_vector2.id = ""; // Empty ID
    invalid_vector2.values = {1.0f, 0.1f, 0.2f, 0.3f};
    invalid_vector2.metadata.custom["category"] = "finance";
    invalid_vector2.metadata.custom["score"] = 0.95f;
    invalid_vector2.metadata.custom["tags"] = nlohmann::json::array({"investment", "trading"});
    
    // Validation should fail for an invalid vector
    result = vector_service_->validate_vector(db_id_, invalid_vector2);
    EXPECT_FALSE(result.has_value());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}