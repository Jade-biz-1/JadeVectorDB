#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/auth.h"

namespace jadevectordb {

// Test fixture for vector API integration tests
class VectorAPIIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services 
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        vector_storage_ = std::make_unique<VectorStorageService>(std::move(db_layer_));
        vector_storage_->initialize();
        
        // Create a test database
        Database db;
        db.name = "vector_api_test_db";
        db.description = "Test database for vector API integration";
        db.vectorDimension = 4;
        
        auto result = vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            vector_storage_->db_layer_->delete_database(test_database_id_);
        }
    }

    std::unique_ptr<VectorStorageService> vector_storage_;
    std::string test_database_id_;
};

// Test vector storage functionality
TEST_F(VectorAPIIntegrationTest, StoreAndRetrieveVector) {
    Vector test_vector;
    test_vector.id = "test_vec_1";
    test_vector.values = {0.1f, 0.2f, 0.3f, 0.4f};
    
    // Store the vector
    auto store_result = vector_storage_->store_vector(test_database_id_, test_vector);
    EXPECT_TRUE(store_result.has_value()) << "Failed to store vector: " 
                                         << ErrorHandler::format_error(store_result.error());
    
    // Verify the vector exists
    auto exists_result = vector_storage_->vector_exists(test_database_id_, "test_vec_1");
    EXPECT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
    
    // Retrieve the vector
    auto retrieve_result = vector_storage_->retrieve_vector(test_database_id_, "test_vec_1");
    EXPECT_TRUE(retrieve_result.has_value()) << "Failed to retrieve vector: " 
                                           << ErrorHandler::format_error(retrieve_result.error());
    
    EXPECT_EQ(retrieve_result.value().id, "test_vec_1");
    ASSERT_EQ(retrieve_result.value().values.size(), 4);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[0], 0.1f);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[1], 0.2f);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[2], 0.3f);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[3], 0.4f);
}

TEST_F(VectorAPIIntegrationTest, UpdateVector) {
    // First store a vector
    Vector original_vector;
    original_vector.id = "update_test_vec";
    original_vector.values = {0.1f, 0.2f, 0.3f, 0.4f};
    
    auto store_result = vector_storage_->store_vector(test_database_id_, original_vector);
    EXPECT_TRUE(store_result.has_value());
    
    // Retrieve and verify original values
    auto retrieve_result = vector_storage_->retrieve_vector(test_database_id_, "update_test_vec");
    ASSERT_TRUE(retrieve_result.has_value());
    EXPECT_FLOAT_EQ(retrieve_result.value().values[0], 0.1f);
    
    // Update the vector with new values
    Vector updated_vector = original_vector;
    updated_vector.values = {0.9f, 0.8f, 0.7f, 0.6f};
    
    auto update_result = vector_storage_->update_vector(test_database_id_, updated_vector);
    EXPECT_TRUE(update_result.has_value()) << "Failed to update vector: " 
                                         << ErrorHandler::format_error(update_result.error());
    
    // Retrieve and verify updated values
    auto updated_retrieve = vector_storage_->retrieve_vector(test_database_id_, "update_test_vec");
    ASSERT_TRUE(updated_retrieve.has_value());
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[0], 0.9f);
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[1], 0.8f);
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[2], 0.7f);
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[3], 0.6f);
}

TEST_F(VectorAPIIntegrationTest, DeleteVector) {
    // Store a vector
    Vector vector;
    vector.id = "delete_test_vec";
    vector.values = {0.5f, 0.6f, 0.7f, 0.8f};
    
    auto store_result = vector_storage_->store_vector(test_database_id_, vector);
    EXPECT_TRUE(store_result.has_value());
    
    // Verify it exists
    auto exists_before = vector_storage_->vector_exists(test_database_id_, "delete_test_vec");
    EXPECT_TRUE(exists_before.has_value());
    EXPECT_TRUE(exists_before.value());
    
    // Delete the vector
    auto delete_result = vector_storage_->delete_vector(test_database_id_, "delete_test_vec");
    EXPECT_TRUE(delete_result.has_value()) << "Failed to delete vector: " 
                                         << ErrorHandler::format_error(delete_result.error());
    
    // Verify it no longer exists
    auto exists_after = vector_storage_->vector_exists(test_database_id_, "delete_test_vec");
    EXPECT_TRUE(exists_after.has_value());
    EXPECT_FALSE(exists_after.value());
    
    // Try to retrieve the deleted vector
    auto retrieve_result = vector_storage_->retrieve_vector(test_database_id_, "delete_test_vec");
    EXPECT_FALSE(retrieve_result.has_value());
}

TEST_F(VectorAPIIntegrationTest, BatchStoreVectors) {
    // Create multiple vectors for batch storage
    std::vector<Vector> vectors;
    
    for (int i = 0; i < 5; ++i) {
        Vector v;
        v.id = "batch_vec_" + std::to_string(i);
        v.values = {static_cast<float>(i)*0.1f, 
                    static_cast<float>(i)*0.2f, 
                    static_cast<float>(i)*0.3f, 
                    static_cast<float>(i)*0.4f};
        vectors.push_back(v);
    }
    
    // Batch store vectors
    auto batch_result = vector_storage_->batch_store_vectors(test_database_id_, vectors);
    EXPECT_TRUE(batch_result.has_value()) << "Failed to batch store vectors: " 
                                        << ErrorHandler::format_error(batch_result.error());
    
    // Verify all vectors were stored
    for (const auto& v : vectors) {
        auto exists_result = vector_storage_->vector_exists(test_database_id_, v.id);
        EXPECT_TRUE(exists_result.has_value());
        EXPECT_TRUE(exists_result.value()) << "Vector " << v.id << " was not stored";
        
        auto retrieve_result = vector_storage_->retrieve_vector(test_database_id_, v.id);
        ASSERT_TRUE(retrieve_result.has_value()) << "Failed to retrieve vector " << v.id;
        EXPECT_EQ(retrieve_result.value().id, v.id);
    }
}

TEST_F(VectorAPIIntegrationTest, BatchDeleteVectors) {
    // First store multiple vectors
    std::vector<Vector> vectors;
    
    for (int i = 0; i < 3; ++i) {
        Vector v;
        v.id = "batch_delete_vec_" + std::to_string(i);
        v.values = {static_cast<float>(i)*0.1f, 
                    static_cast<float>(i)*0.2f, 
                    static_cast<float>(i)*0.3f, 
                    static_cast<float>(i)*0.4f};
        vectors.push_back(v);
    }
    
    // Store all vectors
    for (const auto& v : vectors) {
        auto store_result = vector_storage_->store_vector(test_database_id_, v);
        EXPECT_TRUE(store_result.has_value()) << "Failed to store vector " << v.id;
    }
    
    // Verify all vectors exist
    for (const auto& v : vectors) {
        auto exists_result = vector_storage_->vector_exists(test_database_id_, v.id);
        EXPECT_TRUE(exists_result.has_value());
        EXPECT_TRUE(exists_result.value());
    }
    
    // Extract IDs for batch deletion
    std::vector<std::string> vector_ids;
    for (const auto& v : vectors) {
        vector_ids.push_back(v.id);
    }
    
    // Batch delete vectors
    auto batch_delete_result = vector_storage_->batch_delete_vectors(test_database_id_, vector_ids);
    EXPECT_TRUE(batch_delete_result.has_value()) << "Failed to batch delete vectors: " 
                                               << ErrorHandler::format_error(batch_delete_result.error());
    
    // Verify all vectors were deleted
    for (const auto& v : vectors) {
        auto exists_result = vector_storage_->vector_exists(test_database_id_, v.id);
        EXPECT_TRUE(exists_result.has_value());
        EXPECT_FALSE(exists_result.value()) << "Vector " << v.id << " was not deleted";
    }
}

TEST_F(VectorAPIIntegrationTest, VectorValidation) {
    // Test with correct dimension
    Vector valid_vector;
    valid_vector.id = "valid_vec";
    valid_vector.values = {0.1f, 0.2f, 0.3f, 0.4f};  // 4 dimensions, matching database
    
    auto valid_result = vector_storage_->validate_vector(test_database_id_, valid_vector);
    EXPECT_TRUE(valid_result.has_value()) << "Valid vector should pass validation";
    
    // Test with incorrect dimension
    Vector invalid_vector;
    invalid_vector.id = "invalid_vec";
    invalid_vector.values = {0.1f, 0.2f};  // 2 dimensions, should fail
    
    auto invalid_result = vector_storage_->validate_vector(test_database_id_, invalid_vector);
    EXPECT_FALSE(invalid_result.has_value()) << "Invalid vector should fail validation";
    
    // Test with empty ID
    Vector empty_id_vector;
    empty_id_vector.id = "";  // Empty ID should fail
    empty_id_vector.values = {0.1f, 0.2f, 0.3f, 0.4f};
    
    auto empty_id_result = vector_storage_->validate_vector(test_database_id_, empty_id_vector);
    EXPECT_FALSE(empty_id_result.has_value()) << "Vector with empty ID should fail validation";
}

TEST_F(VectorAPIIntegrationTest, MultipleDatabaseIsolation) {
    // Create a second test database with different dimension
    Database db2;
    db2.name = "vector_api_test_db_2";
    db2.description = "Second test database";
    db2.vectorDimension = 2;  // Different dimension
    
    auto db2_result = vector_storage_->db_layer_->create_database(db2);
    ASSERT_TRUE(db2_result.has_value());
    std::string test_database_id_2 = db2_result.value();
    
    // Store a vector in first database
    Vector vec1;
    vec1.id = "vec_in_db1";
    vec1.values = {0.1f, 0.2f, 0.3f, 0.4f};
    
    auto store_result1 = vector_storage_->store_vector(test_database_id_, vec1);
    EXPECT_TRUE(store_result1.has_value());
    
    // Store a vector in second database (with correct dimensions for that DB)
    Vector vec2;
    vec2.id = "vec_in_db2";
    vec2.values = {0.5f, 0.6f};  // 2 dimensions for DB2
    
    auto store_result2 = vector_storage_->store_vector(test_database_id_2, vec2);
    EXPECT_TRUE(store_result2.has_value());
    
    // Verify vectors are isolated between databases
    auto retrieve_from_db1 = vector_storage_->retrieve_vector(test_database_id_, "vec_in_db2");
    EXPECT_FALSE(retrieve_from_db1.has_value()) << "Vector should not exist in database 1";
    
    auto retrieve_from_db2 = vector_storage_->retrieve_vector(test_database_id_2, "vec_in_db1");
    EXPECT_FALSE(retrieve_from_db2.has_value()) << "Vector should not exist in database 2";
    
    // Clean up second database
    vector_storage_->db_layer_->delete_database(test_database_id_2);
}

} // namespace jadevectordb