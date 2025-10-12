#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Include the headers we want to test
#include "services/vector_storage.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test fixture for VectorStorageService
class VectorStorageServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test database configuration
        test_db_.databaseId = "test_db_123";
        test_db_.name = "test_database";
        test_db_.description = "Test database for unit tests";
        test_db_.vectorDimension = 4; // Small dimension for testing
        test_db_.indexType = "HNSW";
        
        // Create vector storage service
        vector_storage_service_ = std::make_unique<VectorStorageService>();
        
        // Initialize the service
        auto init_result = vector_storage_service_->initialize();
        ASSERT_TRUE(init_result.has_value());
    }
    
    void TearDown() override {
        // Clean up
        vector_storage_service_.reset();
    }
    
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    Database test_db_;
};

// Test that the service initializes correctly
TEST_F(VectorStorageServiceTest, InitializeService) {
    // Service should already be initialized in SetUp
    EXPECT_NE(vector_storage_service_, nullptr);
}

// Test storing a single vector
TEST_F(VectorStorageServiceTest, StoreSingleVector) {
    // Create a test vector
    Vector test_vector;
    test_vector.id = "vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Store the vector
    auto result = vector_storage_service_->store_vector(test_db_.databaseId, test_vector);
    EXPECT_TRUE(result.has_value());
}

// Test retrieving a vector by ID
TEST_F(VectorStorageServiceTest, RetrieveVectorById) {
    // Create and store a test vector
    Vector test_vector;
    test_vector.id = "vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, test_vector);
    ASSERT_TRUE(store_result.has_value());
    
    // Retrieve the vector
    auto retrieve_result = vector_storage_service_->retrieve_vector(test_db_.databaseId, test_vector.id);
    ASSERT_TRUE(retrieve_result.has_value());
    
    Vector retrieved_vector = retrieve_result.value();
    EXPECT_EQ(retrieved_vector.id, test_vector.id);
    EXPECT_EQ(retrieved_vector.values.size(), test_vector.values.size());
    
    for (size_t i = 0; i < test_vector.values.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_vector.values[i], test_vector.values[i]);
    }
}

// Test updating a vector
TEST_F(VectorStorageServiceTest, UpdateVector) {
    // Create and store a test vector
    Vector test_vector;
    test_vector.id = "vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, test_vector);
    ASSERT_TRUE(store_result.has_value());
    
    // Update the vector with new values
    Vector updated_vector = test_vector;
    updated_vector.values = {5.0f, 6.0f, 7.0f, 8.0f};
    
    auto update_result = vector_storage_service_->update_vector(test_db_.databaseId, updated_vector);
    EXPECT_TRUE(update_result.has_value());
    
    // Retrieve the vector to verify it was updated
    auto retrieve_result = vector_storage_service_->retrieve_vector(test_db_.databaseId, test_vector.id);
    ASSERT_TRUE(retrieve_result.has_value());
    
    Vector retrieved_vector = retrieve_result.value();
    EXPECT_EQ(retrieved_vector.id, test_vector.id);
    EXPECT_EQ(retrieved_vector.values.size(), updated_vector.values.size());
    
    for (size_t i = 0; i < updated_vector.values.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_vector.values[i], updated_vector.values[i]);
    }
}

// Test deleting a vector
TEST_F(VectorStorageServiceTest, DeleteVector) {
    // Create and store a test vector
    Vector test_vector;
    test_vector.id = "vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, test_vector);
    ASSERT_TRUE(store_result.has_value());
    
    // Delete the vector
    auto delete_result = vector_storage_service_->delete_vector(test_db_.databaseId, test_vector.id);
    EXPECT_TRUE(delete_result.has_value());
    
    // Try to retrieve the deleted vector (should fail)
    auto retrieve_result = vector_storage_service_->retrieve_vector(test_db_.databaseId, test_vector.id);
    EXPECT_FALSE(retrieve_result.has_value());
}

// Test batch storing vectors
TEST_F(VectorStorageServiceTest, BatchStoreVectors) {
    // Create test vectors
    std::vector<Vector> test_vectors;
    
    for (int i = 0; i < 5; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = {
            static_cast<float>(i + 1), 
            static_cast<float>(i + 2), 
            static_cast<float>(i + 3), 
            static_cast<float>(i + 4)
        };
        test_vectors.push_back(v);
    }
    
    // Store all vectors in a batch
    auto result = vector_storage_service_->batch_store_vectors(test_db_.databaseId, test_vectors);
    EXPECT_TRUE(result.has_value());
}

// Test retrieving multiple vectors by ID
TEST_F(VectorStorageServiceTest, RetrieveMultipleVectors) {
    // Create and store test vectors
    std::vector<Vector> test_vectors;
    
    for (int i = 0; i < 3; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = {
            static_cast<float>(i + 1), 
            static_cast<float>(i + 2), 
            static_cast<float>(i + 3), 
            static_cast<float>(i + 4)
        };
        test_vectors.push_back(v);
        
        auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, v);
        ASSERT_TRUE(store_result.has_value());
    }
    
    // Retrieve all vectors by their IDs
    std::vector<std::string> vector_ids;
    for (const auto& v : test_vectors) {
        vector_ids.push_back(v.id);
    }
    
    auto result = vector_storage_service_->retrieve_vectors(test_db_.databaseId, vector_ids);
    ASSERT_TRUE(result.has_value());
    
    std::vector<Vector> retrieved_vectors = result.value();
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

// Test vector validation
TEST_F(VectorStorageServiceTest, ValidateVector) {
    // Create a valid vector
    Vector valid_vector;
    valid_vector.id = "valid_vector";
    valid_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Validation should pass for a valid vector
    auto result = vector_storage_service_->validate_vector(test_db_.databaseId, valid_vector);
    EXPECT_TRUE(result.has_value());
    
    // Create an invalid vector (wrong dimension)
    Vector invalid_vector;
    invalid_vector.id = "invalid_vector";
    invalid_vector.values = {1.0f, 2.0f, 3.0f}; // Only 3 dimensions, but database expects 4
    
    // Validation should fail for an invalid vector
    result = vector_storage_service_->validate_vector(test_db_.databaseId, invalid_vector);
    EXPECT_FALSE(result.has_value());
    
    // Create an invalid vector (empty ID)
    Vector invalid_vector2;
    invalid_vector2.id = ""; // Empty ID
    invalid_vector2.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Validation should fail for an invalid vector
    result = vector_storage_service_->validate_vector(test_db_.databaseId, invalid_vector2);
    EXPECT_FALSE(result.has_value());
}

// Test checking if a vector exists
TEST_F(VectorStorageServiceTest, VectorExists) {
    // Create and store a test vector
    Vector test_vector;
    test_vector.id = "vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, test_vector);
    ASSERT_TRUE(store_result.has_value());
    
    // Check if the vector exists (should be true)
    auto exists_result = vector_storage_service_->vector_exists(test_db_.databaseId, test_vector.id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
    
    // Check if a non-existent vector exists (should be false)
    exists_result = vector_storage_service_->vector_exists(test_db_.databaseId, "non_existent_vector");
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
}

// Test getting vector count
TEST_F(VectorStorageServiceTest, GetVectorCount) {
    // Initially, there should be 0 vectors
    auto count_result = vector_storage_service_->get_vector_count(test_db_.databaseId);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 0);
    
    // Store some test vectors
    for (int i = 0; i < 3; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = {
            static_cast<float>(i + 1), 
            static_cast<float>(i + 2), 
            static_cast<float>(i + 3), 
            static_cast<float>(i + 4)
        };
        
        auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, v);
        ASSERT_TRUE(store_result.has_value());
    }
    
    // Now there should be 3 vectors
    count_result = vector_storage_service_->get_vector_count(test_db_.databaseId);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 3);
}

// Test getting all vector IDs
TEST_F(VectorStorageServiceTest, GetAllVectorIds) {
    // Initially, there should be no vector IDs
    auto ids_result = vector_storage_service_->get_all_vector_ids(test_db_.databaseId);
    ASSERT_TRUE(ids_result.has_value());
    EXPECT_EQ(ids_result.value().size(), 0);
    
    // Store some test vectors
    std::vector<std::string> expected_ids;
    for (int i = 0; i < 3; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        expected_ids.push_back(v.id);
        v.values = {
            static_cast<float>(i + 1), 
            static_cast<float>(i + 2), 
            static_cast<float>(i + 3), 
            static_cast<float>(i + 4)
        };
        
        auto store_result = vector_storage_service_->store_vector(test_db_.databaseId, v);
        ASSERT_TRUE(store_result.has_value());
    }
    
    // Now there should be 3 vector IDs
    ids_result = vector_storage_service_->get_all_vector_ids(test_db_.databaseId);
    ASSERT_TRUE(ids_result.has_value());
    EXPECT_EQ(ids_result.value().size(), 3);
    
    // Verify all expected IDs are present
    auto retrieved_ids = ids_result.value();
    for (const auto& expected_id : expected_ids) {
        bool found = false;
        for (const auto& retrieved_id : retrieved_ids) {
            if (retrieved_id == expected_id) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Expected ID " << expected_id << " not found in retrieved IDs";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}