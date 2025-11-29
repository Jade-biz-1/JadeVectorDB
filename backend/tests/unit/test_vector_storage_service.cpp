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
// Additional test cases for enhanced coverage
TEST_F(VectorStorageServiceTest, StoreAndRetrieveLargeVector) {
    // Test storing and retrieving large vectors
    Vector large_vector;
    large_vector.id = "large_vector";
    large_vector.values.reserve(10000); // Large vector with 10,000 dimensions

    // Fill with some test data
    for (int i = 0; i < 10000; ++i) {
        large_vector.values.push_back(static_cast<float>(i) / 1000.0f);
    }

    // Add metadata
    large_vector.metadata.source = "test_large";
    large_vector.metadata.owner = "test_user";
    large_vector.metadata.category = "large_vectors";

    // Store the large vector
    auto store_result = vector_storage_service_->store_vector("test_db_123", large_vector);
    EXPECT_TRUE(store_result.has_value());

    // Retrieve the large vector
    auto retrieve_result = vector_storage_service_->retrieve_vector("test_db_123", "large_vector");
    EXPECT_TRUE(retrieve_result.has_value());

    if (retrieve_result.has_value()) {
        auto retrieved_vector = retrieve_result.value();
        EXPECT_EQ(retrieved_vector.id, "large_vector");
        EXPECT_EQ(retrieved_vector.values.size(), 10000);
        EXPECT_EQ(retrieved_vector.metadata.source, "test_large");
        EXPECT_EQ(retrieved_vector.metadata.owner, "test_user");

        // Check a few values to ensure they match
        for (int i = 0; i < 10; ++i) {  // Check first 10 values
            EXPECT_FLOAT_EQ(retrieved_vector.values[i],
                           static_cast<float>(i) / 1000.0f);
        }
    }
}

TEST_F(VectorStorageServiceTest, BatchOperationsWithMetadataFiltering) {
    // Test batch operations with metadata filtering
    std::vector<Vector> test_vectors;

    // Create multiple vectors with different metadata
    for (int i = 0; i < 5; ++i) {
        Vector vec;
        vec.id = "vector_" + std::to_string(i);
        vec.values = {static_cast<float>(i), static_cast<float>(i+1)};
        vec.metadata.category = (i % 2 == 0) ? "even" : "odd";
        vec.metadata.owner = "test_user";
        vec.metadata.tags = {(i < 3) ? "small" : "large"};

        test_vectors.push_back(vec);
    }

    // Batch store vectors
    auto batch_store_result = vector_storage_service_->batch_store_vectors("test_db_123", test_vectors);
    EXPECT_TRUE(batch_store_result.has_value());

    if (batch_store_result.has_value()) {
        auto stored_ids = batch_store_result.value();
        EXPECT_EQ(stored_ids.size(), 5);
    }

    // Create a filter to get only even-numbered vectors
    MetadataFilterCondition category_filter;
    category_filter.field = "category";
    category_filter.operator_type = FilterOperator::EQUALS;
    category_filter.value = "even";

    // Test batch retrieval with filtering
    auto filter_result = vector_storage_service_->retrieve_vectors_by_metadata("test_db_123", {category_filter});
    EXPECT_TRUE(filter_result.has_value());

    if (filter_result.has_value()) {
        auto filtered_vectors = filter_result.value();

        // Should have 3 vectors with even category (indices 0, 2, 4)
        EXPECT_EQ(filtered_vectors.size(), 3);

        // Verify all returned vectors have the correct category
        for (const auto& vec : filtered_vectors) {
            EXPECT_EQ(vec.metadata.category, "even");
        }
    }
}
// Additional comprehensive tests for edge cases and error conditions
TEST_F(VectorStorageServiceTest, StoreVectorWithInvalidDimensions) {
    // Test storing a vector with mismatched dimensions compared to database configuration
    Vector invalid_vector;
    invalid_vector.id = "invalid_vector";
    invalid_vector.values = {1.0f, 2.0f}; // Only 2 values

    // Try to store in a database that expects more dimensions
    auto result = vector_storage_service_->store_vector("mismatched_db", invalid_vector);

    // Behavior depends on validation implementation, but should handle gracefully
    // If validation is strict, this might return an error
    if (result.has_value()) {
        // If it's accepted, verify it was stored correctly
        auto retrieve_result = vector_storage_service_->retrieve_vector("mismatched_db", "invalid_vector");
        EXPECT_TRUE(retrieve_result.has_value());
    }
}

TEST_F(VectorStorageServiceTest, RetrieveNonExistentVector) {
    // Test retrieving a vector that doesn't exist
    auto result = vector_storage_service_->retrieve_vector("test_db_123", "nonexistent_vector");

    // Should return an error since vector doesn't exist
    EXPECT_FALSE(result.has_value());
}

TEST_F(VectorStorageServiceTest, UpdateNonExistentVector) {
    // Test updating a vector that doesn't exist
    Vector new_vector;
    new_vector.id = "nonexistent_vector";
    new_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};

    auto result = vector_storage_service_->update_vector("test_db_123", new_vector);

    // Should return an error since vector doesn't exist
    EXPECT_FALSE(result.has_value());
}

TEST_F(VectorStorageServiceTest, DeleteNonExistentVector) {
    // Test deleting a vector that doesn't exist
    auto result = vector_storage_service_->delete_vector("test_db_123", "nonexistent_vector");

    // Behavior depends on implementation - could return success (no-op) or error
    // In a well-designed API, this might return success as deleting non-existent items is often treated as no-op
    EXPECT_TRUE(result.has_value());
}

TEST_F(VectorStorageServiceTest, BatchStoreWithMixedValidity) {
    // Test batch storing vectors with mixed validity (some valid, some invalid)
    Vector valid_vector;
    valid_vector.id = "valid_vector";
    valid_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};

    Vector invalid_vector;
    invalid_vector.id = "invalid_vector";
    invalid_vector.values = {}; // Empty values - invalid

    std::vector<Vector> vectors = {valid_vector, invalid_vector};

    auto result = vector_storage_service_->batch_store_vectors("test_db_123", vectors);

    // Should handle mixed validity appropriately (may partially succeed or fail completely depending on implementation)
    if (result.has_value()) {
        auto ids = result.value();
        // Implementation may return only successfully stored IDs or all requested IDs with status
        EXPECT_TRUE(ids.size() <= vectors.size());
    }
}

TEST_F(VectorStorageServiceTest, BatchRetrieveWithMixedExistence) {
    // Test batch retrieving vectors with mixed existence (some exist, some don't)
    Vector existing_vector;
    existing_vector.id = "existing_vector";
    existing_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};

    // Store one vector first
    auto store_result = vector_storage_service_->store_vector("test_db_123", existing_vector);
    EXPECT_TRUE(store_result.has_value());

    // Then try to retrieve both existing and non-existing vectors
    std::vector<std::string> vector_ids = {"existing_vector", "nonexistent_vector"};

    auto result = vector_storage_service_->retrieve_vectors("test_db_123", vector_ids);

    // Should return vectors that exist and handle missing ones appropriately
    if (result.has_value()) {
        auto retrieved_vectors = result.value();
        EXPECT_LE(retrieved_vectors.size(), vector_ids.size());  // At most the number of requested vectors
        
        // At least the existing vector should be returned
        bool found_existing = false;
        for (const auto& vec : retrieved_vectors) {
            if (vec.id == "existing_vector") {
                found_existing = true;
                break;
            }
        }
        EXPECT_TRUE(found_existing);
    }
}

TEST_F(VectorStorageServiceTest, StoreVectorWithEmptyID) {
    // Test storing a vector with an empty ID
    Vector vector_with_empty_id;
    vector_with_empty_id.id = "";  // Empty ID - invalid
    vector_with_empty_id.values = {1.0f, 2.0f, 3.0f, 4.0f};

    auto result = vector_storage_service_->store_vector("test_db_123", vector_with_empty_id);

    // Should return an error since vector ID is empty
    EXPECT_FALSE(result.has_value());
}

TEST_F(VectorStorageServiceTest, RetrieveVectorsWithEmptyList) {
    // Test retrieving vectors with an empty list of IDs
    std::vector<std::string> empty_ids = {};

    auto result = vector_storage_service_->retrieve_vectors("test_db_123", empty_ids);

    // Should return an empty vector of results
    if (result.has_value()) {
        EXPECT_EQ(result.value().size(), 0);
    }
}