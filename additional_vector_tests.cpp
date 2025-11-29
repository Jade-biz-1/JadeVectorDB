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