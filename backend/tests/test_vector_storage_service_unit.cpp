#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Mock class for DatabaseLayer to use in unit tests
class MockDatabaseLayer : public DatabaseLayer {
public:
    MOCK_METHOD(Result<void>, initialize, (), (override));
    MOCK_METHOD(Result<std::string>, create_database, (const Database& db_config), (override));
    MOCK_METHOD(Result<Database>, get_database, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<std::vector<Database>>, list_databases, (), (const, override));
    MOCK_METHOD(Result<void>, update_database, (const std::string& database_id, const Database& new_config), (override));
    MOCK_METHOD(Result<void>, delete_database, (const std::string& database_id), (override));
    MOCK_METHOD(Result<void>, store_vector, (const std::string& database_id, const Vector& vector), (override));
    MOCK_METHOD(Result<Vector>, retrieve_vector, (const std::string& database_id, const std::string& vector_id), (const, override));
    MOCK_METHOD(Result<std::vector<Vector>>, retrieve_vectors, (const std::string& database_id, const std::vector<std::string>& vector_ids), (const, override));
    MOCK_METHOD(Result<void>, update_vector, (const std::string& database_id, const Vector& vector), (override));
    MOCK_METHOD(Result<void>, delete_vector, (const std::string& database_id, const std::string& vector_id), (override));
    MOCK_METHOD(Result<void>, batch_store_vectors, (const std::string& database_id, const std::vector<Vector>& vectors), (override));
    MOCK_METHOD(Result<void>, batch_delete_vectors, (const std::string& database_id, const std::vector<std::string>& vector_ids), (override));
    MOCK_METHOD(Result<void>, create_index, (const std::string& database_id, const Index& index), (override));
    MOCK_METHOD(Result<Index>, get_index, (const std::string& database_id, const std::string& index_id), (const, override));
    MOCK_METHOD(Result<std::vector<Index>>, list_indexes, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<void>, update_index, (const std::string& database_id, const std::string& index_id, const Index& index), (override));
    MOCK_METHOD(Result<void>, delete_index, (const std::string& database_id, const std::string& index_id), (override));
    MOCK_METHOD(Result<bool>, database_exists, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<bool>, vector_exists, (const std::string& database_id, const std::string& vector_id), (const, override));
    MOCK_METHOD(Result<bool>, index_exists, (const std::string& database_id, const std::string& index_id), (const, override));
    MOCK_METHOD(Result<size_t>, get_vector_count, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<std::vector<std::string>>, get_all_vector_ids, (const std::string& database_id), (const, override));
};

// Test fixture for VectorStorageService
class VectorStorageServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock database layer
        mock_db_layer_ = std::make_unique<MockDatabaseLayer>();
        
        // Create vector storage service with mock database layer
        vector_storage_service_ = std::make_unique<VectorStorageService>(std::move(mock_db_layer_));
        
        // Initialize the service
        auto init_result = vector_storage_service_->initialize();
        ASSERT_TRUE(init_result.has_value());
    }
    
    void TearDown() override {
        // Clean up
        vector_storage_service_.reset();
    }
    
    std::unique_ptr<MockDatabaseLayer> mock_db_layer_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
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
    test_vector.id = "test_vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    test_vector.metadata["category"] = "finance";
    test_vector.metadata["score"] = 0.95f;
    
    std::string database_id = "test_db_123";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, database_exists(database_id))
        .WillOnce(Return(Result<bool>{true}));
    
    Database test_db;
    test_db.databaseId = database_id;
    test_db.vectorDimension = 4;
    EXPECT_CALL(*mock_db_layer_, get_database(database_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    EXPECT_CALL(*mock_db_layer_, store_vector(database_id, test_vector))
        .WillOnce(Return(Result<void>{}));
    
    // Test storing the vector
    auto result = vector_storage_service_->store_vector(database_id, test_vector);
    EXPECT_TRUE(result.has_value());
}

// Test retrieving a vector by ID
TEST_F(VectorStorageServiceTest, RetrieveVectorById) {
    // Create a test vector
    Vector test_vector;
    test_vector.id = "test_vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    test_vector.metadata["category"] = "finance";
    test_vector.metadata["score"] = 0.95f;
    
    std::string database_id = "test_db_123";
    std::string vector_id = "test_vector_1";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, retrieve_vector(database_id, vector_id))
        .WillOnce(Return(Result<Vector>{test_vector}));
    
    // Test retrieving the vector
    auto result = vector_storage_service_->retrieve_vector(database_id, vector_id);
    ASSERT_TRUE(result.has_value());
    
    Vector retrieved_vector = result.value();
    EXPECT_EQ(retrieved_vector.id, test_vector.id);
    EXPECT_EQ(retrieved_vector.values.size(), test_vector.values.size());
    
    for (size_t i = 0; i < test_vector.values.size(); ++i) {
        EXPECT_FLOAT_EQ(retrieved_vector.values[i], test_vector.values[i]);
    }
    
    EXPECT_EQ(retrieved_vector.metadata["category"].get<std::string>(), "finance");
    EXPECT_FLOAT_EQ(retrieved_vector.metadata["score"].get<float>(), 0.95f);
}

// Test updating a vector
TEST_F(VectorStorageServiceTest, UpdateVector) {
    // Create a test vector
    Vector test_vector;
    test_vector.id = "test_vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    test_vector.metadata["category"] = "finance";
    test_vector.metadata["score"] = 0.95f;
    
    std::string database_id = "test_db_123";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, database_exists(database_id))
        .WillOnce(Return(Result<bool>{true}));
    
    Database test_db;
    test_db.databaseId = database_id;
    test_db.vectorDimension = 4;
    EXPECT_CALL(*mock_db_layer_, get_database(database_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    EXPECT_CALL(*mock_db_layer_, update_vector(database_id, test_vector))
        .WillOnce(Return(Result<void>{}));
    
    // Test updating the vector
    auto result = vector_storage_service_->update_vector(database_id, test_vector);
    EXPECT_TRUE(result.has_value());
}

// Test deleting a vector
TEST_F(VectorStorageServiceTest, DeleteVector) {
    std::string database_id = "test_db_123";
    std::string vector_id = "test_vector_1";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, delete_vector(database_id, vector_id))
        .WillOnce(Return(Result<void>{}));
    
    // Test deleting the vector
    auto result = vector_storage_service_->delete_vector(database_id, vector_id);
    EXPECT_TRUE(result.has_value());
}

// Test batch storing vectors
TEST_F(VectorStorageServiceTest, BatchStoreVectors) {
    // Create test vectors
    std::vector<Vector> test_vectors;
    
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 2.0f, 3.0f, 4.0f};
    v1.metadata["category"] = "finance";
    v1.metadata["score"] = 0.95f;
    test_vectors.push_back(v1);
    
    Vector v2;
    v2.id = "vector_2";
    v2.values = {5.0f, 6.0f, 7.0f, 8.0f};
    v2.metadata["category"] = "technology";
    v2.metadata["score"] = 0.85f;
    test_vectors.push_back(v2);
    
    std::string database_id = "test_db_123";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, database_exists(database_id))
        .WillOnce(Return(Result<bool>{true}));
    
    Database test_db;
    test_db.databaseId = database_id;
    test_db.vectorDimension = 4;
    EXPECT_CALL(*mock_db_layer_, get_database(database_id))
        .Times(2)  // Called twice - once for each vector validation
        .WillRepeatedly(Return(Result<Database>{test_db}));
    
    EXPECT_CALL(*mock_db_layer_, batch_store_vectors(database_id, test_vectors))
        .WillOnce(Return(Result<void>{}));
    
    // Test batch storing vectors
    auto result = vector_storage_service_->batch_store_vectors(database_id, test_vectors);
    EXPECT_TRUE(result.has_value());
}

// Test batch deleting vectors
TEST_F(VectorStorageServiceTest, BatchDeleteVectors) {
    std::vector<std::string> vector_ids = {"vector_1", "vector_2", "vector_3"};
    std::string database_id = "test_db_123";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, batch_delete_vectors(database_id, vector_ids))
        .WillOnce(Return(Result<void>{}));
    
    // Test batch deleting vectors
    auto result = vector_storage_service_->batch_delete_vectors(database_id, vector_ids);
    EXPECT_TRUE(result.has_value());
}

// Test checking if a vector exists
TEST_F(VectorStorageServiceTest, VectorExists) {
    std::string database_id = "test_db_123";
    std::string vector_id = "test_vector_1";
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, vector_exists(database_id, vector_id))
        .WillOnce(Return(Result<bool>{true}));
    
    // Test checking if vector exists
    auto result = vector_storage_service_->vector_exists(database_id, vector_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test getting vector count
TEST_F(VectorStorageServiceTest, GetVectorCount) {
    std::string database_id = "test_db_123";
    size_t expected_count = 42;
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, get_vector_count(database_id))
        .WillOnce(Return(Result<size_t>{expected_count}));
    
    // Test getting vector count
    auto result = vector_storage_service_->get_vector_count(database_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), expected_count);
}

// Test getting all vector IDs
TEST_F(VectorStorageServiceTest, GetAllVectorIds) {
    std::string database_id = "test_db_123";
    std::vector<std::string> expected_ids = {"id1", "id2", "id3"};
    
    // Set up mock expectations
    EXPECT_CALL(*mock_db_layer_, get_all_vector_ids(database_id))
        .WillOnce(Return(Result<std::vector<std::string>>{expected_ids}));
    
    // Test getting all vector IDs
    auto result = vector_storage_service_->get_all_vector_ids(database_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), expected_ids);
}

// Test validating a vector
TEST_F(VectorStorageServiceTest, ValidateVector) {
    // Create a valid test vector
    Vector valid_vector;
    valid_vector.id = "valid_vector";
    valid_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    valid_vector.metadata["category"] = "finance";
    valid_vector.metadata["score"] = 0.95f;
    
    std::string database_id = "test_db_123";
    
    // Set up mock expectations for validation
    EXPECT_CALL(*mock_db_layer_, database_exists(database_id))
        .WillOnce(Return(Result<bool>{true}));
    
    Database test_db;
    test_db.databaseId = database_id;
    test_db.vectorDimension = 4;
    EXPECT_CALL(*mock_db_layer_, get_database(database_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    // Test validating a valid vector
    auto result = vector_storage_service_->validate_vector(database_id, valid_vector);
    EXPECT_TRUE(result.has_value());
    
    // Create an invalid test vector (wrong dimension)
    Vector invalid_vector;
    invalid_vector.id = "invalid_vector";
    invalid_vector.values = {1.0f, 2.0f, 3.0f}; // Only 3 dimensions, but database expects 4
    invalid_vector.metadata["category"] = "finance";
    invalid_vector.metadata["score"] = 0.95f;
    
    // Set up mock expectations for validation
    EXPECT_CALL(*mock_db_layer_, database_exists(database_id))
        .WillOnce(Return(Result<bool>{true}));
    
    EXPECT_CALL(*mock_db_layer_, get_database(database_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    // Test validating an invalid vector
    result = vector_storage_service_->validate_vector(database_id, invalid_vector);
    EXPECT_FALSE(result.has_value());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}