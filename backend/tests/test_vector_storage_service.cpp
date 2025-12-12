#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;
using ::testing::Eq;
using ::testing::ByRef;

// Mock class for DatabasePersistenceInterface to use in unit tests
class MockDatabasePersistence : public DatabasePersistenceInterface {
public:
    MOCK_METHOD(Result<std::string>, create_database, (const Database& db_config), (override));
    MOCK_METHOD(Result<Database>, get_database, (const std::string& database_id), (override));
    MOCK_METHOD(Result<std::vector<Database>>, list_databases, (), (override));
    MOCK_METHOD(Result<void>, update_database, (const std::string& database_id, const Database& new_config), (override));
    MOCK_METHOD(Result<void>, delete_database, (const std::string& database_id), (override));
    MOCK_METHOD(Result<void>, store_vector, (const std::string& database_id, const Vector& vector), (override));
    MOCK_METHOD(Result<Vector>, retrieve_vector, (const std::string& database_id, const std::string& vector_id), (override));
    MOCK_METHOD(Result<std::vector<Vector>>, retrieve_vectors, (const std::string& database_id, const std::vector<std::string>& vector_ids), (override));
    MOCK_METHOD(Result<void>, update_vector, (const std::string& database_id, const Vector& vector), (override));
    MOCK_METHOD(Result<void>, delete_vector, (const std::string& database_id, const std::string& vector_id), (override));
    MOCK_METHOD(Result<void>, batch_store_vectors, (const std::string& database_id, const std::vector<Vector>& vectors), (override));
    MOCK_METHOD(Result<void>, batch_delete_vectors, (const std::string& database_id, const std::vector<std::string>& vector_ids), (override));
    MOCK_METHOD(Result<void>, create_index, (const std::string& database_id, const Index& index), (override));
    MOCK_METHOD(Result<Index>, get_index, (const std::string& database_id, const std::string& index_id), (override));
    MOCK_METHOD(Result<std::vector<Index>>, list_indexes, (const std::string& database_id), (override));
    MOCK_METHOD(Result<void>, update_index, (const std::string& database_id, const std::string& index_id, const Index& index), (override));
    MOCK_METHOD(Result<void>, delete_index, (const std::string& database_id, const std::string& index_id), (override));
    MOCK_METHOD(Result<bool>, database_exists, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<bool>, vector_exists, (const std::string& database_id, const std::string& vector_id), (const, override));
    MOCK_METHOD(Result<bool>, index_exists, (const std::string& database_id, const std::string& index_id), (const, override));
    MOCK_METHOD(Result<size_t>, get_vector_count, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<std::vector<std::string>>, get_all_vector_ids, (const std::string& database_id), (const, override));
};

class VectorStorageServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto mock_persistence = std::make_unique<MockDatabasePersistence>();
        mock_db_persistence_ = mock_persistence.get(); // Keep raw pointer for expectations

        // Wrap mock in DatabaseLayer
        auto db_layer = std::make_unique<DatabaseLayer>(std::move(mock_persistence));

        // Pass DatabaseLayer to VectorStorageService
        vector_storage_service_ = std::make_unique<VectorStorageService>(std::move(db_layer));
    }

    void TearDown() override {
        vector_storage_service_.reset();
        // mock_db_persistence_ is owned by DatabaseLayer now, don't delete
    }

    MockDatabasePersistence* mock_db_persistence_; // Raw pointer for EXPECT_CALL
    std::unique_ptr<VectorStorageService> vector_storage_service_;
};

// Test the initialization of the VectorStorageService
// DISABLED: initialize() method not in DatabasePersistenceInterface
/*
TEST_F(VectorStorageServiceTest, InitializeService) {
    EXPECT_CALL(*mock_db_persistence_, initialize())
        .WillOnce(Return(Result<void>{}));

    auto result = vector_storage_service_->initialize();
    EXPECT_TRUE(result.has_value());
}
*/

// Test vector storage functionality
TEST_F(VectorStorageServiceTest, StoreVectorSuccess) {
    std::string db_id = "test_db";
    Vector test_vector;
    test_vector.id = "test_vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f};
    
    // Mock database existence check
    EXPECT_CALL(*mock_db_persistence_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    
    // Mock database retrieval to validate vector dimensions
    Database test_db;
    test_db.databaseId = db_id;
    test_db.vectorDimension = 3; // Should match vector size
    EXPECT_CALL(*mock_db_persistence_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    // Mock the actual storage operation
    EXPECT_CALL(*mock_db_persistence_, store_vector(db_id, test_vector))
        .WillOnce(Return(Result<void>{}));
    
    auto result = vector_storage_service_->store_vector(db_id, test_vector);
    EXPECT_TRUE(result.has_value());
}

TEST_F(VectorStorageServiceTest, StoreVectorFailureOnValidation) {
    std::string db_id = "test_db";
    Vector test_vector;
    test_vector.id = "test_vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f}; // 4 dimensions
    
    // Mock database existence check
    EXPECT_CALL(*mock_db_persistence_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    
    // Mock database retrieval with different dimensions (validation should fail)
    Database test_db;
    test_db.databaseId = db_id;
    test_db.vectorDimension = 3; // Only 3 dimensions expected
    EXPECT_CALL(*mock_db_persistence_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    auto result = vector_storage_service_->store_vector(db_id, test_vector);
    EXPECT_FALSE(result.has_value());
}

TEST_F(VectorStorageServiceTest, RetrieveVectorSuccess) {
    std::string db_id = "test_db";
    std::string vector_id = "test_vector_1";
    
    Vector expected_vector;
    expected_vector.id = vector_id;
    expected_vector.values = {1.0f, 2.0f, 3.0f};
    
    EXPECT_CALL(*mock_db_persistence_, retrieve_vector(db_id, vector_id))
        .WillOnce(Return(Result<Vector>{expected_vector}));
    
    auto result = vector_storage_service_->retrieve_vector(db_id, vector_id);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().id, vector_id);
    EXPECT_THAT(result.value().values, ::testing::ElementsAre(1.0f, 2.0f, 3.0f));
}

TEST_F(VectorStorageServiceTest, RetrieveVectorFailure) {
    std::string db_id = "test_db";
    std::string vector_id = "nonexistent_vector";
    
    EXPECT_CALL(*mock_db_persistence_, retrieve_vector(db_id, vector_id))
        .WillOnce(Return(Result<Vector>{})); // This will return an error
    
    auto result = vector_storage_service_->retrieve_vector(db_id, vector_id);
    EXPECT_FALSE(result.has_value());
}

// Test batch vector storage functionality
TEST_F(VectorStorageServiceTest, BatchStoreVectorsSuccess) {
    std::string db_id = "test_db";
    std::vector<Vector> test_vectors;
    
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 2.0f, 3.0f};
    test_vectors.push_back(v1);
    
    Vector v2;
    v2.id = "vector_2";
    v2.values = {4.0f, 5.0f, 6.0f};
    test_vectors.push_back(v2);
    
    // Mock database existence check
    EXPECT_CALL(*mock_db_persistence_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    
    // Mock database retrieval for validation
    Database test_db;
    test_db.databaseId = db_id;
    test_db.vectorDimension = 3;
    EXPECT_CALL(*mock_db_persistence_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    EXPECT_CALL(*mock_db_persistence_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    // Mock the batch storage operation
    EXPECT_CALL(*mock_db_persistence_, batch_store_vectors(db_id, test_vectors))
        .WillOnce(Return(Result<void>{}));
    
    auto result = vector_storage_service_->batch_store_vectors(db_id, test_vectors);
    EXPECT_TRUE(result.has_value());
}

// Test vector update functionality
TEST_F(VectorStorageServiceTest, UpdateVectorSuccess) {
    std::string db_id = "test_db";
    Vector test_vector;
    test_vector.id = "test_vector_1";
    test_vector.values = {1.0f, 2.0f, 3.0f};
    
    // Mock database existence check
    EXPECT_CALL(*mock_db_persistence_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    
    // Mock database retrieval for validation
    Database test_db;
    test_db.databaseId = db_id;
    test_db.vectorDimension = 3;
    EXPECT_CALL(*mock_db_persistence_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    // Mock the update operation
    EXPECT_CALL(*mock_db_persistence_, update_vector(db_id, test_vector))
        .WillOnce(Return(Result<void>{}));
    
    auto result = vector_storage_service_->update_vector(db_id, test_vector);
    EXPECT_TRUE(result.has_value());
}

// Test vector deletion functionality
TEST_F(VectorStorageServiceTest, DeleteVectorSuccess) {
    std::string db_id = "test_db";
    std::string vector_id = "vector_to_delete";
    
    EXPECT_CALL(*mock_db_persistence_, delete_vector(db_id, vector_id))
        .WillOnce(Return(Result<void>{}));
    
    auto result = vector_storage_service_->delete_vector(db_id, vector_id);
    EXPECT_TRUE(result.has_value());
}

// Test vector existence check
TEST_F(VectorStorageServiceTest, VectorExistsCheck) {
    std::string db_id = "test_db";
    std::string vector_id = "existing_vector";
    
    EXPECT_CALL(*mock_db_persistence_, vector_exists(db_id, vector_id))
        .WillOnce(Return(Result<bool>{true}));
    
    auto result = vector_storage_service_->vector_exists(db_id, vector_id);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test get vector count
TEST_F(VectorStorageServiceTest, GetVectorCount) {
    std::string db_id = "test_db";
    size_t expected_count = 42;
    
    EXPECT_CALL(*mock_db_persistence_, get_vector_count(db_id))
        .WillOnce(Return(Result<size_t>{expected_count}));
    
    auto result = vector_storage_service_->get_vector_count(db_id);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), expected_count);
}

// Test get all vector IDs
TEST_F(VectorStorageServiceTest, GetAllVectorIds) {
    std::string db_id = "test_db";
    std::vector<std::string> expected_ids = {"id1", "id2", "id3"};
    
    EXPECT_CALL(*mock_db_persistence_, get_all_vector_ids(db_id))
        .WillOnce(Return(Result<std::vector<std::string>>{expected_ids}));
    
    auto result = vector_storage_service_->get_all_vector_ids(db_id);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), expected_ids);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}