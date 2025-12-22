#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/vector_storage.h"
#include "services/database_service.h"
#include "services/similarity_search.h"
#include "services/database_layer.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"


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
using ::testing::Eq;
using ::testing::ByRef;
using ::testing::Invoke;

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
    MOCK_METHOD(Result<size_t>, get_database_count, (), (const, override));
    MOCK_METHOD(Result<size_t>, get_vector_count, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<size_t>, get_index_count, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<std::vector<std::string>>, get_all_vector_ids, (const std::string& database_id), (const, override));
};

class CoreServicesComprehensiveTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_db_layer_ = std::make_unique<MockDatabaseLayer>();
        
        // Initialize services for testing
        vector_storage_service_ = std::make_unique<VectorStorageService>(std::move(mock_db_layer_));
        database_service_ = std::make_unique<DatabaseService>();
        similarity_search_service_ = std::make_unique<SimilaritySearchService>();
    }
    
    void TearDown() override {
        similarity_search_service_.reset();
        database_service_.reset();
        vector_storage_service_.reset();
    }
    
    std::unique_ptr<MockDatabaseLayer> mock_db_layer_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    std::unique_ptr<DatabaseService> database_service_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
};

// Comprehensive test for database service
TEST_F(CoreServicesComprehensiveTest, DatabaseServiceOperationsTest) {
    // Test database creation
    Database new_db;
    new_db.name = "test_database";
    new_db.vectorDimension = 128;
    new_db.indexType = "HNSW";
    
    EXPECT_CALL(*mock_db_layer_, create_database(_))
        .WillOnce(Return(Result<std::string>{"test_db_id"}));
    
    auto create_result = database_service_->create_database(to_creation_params(new_db));
    EXPECT_TRUE(create_result.has_value());
    EXPECT_EQ(create_result.value(), "test_db_id");
    
    // Test database retrieval
    Database retrieved_db;
    retrieved_db.databaseId = "test_db_id";
    retrieved_db.name = "test_database";
    retrieved_db.vectorDimension = 128;
    
    EXPECT_CALL(*mock_db_layer_, get_database("test_db_id"))
        .WillOnce(Return(Result<Database>{retrieved_db}));
    
    auto get_result = database_service_->get_database("test_db_id");
    EXPECT_TRUE(get_result.has_value());
    EXPECT_EQ(get_result.value().name, "test_database");
}

// Comprehensive test for vector storage with edge cases
TEST_F(CoreServicesComprehensiveTest, VectorStorageEdgeCasesTest) {
    // Test with empty vector - should fail validation
    std::string db_id = "test_db";
    Vector empty_vector;
    empty_vector.id = "empty_vector";
    empty_vector.values = {}; // Empty vector should cause validation to fail
    
    // Mock database for validation
    Database test_db;
    test_db.databaseId = db_id;
    test_db.vectorDimension = 3;
    EXPECT_CALL(*mock_db_layer_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    EXPECT_CALL(*mock_db_layer_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    auto result = vector_storage_service_->store_vector(db_id, empty_vector);
    EXPECT_FALSE(result.has_value());
    
    // Test with vector of wrong dimension - should fail validation
    Vector wrong_dim_vector;
    wrong_dim_vector.id = "wrong_dim_vector";
    wrong_dim_vector.values = {1.0f, 2.0f}; // Only 2 dimensions, but db expects 3
    
    EXPECT_CALL(*mock_db_layer_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    EXPECT_CALL(*mock_db_layer_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    
    result = vector_storage_service_->store_vector(db_id, wrong_dim_vector);
    EXPECT_FALSE(result.has_value());
    
    // Test with valid vector - should succeed
    Vector valid_vector;
    valid_vector.id = "valid_vector";
    valid_vector.values = {1.0f, 2.0f, 3.0f};
    
    EXPECT_CALL(*mock_db_layer_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    EXPECT_CALL(*mock_db_layer_, get_database(db_id))
        .WillOnce(Return(Result<Database>{test_db}));
    EXPECT_CALL(*mock_db_layer_, store_vector(db_id, valid_vector))
        .WillOnce(Return(Result<void>{}));
    
    result = vector_storage_service_->store_vector(db_id, valid_vector);
    EXPECT_TRUE(result.has_value());
}

// Comprehensive test for similarity search with various parameters
TEST_F(CoreServicesComprehensiveTest, SimilaritySearchComprehensiveTest) {
    // Initialize the service first
    similarity_search_service_->initialize();
    
    // Test search with empty query vector
    std::string db_id = "test_db";
    Vector empty_query_vector;
    empty_query_vector.values = {};
    
    SearchParams empty_params;
    empty_params.top_k = 5;
    
    auto empty_result = similarity_search_service_->similarity_search(db_id, empty_query_vector, empty_params);
    // This should fail gracefully
    EXPECT_FALSE(empty_result.has_value());
    
    // Test with valid query vector but invalid parameters
    Vector valid_query_vector;
    valid_query_vector.values = {1.0f, 2.0f, 3.0f};
    
    SearchParams negative_k_params;
    negative_k_params.top_k = -1; // Invalid
    
    auto negative_result = similarity_search_service_->similarity_search(db_id, valid_query_vector, negative_params);
    EXPECT_FALSE(negative_result.has_value());
    
    // Test with valid parameters
    SearchParams valid_params;
    valid_params.top_k = 5;
    valid_params.threshold = 0.5;
    
    auto validation_result = similarity_search_service_->validate_search_params(valid_params);
    EXPECT_TRUE(validation_result.has_value());
}

// Test interaction between services
TEST_F(CoreServicesComprehensiveTest, ServiceInteractionTest) {
    // Create a database first
    Database db_config;
    db_config.name = "integration_test_db";
    db_config.vectorDimension = 4;
    db_config.indexType = "FLAT";
    
    EXPECT_CALL(*mock_db_layer_, create_database(_))
        .WillOnce(Return(Result<std::string>{"integration_test_db_id"}));
    
    auto db_result = database_service_->create_database(to_creation_params(db_config));
    ASSERT_TRUE(db_result.has_value());
    std::string db_id = db_result.value();
    
    // Store a vector in the created database
    Vector test_vector;
    test_vector.id = "integration_test_vector";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    EXPECT_CALL(*mock_db_layer_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    EXPECT_CALL(*mock_db_layer_, get_database(db_id))
        .WillOnce(Return(Result<Database>{db_config})); // Use original config for validation
    EXPECT_CALL(*mock_db_layer_, store_vector(db_id, test_vector))
        .WillOnce(Return(Result<void>{}));
    
    auto store_result = vector_storage_service_->store_vector(db_id, test_vector);
    EXPECT_TRUE(store_result.has_value());
    
    // Retrieve the stored vector
    EXPECT_CALL(*mock_db_layer_, retrieve_vector(db_id, "integration_test_vector"))
        .WillOnce(Return(Result<Vector>{test_vector}));
    
    auto retrieve_result = vector_storage_service_->retrieve_vector(db_id, "integration_test_vector");
    EXPECT_TRUE(retrieve_result.has_value());
    EXPECT_EQ(retrieve_result.value().id, "integration_test_vector");
    EXPECT_THAT(retrieve_result.value().values, ::testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
    
    // Perform a similarity search using the retrieved vector as query
    SearchParams search_params;
    search_params.top_k = 10;
    search_params.threshold = 0.0;
    
    auto validation_result = similarity_search_service_->validate_search_params(search_params);
    EXPECT_TRUE(validation_result.has_value());
    
    // Initialize similarity search service with mock data to return results
    // Note: For this test, we're checking validation and parameters rather than the full search
    // since the implementation details might vary
}

// Test error handling in core services
TEST_F(CoreServicesComprehensiveTest, ErrorHandlingTest) {
    std::string db_id = "error_test_db";
    std::string vector_id = "error_test_vector";
    
    // Simulate database not existing
    EXPECT_CALL(*mock_db_layer_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{false}));
    
    Vector test_vector;
    test_vector.id = vector_id;
    test_vector.values = {1.0f, 2.0f, 3.0f};
    
    auto result = vector_storage_service_->store_vector(db_id, test_vector);
    EXPECT_FALSE(result.has_value());
    
    // Simulate vector retrieval failure
    EXPECT_CALL(*mock_db_layer_, retrieve_vector(db_id, vector_id))
        .WillOnce(Return(Result<Vector>{})); // Simulate error return
    
    auto retrieve_result = vector_storage_service_->retrieve_vector(db_id, vector_id);
    EXPECT_FALSE(retrieve_result.has_value());
    
    // Test parameter validation failures
    SearchParams invalid_params;
    invalid_params.top_k = 0; // Invalid: should be > 0
    
    auto validation_result = similarity_search_service_->validate_search_params(invalid_params);
    EXPECT_FALSE(validation_result.has_value());
}

// Performance test for batch operations
TEST_F(CoreServicesComprehensiveTest, BatchOperationsPerformanceTest) {
    std::string db_id = "batch_test_db";
    
    // Create a batch of vectors
    std::vector<Vector> batch_vectors;
    for (int i = 0; i < 100; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.values = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)};
        batch_vectors.push_back(v);
    }
    
    // Mock database validation for all vectors
    Database test_db;
    test_db.databaseId = db_id;
    test_db.vectorDimension = 3;
    
    EXPECT_CALL(*mock_db_layer_, database_exists(db_id))
        .WillOnce(Return(Result<bool>{true}));
    
    // Expect validation check for each vector
    EXPECT_CALL(*mock_db_layer_, get_database(db_id))
        .Times(batch_vectors.size())
        .WillRepeatedly(Return(Result<Database>{test_db}));
    
    // Mock batch storage
    EXPECT_CALL(*mock_db_layer_, batch_store_vectors(db_id, batch_vectors))
        .WillOnce(Return(Result<void>{}));
    
    auto result = vector_storage_service_->batch_store_vectors(db_id, batch_vectors);
    EXPECT_TRUE(result.has_value());
}
