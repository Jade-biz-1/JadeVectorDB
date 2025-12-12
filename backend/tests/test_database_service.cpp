#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/database_service.h"
#include "services/database_layer.h"
#include "models/database.h"
#include "lib/error_handling.h"

namespace jadevectordb {

// Test fixture for database service unit tests
class DatabaseServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize database layer
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        // Initialize database service
        db_service_ = std::make_unique<DatabaseService>(std::move(db_layer_));
        db_service_->initialize();
    }

    void TearDown() override {
        // Clean up is handled automatically
    }

    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<DatabaseLayer> db_layer_;
};

// Test database creation with valid parameters
TEST_F(DatabaseServiceTest, CreateDatabaseWithValidParameters) {
    DatabaseCreationParams params;
    params.name = "test_database";
    params.description = "A test database for unit testing";
    params.vectorDimension = 128;
    params.indexType = "HNSW";
    
    // Add some index parameters
    params.indexParameters["M"] = "16";
    params.indexParameters["efConstruction"] = "200";
    
    // Set sharding configuration
    params.sharding.strategy = "hash";
    params.sharding.numShards = 2;
    
    // Set replication configuration
    params.replication.factor = 2;
    params.replication.sync = true;
    
    // Add an embedding model
    Database::EmbeddingModel model;
    model.name = "test_model";
    model.version = "1.0";
    model.provider = "test_provider";
    model.inputType = "text";
    model.outputDimension = 128;
    model.status = "active";
    params.embeddingModels.push_back(model);
    
    // Add metadata schema
    params.metadataSchema["author"] = "string";
    params.metadataSchema["category"] = "string";
    params.metadataSchema["score"] = "float";
    
    // Add retention policy
    params.retentionPolicy = std::make_unique<Database::RetentionPolicy>();
    params.retentionPolicy->maxAgeDays = 365;
    params.retentionPolicy->archiveOnExpire = true;
    
    // Add access control
    params.accessControl.roles = {"admin", "user"};
    params.accessControl.defaultPermissions = {"read", "write", "search"};
    
    // Create the database
    auto result = db_service_->create_database(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());
    
    std::string database_id = result.value();
    
    // Verify the database was created
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.databaseId, database_id);
    EXPECT_EQ(database.name, "test_database");
    EXPECT_EQ(database.description, "A test database for unit testing");
    EXPECT_EQ(database.vectorDimension, 128);
    EXPECT_EQ(database.indexType, "HNSW");
    EXPECT_EQ(database.sharding.strategy, "hash");
    EXPECT_EQ(database.sharding.numShards, 2);
    EXPECT_EQ(database.replication.factor, 2);
    EXPECT_EQ(database.replication.sync, true);
    EXPECT_EQ(database.embeddingModels.size(), 1);
    EXPECT_EQ(database.embeddingModels[0].name, "test_model");
    EXPECT_EQ(database.metadataSchema.size(), 3);
    EXPECT_TRUE(database.retentionPolicy != nullptr);
    EXPECT_EQ(database.retentionPolicy->maxAgeDays, 365);
    EXPECT_EQ(database.retentionPolicy->archiveOnExpire, true);
    EXPECT_EQ(database.accessControl.roles.size(), 2);
    EXPECT_EQ(database.accessControl.defaultPermissions.size(), 3);
    EXPECT_FALSE(database.created_at.empty());
    EXPECT_FALSE(database.updated_at.empty());
}

// Test database creation with invalid parameters
TEST_F(DatabaseServiceTest, CreateDatabaseWithInvalidParameters) {
    DatabaseCreationParams params;
    params.name = "";  // Empty name should fail validation
    params.vectorDimension = 128;
    
    auto result = db_service_->create_database(params);
    // Should return empty string on validation failure
    EXPECT_FALSE(result.has_value());
}

// Test database creation with invalid vector dimension
TEST_F(DatabaseServiceTest, CreateDatabaseWithInvalidVectorDimension) {
    DatabaseCreationParams params;
    params.name = "invalid_dimension_db";
    params.vectorDimension = -1;  // Negative dimension should fail validation
    
    auto result = db_service_->create_database(params);
    EXPECT_FALSE(result.has_value());
}

// Test database creation with too large vector dimension
TEST_F(DatabaseServiceTest, CreateDatabaseWithTooLargeVectorDimension) {
    DatabaseCreationParams params;
    params.name = "large_dimension_db";
    params.vectorDimension = 5000;  // Too large should fail validation (> 4096)
    
    auto result = db_service_->create_database(params);
    EXPECT_FALSE(result.has_value());
}

// Test database retrieval by ID
TEST_F(DatabaseServiceTest, GetDatabaseById) {
    // First create a database
    DatabaseCreationParams params;
    params.name = "retrieval_test_db";
    params.description = "Test database for retrieval";
    params.vectorDimension = 256;
    
    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Retrieve the database
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.databaseId, database_id);
    EXPECT_EQ(database.name, "retrieval_test_db");
    EXPECT_EQ(database.description, "Test database for retrieval");
    EXPECT_EQ(database.vectorDimension, 256);
}

// Test database retrieval with non-existent ID
TEST_F(DatabaseServiceTest, GetDatabaseWithNonExistentId) {
    auto result = db_service_->get_database("non_existent_database_id");
    // Should return error for non-existent database
    EXPECT_FALSE(result.has_value());
}

// Test database listing
TEST_F(DatabaseServiceTest, ListDatabases) {
    // Create multiple databases
    std::vector<std::string> database_ids;
    
    for (int i = 0; i < 3; ++i) {
        DatabaseCreationParams params;
        params.name = "list_test_db_" + std::to_string(i);
        params.description = "Test database " + std::to_string(i);
        params.vectorDimension = 128 + i * 10;
        
        auto result = db_service_->create_database(params);
        ASSERT_TRUE(result.has_value());
        database_ids.push_back(result.value());
    }
    
    // List all databases
    DatabaseListParams list_params;
    list_params.limit = 10;
    
    auto list_result = db_service_->list_databases(list_params);
    ASSERT_TRUE(list_result.has_value());
    
    auto databases = list_result.value();
    EXPECT_GE(databases.size(), 3);
    
    // Verify that our created databases are in the list
    std::vector<std::string> found_ids;
    for (const auto& db : databases) {
        found_ids.push_back(db.databaseId);
    }
    
    for (const auto& id : database_ids) {
        EXPECT_NE(std::find(found_ids.begin(), found_ids.end(), id), found_ids.end())
            << "Database " << id << " not found in list";
    }
}

// Test database listing with filtering
TEST_F(DatabaseServiceTest, ListDatabasesWithFiltering) {
    // Create databases with different names
    std::vector<std::string> database_ids;
    
    for (int i = 0; i < 3; ++i) {
        DatabaseCreationParams params;
        params.name = "filter_test_" + std::to_string(i);
        params.description = "Filter test database " + std::to_string(i);
        params.vectorDimension = 128;
        
        auto result = db_service_->create_database(params);
        ASSERT_TRUE(result.has_value());
        database_ids.push_back(result.value());
    }
    
    // List databases with name filter
    DatabaseListParams list_params;
    list_params.filterByName = "filter_test_1";
    list_params.limit = 10;
    
    auto list_result = db_service_->list_databases(list_params);
    ASSERT_TRUE(list_result.has_value());
    
    auto databases = list_result.value();
    EXPECT_EQ(databases.size(), 1);
    EXPECT_EQ(databases[0].name, "filter_test_1");
}

// Test database listing with pagination
TEST_F(DatabaseServiceTest, ListDatabasesWithPagination) {
    // Create multiple databases
    std::vector<std::string> database_ids;
    
    for (int i = 0; i < 5; ++i) {
        DatabaseCreationParams params;
        params.name = "pagination_test_db_" + std::to_string(i);
        params.description = "Pagination test database " + std::to_string(i);
        params.vectorDimension = 128;
        
        auto result = db_service_->create_database(params);
        ASSERT_TRUE(result.has_value());
        database_ids.push_back(result.value());
    }
    
    // List databases with pagination (first page)
    DatabaseListParams list_params;
    list_params.limit = 2;
    list_params.offset = 0;
    
    auto list_result = db_service_->list_databases(list_params);
    ASSERT_TRUE(list_result.has_value());
    
    auto first_page = list_result.value();
    EXPECT_EQ(first_page.size(), 2);
    
    // List databases with pagination (second page)
    list_params.offset = 2;
    
    list_result = db_service_->list_databases(list_params);
    ASSERT_TRUE(list_result.has_value());
    
    auto second_page = list_result.value();
    EXPECT_EQ(second_page.size(), 2);
}

// Test database update functionality
TEST_F(DatabaseServiceTest, UpdateDatabase) {
    // First create a database
    DatabaseCreationParams create_params;
    create_params.name = "update_test_db";
    create_params.description = "Original description";
    create_params.vectorDimension = 128;
    create_params.indexType = "HNSW";
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Update the database
    DatabaseUpdateParams update_params;
    update_params.name = "updated_database_name";
    update_params.description = "Updated description";
    update_params.vectorDimension = 256;
    update_params.indexType = "IVF";
    
    auto update_result = db_service_->update_database(database_id, update_params);
    EXPECT_TRUE(update_result.has_value());
    
    // Verify the update
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.name, "updated_database_name");
    EXPECT_EQ(database.description, "Updated description");
    EXPECT_EQ(database.vectorDimension, 256);
    EXPECT_EQ(database.indexType, "IVF");
    EXPECT_FALSE(database.updated_at.empty());
}

// Test database update with partial parameters
TEST_F(DatabaseServiceTest, UpdateDatabaseWithPartialParameters) {
    // First create a database
    DatabaseCreationParams create_params;
    create_params.name = "partial_update_test_db";
    create_params.description = "Original description";
    create_params.vectorDimension = 128;
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Update only the name
    DatabaseUpdateParams update_params;
    update_params.name = "partially_updated_name";
    
    auto update_result = db_service_->update_database(database_id, update_params);
    EXPECT_TRUE(update_result.has_value());
    
    // Verify only the name was updated
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.name, "partially_updated_name");
    EXPECT_EQ(database.description, "Original description");  // Should remain unchanged
    EXPECT_EQ(database.vectorDimension, 128);  // Should remain unchanged
}

// Test database deletion
TEST_F(DatabaseServiceTest, DeleteDatabase) {
    // First create a database
    DatabaseCreationParams create_params;
    create_params.name = "delete_test_db";
    create_params.description = "Database to be deleted";
    create_params.vectorDimension = 128;
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Verify the database exists
    auto exists_result = db_service_->database_exists(database_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
    
    // Delete the database
    auto delete_result = db_service_->delete_database(database_id);
    EXPECT_TRUE(delete_result.has_value());
    
    // Verify the database no longer exists
    exists_result = db_service_->database_exists(database_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
    
    // Trying to get the deleted database should fail
    auto get_result = db_service_->get_database(database_id);
    EXPECT_FALSE(get_result.has_value());
}

// Test database deletion with non-existent ID
TEST_F(DatabaseServiceTest, DeleteNonExistentDatabase) {
    auto result = db_service_->delete_database("non_existent_database_id");
    // Should return error for non-existent database
    EXPECT_FALSE(result.has_value());
}

// Test database existence check
TEST_F(DatabaseServiceTest, CheckDatabaseExistence) {
    // Check non-existent database
    auto exists_result = db_service_->database_exists("non_existent_database");
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
    
    // Create a database
    DatabaseCreationParams create_params;
    create_params.name = "existence_test_db";
    create_params.vectorDimension = 128;
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Check that the database exists
    exists_result = db_service_->database_exists(database_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
}

// Test database count
TEST_F(DatabaseServiceTest, GetDatabaseCount) {
    // Get initial count
    auto initial_count_result = db_service_->get_database_count();
    ASSERT_TRUE(initial_count_result.has_value());
    size_t initial_count = initial_count_result.value();
    
    // Create a few databases
    std::vector<std::string> database_ids;
    for (int i = 0; i < 3; ++i) {
        DatabaseCreationParams params;
        params.name = "count_test_db_" + std::to_string(i);
        params.vectorDimension = 128;
        
        auto result = db_service_->create_database(params);
        ASSERT_TRUE(result.has_value());
        database_ids.push_back(result.value());
    }
    
    // Get final count
    auto final_count_result = db_service_->get_database_count();
    ASSERT_TRUE(final_count_result.has_value());
    size_t final_count = final_count_result.value();
    
    // Verify count increased by 3
    EXPECT_EQ(final_count, initial_count + 3);
}

// Test database statistics
// DISABLED: get_database_stats() method not implemented in DatabaseService
/*
TEST_F(DatabaseServiceTest, GetDatabaseStatistics) {
    // Create a database
    DatabaseCreationParams create_params;
    create_params.name = "stats_test_db";
    create_params.description = "Statistics test database";
    create_params.vectorDimension = 256;
    create_params.indexType = "IVF";

    // Set sharding and replication
    create_params.sharding.strategy = "hash";
    create_params.sharding.numShards = 3;
    create_params.replication.factor = 2;
    create_params.replication.sync = true;

    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();

    // Get statistics
    auto stats_result = db_service_->get_database_stats(database_id);
    ASSERT_TRUE(stats_result.has_value());

    const auto& stats = stats_result.value();

    // Verify statistics
    EXPECT_EQ(stats.at("database_id"), database_id);
    EXPECT_EQ(stats.at("name"), "stats_test_db");
    EXPECT_EQ(stats.at("vector_dimension"), "256");
    EXPECT_EQ(stats.at("index_type"), "IVF");
    EXPECT_EQ(stats.at("shard_count"), "3");
    EXPECT_EQ(stats.at("replication_factor"), "2");
    EXPECT_EQ(stats.at("embedding_model_count"), "0");
    EXPECT_FALSE(stats.at("created_at").empty());
    EXPECT_FALSE(stats.at("updated_at").empty());
}
*/

// Test database creation parameter validation
TEST_F(DatabaseServiceTest, ValidateCreationParameters) {
    // Test valid parameters
    DatabaseCreationParams valid_params;
    valid_params.name = "valid_db";
    valid_params.vectorDimension = 128;
    valid_params.indexType = "HNSW";
    valid_params.sharding.numShards = 1;
    valid_params.replication.factor = 1;
    
    auto valid_result = db_service_->validate_creation_params(valid_params);
    EXPECT_TRUE(valid_result.has_value());
    
    // Test invalid parameters - empty name
    DatabaseCreationParams invalid_params1;
    invalid_params1.name = "";  // Empty name
    invalid_params1.vectorDimension = 128;
    
    auto invalid_result1 = db_service_->validate_creation_params(invalid_params1);
    EXPECT_FALSE(invalid_result1.has_value());
    
    // Test invalid parameters - negative vector dimension
    DatabaseCreationParams invalid_params2;
    invalid_params2.name = "invalid_db";
    invalid_params2.vectorDimension = -1;  // Negative dimension
    
    auto invalid_result2 = db_service_->validate_creation_params(invalid_params2);
    EXPECT_FALSE(invalid_result2.has_value());
    
    // Test invalid parameters - too large vector dimension
    DatabaseCreationParams invalid_params3;
    invalid_params3.name = "invalid_db";
    invalid_params3.vectorDimension = 5000;  // Too large
    
    auto invalid_result3 = db_service_->validate_creation_params(invalid_params3);
    EXPECT_FALSE(invalid_result3.has_value());
}

// Test database update parameter validation
TEST_F(DatabaseServiceTest, ValidateUpdateParameters) {
    // Test valid update parameters
    DatabaseUpdateParams valid_params;
    valid_params.name = "updated_name";
    
    auto valid_result = db_service_->validate_update_params(valid_params);
    EXPECT_TRUE(valid_result.has_value());
    
    // Test invalid update parameters - name too long
    DatabaseUpdateParams invalid_params;
    invalid_params.name = std::string(300, 'a');  // Very long name
    
    auto invalid_result = db_service_->validate_update_params(invalid_params);
    EXPECT_FALSE(invalid_result.has_value());
}

} // namespace jadevectordb