#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>

#include "api/rest/rest_api.h"
#include "services/database_service.h"
#include "services/database_layer.h"
#include "lib/auth.h"
#include "models/database.h"

namespace jadevectordb {

// Test fixture for database API integration tests
class DatabaseApiIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize database layer
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        // Initialize database service
        db_service_ = std::make_unique<DatabaseService>(std::move(db_layer_));
        db_service_->initialize();
        
        // Initialize REST API service
        rest_api_ = std::make_unique<RestApiService>(8081); // Use different port for testing
        
        // Note: In a real integration test, we would actually start the server
        // and make HTTP requests to test the endpoints. For now, we'll test
        // the service layer integration directly.
    }

    void TearDown() override {
        // Clean up
        if (rest_api_ && rest_api_->is_running()) {
            rest_api_->stop();
        }
    }

    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<RestApiService> rest_api_;
};

// Test database creation API endpoint
TEST_F(DatabaseApiIntegrationTest, CreateDatabaseEndpoint) {
    // This test would normally make an HTTP POST request to /v1/databases
    // with a JSON payload containing database creation parameters.
    // Since we're not running a real HTTP server in this test, we'll test
    // the service layer directly.
    
    DatabaseCreationParams params;
    params.name = "integration_test_db";
    params.description = "Database for integration testing";
    params.vectorDimension = 128;
    params.indexType = "HNSW";
    params.indexParameters = {{"M", "16"}, {"efConstruction", "200"}};
    params.sharding = {"hash", 1};
    params.replication = {1, true};
    
    // Test the service directly
    auto result = db_service_->create_database(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());
    
    std::string database_id = result.value();
    
    // Verify the database was created
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.databaseId, database_id);
    EXPECT_EQ(database.name, "integration_test_db");
    EXPECT_EQ(database.description, "Database for integration testing");
    EXPECT_EQ(database.vectorDimension, 128);
    EXPECT_EQ(database.indexType, "HNSW");
    EXPECT_EQ(database.indexParameters.size(), 2);
    EXPECT_EQ(database.sharding.strategy, "hash");
    EXPECT_EQ(database.sharding.numShards, 1);
    EXPECT_EQ(database.replication.factor, 1);
    EXPECT_EQ(database.replication.sync, true);
    EXPECT_FALSE(database.created_at.empty());
    EXPECT_FALSE(database.updated_at.empty());
}

// Test database listing API endpoint
TEST_F(DatabaseApiIntegrationTest, ListDatabasesEndpoint) {
    // Create multiple databases for testing
    std::vector<std::string> database_ids;
    
    for (int i = 0; i < 3; ++i) {
        DatabaseCreationParams params;
        params.name = "list_test_db_" + std::to_string(i);
        params.description = "List test database " + std::to_string(i);
        params.vectorDimension = 128 + i * 10;
        
        auto result = db_service_->create_database(params);
        ASSERT_TRUE(result.has_value());
        database_ids.push_back(result.value());
    }
    
    // Test listing databases
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

// Test database retrieval API endpoint
TEST_F(DatabaseApiIntegrationTest, GetDatabaseEndpoint) {
    // Create a database
    DatabaseCreationParams create_params;
    create_params.name = "get_test_db";
    create_params.description = "Get test database";
    create_params.vectorDimension = 256;
    create_params.indexType = "IVF";
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Test retrieving the database
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.databaseId, database_id);
    EXPECT_EQ(database.name, "get_test_db");
    EXPECT_EQ(database.description, "Get test database");
    EXPECT_EQ(database.vectorDimension, 256);
    EXPECT_EQ(database.indexType, "IVF");
}

// Test database update API endpoint
TEST_F(DatabaseApiIntegrationTest, UpdateDatabaseEndpoint) {
    // Create a database
    DatabaseCreationParams create_params;
    create_params.name = "update_test_db";
    create_params.description = "Original description";
    create_params.vectorDimension = 128;
    create_params.indexType = "HNSW";
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Test updating the database
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
    // Updated timestamp should be different from created timestamp
    EXPECT_NE(database.created_at, database.updated_at);
}

// Test database deletion API endpoint
TEST_F(DatabaseApiIntegrationTest, DeleteDatabaseEndpoint) {
    // Create a database
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
    
    // Test deleting the database
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

// Test database API with authentication
TEST_F(DatabaseApiIntegrationTest, DatabaseApiWithAuthentication) {
    // Create a test user and API key
    auto auth_manager = AuthManager::get_instance();
    
    // Create a user
    auto user_result = auth_manager->create_user("test_user", "test@example.com", {"user"});
    ASSERT_TRUE(user_result.has_value());
    std::string user_id = user_result.value();
    
    // Generate an API key for the user with database permissions
    std::vector<std::string> permissions = {
        "database:create", 
        "database:list", 
        "database:read", 
        "database:update", 
        "database:delete"
    };
    
    auto key_result = auth_manager->generate_api_key(user_id, permissions, "Test API key");
    ASSERT_TRUE(key_result.has_value());
    std::string api_key = key_result.value();
    
    // Test authenticating with the API key
    auto auth_result = auth_manager->validate_api_key(api_key);
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_TRUE(auth_result.value());
    
    // Test getting user from API key
    auto user_from_key_result = auth_manager->get_user_from_api_key(api_key);
    ASSERT_TRUE(user_from_key_result.has_value());
    EXPECT_EQ(user_from_key_result.value(), user_id);
    
    // Test permission checking
    auto perm_result = auth_manager->has_permission_with_api_key(api_key, "database:create");
    ASSERT_TRUE(perm_result.has_value());
    EXPECT_TRUE(perm_result.value());
    
    // Test invalid API key
    auto invalid_auth_result = auth_manager->validate_api_key("invalid_key");
    ASSERT_FALSE(invalid_auth_result.has_value());
    
    // Test permission checking with invalid API key
    auto invalid_perm_result = auth_manager->has_permission_with_api_key("invalid_key", "database:create");
    EXPECT_FALSE(invalid_perm_result.has_value());
}

// Test database API parameter validation
TEST_F(DatabaseApiIntegrationTest, DatabaseApiParameterValidation) {
    // Test creating database with invalid parameters
    DatabaseCreationParams invalid_params;
    invalid_params.name = "";  // Empty name should fail
    invalid_params.vectorDimension = 128;
    
    auto result = db_service_->create_database(invalid_params);
    EXPECT_FALSE(result.has_value());
    
    // Test creating database with negative vector dimension
    DatabaseCreationParams invalid_params2;
    invalid_params2.name = "test_db";
    invalid_params2.vectorDimension = -1;  // Negative dimension should fail
    
    auto result2 = db_service_->create_database(invalid_params2);
    EXPECT_FALSE(result2.has_value());
    
    // Test creating database with too large vector dimension
    DatabaseCreationParams invalid_params3;
    invalid_params3.name = "test_db";
    invalid_params3.vectorDimension = 5000;  // Too large should fail (> 4096)
    
    auto result3 = db_service_->create_database(invalid_params3);
    EXPECT_FALSE(result3.has_value());
    
    // Test updating database with invalid parameters
    DatabaseUpdateParams invalid_update_params;
    invalid_update_params.name = std::string(300, 'a');  // Very long name should fail
    
    auto update_result = db_service_->validate_update_params(invalid_update_params);
    EXPECT_FALSE(update_result.has_value());
}

// Test database API error handling
TEST_F(DatabaseApiIntegrationTest, DatabaseApiErrorHandling) {
    // Test getting non-existent database
    auto get_result = db_service_->get_database("non_existent_database");
    EXPECT_FALSE(get_result.has_value());
    
    // Test updating non-existent database
    DatabaseUpdateParams update_params;
    update_params.name = "updated_name";
    
    auto update_result = db_service_->update_database("non_existent_database", update_params);
    EXPECT_FALSE(update_result.has_value());
    
    // Test deleting non-existent database
    auto delete_result = db_service_->delete_database("non_existent_database");
    EXPECT_FALSE(delete_result.has_value());
    
    // Test checking existence of non-existent database
    auto exists_result = db_service_->database_exists("non_existent_database");
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
}

// Test database API statistics
TEST_F(DatabaseApiIntegrationTest, DatabaseApiStatistics) {
    // Create a database
    DatabaseCreationParams create_params;
    create_params.name = "stats_test_db";
    create_params.description = "Statistics test database";
    create_params.vectorDimension = 256;
    create_params.indexType = "IVF";
    create_params.sharding = {"hash", 2};
    create_params.replication = {2, true};
    
    auto create_result = db_service_->create_database(create_params);
    ASSERT_TRUE(create_result.has_value());
    std::string database_id = create_result.value();
    
    // Test getting database statistics
    auto stats_result = db_service_->get_database_stats(database_id);
    ASSERT_TRUE(stats_result.has_value());
    
    const auto& stats = stats_result.value();
    EXPECT_EQ(stats.at("database_id"), database_id);
    EXPECT_EQ(stats.at("name"), "stats_test_db");
    EXPECT_EQ(stats.at("vector_dimension"), "256");
    EXPECT_EQ(stats.at("index_type"), "IVF");
    EXPECT_EQ(stats.at("shard_count"), "2");
    EXPECT_EQ(stats.at("replication_factor"), "2");
    EXPECT_FALSE(stats.at("created_at").empty());
    EXPECT_FALSE(stats.at("updated_at").empty());
    
    // Test getting database count
    auto count_result = db_service_->get_database_count();
    ASSERT_TRUE(count_result.has_value());
    EXPECT_GE(count_result.value(), 1);
}

// Test database API with complex configurations
TEST_F(DatabaseApiIntegrationTest, DatabaseApiComplexConfigurations) {
    // Create a database with complex configuration
    DatabaseCreationParams params;
    params.name = "complex_config_db";
    params.description = "Database with complex configuration";
    params.vectorDimension = 512;
    params.indexType = "HNSW";
    
    // Add complex index parameters
    params.indexParameters = {
        {"M", "32"},
        {"efConstruction", "400"},
        {"efSearch", "100"}
    };
    
    // Set sharding configuration
    params.sharding = {"hash", 4};
    
    // Set replication configuration
    params.replication = {3, true};
    
    // Add embedding models
    Database::EmbeddingModel model1;
    model1.name = "bert-base";
    model1.version = "1.0";
    model1.provider = "huggingface";
    model1.inputType = "text";
    model1.outputDimension = 768;
    model1.status = "active";
    model1.parameters = {{"max_tokens", "512"}};
    
    Database::EmbeddingModel model2;
    model2.name = "resnet50";
    model2.version = "1.0";
    model2.provider = "torchvision";
    model2.inputType = "image";
    model2.outputDimension = 2048;
    model2.status = "active";
    
    params.embeddingModels = {model1, model2};
    
    // Add metadata schema
    params.metadataSchema = {
        {"author", "string"},
        {"category", "string"},
        {"score", "float"},
        {"tags", "array"},
        {"created_at", "datetime"}
    };
    
    // Add retention policy
    params.retentionPolicy = std::make_unique<Database::RetentionPolicy>();
    params.retentionPolicy->maxAgeDays = 365;
    params.retentionPolicy->archiveOnExpire = true;
    
    // Add access control
    params.accessControl.roles = {"admin", "user", "reader"};
    params.accessControl.defaultPermissions = {"read", "search"};
    
    // Create the database
    auto result = db_service_->create_database(params);
    ASSERT_TRUE(result.has_value());
    std::string database_id = result.value();
    
    // Verify the complex configuration
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    const auto& database = get_result.value();
    EXPECT_EQ(database.databaseId, database_id);
    EXPECT_EQ(database.name, "complex_config_db");
    EXPECT_EQ(database.vectorDimension, 512);
    EXPECT_EQ(database.indexType, "HNSW");
    EXPECT_EQ(database.indexParameters.size(), 3);
    EXPECT_EQ(database.indexParameters.at("M"), "32");
    EXPECT_EQ(database.indexParameters.at("efConstruction"), "400");
    EXPECT_EQ(database.indexParameters.at("efSearch"), "100");
    EXPECT_EQ(database.sharding.strategy, "hash");
    EXPECT_EQ(database.sharding.numShards, 4);
    EXPECT_EQ(database.replication.factor, 3);
    EXPECT_EQ(database.replication.sync, true);
    EXPECT_EQ(database.embeddingModels.size(), 2);
    EXPECT_EQ(database.embeddingModels[0].name, "bert-base");
    EXPECT_EQ(database.embeddingModels[1].name, "resnet50");
    EXPECT_EQ(database.metadataSchema.size(), 5);
    EXPECT_TRUE(database.retentionPolicy != nullptr);
    EXPECT_EQ(database.retentionPolicy->maxAgeDays, 365);
    EXPECT_EQ(database.retentionPolicy->archiveOnExpire, true);
    EXPECT_EQ(database.accessControl.roles.size(), 3);
    EXPECT_EQ(database.accessControl.defaultPermissions.size(), 2);
}

} // namespace jadevectordb