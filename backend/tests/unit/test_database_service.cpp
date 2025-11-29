#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/database_service.h"
#include "services/database_layer.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Test fixture for DatabaseService
class DatabaseServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create database service
        db_service_ = std::make_unique<DatabaseService>();
        
        // Initialize the service
        auto init_result = db_service_->initialize();
        ASSERT_TRUE(init_result.has_value());
    }
    
    void TearDown() override {
        // Clean up
        db_service_.reset();
    }
    
    std::unique_ptr<DatabaseService> db_service_;
};

// Test that the service initializes correctly
TEST_F(DatabaseServiceTest, InitializeService) {
    // Service should already be initialized in SetUp
    EXPECT_NE(db_service_, nullptr);
}

// Test creating a database
TEST_F(DatabaseServiceTest, CreateDatabase) {
    // Create a valid database configuration
    Database db_config;
    db_config.name = "test_database";
    db_config.description = "Test database for unit tests";
    db_config.vectorDimension = 128;
    db_config.indexType = "HNSW";
    
    // Validate the database configuration
    auto validation_result = db_service_->validate_creation_params(db_config);
    EXPECT_TRUE(validation_result.has_value());
    
    // Create the database
    auto result = db_service_->create_database(db_config);
    ASSERT_TRUE(result.has_value());
    
    std::string database_id = result.value();
    EXPECT_FALSE(database_id.empty());
    
    // Verify the database was created by retrieving it
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    Database retrieved_db = get_result.value();
    EXPECT_EQ(retrieved_db.name, db_config.name);
    EXPECT_EQ(retrieved_db.description, db_config.description);
    EXPECT_EQ(retrieved_db.vectorDimension, db_config.vectorDimension);
    EXPECT_EQ(retrieved_db.indexType, db_config.indexType);
    EXPECT_EQ(retrieved_db.databaseId, database_id);
}

// Test listing databases
TEST_F(DatabaseServiceTest, ListDatabases) {
    // Initially, there should be no databases
    DatabaseListParams list_params;
    auto initial_result = db_service_->list_databases(list_params);
    ASSERT_TRUE(initial_result.has_value());
    
    auto initial_databases = initial_result.value();
    size_t initial_count = initial_databases.size();
    
    // Create a few test databases
    std::vector<std::string> created_db_ids;
    
    for (int i = 0; i < 3; ++i) {
        Database db_config;
        db_config.name = "test_database_" + std::to_string(i);
        db_config.description = "Test database " + std::to_string(i);
        db_config.vectorDimension = 64 + i * 32; // 64, 96, 128
        db_config.indexType = (i % 2 == 0) ? "HNSW" : "IVF";
        
        auto validation_result = db_service_->validate_creation_params(db_config);
        ASSERT_TRUE(validation_result.has_value());
        
        auto create_result = db_service_->create_database(db_config);
        ASSERT_TRUE(create_result.has_value());
        
        created_db_ids.push_back(create_result.value());
    }
    
    // List databases again
    auto result = db_service_->list_databases(list_params);
    ASSERT_TRUE(result.has_value());
    
    auto databases = result.value();
    EXPECT_GE(databases.size(), initial_count + 3);
    
    // Verify all created databases are in the list
    for (const auto& db_id : created_db_ids) {
        bool found = false;
        for (const auto& db : databases) {
            if (db.databaseId == db_id) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Database " << db_id << " not found in list";
    }
}

// Test getting a database by ID
TEST_F(DatabaseServiceTest, GetDatabase) {
    // Create a test database
    Database db_config;
    db_config.name = "get_test_database";
    db_config.description = "Database for get tests";
    db_config.vectorDimension = 256;
    db_config.indexType = "LSH";
    
    auto validation_result = db_service_->validate_creation_params(db_config);
    ASSERT_TRUE(validation_result.has_value());
    
    auto create_result = db_service_->create_database(db_config);
    ASSERT_TRUE(create_result.has_value());
    
    std::string database_id = create_result.value();
    
    // Get the database by ID
    auto result = db_service_->get_database(database_id);
    ASSERT_TRUE(result.has_value());
    
    Database retrieved_db = result.value();
    EXPECT_EQ(retrieved_db.name, db_config.name);
    EXPECT_EQ(retrieved_db.description, db_config.description);
    EXPECT_EQ(retrieved_db.vectorDimension, db_config.vectorDimension);
    EXPECT_EQ(retrieved_db.indexType, db_config.indexType);
    EXPECT_EQ(retrieved_db.databaseId, database_id);
    
    // Try to get a non-existent database (should fail)
    auto non_existent_result = db_service_->get_database("non_existent_db_id");
    EXPECT_FALSE(non_existent_result.has_value());
}

// Test updating a database
TEST_F(DatabaseServiceTest, UpdateDatabase) {
    // Create a test database
    Database db_config;
    db_config.name = "update_test_database";
    db_config.description = "Database for update tests";
    db_config.vectorDimension = 128;
    db_config.indexType = "HNSW";
    
    auto validation_result = db_service_->validate_creation_params(db_config);
    ASSERT_TRUE(validation_result.has_value());
    
    auto create_result = db_service_->create_database(db_config);
    ASSERT_TRUE(create_result.has_value());
    
    std::string database_id = create_result.value();
    
    // Verify initial state
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    Database initial_db = get_result.value();
    EXPECT_EQ(initial_db.name, db_config.name);
    EXPECT_EQ(initial_db.description, db_config.description);
    
    // Update the database
    DatabaseUpdateParams update_params;
    update_params.name = "updated_database_name";
    update_params.description = "Updated database description";
    
    auto update_result = db_service_->update_database(database_id, update_params);
    EXPECT_TRUE(update_result.has_value());
    
    // Verify the update
    get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    Database updated_db = get_result.value();
    EXPECT_EQ(updated_db.name, "updated_database_name");
    EXPECT_EQ(updated_db.description, "Updated database description");
    EXPECT_EQ(updated_db.vectorDimension, db_config.vectorDimension); // Should remain unchanged
    EXPECT_EQ(updated_db.indexType, db_config.indexType); // Should remain unchanged
}

// Test deleting a database
TEST_F(DatabaseServiceTest, DeleteDatabase) {
    // Create a test database
    Database db_config;
    db_config.name = "delete_test_database";
    db_config.description = "Database for delete tests";
    db_config.vectorDimension = 128;
    db_config.indexType = "HNSW";
    
    auto validation_result = db_service_->validate_creation_params(db_config);
    ASSERT_TRUE(validation_result.has_value());
    
    auto create_result = db_service_->create_database(db_config);
    ASSERT_TRUE(create_result.has_value());
    
    std::string database_id = create_result.value();
    
    // Verify the database exists
    auto get_result = db_service_->get_database(database_id);
    ASSERT_TRUE(get_result.has_value());
    
    // Delete the database
    auto delete_result = db_service_->delete_database(database_id);
    EXPECT_TRUE(delete_result.has_value());
    
    // Verify the database no longer exists
    get_result = db_service_->get_database(database_id);
    EXPECT_FALSE(get_result.has_value());
    
    // Try to delete a non-existent database (should fail)
    auto non_existent_delete_result = db_service_->delete_database("non_existent_db_id");
    EXPECT_FALSE(non_existent_delete_result.has_value());
}

// Test database creation parameter validation
TEST_F(DatabaseServiceTest, ValidateDatabaseCreationParams) {
    // Valid database configuration
    Database valid_db;
    valid_db.name = "valid_database";
    valid_db.description = "Valid database configuration";
    valid_db.vectorDimension = 128;
    valid_db.indexType = "HNSW";
    
    auto result = db_service_->validate_creation_params(valid_db);
    EXPECT_TRUE(result.has_value());
    
    // Invalid database configuration - empty name
    Database invalid_db1 = valid_db;
    invalid_db1.name = "";
    
    result = db_service_->validate_creation_params(invalid_db1);
    EXPECT_FALSE(result.has_value());
    
    // Invalid database configuration - negative vector dimension
    Database invalid_db2 = valid_db;
    invalid_db2.vectorDimension = -1;
    
    result = db_service_->validate_creation_params(invalid_db2);
    EXPECT_FALSE(result.has_value());
    
    // Invalid database configuration - zero vector dimension
    Database invalid_db3 = valid_db;
    invalid_db3.vectorDimension = 0;
    
    result = db_service_->validate_creation_params(invalid_db3);
    EXPECT_FALSE(result.has_value());
    
    // Invalid database configuration - unsupported index type
    Database invalid_db4 = valid_db;
    invalid_db4.indexType = "UNSUPPORTED_INDEX";
    
    result = db_service_->validate_creation_params(invalid_db4);
    EXPECT_FALSE(result.has_value());
    
    // Valid database configuration - maximum allowed vector dimension
    Database valid_db_max_dim = valid_db;
    valid_db_max_dim.vectorDimension = 4096; // Maximum allowed dimension
    
    result = db_service_->validate_creation_params(valid_db_max_dim);
    EXPECT_TRUE(result.has_value());
    
    // Invalid database configuration - vector dimension too large
    Database invalid_db5 = valid_db;
    invalid_db5.vectorDimension = 5000; // Exceeds maximum allowed dimension
    
    result = db_service_->validate_creation_params(invalid_db5);
    EXPECT_FALSE(result.has_value());
}

// Test database update parameter validation
TEST_F(DatabaseServiceTest, ValidateDatabaseUpdateParams) {
    // Valid update parameters
    DatabaseUpdateParams valid_params;
    valid_params.name = "updated_name";
    valid_params.description = "Updated description";
    valid_params.vectorDimension = 256;
    valid_params.indexType = "IVF";
    
    auto result = db_service_->validate_update_params(valid_params);
    EXPECT_TRUE(result.has_value());
    
    // Valid update parameters - only name
    DatabaseUpdateParams valid_params_name_only;
    valid_params_name_only.name = "new_name_only";
    
    result = db_service_->validate_update_params(valid_params_name_only);
    EXPECT_TRUE(result.has_value());
    
    // Invalid update parameters - negative vector dimension
    DatabaseUpdateParams invalid_params1;
    invalid_params1.vectorDimension = -1;
    
    result = db_service_->validate_update_params(invalid_params1);
    EXPECT_FALSE(result.has_value());
    
    // Invalid update parameters - zero vector dimension
    DatabaseUpdateParams invalid_params2;
    invalid_params2.vectorDimension = 0;
    
    result = db_service_->validate_update_params(invalid_params2);
    EXPECT_FALSE(result.has_value());
    
    // Invalid update parameters - unsupported index type
    DatabaseUpdateParams invalid_params3;
    invalid_params3.indexType = "UNSUPPORTED_INDEX";
    
    result = db_service_->validate_update_params(invalid_params3);
    EXPECT_FALSE(result.has_value());
    
    // Valid update parameters - maximum allowed vector dimension
    DatabaseUpdateParams valid_params_max_dim;
    valid_params_max_dim.vectorDimension = 4096; // Maximum allowed dimension
    
    result = db_service_->validate_update_params(valid_params_max_dim);
    EXPECT_TRUE(result.has_value());
    
    // Invalid update parameters - vector dimension too large
    DatabaseUpdateParams invalid_params4;
    invalid_params4.vectorDimension = 5000; // Exceeds maximum allowed dimension
    
    result = db_service_->validate_update_params(invalid_params4);
    EXPECT_FALSE(result.has_value());
}

// Test checking if database exists
TEST_F(DatabaseServiceTest, DatabaseExists) {
    // Create a test database
    Database db_config;
    db_config.name = "exists_test_database";
    db_config.description = "Database for exists tests";
    db_config.vectorDimension = 128;
    db_config.indexType = "HNSW";
    
    auto validation_result = db_service_->validate_creation_params(db_config);
    ASSERT_TRUE(validation_result.has_value());
    
    auto create_result = db_service_->create_database(db_config);
    ASSERT_TRUE(create_result.has_value());
    
    std::string database_id = create_result.value();
    
    // Check if the database exists (should be true)
    auto exists_result = db_service_->database_exists(database_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
    
    // Check if a non-existent database exists (should be false)
    exists_result = db_service_->database_exists("non_existent_db_id");
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
}

// Test getting database count
TEST_F(DatabaseServiceTest, GetDatabaseCount) {
    // Get initial count
    auto initial_count_result = db_service_->get_database_count();
    ASSERT_TRUE(initial_count_result.has_value());
    
    size_t initial_count = initial_count_result.value();
    
    // Create a few test databases
    std::vector<std::string> created_db_ids;
    
    for (int i = 0; i < 2; ++i) {
        Database db_config;
        db_config.name = "count_test_database_" + std::to_string(i);
        db_config.description = "Database for count tests " + std::to_string(i);
        db_config.vectorDimension = 128;
        db_config.indexType = "HNSW";
        
        auto validation_result = db_service_->validate_creation_params(db_config);
        ASSERT_TRUE(validation_result.has_value());
        
        auto create_result = db_service_->create_database(db_config);
        ASSERT_TRUE(create_result.has_value());
        
        created_db_ids.push_back(create_result.value());
    }
    
    // Get count after creating databases
    auto count_result = db_service_->get_database_count();
    ASSERT_TRUE(count_result.has_value());
    
    size_t final_count = count_result.value();
    EXPECT_EQ(final_count, initial_count + 2);
    
    // Delete one database
    auto delete_result = db_service_->delete_database(created_db_ids[0]);
    ASSERT_TRUE(delete_result.has_value());
    
    // Get count after deleting one database
    count_result = db_service_->get_database_count();
    ASSERT_TRUE(count_result.has_value());
    
    size_t count_after_delete = count_result.value();
    EXPECT_EQ(count_after_delete, initial_count + 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
// Additional test cases for enhanced coverage
TEST_F(DatabaseServiceTest, CreateAndConfigureDatabaseWithSpecificSettings) {
    // Test creating a database with specific configuration settings
    DatabaseConfig config;
    config.vector_dimension = 768; // Common embedding dimension
    config.index_type = "HNSW";
    config.storage_format = "custom_binary";
    config.replication_factor = 3;
    config.sharding_enabled = true;
    config.retention_policy.days = 365;
    config.performance_tier = "standard";

    auto result = database_service_->create_database("configured_test_db", config);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        std::string created_db_id = result.value();

        // Verify the database was created with the specified configuration
        auto db_result = database_service_->get_database(created_db_id);
        EXPECT_TRUE(db_result.has_value());

        if (db_result.has_value()) {
            Database created_db = db_result.value();
            EXPECT_EQ(created_db.vectorDimension, 768);
            EXPECT_EQ(created_db.indexType, "HNSW");
            EXPECT_EQ(created_db.config.storage_format, "custom_binary");
            EXPECT_EQ(created_db.config.replication_factor, 3);
            EXPECT_EQ(created_db.config.sharding_enabled, true);
            EXPECT_EQ(created_db.config.retention_policy.days, 365);
            EXPECT_EQ(created_db.config.performance_tier, "standard");
        }
    }
}

TEST_F(DatabaseServiceTest, ListDatabasesWithPagination) {
    // Test pagination functionality when listing databases
    std::vector<std::string> created_dbs;

    // Create multiple test databases
    for (int i = 0; i < 10; ++i) {
        DatabaseConfig config;
        config.vector_dimension = 128;
        config.index_type = "IVF";
        config.sharding_enabled = false;

        auto result = database_service_->create_database("pagination_test_db_" + std::to_string(i), config);
        if (result.has_value()) {
            created_dbs.push_back(result.value());
        }
    }

    // Test listing with pagination
    auto list_result = database_service_->list_databases(0, 5); // First 5
    EXPECT_TRUE(list_result.has_value());

    if (list_result.has_value()) {
        auto page1 = list_result.value();
        EXPECT_EQ(page1.size(), 5);  // Should return first 5 databases

        // Test the next page
        auto list_result_page2 = database_service_->list_databases(5, 5); // Next 5
        EXPECT_TRUE(list_result_page2.has_value());

        if (list_result_page2.has_value()) {
            auto page2 = list_result_page2.value();
            EXPECT_EQ(page2.size(), created_dbs.size() - 5);  // Remaining databases

            // Verify no overlap between pages
            std::set<std::string> page1_ids;
            for (const auto& db : page1) {
                page1_ids.insert(db.databaseId);
            }

            for (const auto& db : page2) {
                EXPECT_FALSE(page1_ids.count(db.databaseId));  // No overlap
            }
        }
    }

    // Clean up test databases
    for (const auto& db_id : created_dbs) {
        database_service_->delete_database(db_id);
    }
}
