#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "api/rest/rest_api.h"
#include "services/database_service.h"
#include "services/vector_storage.h"
#include "models/database.h"
#include "models/vector.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Integration test for database API endpoints
class DatabaseApiIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create shared database layer
        auto db_layer = std::make_shared<DatabaseLayer>();
        db_layer->initialize();
        
        // Create services that share the same database layer
        db_service_ = std::make_unique<DatabaseService>(db_layer);
        vector_service_ = std::make_unique<VectorStorageService>(db_layer);
        
        // Initialize services
        db_service_->initialize();
        vector_service_->initialize();
    }
    
    void TearDown() override {
        // Clean up any databases created during tests
        auto list_result = db_service_->list_databases();
        if (list_result.has_value()) {
            for (const auto& db : list_result.value()) {
                db_service_->delete_database(db.databaseId);
            }
        }
    }
    
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_service_;

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
    DatabaseUpdateParams to_update_params(const Database& db) {
        DatabaseUpdateParams params;
        params.name = db.name;
        params.description = db.description;
        params.vectorDimension = db.vectorDimension;
        params.indexType = db.indexType;
        // Note: indexParameters type mismatch (map vs unordered_map)
        std::unordered_map<std::string, std::string> idx_params;
        for (const auto& [k, v] : db.indexParameters) {
            idx_params[k] = v;
        }
        params.indexParameters = idx_params;
        params.sharding = db.sharding;
        params.replication = db.replication;
        params.embeddingModels = db.embeddingModels;
        // Note: metadataSchema type mismatch (map vs unordered_map)
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

};

// Test database creation and retrieval
TEST_F(DatabaseApiIntegrationTest, CreateAndGetDatabase) {
    // Create a new database
    Database new_db;
    new_db.name = "integration_test_db";
    new_db.description = "A database for integration testing";
    new_db.vectorDimension = 128;
    new_db.indexType = "HNSW";
    
    auto create_result = db_service_->create_database(to_creation_params(new_db));
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = create_result.value();
    
    // Retrieve the created database
    auto get_result = db_service_->get_database(db_id);
    ASSERT_TRUE(get_result.has_value());
    
    Database retrieved_db = get_result.value();
    EXPECT_EQ(retrieved_db.name, "integration_test_db");
    EXPECT_EQ(retrieved_db.description, "A database for integration testing");
    EXPECT_EQ(retrieved_db.vectorDimension, 128);
    EXPECT_EQ(retrieved_db.indexType, "HNSW");
    EXPECT_EQ(retrieved_db.databaseId, db_id);
}

// Test database listing
TEST_F(DatabaseApiIntegrationTest, ListDatabases) {
    // Create multiple databases
    std::vector<std::string> created_db_ids;
    
    for (int i = 0; i < 3; ++i) {
        Database new_db;
        new_db.name = "test_db_" + std::to_string(i);
        new_db.description = "Test database " + std::to_string(i);
        new_db.vectorDimension = 64 + i * 64; // 64, 128, 192
        
        auto create_result = db_service_->create_database(to_creation_params(new_db));
        ASSERT_TRUE(create_result.has_value());
        created_db_ids.push_back(create_result.value());
    }
    
    // List all databases
    auto list_result = db_service_->list_databases();
    ASSERT_TRUE(list_result.has_value());
    
    auto databases = list_result.value();
    
    // Should have at least the 3 databases we created
    EXPECT_GE(databases.size(), 3);
    
    // Verify our created databases are in the list
    for (const auto& created_id : created_db_ids) {
        bool found = false;
        for (const auto& db : databases) {
            if (db.databaseId == created_id) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Database " << created_id << " not found in list";
    }
}

// Test database update
TEST_F(DatabaseApiIntegrationTest, UpdateDatabase) {
    // Create a database
    Database initial_db;
    initial_db.name = "initial_db_name";
    initial_db.description = "Initial description";
    initial_db.vectorDimension = 128;
    initial_db.indexType = "flat";  // Use lowercase index type
    
    auto create_result = db_service_->create_database(to_creation_params(initial_db));
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = create_result.value();
    
    // Verify the database was created with initial values
    auto get_result = db_service_->get_database(db_id);
    ASSERT_TRUE(get_result.has_value());
    EXPECT_EQ(get_result.value().name, "initial_db_name");
    
    // Update the database
    Database updated_db = initial_db;
    updated_db.databaseId = db_id;  // Set the database ID
    updated_db.name = "updated_db_name";
    updated_db.description = "Updated description";
    updated_db.vectorDimension = 256;
    
    auto update_result = db_service_->update_database(db_id, to_update_params(updated_db));
    ASSERT_TRUE(update_result.has_value()) << "Update failed: " << update_result.error().message;
    
    // Retrieve the database again to verify the update
    get_result = db_service_->get_database(db_id);
    ASSERT_TRUE(get_result.has_value());
    
    Database retrieved_db = get_result.value();
    EXPECT_EQ(retrieved_db.name, "updated_db_name");
    EXPECT_EQ(retrieved_db.description, "Updated description");
    EXPECT_EQ(retrieved_db.vectorDimension, 256);
}

// Test database deletion
TEST_F(DatabaseApiIntegrationTest, DeleteDatabase) {
    // Create a database
    Database new_db;
    new_db.name = "db_to_delete";
    new_db.description = "Database for deletion test";
    new_db.vectorDimension = 128;
    
    auto create_result = db_service_->create_database(to_creation_params(new_db));
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = create_result.value();
    
    // Verify the database exists
    auto exists_result = db_service_->database_exists(db_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_TRUE(exists_result.value());
    
    // Delete the database
    auto delete_result = db_service_->delete_database(db_id);
    ASSERT_TRUE(delete_result.has_value());
    
    // Verify the database no longer exists
    exists_result = db_service_->database_exists(db_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
    
    // Verify the database is not in the list
    auto list_result = db_service_->list_databases();
    ASSERT_TRUE(list_result.has_value());
    
    for (const auto& db : list_result.value()) {
        EXPECT_NE(db.databaseId, db_id);
    }
}

// Test database with vectors integration
TEST_F(DatabaseApiIntegrationTest, DatabaseWithVectors) {
    // Create a database
    Database vectors_db;
    vectors_db.name = "vectors_test_db";
    vectors_db.description = "Database for testing vectors";
    vectors_db.vectorDimension = 3;  // 3-dimensional vectors
    
    auto create_result = db_service_->create_database(to_creation_params(vectors_db));
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = create_result.value();
    
    // Create and store some vectors in this database
    std::vector<Vector> test_vectors;
    
    for (int i = 0; i < 3; ++i) {
        Vector v;
        v.id = "vector_" + std::to_string(i);
        v.databaseId = db_id;
        v.values = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)};
        v.metadata.status = "active";
        v.metadata.custom["test_id"] = i;
        v.metadata.custom["category"] = "test_vector";
        
        test_vectors.push_back(v);
    }
    
    // Store the vectors
    for (const auto& v : test_vectors) {
        auto store_result = vector_service_->store_vector(db_id, v);
        ASSERT_TRUE(store_result.has_value()) << "Failed to store vector: " << v.id 
            << " Error: " << (store_result.has_value() ? "" : store_result.error().message);
    }
    
    // Verify vectors were stored by retrieving them
    std::vector<std::string> vector_ids;
    for (const auto& v : test_vectors) {
        vector_ids.push_back(v.id);
    }
    
    auto retrieve_result = vector_service_->retrieve_vectors(db_id, vector_ids);
    ASSERT_TRUE(retrieve_result.has_value());
    
    auto retrieved_vectors = retrieve_result.value();
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
    
    // Test vector count in database
    auto count_result = vector_service_->get_vector_count(db_id);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), test_vectors.size());
    
    // Clean up: delete the database (which should also remove associated vectors)
    auto delete_result = db_service_->delete_database(db_id);
    ASSERT_TRUE(delete_result.has_value());
    
    // Verify database and vectors are gone
    auto exists_result = db_service_->database_exists(db_id);
    ASSERT_TRUE(exists_result.has_value());
    EXPECT_FALSE(exists_result.value());
}

// Test multiple databases with different configurations
TEST_F(DatabaseApiIntegrationTest, MultipleDatabasesWithDifferentConfigs) {
    std::vector<std::pair<std::string, Database>> created_dbs;
    
    // Create databases with different configurations
    struct DbConfig {
        std::string name;
        std::string description;
        int vectorDimension;
        std::string indexType;
    };
    
    std::vector<DbConfig> configs = {
        {"db_config_1", "Config 1 DB", 64, "flat"},
        {"db_config_2", "Config 2 DB", 128, "hnsw"},
        {"db_config_3", "Config 3 DB", 256, "ivf"}
    };
    
    for (const auto& config : configs) {
        Database db;
        db.name = config.name;
        db.description = config.description;
        db.vectorDimension = config.vectorDimension;
        db.indexType = config.indexType;
        
        auto create_result = db_service_->create_database(to_creation_params(db));
        ASSERT_TRUE(create_result.has_value()) << "Failed to create database: " << db.name;
        
        created_dbs.push_back({create_result.value(), db});
    }
    
    // Verify all databases were created with correct configurations
    for (const auto& [db_id, expected_db] : created_dbs) {
        auto get_result = db_service_->get_database(db_id);
        ASSERT_TRUE(get_result.has_value()) << "Failed to retrieve database: " << db_id;
        
        Database actual_db = get_result.value();
        EXPECT_EQ(actual_db.name, expected_db.name);
        EXPECT_EQ(actual_db.description, expected_db.description);
        EXPECT_EQ(actual_db.vectorDimension, expected_db.vectorDimension);
        EXPECT_EQ(actual_db.indexType, expected_db.indexType);
    }
    
    // List databases and verify count
    auto list_result = db_service_->list_databases();
    ASSERT_TRUE(list_result.has_value());
    EXPECT_GE(list_result.value().size(), configs.size());
    
    // Clean up all created databases
    for (const auto& [db_id, _] : created_dbs) {
        auto delete_result = db_service_->delete_database(db_id);
        ASSERT_TRUE(delete_result.has_value()) << "Failed to delete database: " << db_id;
    }
}

// Test database validation during creation
TEST_F(DatabaseApiIntegrationTest, DatabaseValidation) {
    // Test creating a database with invalid dimensions
    Database invalid_db;
    invalid_db.name = "invalid_db";
    invalid_db.description = "Database with invalid config";
    invalid_db.vectorDimension = 0;  // Invalid dimension
    
    auto create_result = db_service_->create_database(to_creation_params(invalid_db));
    EXPECT_FALSE(create_result.has_value());
    
    // Test creating a database with valid but large dimensions
    Database large_db;
    large_db.name = "large_db";
    large_db.description = "Database with large dimensions";
    large_db.vectorDimension = 4096;  // This is at the upper limit from spec
    
    create_result = db_service_->create_database(to_creation_params(large_db));
    ASSERT_TRUE(create_result.has_value());
    
    std::string large_db_id = create_result.value();
    
    // Verify it was created correctly
    auto get_result = db_service_->get_database(large_db_id);
    ASSERT_TRUE(get_result.has_value());
    EXPECT_EQ(get_result.value().vectorDimension, 4096);
    
    // Clean up
    auto delete_result = db_service_->delete_database(large_db_id);
    ASSERT_TRUE(delete_result.has_value());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}