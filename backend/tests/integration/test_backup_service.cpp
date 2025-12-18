/**
 * Integration tests for Incremental Backup using Service Layer (Sprint 2.2)
 */

#include <gtest/gtest.h>
#include <memory>
#include <filesystem>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "storage/incremental_backup_manager.h"
#include "storage/memory_mapped_vector_store.h"
#include "models/vector.h"
#include "models/database.h"

namespace fs = std::filesystem;

namespace jadevectordb {

// Helper function
static DatabaseCreationParams to_creation_params_backup(const Database& db) {
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

namespace test {

class BackupServiceIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directories
        test_dir_ = "./test_sprint22_backup_" + std::to_string(std::time(nullptr));
        backup_dir_ = test_dir_ + "/backups";
        fs::create_directories(test_dir_);
        fs::create_directories(backup_dir_);
        
        // Create persistence layer with file-based storage
        auto persistence = std::make_unique<PersistentDatabasePersistence>(test_dir_);
        auto db_layer = std::make_unique<DatabaseLayer>(std::move(persistence));
        
        // Initialize database layer
        auto db_init = db_layer->initialize();
        ASSERT_TRUE(db_init.has_value());
        
        // Store the layer
        db_layer_ = std::move(db_layer);
        
        // Create services (shared_ptr with no-op deleter)
        auto db_layer_shared = std::shared_ptr<DatabaseLayer>(db_layer_.get(), [](DatabaseLayer*){});
        db_service_ = std::make_unique<DatabaseService>(db_layer_shared);
        vector_service_ = std::make_unique<VectorStorageService>(db_layer_shared);
        
        // Initialize services
        auto db_svc_init = db_service_->initialize();
        ASSERT_TRUE(db_svc_init.has_value());
        
        auto vec_init = vector_service_->initialize();
        ASSERT_TRUE(vec_init.has_value());
        
        // Create test database
        Database test_db;
        test_db.name = "backup_test_db";
        test_db.vectorDimension = 128;
        test_db.description = "Test database for backup";
        test_db.indexType = "FLAT";
        
        auto create_result = db_service_->create_database(to_creation_params_backup(test_db));
        ASSERT_TRUE(create_result.has_value());
        test_db_id_ = create_result.value();
        
        dimension_ = 128;
    }
    
    void TearDown() override {
        if (!test_db_id_.empty()) {
            auto del_result = db_layer_->delete_database(test_db_id_);
        }
        
        // Reset services
        vector_service_.reset();
        db_service_.reset();
        db_layer_.reset();
        
        // Cleanup test directories
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }
    
    // Helper: Add vectors
    void add_vectors(int count, const std::string& prefix = "vec_") {
        for (int i = 0; i < count; i++) {
            Vector v;
            v.id = prefix + std::to_string(i);
            v.databaseId = test_db_id_;
            v.values.resize(dimension_, static_cast<float>(i) * 0.1f);
            v.metadata.status = "active";
            
            auto result = vector_service_->store_vector(test_db_id_, v);
            ASSERT_TRUE(result.has_value()) << "Failed to store vector " << v.id;
        }
    }
    
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_service_;
    std::string test_db_id_;
    std::string test_dir_;
    std::string backup_dir_;
    int dimension_;
};

TEST_F(BackupServiceIntegrationTest, VectorPersistenceAcrossSessions) {
    // Add vectors
    add_vectors(30);
    
    auto count_result = vector_service_->get_vector_count(test_db_id_);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 30);
    
    // Verify vectors are retrievable
    for (int i = 0; i < 30; i++) {
        auto result = vector_service_->retrieve_vector(test_db_id_, "vec_" + std::to_string(i));
        EXPECT_TRUE(result.has_value());
    }
}

TEST_F(BackupServiceIntegrationTest, VectorDataIntegrity) {
    // Add vectors with specific data
    for (int i = 0; i < 20; i++) {
        Vector v;
        v.id = "data_" + std::to_string(i);
        for (int j = 0; j < dimension_; j++) {
            v.values.push_back(static_cast<float>(i + j * 0.01f));
        }
        
        auto result = vector_service_->store_vector(test_db_id_, v);
        ASSERT_TRUE(result.has_value());
    }
    
    // Retrieve and verify data
    for (int i = 0; i < 20; i++) {
        auto result = vector_service_->retrieve_vector(test_db_id_, "data_" + std::to_string(i));
        ASSERT_TRUE(result.has_value());
        
        const auto& v = result.value();
        EXPECT_EQ(v.values.size(), dimension_);
        EXPECT_FLOAT_EQ(v.values[0], static_cast<float>(i));
        EXPECT_FLOAT_EQ(v.values[10], static_cast<float>(i + 0.1f));
    }
}

TEST_F(BackupServiceIntegrationTest, BatchOperations) {
    // Test batch vector storage
    std::vector<Vector> batch_vectors;
    for (int i = 0; i < 25; i++) {
        Vector v;
        v.id = "batch_" + std::to_string(i);
        v.values.resize(dimension_, static_cast<float>(i));
        batch_vectors.push_back(v);
    }
    
    auto batch_result = vector_service_->batch_store_vectors(test_db_id_, batch_vectors);
    EXPECT_TRUE(batch_result.has_value());
    
    // Verify all vectors stored
    auto count_result = vector_service_->get_vector_count(test_db_id_);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 25);
}

TEST_F(BackupServiceIntegrationTest, VectorDeletion) {
    add_vectors(30);
    
    // Delete half
    for (int i = 0; i < 15; i++) {
        auto result = vector_service_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
        EXPECT_TRUE(result.has_value());
    }
    
    // Verify deleted vectors don't exist
    for (int i = 0; i < 15; i++) {
        auto exists = vector_service_->vector_exists(test_db_id_, "vec_" + std::to_string(i));
        ASSERT_TRUE(exists.has_value());
        EXPECT_FALSE(exists.value());
    }
    
    // Verify remaining vectors exist
    for (int i = 15; i < 30; i++) {
        auto exists = vector_service_->vector_exists(test_db_id_, "vec_" + std::to_string(i));
        ASSERT_TRUE(exists.has_value());
        EXPECT_TRUE(exists.value());
    }
}

TEST_F(BackupServiceIntegrationTest, VectorUpdate) {
    // Add vector
    Vector v;
    v.id = "update_test";
    v.values.resize(dimension_, 1.0f);
    
    auto store_result = vector_service_->store_vector(test_db_id_, v);
    ASSERT_TRUE(store_result.has_value());
    
    // Update vector
    v.values[0] = 2.0f;
    v.values[1] = 3.0f;
    
    auto update_result = vector_service_->update_vector(test_db_id_, v);
    EXPECT_TRUE(update_result.has_value());
    
    // Verify updated data
    auto retrieve_result = vector_service_->retrieve_vector(test_db_id_, "update_test");
    ASSERT_TRUE(retrieve_result.has_value());
    
    const auto& updated = retrieve_result.value();
    EXPECT_FLOAT_EQ(updated.values[0], 2.0f);
    EXPECT_FLOAT_EQ(updated.values[1], 3.0f);
    EXPECT_FLOAT_EQ(updated.values[2], 1.0f);  // Unchanged
}

} // namespace test
} // namespace jadevectordb
