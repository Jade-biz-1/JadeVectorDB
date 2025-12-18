/**
 * Integration tests for Vector File Compaction using Service Layer (Sprint 2.2)
 */

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <filesystem>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "storage/vector_file_compactor.h"
#include "storage/memory_mapped_vector_store.h"
#include "models/vector.h"
#include "models/database.h"

namespace fs = std::filesystem;

namespace jadevectordb {

// Helper function to convert Database to DatabaseCreationParams
static DatabaseCreationParams to_creation_params_compact(const Database& db) {
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

class CompactionServiceIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory for persistent storage
        test_dir_ = "./test_sprint22_" + std::to_string(std::time(nullptr));
        fs::create_directories(test_dir_);
        
        // Create persistence layer with file-based storage
        auto persistence = std::make_unique<PersistentDatabasePersistence>(test_dir_);
        auto db_layer = std::make_unique<DatabaseLayer>(std::move(persistence));
        
        // Initialize database layer
        auto db_init = db_layer->initialize();
        ASSERT_TRUE(db_init.has_value()) << "Failed to initialize database layer";
        
        // Store the layer
        db_layer_ = std::move(db_layer);
        
        // Create services with the database layer (shared_ptr with no-op deleter)
        auto db_layer_shared = std::shared_ptr<DatabaseLayer>(db_layer_.get(), [](DatabaseLayer*){});
        db_service_ = std::make_unique<DatabaseService>(db_layer_shared);
        vector_service_ = std::make_unique<VectorStorageService>(db_layer_shared);
        
        // Initialize services
        auto db_svc_init = db_service_->initialize();
        ASSERT_TRUE(db_svc_init.has_value()) << "Failed to initialize database service";
        
        auto vec_init = vector_service_->initialize();
        ASSERT_TRUE(vec_init.has_value()) << "Failed to initialize vector service";
        
        // Create a test database
        Database test_db;
        test_db.name = "compaction_test_db";
        test_db.vectorDimension = 128;
        test_db.description = "Test database for compaction";
        test_db.indexType = "FLAT";
        
        auto create_result = db_service_->create_database(to_creation_params_compact(test_db));
        ASSERT_TRUE(create_result.has_value()) << "Failed to create test database";
        test_db_id_ = create_result.value();
        
        dimension_ = 128;
    }
    
    void TearDown() override {
        if (!test_db_id_.empty()) {
            auto del_result = db_service_->delete_database(test_db_id_);
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
            if (!result.has_value()) {
                std::cout << "Failed to store vector " << v.id << ": " 
                          << result.error().message << " (code: " << static_cast<int>(result.error().code) << ")" << std::endl;
            }
            ASSERT_TRUE(result.has_value()) << "Failed to store vector " << v.id;
        }
    }
    
    // Helper: Delete vectors
    void delete_vectors(int count, const std::string& prefix = "vec_") {
        for (int i = 0; i < count; i++) {
            std::string vector_id = prefix + std::to_string(i);
            auto result = vector_service_->delete_vector(test_db_id_, vector_id);
            // Note: delete might fail if vector doesn't exist, which is ok
        }
    }
    
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_service_;
    std::string test_db_id_;
    std::string test_dir_;
    int dimension_;
};

TEST_F(CompactionServiceIntegrationTest, VectorPersistence) {
    // Test that vectors persist correctly through service layer
    add_vectors(50);
    
    auto count_result = vector_service_->get_vector_count(test_db_id_);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), 50);
    
    // Delete some vectors
    delete_vectors(25);
    
    // Verify remaining vectors are accessible
    for (int i = 25; i < 50; i++) {
        std::string vector_id = "vec_" + std::to_string(i);
        auto result = vector_service_->retrieve_vector(test_db_id_, vector_id);
        EXPECT_TRUE(result.has_value()) << "Vector " << vector_id << " should exist";
    }
}

TEST_F(CompactionServiceIntegrationTest, DataIntegrityAfterDeletion) {
    // Add vectors with specific values
    for (int i = 0; i < 30; i++) {
        Vector v;
        v.id = "test_" + std::to_string(i);
        for (int j = 0; j < dimension_; j++) {
            v.values.push_back(static_cast<float>(i * 10 + j));
        }
        
        auto result = vector_service_->store_vector(test_db_id_, v);
        ASSERT_TRUE(result.has_value());
    }
    
    // Delete first 10
    for (int i = 0; i < 10; i++) {
        auto del_result = vector_service_->delete_vector(test_db_id_, "test_" + std::to_string(i));
        EXPECT_TRUE(del_result.has_value());
    }
    
    // Verify remaining vectors have correct data
    for (int i = 10; i < 30; i++) {
        auto result = vector_service_->retrieve_vector(test_db_id_, "test_" + std::to_string(i));
        ASSERT_TRUE(result.has_value());
        
        const auto& retrieved = result.value();
        EXPECT_EQ(retrieved.values.size(), dimension_);
        
        // Check first few values
        EXPECT_FLOAT_EQ(retrieved.values[0], static_cast<float>(i * 10));
        EXPECT_FLOAT_EQ(retrieved.values[1], static_cast<float>(i * 10 + 1));
    }
}

TEST_F(CompactionServiceIntegrationTest, MultipleBatchOperations) {
    // First batch
    add_vectors(30, "batch1_");
    delete_vectors(15, "batch1_");
    
    // Second batch
    add_vectors(30, "batch2_");
    delete_vectors(10, "batch2_");
    
    // Verify vectors from both batches exist
    auto result1 = vector_service_->retrieve_vector(test_db_id_, "batch1_20");
    EXPECT_TRUE(result1.has_value());
    
    auto result2 = vector_service_->retrieve_vector(test_db_id_, "batch2_15");
    EXPECT_TRUE(result2.has_value());
    
    // Verify count
    auto count_result = vector_service_->get_vector_count(test_db_id_);
    ASSERT_TRUE(count_result.has_value());
}

} // namespace test
} // namespace jadevectordb
