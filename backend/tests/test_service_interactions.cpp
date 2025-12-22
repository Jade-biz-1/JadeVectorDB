#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <future>

#include "services/vector_storage.h"
#include "services/database_service.h"
#include "services/similarity_search.h"
#include "services/index_service.h"
#include "services/lifecycle_service.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;
using ::testing::Eq;

// Additional comprehensive integration test focusing on service interactions
class ServiceInteractionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services for integration testing
        logging::LoggerManager::initialize();
        logger_ = logging::LoggerManager::get_logger("ServiceInteractionIntegrationTest");
        
        db_service_ = std::make_unique<DatabaseService>();
        vector_storage_service_ = std::make_unique<VectorStorageService>();
        similarity_search_service_ = std::make_unique<SimilaritySearchService>();
        index_service_ = std::make_unique<IndexService>();
        lifecycle_service_ = std::make_unique<LifecycleService>();
        
        // Initialize services
        EXPECT_TRUE(db_service_->initialize().has_value());
        EXPECT_TRUE(vector_storage_service_->initialize().has_value());
        EXPECT_TRUE(similarity_search_service_->initialize().has_value());
        EXPECT_TRUE(index_service_->initialize().has_value());
        EXPECT_TRUE(lifecycle_service_->initialize().has_value());
    }

    void TearDown() override {
        // Cleanup - try to delete any test databases created
        if (db_id_.size() > 0) {
            db_service_->delete_database(db_id_);
        }
        
        lifecycle_service_.reset();
        index_service_.reset();
        similarity_search_service_.reset();
        vector_storage_service_.reset();
        db_service_.reset();
    }

    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
    std::unique_ptr<IndexService> index_service_;
    std::unique_ptr<LifecycleService> lifecycle_service_;
    std::shared_ptr<logging::Logger> logger_;
    std::string db_id_;
};

// Test complete workflow: create database -> store vectors -> build index -> search -> lifecycle management
TEST_F(ServiceInteractionIntegrationTest, CompleteWorkflowTest) {
    LOG_INFO(logger_, "Starting complete workflow integration test");
    
    // Step 1: Create a database
    DatabaseCreationParams db_params;
    db_params.name = "workflow_integration_test_db";
    db_params.description = "Database for complete workflow integration testing";
    db_params.vectorDimension = 32;
    db_params.indexType = "FLAT"; // Start with simple index type
    
    auto create_result = db_service_->create_database(db_params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for workflow test";
    
    db_id_ = create_result.value();
    LOG_DEBUG(logger_, "Created database with ID: " << db_id_);
    
    // Step 2: Store a set of vectors
    std::vector<Vector> test_vectors;
    for (int i = 0; i < 20; ++i) {
        Vector v;
        v.id = "workflow_vector_" + std::to_string(i);
        
        // Create vectors with a pattern to make similarity search meaningful
        for (int j = 0; j < 32; ++j) {
            if (i < 10) {
                // First 10 vectors will be similar to each other
                v.values.push_back(0.1f * i + 0.01f * j);
            } else {
                // Next 10 vectors will be different
                v.values.push_back(1.0f + 0.05f * i + 0.01f * j);
            }
        }
        
        // Add metadata for filtering tests later
        v.metadata["group"] = (i < 10) ? "group_a" : "group_b";
        v.metadata["index"] = std::to_string(i);
        v.metadata["timestamp"] = std::to_string(std::time(nullptr));
        
        test_vectors.push_back(v);
    }
    
    // Store vectors in batch for efficiency
    auto batch_store_result = vector_storage_service_->batch_store_vectors(db_id_, test_vectors);
    EXPECT_TRUE(batch_store_result.has_value()) << "Batch store failed";
    
    LOG_DEBUG(logger_, "Stored " << test_vectors.size() << " vectors in database");
    
    // Verify all vectors were stored by checking count
    auto count_result = vector_storage_service_->get_vector_count(db_id_);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(count_result.value(), test_vectors.size());
    
    // Step 3: Create an index for the database
    IndexCreationParams index_params;
    index_params.databaseId = db_id_;
    index_params.type = "FLAT"; // Use simple index for test
    index_params.name = "workflow_index";
    index_params.description = "Index for workflow test";
    index_params.config = {{"build_parallel", "false"}};
    
    auto index_create_result = index_service_->create_index(index_params);
    EXPECT_TRUE(index_create_result.has_value()) << "Failed to create index";
    
    std::string index_id = index_create_result.value();
    LOG_DEBUG(logger_, "Created index with ID: " << index_id);
    
    // Verify index exists
    auto index_exists_result = index_service_->index_exists(db_id_, index_id);
    EXPECT_TRUE(index_exists_result.has_value());
    EXPECT_TRUE(index_exists_result.value());
    
    // Step 4: Perform similarity searches
    SearchParams search_params;
    search_params.top_k = 5;
    search_params.threshold = 0.0; // No threshold to see all results
    search_params.include_metadata = true;
    search_params.include_vector_data = false; // Don't include vector data to be efficient
    
    // Search with first vector (should find similar vectors from the same group)
    auto search_result = similarity_search_service_->similarity_search(db_id_, test_vectors[0], search_params);
    ASSERT_TRUE(search_result.has_value()) << "Similarity search failed";
    
    auto results = search_result.value();
    LOG_DEBUG(logger_, "Search returned " << results.size() << " results");
    
    // Verify that we got results
    ASSERT_GT(results.size(), 0);
    
    // Step 5: Test lifecycle configuration and operations
    RetentionPolicy retention_policy;
    retention_policy.max_age_days = 30; // 30 days
    retention_policy.archive_on_expire = false;
    retention_policy.enable_cleanup = false; // Don't actually cleanup in test
    
    auto lifecycle_result = lifecycle_service_->configure_retention_policy(db_id_, 
        std::chrono::hours(retention_policy.max_age_days * 24), 
        retention_policy.archive_on_expire);
    EXPECT_TRUE(lifecycle_result.has_value()) << "Failed to configure retention policy";
    
    LOG_DEBUG(logger_, "Configured retention policy for database: " << db_id_);
    
    // Verify retention policy was set
    auto get_policy_result = lifecycle_service_->get_retention_policy(db_id_);
    EXPECT_TRUE(get_policy_result.has_value());
    
    LOG_INFO(logger_, "Complete workflow integration test completed successfully");
}

// Test error handling across service boundaries
TEST_F(ServiceInteractionIntegrationTest, CrossServiceErrorHandling) {
    LOG_INFO(logger_, "Starting cross-service error handling test");
    
    // Create a valid database first
    DatabaseCreationParams db_params;
    db_params.name = "error_handling_test_db";
    db_params.description = "Database for error handling testing";
    db_params.vectorDimension = 8;
    db_params.indexType = "FLAT";
    
    auto create_result = db_service_->create_database(db_params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for error test";
    
    std::string test_db_id = create_result.value();
    LOG_DEBUG(logger_, "Created database with ID: " << test_db_id);
    
    // Test operations with invalid data to ensure proper error propagation
    Vector invalid_vector;
    invalid_vector.id = "invalid_vector";
    // Leave values empty to trigger validation error
    // (vector dimension doesn't match database dimension)
    
    auto store_result = vector_storage_service_->store_vector(test_db_id, invalid_vector);
    // This should fail because vector has 0 dimensions but database expects 8
    EXPECT_FALSE(store_result.has_value()) << "Store should fail with dimension mismatch";
    
    // Test search with invalid query vector
    Vector invalid_query;
    invalid_query.id = "invalid_query";
    // Leave values empty to trigger validation error
    
    SearchParams search_params;
    search_params.top_k = 5;
    
    auto search_result = similarity_search_service_->similarity_search(test_db_id, invalid_query, search_params);
    EXPECT_FALSE(search_result.has_value()) << "Search should fail with invalid query";
    
    // Test operations on non-existent database across services
    std::string fake_db_id = "fake_database_id_that_does_not_exist";
    
    Vector fake_vector;
    fake_vector.id = "fake_vector";
    for (int i = 0; i < 8; ++i) {
        fake_vector.values.push_back(1.0f);
    }
    
    // These operations should fail gracefully
    auto fake_store_result = vector_storage_service_->store_vector(fake_db_id, fake_vector);
    EXPECT_FALSE(fake_store_result.has_value());
    
    auto fake_search_result = similarity_search_service_->similarity_search(fake_db_id, fake_vector, search_params);
    EXPECT_FALSE(fake_search_result.has_value());
    
    auto fake_index_result = index_service_->create_index({fake_db_id, "FLAT", "test_index", "Test index", {{"build_parallel", "false"}}});
    EXPECT_FALSE(fake_index_result.has_value());
    
    // Clean up
    db_service_->delete_database(test_db_id);
    
    LOG_INFO(logger_, "Cross-service error handling test completed successfully");
}

// Test concurrent access to services
TEST_F(ServiceInteractionIntegrationTest, ConcurrentServiceAccess) {
    LOG_INFO(logger_, "Starting concurrent service access test");
    
    // Create a database for concurrent testing
    DatabaseCreationParams db_params;
    db_params.name = "concurrent_access_test_db";
    db_params.description = "Database for concurrent access testing";
    db_params.vectorDimension = 4;
    db_params.indexType = "FLAT";
    
    auto create_result = db_service_->create_database(db_params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for concurrent test";
    
    std::string test_db_id = create_result.value();
    LOG_DEBUG(logger_, "Created database with ID: " << test_db_id);
    
    const int num_threads = 6; // More threads for comprehensive testing
    std::vector<std::thread> threads;
    
    std::atomic<int> vectors_stored{0};
    std::atomic<int> search_successes{0};
    std::atomic<int> total_operations{0};
    
    // Launch threads that perform various operations simultaneously
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 3; ++i) { // Each thread does 3 operations
                total_operations++;
                
                // Operation 1: Store a vector
                Vector v;
                v.id = "thread_" + std::to_string(t) + "_op_" + std::to_string(i);
                for (int j = 0; j < 4; ++j) {
                    v.values.push_back(static_cast<float>(t + i + j * 0.1));
                }
                
                auto store_result = vector_storage_service_->store_vector(test_db_id, v);
                if (store_result.has_value()) {
                    vectors_stored++;
                }
                
                // Operation 2: Retrieve the stored vector
                auto retrieve_result = vector_storage_service_->retrieve_vector(test_db_id, v.id);
                if (retrieve_result.has_value()) {
                    total_operations++;
                }
                
                // Operation 3: Perform a search (if we have enough vectors)
                if (vectors_stored.load() > 1) {
                    SearchParams sp;
                    sp.top_k = 2;
                    auto search_result = similarity_search_service_->similarity_search(test_db_id, v, sp);
                    if (search_result.has_value()) {
                        search_successes++;
                    }
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify that operations were performed successfully
    EXPECT_GT(vectors_stored.load(), 0) << "At least some vectors should have been stored";
    
    // Count the actual number of vectors in the database
    auto count_result = vector_storage_service_->get_vector_count(test_db_id);
    ASSERT_TRUE(count_result.has_value());
    int actual_count = count_result.value();
    
    LOG_DEBUG(logger_, "Concurrent operations summary - Attempted: " << total_operations.load() 
             << ", Stored: " << vectors_stored.load() 
             << ", Search successes: " << search_successes.load()
             << ", Actual vectors in DB: " << actual_count);
    
    // Clean up
    db_service_->delete_database(test_db_id);
    
    LOG_INFO(logger_, "Concurrent service access test completed successfully");
}

// Test service configuration and parameter validation
TEST_F(ServiceInteractionIntegrationTest, ServiceConfigurationTest) {
    LOG_INFO(logger_, "Starting service configuration test");
    
    // Test database creation with various configurations
    struct TestCase {
        std::string name;
        int dimension;
        std::string index_type;
        bool should_succeed;
    };
    
    std::vector<TestCase> test_cases = {
        {"config_test_1", 16, "FLAT", true},
        {"config_test_2", 32, "HNSW", true}, 
        {"config_test_3", 64, "IVF", true},
        {"config_test_4", 128, "LSH", true},
        {"config_test_5", 0, "FLAT", false},    // Invalid dimension
        {"config_test_6", 1001, "FLAT", false}  // Dimension too large (if there's a limit)
    };
    
    for (const auto& test_case : test_cases) {
        DatabaseCreationParams params;
        params.name = test_case.name;
        params.description = "Configuration test database";
        params.vectorDimension = test_case.dimension;
        params.indexType = test_case.index_type;
        
        auto create_result = db_service_->create_database(params);
        
        if (test_case.should_succeed) {
            ASSERT_TRUE(create_result.has_value()) << "Database creation should succeed for test case: " << test_case.name;
            
            std::string db_id = create_result.value();
            
            // Verify the database has correct configuration
            auto get_result = db_service_->get_database(db_id);
            ASSERT_TRUE(get_result.has_value());
            
            Database db = get_result.value();
            EXPECT_EQ(db.vectorDimension, test_case.dimension);
            EXPECT_EQ(db.indexType, test_case.index_type);
            
            // Clean up
            db_service_->delete_database(db_id);
        } else {
            // For cases that should fail
            EXPECT_FALSE(create_result.has_value()) << "Database creation should fail for test case: " << test_case.name;
        }
    }
    
    LOG_INFO(logger_, "Service configuration test completed successfully");
}
