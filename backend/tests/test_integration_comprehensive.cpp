#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>

#include "services/vector_storage.h"
#include "services/database_service.h"
#include "services/similarity_search.h"
#include "services/database_layer.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;
using ::testing::Eq;

// Integration test for full system functionality
class FullSystemIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services for integration testing
        logging::LoggerManager::initialize();
        logger_ = logging::LoggerManager::get_logger("FullSystemIntegrationTest");
        
        db_service_ = std::make_unique<DatabaseService>();
        vector_storage_service_ = std::make_unique<VectorStorageService>();
        similarity_search_service_ = std::make_unique<SimilaritySearchService>();
        
        // Initialize services
        EXPECT_TRUE(db_service_->initialize().has_value());
        EXPECT_TRUE(vector_storage_service_->initialize().has_value());
        EXPECT_TRUE(similarity_search_service_->initialize().has_value());
    }

    void TearDown() override {
        // Cleanup
        if (db_id_.size() > 0) {
            db_service_->delete_database(db_id_);
        }
        similarity_search_service_.reset();
        vector_storage_service_.reset();
        db_service_.reset();
    }

    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
    std::shared_ptr<logging::Logger> logger_;
    std::string db_id_;
};

// Test database creation and management
TEST_F(FullSystemIntegrationTest, DatabaseCreateAndManage) {
    LOG_INFO(logger_, "Starting database create and manage test");
    
    // Create a database
    DatabaseCreationParams params;
    params.name = "integration_test_db";
    params.description = "Database for integration testing";
    params.vectorDimension = 128;
    params.indexType = "FLAT";
    params.config = {{"max_vectors", "1000000"}};
    
    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database";
    
    db_id_ = create_result.value();
    LOG_DEBUG(logger_, "Created database with ID: " << db_id_);
    
    // Verify database exists
    auto exists_result = db_service_->database_exists(db_id_);
    ASSERT_TRUE(exists_result.has_value());
    ASSERT_TRUE(exists_result.value());
    
    // Get database details
    auto get_result = db_service_->get_database(db_id_);
    ASSERT_TRUE(get_result.has_value());
    
    Database retrieved_db = get_result.value();
    EXPECT_EQ(retrieved_db.name, params.name);
    EXPECT_EQ(retrieved_db.description, params.description);
    EXPECT_EQ(retrieved_db.vectorDimension, params.vectorDimension);
    
    LOG_INFO(logger_, "Database create and manage test completed successfully");
}

// Test vector storage and retrieval
TEST_F(FullSystemIntegrationTest, VectorStorageAndRetrieval) {
    LOG_INFO(logger_, "Starting vector storage and retrieval test");
    
    // First create a database
    DatabaseCreationParams params;
    params.name = "vector_storage_test_db";
    params.description = "Database for vector storage testing";
    params.vectorDimension = 16;
    params.indexType = "FLAT";
    
    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for vector test";
    
    db_id_ = create_result.value();
    LOG_DEBUG(logger_, "Created database with ID: " << db_id_);
    
    // Create some test vectors
    std::vector<Vector> test_vectors;
    for (int i = 0; i < 10; ++i) {
        Vector v;
        v.id = "test_vector_" + std::to_string(i);
        
        // Create a simple pattern for vectors to make similarity search meaningful
        for (int j = 0; j < 16; ++j) {
            v.values.push_back(static_cast<float>(i + j * 0.1));
        }
        
        // Add some metadata
        v.metadata["category"] = "test";
        v.metadata["index"] = std::to_string(i);
        v.metadata["timestamp"] = std::to_string(std::time(nullptr));
        
        test_vectors.push_back(v);
    }
    
    // Store vectors one by one
    for (const auto& vector : test_vectors) {
        auto store_result = vector_storage_service_->store_vector(db_id_, vector);
        EXPECT_TRUE(store_result.has_value()) << "Failed to store vector: " << vector.id;
    }
    
    LOG_DEBUG(logger_, "Stored " << test_vectors.size() << " vectors");
    
    // Verify vectors were stored by retrieving them
    for (const auto& original_vector : test_vectors) {
        auto retrieve_result = vector_storage_service_->retrieve_vector(db_id_, original_vector.id);
        ASSERT_TRUE(retrieve_result.has_value()) << "Failed to retrieve vector: " << original_vector.id;
        
        Vector retrieved_vector = retrieve_result.value();
        EXPECT_EQ(retrieved_vector.id, original_vector.id);
        ASSERT_EQ(retrieved_vector.values.size(), original_vector.values.size());
        
        // Check that values are approximately equal
        for (size_t i = 0; i < original_vector.values.size(); ++i) {
            EXPECT_NEAR(retrieved_vector.values[i], original_vector.values[i], 1e-6);
        }
    }
    
    // Test batch storage
    std::vector<Vector> batch_vectors;
    for (int i = 10; i < 15; ++i) {
        Vector v;
        v.id = "batch_vector_" + std::to_string(i);
        for (int j = 0; j < 16; ++j) {
            v.values.push_back(static_cast<float>(i + j * 0.15));
        }
        v.metadata["category"] = "batch_test";
        v.metadata["index"] = std::to_string(i);
        batch_vectors.push_back(v);
    }
    
    auto batch_store_result = vector_storage_service_->batch_store_vectors(db_id_, batch_vectors);
    EXPECT_TRUE(batch_store_result.has_value());
    
    LOG_INFO(logger_, "Vector storage and retrieval test completed successfully");
}

// Test similarity search functionality
TEST_F(FullSystemIntegrationTest, SimilaritySearch) {
    LOG_INFO(logger_, "Starting similarity search test");
    
    // First create a database
    DatabaseCreationParams params;
    params.name = "similarity_search_test_db";
    params.description = "Database for similarity search testing";
    params.vectorDimension = 8;
    params.indexType = "FLAT"; // Use FLAT index for predictable results
    
    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for similarity test";
    
    db_id_ = create_result.value();
    LOG_DEBUG(logger_, "Created database with ID: " << db_id_);
    
    // Create and store test vectors with known relationships
    std::vector<Vector> test_vectors;
    
    // Create base vector
    Vector base_vector;
    base_vector.id = "base_vector";
    for (int i = 0; i < 8; ++i) {
        base_vector.values.push_back(1.0f);
    }
    test_vectors.push_back(base_vector);
    
    // Create similar vectors (small differences)
    for (int i = 1; i <= 3; ++i) {
        Vector similar_vector;
        similar_vector.id = "similar_vector_" + std::to_string(i);
        for (int j = 0; j < 8; ++j) {
            similar_vector.values.push_back(1.0f + (i * 0.1f));  // Small variation
        }
        test_vectors.push_back(similar_vector);
    }
    
    // Create dissimilar vectors
    for (int i = 1; i <= 3; ++i) {
        Vector dissimilar_vector;
        dissimilar_vector.id = "dissimilar_vector_" + std::to_string(i);
        for (int j = 0; j < 8; ++j) {
            dissimilar_vector.values.push_back(0.1f * i);  // Very different
        }
        test_vectors.push_back(dissimilar_vector);
    }
    
    // Store all vectors
    for (const auto& vector : test_vectors) {
        auto store_result = vector_storage_service_->store_vector(db_id_, vector);
        EXPECT_TRUE(store_result.has_value()) << "Failed to store vector: " << vector.id;
    }
    
    LOG_DEBUG(logger_, "Stored " << test_vectors.size() << " vectors for similarity search");
    
    // Perform similarity search using the base vector as query
    SearchParams search_params;
    search_params.top_k = 5;
    search_params.threshold = 0.5;  // Minimum similarity threshold
    search_params.include_metadata = true;
    search_params.include_vector_data = true;
    
    auto search_result = similarity_search_service_->similarity_search(db_id_, base_vector, search_params);
    ASSERT_TRUE(search_result.has_value()) << "Similarity search failed";
    
    auto results = search_result.value();
    LOG_DEBUG(logger_, "Search returned " << results.size() << " results");
    
    // Verify that similar vectors are returned with higher similarity scores
    ASSERT_GT(results.size(), 0) << "Expected at least one result";
    
    // The base vector should be most similar to itself (if stored and searched for)
    // or the most similar vectors should have the highest scores
    float last_score = 2.0f; // Start with a value higher than possible similarity
    for (const auto& result : results) {
        EXPECT_LE(result.similarity_score, last_score) << "Results should be in descending similarity order";
        last_score = result.similarity_score;
        LOG_DEBUG(logger_, "Result: " << result.vector_id << " with similarity: " << result.similarity_score);
    }
    
    LOG_INFO(logger_, "Similarity search test completed successfully");
}

// Test concurrent operations
TEST_F(FullSystemIntegrationTest, ConcurrentOperations) {
    LOG_INFO(logger_, "Starting concurrent operations test");
    
    // Create a database
    DatabaseCreationParams params;
    params.name = "concurrent_test_db";
    params.description = "Database for concurrent operations testing";
    params.vectorDimension = 4;
    params.indexType = "FLAT";
    
    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for concurrent test";
    
    db_id_ = create_result.value();
    
    const int num_threads = 4;
    std::vector<std::thread> threads;
    
    // Create a promise/future for each thread to collect results
    std::vector<std::future<bool>> futures;
    std::vector<std::promise<bool>> promises(num_threads);
    
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    
    // Launch multiple threads performing different operations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            bool thread_success = true;
            
            try {
                // Each thread performs multiple operations
                for (int i = 0; i < 5; ++i) {
                    Vector v;
                    v.id = "thread_" + std::to_string(t) + "_vector_" + std::to_string(i);
                    
                    for (int j = 0; j < 4; ++j) {
                        v.values.push_back(static_cast<float>((t + 1) * (i + 1) + j * 0.1));
                    }
                    
                    // Store vector
                    auto store_result = vector_storage_service_->store_vector(db_id_, v);
                    if (!store_result.has_value()) {
                        thread_success = false;
                        break;
                    }
                    
                    // Retrieve vector
                    auto retrieve_result = vector_storage_service_->retrieve_vector(db_id_, v.id);
                    if (!retrieve_result.has_value()) {
                        thread_success = false;
                        break;
                    }
                    
                    // Verify retrieved vector matches original
                    Vector retrieved = retrieve_result.value();
                    if (retrieved.id != v.id || retrieved.values.size() != v.values.size()) {
                        thread_success = false;
                        break;
                    }
                }
            } catch (const std::exception& e) {
                thread_success = false;
            }
            
            if (thread_success) {
                success_count++;
            } else {
                failure_count++;
            }
            
            // Set promise value
            promises[t].set_value(thread_success);
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Check results
    EXPECT_GT(success_count.load(), 0) << "At least some threads should succeed";
    LOG_INFO(logger_, "Concurrent operations test completed. Success: " << 
             success_count.load() << ", Failures: " << failure_count.load());
}

// Test error handling across the system
TEST_F(FullSystemIntegrationTest, ErrorHandling) {
    LOG_INFO(logger_, "Starting error handling test");
    
    // Test operations on non-existent database
    Vector test_vector;
    test_vector.id = "should_fail_vector";
    test_vector.values = {1.0f, 2.0f, 3.0f, 4.0f};
    
    std::string fake_db_id = "non_existent_database_id";
    
    // These operations should fail gracefully
    auto store_result = vector_storage_service_->store_vector(fake_db_id, test_vector);
    EXPECT_FALSE(store_result.has_value()) << "Store operation should fail for non-existent database";
    
    auto retrieve_result = vector_storage_service_->retrieve_vector(fake_db_id, test_vector.id);
    EXPECT_FALSE(retrieve_result.has_value()) << "Retrieve operation should fail for non-existent database";
    
    auto search_result = similarity_search_service_->similarity_search(fake_db_id, test_vector, SearchParams());
    EXPECT_FALSE(search_result.has_value()) << "Search operation should fail for non-existent database";
    
    LOG_INFO(logger_, "Error handling test completed successfully");
}

// Performance test to measure operations under load
TEST_F(FullSystemIntegrationTest, PerformanceTest) {
    LOG_INFO(logger_, "Starting performance test");
    
    // Create a database for performance testing
    DatabaseCreationParams params;
    params.name = "performance_test_db";
    params.description = "Database for performance testing";
    params.vectorDimension = 64;
    params.indexType = "FLAT";
    
    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value()) << "Failed to create database for performance test";
    
    db_id_ = create_result.value();
    
    // Measure time to store multiple vectors
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int num_vectors = 100;
    for (int i = 0; i < num_vectors; ++i) {
        Vector v;
        v.id = "perf_vector_" + std::to_string(i);
        
        for (int j = 0; j < 64; ++j) {
            v.values.push_back(static_cast<float>((i + 1) * 0.01 + j * 0.02));
        }
        
        auto result = vector_storage_service_->store_vector(db_id_, v);
        EXPECT_TRUE(result.has_value()) << "Failed to store performance test vector: " << v.id;
    }
    
    auto store_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    
    LOG_DEBUG(logger_, "Stored " << num_vectors << " vectors in " << store_duration.count() << " ms");
    
    // Measure time to perform multiple searches
    Vector query_vector;
    query_vector.id = "perf_query";
    for (int j = 0; j < 64; ++j) {
        query_vector.values.push_back(0.5f);
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    const int num_searches = 10;
    for (int i = 0; i < num_searches; ++i) {
        SearchParams search_params;
        search_params.top_k = 5;
        auto result = similarity_search_service_->similarity_search(db_id_, query_vector, search_params);
        EXPECT_TRUE(result.has_value()) << "Search " << i << " failed";
    }
    
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    
    LOG_DEBUG(logger_, "Performed " << num_searches << " searches in " << search_duration.count() << " ms");
    
    LOG_INFO(logger_, "Performance test completed successfully");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}