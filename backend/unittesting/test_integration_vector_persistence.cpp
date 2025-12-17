#include <gtest/gtest.h>
#include "services/database_layer.h"
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <thread>
#include <atomic>
#include <random>

class VectorPersistenceIntegrationTest : public ::testing::Test {
protected:
    std::string test_storage_path_;
    
    void SetUp() override {
        test_storage_path_ = "/tmp/jade_persistence_integration_test";
        std::filesystem::remove_all(test_storage_path_);
        std::filesystem::create_directories(test_storage_path_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_storage_path_);
    }
    
    std::vector<float> generate_random_vector(size_t dimension, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        std::vector<float> vec(dimension);
        for (size_t i = 0; i < dimension; i++) {
            vec[i] = dis(gen);
        }
        return vec;
    }
};

// Test 1: Basic Persistence - Store, Flush, Restart, Retrieve
TEST_F(VectorPersistenceIntegrationTest, StoreFlushRestartRetrieve) {
    const int DIMENSION = 128;
    const int NUM_VECTORS = 100;
    
    // Phase 1: Store vectors and flush
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        // Create database
        jadevectordb::Database db;
        db.name = "test_db";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        ASSERT_TRUE(create_result.has_value());
        std::string db_id = *create_result;
        
        // Store vectors
        for (int i = 0; i < NUM_VECTORS; i++) {
            jadevectordb::Vector vec;
            vec.id = "vec_" + std::to_string(i);
            vec.values = generate_random_vector(DIMENSION, i);
            vec.databaseId = db_id;
            
            auto store_result = persistence->store_vector(db_id, vec);
            ASSERT_TRUE(store_result.has_value()) << "Failed to store vector " << i;
        }
        
        // Flush all data
        persistence->flush_all();
    }
    
    // Phase 2: Create new persistence layer (simulates restart)
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        // Database metadata is in-memory, so we need to recreate it
        // In production, this would be loaded from SQLite
        jadevectordb::Database db;
        db.name = "test_db";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        ASSERT_TRUE(create_result.has_value());
        std::string db_id = *create_result;
        
        // Retrieve and verify vectors
        for (int i = 0; i < NUM_VECTORS; i++) {
            std::string vec_id = "vec_" + std::to_string(i);
            auto retrieve_result = persistence->retrieve_vector(db_id, vec_id);
            ASSERT_TRUE(retrieve_result.has_value()) << "Failed to retrieve vector " << i;
            
            // Verify values match
            auto expected = generate_random_vector(DIMENSION, i);
            ASSERT_EQ(retrieve_result->values.size(), expected.size());
            for (size_t j = 0; j < expected.size(); j++) {
                EXPECT_FLOAT_EQ(retrieve_result->values[j], expected[j]);
            }
        }
    }
}

// Test 2: Multiple Databases with Different Dimensions
TEST_F(VectorPersistenceIntegrationTest, MultipleDatabasesPersistence) {
    struct DatabaseSpec {
        std::string name;
        int dimension;
        int num_vectors;
    };
    
    std::vector<DatabaseSpec> databases = {
        {"db_small", 64, 50},
        {"db_medium", 256, 30},
        {"db_large", 1024, 20}
    };
    
    std::vector<std::string> db_ids;
    
    // Phase 1: Create databases and store vectors
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        for (const auto& spec : databases) {
            jadevectordb::Database db;
            db.name = spec.name;
            db.vectorDimension = spec.dimension;
            
            auto create_result = persistence->create_database(db);
            ASSERT_TRUE(create_result.has_value());
            std::string db_id = *create_result;
            db_ids.push_back(db_id);
            
            // Store vectors
            for (int i = 0; i < spec.num_vectors; i++) {
                jadevectordb::Vector vec;
                vec.id = spec.name + "_vec_" + std::to_string(i);
                vec.values = generate_random_vector(spec.dimension, i * 1000);
                vec.databaseId = db_id;
                
                auto store_result = persistence->store_vector(db_id, vec);
                ASSERT_TRUE(store_result.has_value());
            }
        }
        
        persistence->flush_all();
    }
    
    // Phase 2: Restart and verify all databases
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        for (size_t db_idx = 0; db_idx < databases.size(); db_idx++) {
            const auto& spec = databases[db_idx];
            
            // Recreate database metadata
            jadevectordb::Database db;
            db.name = spec.name;
            db.vectorDimension = spec.dimension;
            
            auto create_result = persistence->create_database(db);
            ASSERT_TRUE(create_result.has_value());
            std::string db_id = *create_result;
            
            // Verify vector count
            auto count_result = persistence->get_vector_count(db_id);
            ASSERT_TRUE(count_result.has_value());
            EXPECT_EQ(*count_result, spec.num_vectors);
            
            // Sample verify a few vectors
            for (int i = 0; i < std::min(5, spec.num_vectors); i++) {
                std::string vec_id = spec.name + "_vec_" + std::to_string(i);
                auto retrieve_result = persistence->retrieve_vector(db_id, vec_id);
                ASSERT_TRUE(retrieve_result.has_value());
                EXPECT_EQ(retrieve_result->values.size(), spec.dimension);
            }
        }
    }
}

// Test 3: Concurrent Access from Multiple Threads
TEST_F(VectorPersistenceIntegrationTest, ConcurrentAccess) {
    const int DIMENSION = 128;
    const int NUM_THREADS = 8;
    const int VECTORS_PER_THREAD = 50;
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        test_storage_path_, nullptr, nullptr, nullptr);
    
    // Create database
    jadevectordb::Database db;
    db.name = "concurrent_test";
    db.vectorDimension = DIMENSION;
    
    auto create_result = persistence->create_database(db);
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = *create_result;
    
    // Concurrent write phase
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < VECTORS_PER_THREAD; i++) {
                jadevectordb::Vector vec;
                vec.id = "thread_" + std::to_string(t) + "_vec_" + std::to_string(i);
                vec.values = generate_random_vector(DIMENSION, t * 10000 + i);
                vec.databaseId = db_id;
                
                auto result = persistence->store_vector(db_id, vec);
                if (result.has_value()) {
                    success_count++;
                } else {
                    failure_count++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count.load(), NUM_THREADS * VECTORS_PER_THREAD);
    EXPECT_EQ(failure_count.load(), 0);
    
    // Verify count
    auto count_result = persistence->get_vector_count(db_id);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_EQ(*count_result, NUM_THREADS * VECTORS_PER_THREAD);
    
    // Concurrent read phase
    threads.clear();
    success_count = 0;
    failure_count = 0;
    
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < VECTORS_PER_THREAD; i++) {
                std::string vec_id = "thread_" + std::to_string(t) + "_vec_" + std::to_string(i);
                auto result = persistence->retrieve_vector(db_id, vec_id);
                if (result.has_value()) {
                    success_count++;
                } else {
                    failure_count++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count.load(), NUM_THREADS * VECTORS_PER_THREAD);
    EXPECT_EQ(failure_count.load(), 0);
}

// Test 4: Update and Delete Persistence
TEST_F(VectorPersistenceIntegrationTest, UpdateDeletePersistence) {
    const int DIMENSION = 256;
    
    // Phase 1: Store initial vectors
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        jadevectordb::Database db;
        db.name = "update_test";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        ASSERT_TRUE(create_result.has_value());
        std::string db_id = *create_result;
        
        for (int i = 0; i < 10; i++) {
            jadevectordb::Vector vec;
            vec.id = "vec_" + std::to_string(i);
            vec.values = generate_random_vector(DIMENSION, i);
            vec.databaseId = db_id;
            
            persistence->store_vector(db_id, vec);
        }
        
        persistence->flush_all();
    }
    
    // Phase 2: Update vectors
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        jadevectordb::Database db;
        db.name = "update_test";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        std::string db_id = *create_result;
        
        // Update half the vectors
        for (int i = 0; i < 5; i++) {
            jadevectordb::Vector vec;
            vec.id = "vec_" + std::to_string(i);
            vec.values = generate_random_vector(DIMENSION, i + 1000);
            vec.databaseId = db_id;
            
            auto result = persistence->update_vector(db_id, vec);
            ASSERT_TRUE(result.has_value());
        }
        
        // Delete the other half
        for (int i = 5; i < 10; i++) {
            std::string vec_id = "vec_" + std::to_string(i);
            auto result = persistence->delete_vector(db_id, vec_id);
            ASSERT_TRUE(result.has_value());
        }
        
        persistence->flush_all();
    }
    
    // Phase 3: Verify updates and deletes persisted
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        jadevectordb::Database db;
        db.name = "update_test";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        std::string db_id = *create_result;
        
        // Verify updated vectors have new values
        for (int i = 0; i < 5; i++) {
            std::string vec_id = "vec_" + std::to_string(i);
            auto result = persistence->retrieve_vector(db_id, vec_id);
            ASSERT_TRUE(result.has_value());
            
            auto expected = generate_random_vector(DIMENSION, i + 1000);
            for (size_t j = 0; j < expected.size(); j++) {
                EXPECT_FLOAT_EQ(result->values[j], expected[j]);
            }
        }
        
        // Verify deleted vectors don't exist
        for (int i = 5; i < 10; i++) {
            std::string vec_id = "vec_" + std::to_string(i);
            auto result = persistence->retrieve_vector(db_id, vec_id);
            EXPECT_FALSE(result.has_value());
        }
        
        // Verify count
        auto count_result = persistence->get_vector_count(db_id);
        ASSERT_TRUE(count_result.has_value());
        EXPECT_EQ(*count_result, 5);
    }
}

// Test 5: Batch Operations Persistence
TEST_F(VectorPersistenceIntegrationTest, BatchOperationsPersistence) {
    const int DIMENSION = 512;
    const int BATCH_SIZE = 100;
    
    // Phase 1: Batch store
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        jadevectordb::Database db;
        db.name = "batch_test";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        ASSERT_TRUE(create_result.has_value());
        std::string db_id = *create_result;
        
        // Prepare batch
        std::vector<jadevectordb::Vector> vectors;
        for (int i = 0; i < BATCH_SIZE; i++) {
            jadevectordb::Vector vec;
            vec.id = "batch_vec_" + std::to_string(i);
            vec.values = generate_random_vector(DIMENSION, i * 100);
            vec.databaseId = db_id;
            vectors.push_back(vec);
        }
        
        // Batch store
        auto result = persistence->batch_store_vectors(db_id, vectors);
        ASSERT_TRUE(result.has_value());
        
        persistence->flush_all();
    }
    
    // Phase 2: Restart and batch retrieve
    {
        auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
            test_storage_path_, nullptr, nullptr, nullptr);
        
        jadevectordb::Database db;
        db.name = "batch_test";
        db.vectorDimension = DIMENSION;
        
        auto create_result = persistence->create_database(db);
        std::string db_id = *create_result;
        
        // Prepare IDs for batch retrieve
        std::vector<std::string> ids;
        for (int i = 0; i < BATCH_SIZE; i++) {
            ids.push_back("batch_vec_" + std::to_string(i));
        }
        
        // Batch retrieve
        auto result = persistence->retrieve_vectors(db_id, ids);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->size(), BATCH_SIZE);
        
        // Verify a sample
        for (int i = 0; i < 10; i++) {
            auto expected = generate_random_vector(DIMENSION, i * 100);
            for (size_t j = 0; j < expected.size(); j++) {
                EXPECT_FLOAT_EQ((*result)[i].values[j], expected[j]);
            }
        }
    }
}

// Test 6: Database Deletion Cleanup
TEST_F(VectorPersistenceIntegrationTest, DatabaseDeletionCleanup) {
    const int DIMENSION = 128;
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        test_storage_path_, nullptr, nullptr, nullptr);
    
    // Create and populate database
    jadevectordb::Database db;
    db.name = "temp_db";
    db.vectorDimension = DIMENSION;
    
    auto create_result = persistence->create_database(db);
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = *create_result;
    
    // Store vectors
    for (int i = 0; i < 50; i++) {
        jadevectordb::Vector vec;
        vec.id = "vec_" + std::to_string(i);
        vec.values = generate_random_vector(DIMENSION, i);
        vec.databaseId = db_id;
        
        persistence->store_vector(db_id, vec);
    }
    
    persistence->flush_all();
    
    // Verify database directory exists
    std::string db_dir = test_storage_path_ + "/" + db_id;
    EXPECT_TRUE(std::filesystem::exists(db_dir));
    
    // Delete database
    auto delete_result = persistence->delete_database(db_id);
    ASSERT_TRUE(delete_result.has_value());
    
    // Verify directory is removed
    EXPECT_FALSE(std::filesystem::exists(db_dir));
    
    // Verify vector retrieval fails
    auto retrieve_result = persistence->retrieve_vector(db_id, "vec_0");
    EXPECT_FALSE(retrieve_result.has_value());
}

// Test 7: Stress Test - Many Small Operations
TEST_F(VectorPersistenceIntegrationTest, StressTestManyOperations) {
    const int DIMENSION = 64;
    const int NUM_OPERATIONS = 1000;
    
    auto persistence = std::make_unique<jadevectordb::PersistentDatabasePersistence>(
        test_storage_path_, nullptr, nullptr, nullptr);
    
    jadevectordb::Database db;
    db.name = "stress_test";
    db.vectorDimension = DIMENSION;
    
    auto create_result = persistence->create_database(db);
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = *create_result;
    
    // Interleaved store, retrieve, update operations
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        jadevectordb::Vector vec;
        vec.id = "stress_vec_" + std::to_string(i % 100);
        vec.values = generate_random_vector(DIMENSION, i);
        vec.databaseId = db_id;
        
        if (i % 3 == 0) {
            // Store
            persistence->store_vector(db_id, vec);
        } else if (i % 3 == 1) {
            // Retrieve
            persistence->retrieve_vector(db_id, vec.id);
        } else {
            // Update
            persistence->update_vector(db_id, vec);
        }
        
        // Periodic flush
        if (i % 100 == 0) {
            persistence->flush_database(db_id);
        }
    }
    
    persistence->flush_all();
    
    // Verify at least some vectors exist
    auto count_result = persistence->get_vector_count(db_id);
    ASSERT_TRUE(count_result.has_value());
    EXPECT_GT(*count_result, 0);
}
