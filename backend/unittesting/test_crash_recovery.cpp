#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <cstring>

class CrashRecoveryTest : public ::testing::Test {
protected:
    std::string test_storage_path_;
    
    void SetUp() override {
        test_storage_path_ = "/tmp/jade_crash_recovery_test";
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
    
    // Simulate crash by abruptly destroying store without flush
    void simulate_crash(jadevectordb::MemoryMappedVectorStore* store) {
        // Just destroy the object without calling flush
        // Simulates process termination
        delete store;
    }
    
    // Corrupt a file to simulate partial write during crash
    void corrupt_file(const std::string& file_path, size_t offset, size_t bytes) {
        std::fstream file(file_path, std::ios::in | std::ios::out | std::ios::binary);
        if (file.is_open()) {
            file.seekp(offset);
            std::vector<char> garbage(bytes, 0xFF);
            file.write(garbage.data(), bytes);
            file.close();
        }
    }
};

// Test 1: Recovery after ungraceful shutdown without flush
TEST_F(CrashRecoveryTest, RecoveryWithoutFlush) {
    const int DIMENSION = 128;
    const int NUM_VECTORS = 50;
    const std::string db_id = "crash_test_db";
    
    // Phase 1: Store vectors with flush (baseline)
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < NUM_VECTORS; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        // Explicit flush
        store->flush_all(true);
        delete store;
    }
    
    // Phase 2: Add more vectors WITHOUT flush, then simulate crash
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        for (int i = NUM_VECTORS; i < NUM_VECTORS + 20; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        // Simulate crash - no flush!
        simulate_crash(store);
    }
    
    // Phase 3: Recovery - verify flushed data intact
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        // Original 50 vectors should be intact
        for (int i = 0; i < NUM_VECTORS; i++) {
            auto result = store->retrieve_vector(db_id, "vec_" + std::to_string(i));
            ASSERT_TRUE(result.has_value()) << "Vector " << i << " should exist";
            
            auto expected = generate_random_vector(DIMENSION, i);
            for (size_t j = 0; j < expected.size(); j++) {
                EXPECT_FLOAT_EQ((*result)[j], expected[j]);
            }
        }
        
        // Last 20 vectors may or may not exist (depends on OS page cache)
        // But file should still be valid and readable
        size_t count = store->get_vector_count(db_id);
        EXPECT_GE(count, NUM_VECTORS);
        EXPECT_LE(count, NUM_VECTORS + 20);
        
        delete store;
    }
}

// Test 2: Recovery with periodic flush
TEST_F(CrashRecoveryTest, RecoveryWithPeriodicFlush) {
    const int DIMENSION = 256;
    const std::string db_id = "periodic_flush_db";
    
    // Store vectors with periodic flushing
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < 100; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
            
            // Flush every 10 vectors
            if (i % 10 == 9) {
                store->flush(db_id, true);
            }
        }
        
        // Simulate crash after last flush
        simulate_crash(store);
    }
    
    // Recovery
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        // Most vectors should be recoverable (at least 90)
        size_t recovered = 0;
        for (int i = 0; i < 100; i++) {
            auto result = store->retrieve_vector(db_id, "vec_" + std::to_string(i));
            if (result.has_value()) {
                recovered++;
            }
        }
        
        EXPECT_GE(recovered, 90) << "Should recover most vectors with periodic flush";
        
        delete store;
    }
}

// Test 3: File header integrity check
TEST_F(CrashRecoveryTest, HeaderIntegrityCheck) {
    const int DIMENSION = 128;
    const std::string db_id = "header_test_db";
    
    // Create and populate database
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < 10; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        store->flush_all(true);
        delete store;
    }
    
    // Verify header can be read correctly
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        EXPECT_EQ(store->get_dimension(db_id), DIMENSION);
        EXPECT_EQ(store->get_vector_count(db_id), 10);
        EXPECT_TRUE(store->has_database(db_id));
        
        delete store;
    }
}

// Test 4: Multiple databases crash recovery
TEST_F(CrashRecoveryTest, MultipleDatabasesCrashRecovery) {
    const int NUM_DATABASES = 5;
    std::vector<std::string> db_ids;
    std::vector<int> dimensions = {64, 128, 256, 512, 1024};
    
    // Phase 1: Create multiple databases with varying flush patterns
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
            std::string db_id = "crash_db_" + std::to_string(db_idx);
            db_ids.push_back(db_id);
            
            store->create_vector_file(db_id, dimensions[db_idx]);
            
            for (int i = 0; i < 20; i++) {
                std::vector<float> vec = generate_random_vector(dimensions[db_idx], i);
                store->store_vector(db_id, "vec_" + std::to_string(i), vec);
            }
            
            // Flush some databases but not others
            if (db_idx % 2 == 0) {
                store->flush(db_id, true);
            }
        }
        
        // Simulate crash
        simulate_crash(store);
    }
    
    // Phase 2: Recovery - verify all databases still accessible
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
            const std::string& db_id = db_ids[db_idx];
            
            EXPECT_TRUE(store->has_database(db_id)) << "Database " << db_idx << " should exist";
            EXPECT_EQ(store->get_dimension(db_id), dimensions[db_idx]);
            
            // Flushed databases should have all vectors
            if (db_idx % 2 == 0) {
                EXPECT_EQ(store->get_vector_count(db_id), 20);
            } else {
                // Unflushed databases may have lost recent data
                size_t count = store->get_vector_count(db_id);
                EXPECT_LE(count, 20);
            }
        }
        
        delete store;
    }
}

// Test 5: Concurrent operations during "crash"
TEST_F(CrashRecoveryTest, ConcurrentOperationsCrash) {
    const int DIMENSION = 128;
    const std::string db_id = "concurrent_crash_db";
    
    // Phase 1: Store baseline data
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < 100; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        store->flush_all(true);
        delete store;
    }
    
    // Phase 2: Simulate concurrent writes then crash
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; t++) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < 25; i++) {
                    std::string id = "thread_" + std::to_string(t) + "_vec_" + std::to_string(i);
                    std::vector<float> vec = generate_random_vector(DIMENSION, t * 1000 + i);
                    store->store_vector(db_id, id, vec);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Simulate crash without flush
        simulate_crash(store);
    }
    
    // Phase 3: Recovery
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        // Original 100 vectors should be intact
        size_t original_recovered = 0;
        for (int i = 0; i < 100; i++) {
            auto result = store->retrieve_vector(db_id, "vec_" + std::to_string(i));
            if (result.has_value()) {
                original_recovered++;
            }
        }
        
        EXPECT_EQ(original_recovered, 100) << "All flushed vectors should be recoverable";
        
        // Some of the concurrent writes may have persisted
        size_t total_count = store->get_vector_count(db_id);
        EXPECT_GE(total_count, 100);
        
        delete store;
    }
}

// Test 6: Delete operations during crash
TEST_F(CrashRecoveryTest, DeleteOperationsCrash) {
    const int DIMENSION = 128;
    const std::string db_id = "delete_crash_db";
    
    // Phase 1: Store and flush
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < 50; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        store->flush_all(true);
        delete store;
    }
    
    // Phase 2: Delete some vectors without flush
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        for (int i = 0; i < 25; i++) {
            store->delete_vector(db_id, "vec_" + std::to_string(i));
        }
        
        // Crash without flush
        simulate_crash(store);
    }
    
    // Phase 3: Recovery - deletes may or may not have persisted
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        size_t count = store->get_vector_count(db_id);
        EXPECT_LE(count, 50);
        EXPECT_GE(count, 25); // At least the non-deleted ones
        
        // File should still be valid
        EXPECT_TRUE(store->has_database(db_id));
        EXPECT_EQ(store->get_dimension(db_id), DIMENSION);
        
        delete store;
    }
}

// Test 7: Verify data integrity after recovery
TEST_F(CrashRecoveryTest, DataIntegrityAfterRecovery) {
    const int DIMENSION = 512;
    const std::string db_id = "integrity_db";
    
    // Phase 1: Store with known values
    std::vector<std::vector<float>> expected_vectors;
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < 100; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i * 42);
            expected_vectors.push_back(vec);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        store->flush_all(true);
        delete store;
    }
    
    // Phase 2: Simulate crash scenario
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        // Add more data
        for (int i = 100; i < 120; i++) {
            std::vector<float> vec = generate_random_vector(DIMENSION, i * 42);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        simulate_crash(store);
    }
    
    // Phase 3: Verify flushed data has perfect integrity
    {
        auto store = new jadevectordb::MemoryMappedVectorStore(test_storage_path_);
        
        for (int i = 0; i < 100; i++) {
            auto result = store->retrieve_vector(db_id, "vec_" + std::to_string(i));
            ASSERT_TRUE(result.has_value()) << "Vector " << i << " should exist";
            
            const auto& expected = expected_vectors[i];
            ASSERT_EQ(result->size(), expected.size());
            
            // Verify exact match
            for (size_t j = 0; j < expected.size(); j++) {
                EXPECT_FLOAT_EQ((*result)[j], expected[j]) 
                    << "Mismatch at vector " << i << " index " << j;
            }
        }
        
        delete store;
    }
}
