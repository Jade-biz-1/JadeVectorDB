#include <gtest/gtest.h>
#include "storage/vector_flush_manager.h"
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <thread>
#include <chrono>

class FlushManagerTest : public ::testing::Test {
protected:
    std::string test_storage_path_;
    
    void SetUp() override {
        test_storage_path_ = "/tmp/jade_flush_manager_test";
        std::filesystem::remove_all(test_storage_path_);
        std::filesystem::create_directories(test_storage_path_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_storage_path_);
    }
};

TEST_F(FlushManagerTest, StartAndStop) {
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    jadevectordb::VectorFlushManager manager(store, 2); // 2 second interval
    
    EXPECT_FALSE(manager.is_running());
    
    manager.start();
    EXPECT_TRUE(manager.is_running());
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_TRUE(manager.is_running());
    
    manager.stop();
    EXPECT_FALSE(manager.is_running());
}

TEST_F(FlushManagerTest, PeriodicFlush) {
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    // Create database and store vectors
    store->create_vector_file("db1", 128);
    
    std::vector<float> vec(128, 1.0f);
    store->store_vector("db1", "vec1", vec);
    
    // Start flush manager with 1 second interval
    jadevectordb::VectorFlushManager manager(store, 1);
    manager.start();
    
    // Wait for at least 2 flush cycles
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    
    // Add more vectors
    vec[0] = 2.0f;
    store->store_vector("db1", "vec2", vec);
    
    // Wait for another flush
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    
    manager.stop();
    
    // Verify data persisted
    auto store2 = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    auto retrieved = store2->retrieve_vector("db1", "vec2");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ((*retrieved)[0], 2.0f);
}

TEST_F(FlushManagerTest, ManualFlush) {
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    store->create_vector_file("db1", 64);
    
    std::vector<float> vec(64, 5.0f);
    store->store_vector("db1", "vec1", vec);
    
    jadevectordb::VectorFlushManager manager(store, 60); // Long interval
    manager.start();
    
    // Trigger manual flush immediately
    manager.flush_now(true);
    
    manager.stop();
    
    // Verify data persisted
    auto store2 = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    auto retrieved = store2->retrieve_vector("db1", "vec1");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ((*retrieved)[0], 5.0f);
}

TEST_F(FlushManagerTest, FinalFlushOnShutdown) {
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    store->create_vector_file("db1", 256);
    
    {
        jadevectordb::VectorFlushManager manager(store, 60); // Long interval
        manager.start();
        
        // Store vectors but don't wait for periodic flush
        std::vector<float> vec(256, 3.14f);
        store->store_vector("db1", "vec1", vec);
        
        // Manager destructor should trigger final synchronous flush
    }
    
    // Verify data persisted despite no periodic flush
    auto store2 = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    auto retrieved = store2->retrieve_vector("db1", "vec1");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_FLOAT_EQ((*retrieved)[0], 3.14f);
}

TEST_F(FlushManagerTest, MultipleStartStopCycles) {
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    jadevectordb::VectorFlushManager manager(store, 1);
    
    // Cycle 1
    manager.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    manager.stop();
    
    // Cycle 2
    manager.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    manager.stop();
    
    // Cycle 3
    manager.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    manager.stop();
    
    EXPECT_FALSE(manager.is_running());
}

TEST_F(FlushManagerTest, GetFlushInterval) {
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    jadevectordb::VectorFlushManager manager(store, 7);
    
    EXPECT_EQ(manager.get_flush_interval().count(), 7);
}
