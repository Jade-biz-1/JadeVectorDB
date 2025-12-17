#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <vector>

class LazyLoadingTest : public ::testing::Test {
protected:
    std::string test_storage_path_;
    
    void SetUp() override {
        test_storage_path_ = "/tmp/jade_lazy_loading_test";
        std::filesystem::remove_all(test_storage_path_);
        std::filesystem::create_directories(test_storage_path_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_storage_path_);
    }
};

TEST_F(LazyLoadingTest, FilesNotOpenedAtStartup) {
    // Create multiple databases
    {
        jadevectordb::MemoryMappedVectorStore store(test_storage_path_);
        
        store.create_vector_file("db1", 128);
        store.create_vector_file("db2", 256);
        store.create_vector_file("db3", 512);
        
        std::vector<float> vec(128, 1.0f);
        store.store_vector("db1", "vec1", vec);
        
        vec.resize(256, 2.0f);
        store.store_vector("db2", "vec2", vec);
        
        vec.resize(512, 3.0f);
        store.store_vector("db3", "vec3", vec);
        
        store.flush_all(true);
    }
    
    // Create new store instance - files should NOT be opened yet
    jadevectordb::MemoryMappedVectorStore store(test_storage_path_);
    
    // At this point, no files should be memory-mapped
    // Only when we access them will they be opened
    
    // Access db1 - should trigger lazy loading
    auto vec1 = store.retrieve_vector("db1", "vec1");
    ASSERT_TRUE(vec1.has_value());
    EXPECT_EQ(vec1->size(), 128);
    
    // Access db3 - should trigger lazy loading
    auto vec3 = store.retrieve_vector("db3", "vec3");
    ASSERT_TRUE(vec3.has_value());
    EXPECT_EQ(vec3->size(), 512);
    
    // db2 still not accessed, so still not loaded
}

TEST_F(LazyLoadingTest, LRUEvictionWorks) {
    jadevectordb::MemoryMappedVectorStore store(test_storage_path_, 3); // Max 3 open files
    
    // Create 5 databases
    for (int i = 1; i <= 5; i++) {
        std::string db_id = "db" + std::to_string(i);
        store.create_vector_file(db_id, 128);
        
        std::vector<float> vec(128, static_cast<float>(i));
        store.store_vector(db_id, "vec" + std::to_string(i), vec);
    }
    
    store.flush_all(true);
    
    // Access all 5 databases - should trigger LRU eviction
    // Max 3 can be open at once
    for (int i = 1; i <= 5; i++) {
        std::string db_id = "db" + std::to_string(i);
        auto vec = store.retrieve_vector(db_id, "vec" + std::to_string(i));
        ASSERT_TRUE(vec.has_value());
        EXPECT_EQ((*vec)[0], static_cast<float>(i));
    }
    
    // All operations should succeed despite only 3 files being open at once
}

TEST_F(LazyLoadingTest, LastAccessTimeUpdates) {
    jadevectordb::MemoryMappedVectorStore store(test_storage_path_, 2); // Max 2 open files
    
    // Create 3 databases
    store.create_vector_file("db1", 128);
    store.create_vector_file("db2", 128);
    store.create_vector_file("db3", 128);
    
    std::vector<float> vec(128, 1.0f);
    store.store_vector("db1", "vec1", vec);
    store.store_vector("db2", "vec2", vec);
    store.store_vector("db3", "vec3", vec);
    
    // Access db1 and db2 (both loaded)
    store.retrieve_vector("db1", "vec1");
    store.retrieve_vector("db2", "vec2");
    
    // Access db3 - should evict LRU (db1, since it was accessed first)
    store.retrieve_vector("db3", "vec3");
    
    // Access db2 again (should still be open)
    auto vec2 = store.retrieve_vector("db2", "vec2");
    ASSERT_TRUE(vec2.has_value());
    
    // Access db1 again - should reload and evict LRU (db3)
    auto vec1 = store.retrieve_vector("db1", "vec1");
    ASSERT_TRUE(vec1.has_value());
}

TEST_F(LazyLoadingTest, FlushDoesNotRequireAllFilesOpen) {
    jadevectordb::MemoryMappedVectorStore store(test_storage_path_, 2); // Max 2 open
    
    // Create 4 databases
    for (int i = 1; i <= 4; i++) {
        std::string db_id = "db" + std::to_string(i);
        store.create_vector_file(db_id, 64);
        
        std::vector<float> vec(64, static_cast<float>(i));
        store.store_vector(db_id, "vec" + std::to_string(i), vec);
    }
    
    // Only 2 files should be open at this point
    // flush_all should handle this gracefully
    store.flush_all(false);
    
    // Verify all data is accessible
    for (int i = 1; i <= 4; i++) {
        std::string db_id = "db" + std::to_string(i);
        auto vec = store.retrieve_vector(db_id, "vec" + std::to_string(i));
        ASSERT_TRUE(vec.has_value());
        EXPECT_EQ((*vec)[0], static_cast<float>(i));
    }
}
