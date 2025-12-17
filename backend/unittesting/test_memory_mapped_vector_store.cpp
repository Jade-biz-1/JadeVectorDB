#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <vector>

using namespace jadevectordb;

class MemoryMappedVectorStoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_storage_path_ = "./test_vector_storage";
        std::filesystem::remove_all(test_storage_path_);
        store_ = std::make_unique<MemoryMappedVectorStore>(test_storage_path_);
    }

    void TearDown() override {
        store_.reset();
        std::filesystem::remove_all(test_storage_path_);
    }

    std::string test_storage_path_;
    std::unique_ptr<MemoryMappedVectorStore> store_;
};

TEST_F(MemoryMappedVectorStoreTest, CreateVectorFile) {
    // Create a vector file for a test database
    bool created = store_->create_vector_file("test_db", 384, 1000);
    EXPECT_TRUE(created);
    
    // Verify file exists
    EXPECT_TRUE(store_->has_database("test_db"));
    
    // Verify dimension
    EXPECT_EQ(store_->get_dimension("test_db"), 384);
    
    // Verify initial vector count is 0
    EXPECT_EQ(store_->get_vector_count("test_db"), 0);
}

TEST_F(MemoryMappedVectorStoreTest, StoreAndRetrieveVector) {
    // Create database
    store_->create_vector_file("test_db", 128, 100);
    
    // Create test vector
    std::vector<float> test_vector(128);
    for (size_t i = 0; i < 128; i++) {
        test_vector[i] = static_cast<float>(i) / 128.0f;
    }
    
    // Store vector
    bool stored = store_->store_vector("test_db", "vec1", test_vector);
    EXPECT_TRUE(stored);
    
    // Verify count
    EXPECT_EQ(store_->get_vector_count("test_db"), 1);
    
    // Retrieve vector
    auto retrieved = store_->retrieve_vector("test_db", "vec1");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->size(), 128);
    
    // Verify values
    for (size_t i = 0; i < 128; i++) {
        EXPECT_FLOAT_EQ((*retrieved)[i], test_vector[i]);
    }
}

TEST_F(MemoryMappedVectorStoreTest, UpdateVector) {
    // Create database and store initial vector
    store_->create_vector_file("test_db", 64, 100);
    
    std::vector<float> initial(64, 1.0f);
    store_->store_vector("test_db", "vec1", initial);
    
    // Update with new values
    std::vector<float> updated(64, 2.0f);
    bool success = store_->update_vector("test_db", "vec1", updated);
    EXPECT_TRUE(success);
    
    // Verify updated values
    auto retrieved = store_->retrieve_vector("test_db", "vec1");
    ASSERT_TRUE(retrieved.has_value());
    for (float val : *retrieved) {
        EXPECT_FLOAT_EQ(val, 2.0f);
    }
    
    // Count should still be 1 (update, not insert)
    EXPECT_EQ(store_->get_vector_count("test_db"), 1);
}

TEST_F(MemoryMappedVectorStoreTest, DeleteVector) {
    // Create database and store vector
    store_->create_vector_file("test_db", 32, 100);
    
    std::vector<float> vec(32, 1.0f);
    store_->store_vector("test_db", "vec1", vec);
    EXPECT_EQ(store_->get_vector_count("test_db"), 1);
    
    // Delete vector
    bool deleted = store_->delete_vector("test_db", "vec1");
    EXPECT_TRUE(deleted);
    
    // Verify count decreased
    EXPECT_EQ(store_->get_vector_count("test_db"), 0);
    
    // Try to retrieve deleted vector (should fail)
    auto retrieved = store_->retrieve_vector("test_db", "vec1");
    EXPECT_FALSE(retrieved.has_value());
}

TEST_F(MemoryMappedVectorStoreTest, MultipleVectors) {
    // Create database
    store_->create_vector_file("test_db", 256, 100);
    
    // Store 10 vectors
    for (int i = 0; i < 10; i++) {
        std::vector<float> vec(256);
        for (size_t j = 0; j < 256; j++) {
            vec[j] = static_cast<float>(i * 256 + j);
        }
        
        bool stored = store_->store_vector("test_db", "vec" + std::to_string(i), vec);
        EXPECT_TRUE(stored);
    }
    
    // Verify count
    EXPECT_EQ(store_->get_vector_count("test_db"), 10);
    
    // Retrieve and verify all vectors
    for (int i = 0; i < 10; i++) {
        auto retrieved = store_->retrieve_vector("test_db", "vec" + std::to_string(i));
        ASSERT_TRUE(retrieved.has_value());
        EXPECT_EQ(retrieved->size(), 256);
        
        // Verify first value
        EXPECT_FLOAT_EQ((*retrieved)[0], static_cast<float>(i * 256));
    }
}

TEST_F(MemoryMappedVectorStoreTest, BatchStore) {
    // Create database
    store_->create_vector_file("test_db", 64, 100);
    
    // Prepare batch
    std::vector<std::pair<std::string, std::vector<float>>> batch;
    for (int i = 0; i < 20; i++) {
        std::vector<float> vec(64, static_cast<float>(i));
        batch.emplace_back("vec" + std::to_string(i), vec);
    }
    
    // Batch store
    size_t stored = store_->batch_store("test_db", batch);
    EXPECT_EQ(stored, 20);
    EXPECT_EQ(store_->get_vector_count("test_db"), 20);
}

TEST_F(MemoryMappedVectorStoreTest, BatchRetrieve) {
    // Create database and store vectors
    store_->create_vector_file("test_db", 128, 100);
    
    for (int i = 0; i < 5; i++) {
        std::vector<float> vec(128, static_cast<float>(i));
        store_->store_vector("test_db", "vec" + std::to_string(i), vec);
    }
    
    // Batch retrieve
    std::vector<std::string> ids = {"vec0", "vec2", "vec4", "vec99"};  // vec99 doesn't exist
    auto results = store_->batch_retrieve("test_db", ids);
    
    EXPECT_EQ(results.size(), 4);
    EXPECT_TRUE(results[0].has_value());  // vec0 exists
    EXPECT_TRUE(results[1].has_value());  // vec2 exists
    EXPECT_TRUE(results[2].has_value());  // vec4 exists
    EXPECT_FALSE(results[3].has_value()); // vec99 doesn't exist
    
    // Verify values
    EXPECT_FLOAT_EQ((*results[0])[0], 0.0f);
    EXPECT_FLOAT_EQ((*results[1])[0], 2.0f);
    EXPECT_FLOAT_EQ((*results[2])[0], 4.0f);
}

TEST_F(MemoryMappedVectorStoreTest, FlushAndReopen) {
    std::string db_id = "test_db";
    
    // Create database and store vector
    store_->create_vector_file(db_id, 384, 100);
    std::vector<float> vec(384, 3.14f);
    store_->store_vector(db_id, "vec1", vec);
    
    // Flush to disk
    store_->flush(db_id, true);
    
    // Close the store
    store_->close_vector_file(db_id, true);
    
    // Create new store instance (simulating restart)
    auto new_store = std::make_unique<MemoryMappedVectorStore>(test_storage_path_);
    
    // Open existing file
    bool opened = new_store->open_vector_file(db_id);
    EXPECT_TRUE(opened);
    
    // Verify vector persisted
    auto retrieved = new_store->retrieve_vector(db_id, "vec1");
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->size(), 384);
    EXPECT_FLOAT_EQ((*retrieved)[0], 3.14f);
}

TEST_F(MemoryMappedVectorStoreTest, DeleteDatabaseVectors) {
    // Create database and store vectors
    store_->create_vector_file("test_db", 256, 100);
    std::vector<float> vec(256, 1.0f);
    store_->store_vector("test_db", "vec1", vec);
    
    // Verify database exists
    EXPECT_TRUE(store_->has_database("test_db"));
    
    // Delete all vectors for database
    bool deleted = store_->delete_database_vectors("test_db");
    EXPECT_TRUE(deleted);
    
    // Verify database no longer exists
    EXPECT_FALSE(store_->has_database("test_db"));
}

TEST_F(MemoryMappedVectorStoreTest, DimensionMismatch) {
    // Create database with dimension 128
    store_->create_vector_file("test_db", 128, 100);
    
    // Try to store vector with wrong dimension
    std::vector<float> wrong_size(256, 1.0f);  // Wrong: 256 instead of 128
    bool stored = store_->store_vector("test_db", "vec1", wrong_size);
    EXPECT_FALSE(stored);
    
    // Verify no vector was stored
    EXPECT_EQ(store_->get_vector_count("test_db"), 0);
}

TEST_F(MemoryMappedVectorStoreTest, RetrieveNonExistentVector) {
    store_->create_vector_file("test_db", 64, 100);
    
    // Try to retrieve vector that doesn't exist
    auto retrieved = store_->retrieve_vector("test_db", "non_existent");
    EXPECT_FALSE(retrieved.has_value());
}

TEST_F(MemoryMappedVectorStoreTest, MultipleDatabases) {
    // Create multiple databases with different dimensions
    store_->create_vector_file("db1", 128, 100);
    store_->create_vector_file("db2", 256, 100);
    store_->create_vector_file("db3", 512, 100);
    
    // Store vectors in each database
    std::vector<float> vec1(128, 1.0f);
    std::vector<float> vec2(256, 2.0f);
    std::vector<float> vec3(512, 3.0f);
    
    store_->store_vector("db1", "vec", vec1);
    store_->store_vector("db2", "vec", vec2);
    store_->store_vector("db3", "vec", vec3);
    
    // Verify each database
    EXPECT_EQ(store_->get_dimension("db1"), 128);
    EXPECT_EQ(store_->get_dimension("db2"), 256);
    EXPECT_EQ(store_->get_dimension("db3"), 512);
    
    // Retrieve and verify
    auto r1 = store_->retrieve_vector("db1", "vec");
    auto r2 = store_->retrieve_vector("db2", "vec");
    auto r3 = store_->retrieve_vector("db3", "vec");
    
    ASSERT_TRUE(r1.has_value() && r2.has_value() && r3.has_value());
    EXPECT_EQ(r1->size(), 128);
    EXPECT_EQ(r2->size(), 256);
    EXPECT_EQ(r3->size(), 512);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
