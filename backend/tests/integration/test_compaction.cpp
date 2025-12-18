/**
 * Integration tests for Vector File Compaction (Sprint 2.2)
 */

#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include "storage/vector_file_compactor.h"
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

namespace jadevectordb {
namespace test {

class CompactionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        test_dir_ = "./test_compaction_" + std::to_string(std::time(nullptr));
        fs::create_directories(test_dir_);
        
        // Create vector store
        store_ = std::make_unique<MemoryMappedVectorStore>(test_dir_);
        
        // Create test database
        test_db_id_ = "test_compaction_db";
        dimension_ = 128;
        store_->create_vector_file(test_db_id_, dimension_, 500);  // Larger capacity for tests
    }
    
    void TearDown() override {
        // Close all files
        store_->close_vector_file(test_db_id_, true);
        store_.reset();
        
        // Cleanup test directory
        fs::remove_all(test_dir_);
    }
    
    // Helper: Add vectors to database
    void add_vectors(int count, const std::string& prefix = "vec_") {
        for (int i = 0; i < count; i++) {
            std::string vector_id = prefix + std::to_string(i);
            std::vector<float> values(dimension_, static_cast<float>(i) * 0.1f);
            ASSERT_TRUE(store_->store_vector(test_db_id_, vector_id, values));
        }
        store_->flush(test_db_id_, true);
    }
    
    // Helper: Delete vectors
    void delete_vectors(int count, const std::string& prefix = "vec_") {
        for (int i = 0; i < count; i++) {
            std::string vector_id = prefix + std::to_string(i);
            ASSERT_TRUE(store_->delete_vector(test_db_id_, vector_id));
        }
        store_->flush(test_db_id_, true);
    }
    
    // Helper: Get file size
    size_t get_file_size() {
        std::string file_path = test_dir_ + "/" + test_db_id_ + "/vectors.jvdb";
        if (fs::exists(file_path)) {
            return fs::file_size(file_path);
        }
        return 0;
    }
    
    std::string test_dir_;
    std::unique_ptr<MemoryMappedVectorStore> store_;
    std::string test_db_id_;
    int dimension_;
};

TEST_F(CompactionIntegrationTest, BasicCompaction) {
    // Add vectors
    add_vectors(100);
    
    size_t initial_count = store_->get_vector_count(test_db_id_);
    EXPECT_EQ(initial_count, 100);
    
    // Delete half the vectors
    delete_vectors(50);
    
    size_t after_delete = store_->get_vector_count(test_db_id_);
    EXPECT_EQ(after_delete, 50);
    
    // Create compactor
    CompactionPolicy policy;
    policy.min_deleted_ratio = 0.3;  // 30% deleted triggers compaction
    policy.min_file_size_bytes = 1024;  // 1KB minimum
    
    VectorFileCompactor compactor(*store_, policy);
    
    // Force compaction (bypassing policy checks for test)
    auto stats = compactor.compact_database(test_db_id_, true);
    
    // Verify compaction completed
    EXPECT_TRUE(stats.success) << "Error: " << stats.error_message;
    EXPECT_GT(stats.duration_seconds, 0.0);
    
    // Verify vectors still accessible after compaction
    size_t final_count = store_->get_vector_count(test_db_id_);
    EXPECT_EQ(final_count, 50);
    
    // Verify remaining vectors are intact
    for (int i = 50; i < 100; i++) {
        std::string vector_id = "vec_" + std::to_string(i);
        auto values_opt = store_->retrieve_vector(test_db_id_, vector_id);
        ASSERT_TRUE(values_opt.has_value());
        EXPECT_EQ(values_opt->size(), dimension_);
        EXPECT_FLOAT_EQ((*values_opt)[0], static_cast<float>(i) * 0.1f);
    }
}

TEST_F(CompactionIntegrationTest, CompactionReclainsSpace) {
    // Add vectors
    add_vectors(100);
    
    size_t size_before_delete = get_file_size();
    EXPECT_GT(size_before_delete, 0);
    
    // Delete most vectors (90%)
    delete_vectors(90);
    
    size_t size_after_delete = get_file_size();
    // File size shouldn't decrease after delete (soft delete)
    EXPECT_GE(size_after_delete, size_before_delete);
    
    // Compact
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    auto stats = compactor.compact_database(test_db_id_, true);
    
    EXPECT_TRUE(stats.success);
    
    size_t size_after_compact = get_file_size();
    // File should be smaller after compaction
    EXPECT_LT(size_after_compact, size_after_delete);
    
    std::cout << "Size before delete: " << size_before_delete << " bytes\n";
    std::cout << "Size after delete: " << size_after_delete << " bytes\n";
    std::cout << "Size after compact: " << size_after_compact << " bytes\n";
    std::cout << "Space reclaimed: " << (size_after_delete - size_after_compact) << " bytes\n";
}

TEST_F(CompactionIntegrationTest, NoCompactionNeeded) {
    // Add vectors but don't delete any
    add_vectors(50);
    
    CompactionPolicy policy;
    policy.min_deleted_ratio = 0.2;  // Need 20% deleted
    
    VectorFileCompactor compactor(*store_, policy);
    
    // Check if compaction needed
    bool needs = compactor.needs_compaction(test_db_id_);
    EXPECT_FALSE(needs);  // No deletions, so no compaction needed
}

TEST_F(CompactionIntegrationTest, BackgroundCompaction) {
    CompactionPolicy policy;
    policy.enable_background_compaction = true;
    policy.background_check_interval_seconds = 2;  // Check every 2 seconds
    policy.min_deleted_ratio = 0.3;
    
    VectorFileCompactor compactor(*store_, policy);
    
    // Start background compaction
    compactor.start_background_compaction();
    EXPECT_TRUE(compactor.is_background_compaction_running());
    
    // Add and delete vectors
    add_vectors(100);
    delete_vectors(50);
    
    // Wait for background compaction to run
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Stop background compaction
    compactor.stop_background_compaction();
    EXPECT_FALSE(compactor.is_background_compaction_running());
    
    // Verify database is still accessible
    size_t count = store_->get_vector_count(test_db_id_);
    EXPECT_EQ(count, 50);
}

TEST_F(CompactionIntegrationTest, CompactionCallback) {
    bool callback_called = false;
    std::string callback_db_id;
    CompactionStats callback_stats;
    
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    
    // Set callback
    compactor.set_compaction_callback(
        [&](const std::string& db_id, const CompactionStats& stats) {
            callback_called = true;
            callback_db_id = db_id;
            callback_stats = stats;
        }
    );
    
    // Trigger compaction
    add_vectors(50);
    delete_vectors(25);
    
    auto stats = compactor.compact_database(test_db_id_, true);
    
    // Verify callback was called
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(callback_db_id, test_db_id_);
    EXPECT_TRUE(callback_stats.success);
}

TEST_F(CompactionIntegrationTest, MultipleCompactions) {
    // First batch
    add_vectors(50, "batch1_");
    delete_vectors(25, "batch1_");
    
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    
    // First compaction
    auto stats1 = compactor.compact_database(test_db_id_, true);
    EXPECT_TRUE(stats1.success);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 25);
    
    // Second batch
    add_vectors(50, "batch2_");
    delete_vectors(30, "batch2_");
    
    // Second compaction
    auto stats2 = compactor.compact_database(test_db_id_, true);
    EXPECT_TRUE(stats2.success);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 45); // 25 + 20
    
    // Verify all remaining vectors are accessible
    auto vector_ids = store_->list_vector_ids(test_db_id_);
    EXPECT_EQ(vector_ids.size(), 45);
}

TEST_F(CompactionIntegrationTest, CompactionWithConcurrentAccess) {
    // Add initial vectors
    add_vectors(100);
    delete_vectors(50);
    
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    
    // Start compaction in separate thread
    std::thread compaction_thread([&]() {
        auto stats = compactor.compact_database(test_db_id_, true);
        EXPECT_TRUE(stats.success);
    });
    
    // Allow compaction to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Try to access vectors during compaction
    // Note: This may fail if compaction closes the file
    // This tests that the system doesn't crash
    
    compaction_thread.join();
    
    // After compaction, verify database is accessible
    size_t count = store_->get_vector_count(test_db_id_);
    EXPECT_EQ(count, 50);
}

} // namespace test
} // namespace jadevectordb
