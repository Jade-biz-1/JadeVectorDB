/**
 * Integration tests for Compaction + Backup interaction (Sprint 2.2)
 */

#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include "storage/vector_file_compactor.h"
#include "storage/incremental_backup_manager.h"
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

namespace jadevectordb {
namespace test {

class CompactionBackupIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "./test_compact_backup_" + std::to_string(std::time(nullptr));
        backup_dir_ = test_dir_ + "/backups";
        fs::create_directories(test_dir_);
        fs::create_directories(backup_dir_);
        
        store_ = std::make_unique<MemoryMappedVectorStore>(test_dir_);
        backup_mgr_ = std::make_unique<IncrementalBackupManager>(*store_, backup_dir_);
        
        test_db_id_ = "test_db";
        dimension_ = 128;
        store_->create_vector_file(test_db_id_, dimension_, 100);
    }
    
    void TearDown() override {
        store_->close_vector_file(test_db_id_, true);
        backup_mgr_.reset();
        store_.reset();
        fs::remove_all(test_dir_);
    }
    
    void add_vectors(int count) {
        for (int i = 0; i < count; i++) {
            std::vector<float> values(dimension_, static_cast<float>(i) * 0.1f);
            store_->store_vector(test_db_id_, "vec_" + std::to_string(i), values);
        }
        store_->flush(test_db_id_, true);
    }
    
    std::string test_dir_;
    std::string backup_dir_;
    std::unique_ptr<MemoryMappedVectorStore> store_;
    std::unique_ptr<IncrementalBackupManager> backup_mgr_;
    std::string test_db_id_;
    int dimension_;
};

TEST_F(CompactionBackupIntegrationTest, BackupAfterCompaction) {
    // Add and delete vectors
    add_vectors(100);
    for (int i = 0; i < 50; i++) {
        store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    // Compact
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    auto compact_stats = compactor.compact_database(test_db_id_, true);
    ASSERT_TRUE(compact_stats.success);
    
    // Backup after compaction
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    EXPECT_EQ(backup_stats.vectors_backed_up, 50);
    
    // Restore and verify
    store_->delete_database_vectors(test_db_id_);
    auto restore_stats = backup_mgr_->restore_from_backup(
        backup_stats.metadata.backup_id, test_db_id_);
    
    ASSERT_TRUE(restore_stats.success);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 50);
}

TEST_F(CompactionBackupIntegrationTest, CompactAfterRestore) {
    // Create initial backup
    add_vectors(80);
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    
    // Delete vectors and restore
    for (int i = 0; i < 40; i++) {
        store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    auto restore_stats = backup_mgr_->restore_from_backup(
        backup_stats.metadata.backup_id, test_db_id_);
    ASSERT_TRUE(restore_stats.success);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 80);
    
    // Now delete and compact
    for (int i = 0; i < 40; i++) {
        store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    auto compact_stats = compactor.compact_database(test_db_id_, true);
    ASSERT_TRUE(compact_stats.success);
    
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 40);
}

TEST_F(CompactionBackupIntegrationTest, IncrementalBackupAfterCompaction) {
    // Initial data and full backup
    add_vectors(50);
    auto full_backup = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(full_backup.success);
    
    // Enable tracking, delete vectors, and compact
    backup_mgr_->enable_change_tracking(test_db_id_);
    
    for (int i = 0; i < 25; i++) {
        store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    auto compact_stats = compactor.compact_database(test_db_id_, true);
    ASSERT_TRUE(compact_stats.success);
    
    // Add new vectors after compaction
    for (int i = 50; i < 60; i++) {
        std::vector<float> values(dimension_, static_cast<float>(i) * 0.1f);
        store_->store_vector(test_db_id_, "vec_" + std::to_string(i), values);
        backup_mgr_->record_vector_change(test_db_id_, "vec_" + std::to_string(i));
    }
    
    // Create incremental backup
    auto incr_backup = backup_mgr_->create_incremental_backup(test_db_id_);
    ASSERT_TRUE(incr_backup.success);
    EXPECT_EQ(incr_backup.vectors_backed_up, 10);
}

TEST_F(CompactionBackupIntegrationTest, DataIntegrityAcrossOperations) {
    // Add vectors
    add_vectors(60);
    
    // Take snapshot of data
    std::map<std::string, std::vector<float>> original_data;
    for (int i = 0; i < 60; i++) {
        std::string vid = "vec_" + std::to_string(i);
        auto values = store_->retrieve_vector(test_db_id_, vid);
        ASSERT_TRUE(values.has_value());
        original_data[vid] = *values;
    }
    
    // Backup
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    
    // Delete half
    for (int i = 0; i < 30; i++) {
        store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
        original_data.erase("vec_" + std::to_string(i));
    }
    
    // Compact
    CompactionPolicy policy;
    VectorFileCompactor compactor(*store_, policy);
    auto compact_stats = compactor.compact_database(test_db_id_, true);
    ASSERT_TRUE(compact_stats.success);
    
    // Verify remaining data matches original
    for (const auto& [vid, expected_values] : original_data) {
        auto actual_values = store_->retrieve_vector(test_db_id_, vid);
        ASSERT_TRUE(actual_values.has_value());
        EXPECT_EQ(actual_values->size(), expected_values.size());
        for (size_t j = 0; j < expected_values.size(); j++) {
            EXPECT_FLOAT_EQ((*actual_values)[j], expected_values[j]);
        }
    }
    
    std::cout << "Data integrity verified across backup and compaction\n";
}

} // namespace test
} // namespace jadevectordb
