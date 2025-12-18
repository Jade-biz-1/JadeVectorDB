// Sprint 2.2 Direct Integration Tests
// Tests for VectorFileCompactor and IncrementalBackupManager

#include <gtest/gtest.h>
#include <filesystem>
#include <chrono>
#include <thread>

#include "storage/vector_file_compactor.h"
#include "storage/incremental_backup_manager.h"
#include "storage/memory_mapped_vector_store.h"

namespace fs = std::filesystem;
using namespace jadevectordb;

namespace test {

class Sprint22DirectTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        test_dir_ = "./test_sprint22_direct_" + timestamp;
        fs::create_directories(test_dir_);
        
        vector_store_ = std::make_unique<MemoryMappedVectorStore>(test_dir_);
        test_db_id_ = "test_db_" + timestamp;
        dimension_ = 128;
        
        bool created = vector_store_->create_vector_file(test_db_id_, dimension_, 1000);
        ASSERT_TRUE(created) << "Failed to create vector file";
    }
    
    void TearDown() override {
        vector_store_.reset();
        compactor_.reset();
        backup_manager_.reset();
        
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }
    
    void store_vectors(int count) {
        for (int i = 0; i < count; i++) {
            std::string vector_id = "vec_" + std::to_string(i);
            std::vector<float> values(dimension_, static_cast<float>(i) * 0.1f);
            bool stored = vector_store_->store_vector(test_db_id_, vector_id, values);
            ASSERT_TRUE(stored) << "Failed to store vector " << vector_id;
        }
    }
    
    std::unique_ptr<MemoryMappedVectorStore> vector_store_;
    std::unique_ptr<VectorFileCompactor> compactor_;
    std::unique_ptr<IncrementalBackupManager> backup_manager_;
    std::string test_dir_;
    std::string test_db_id_;
    int dimension_;
};

TEST_F(Sprint22DirectTest, CompactorInitialization) {
    compactor_ = std::make_unique<VectorFileCompactor>(*vector_store_);
    EXPECT_NE(compactor_, nullptr);
    
    compactor_->start_background_compaction();
    EXPECT_TRUE(compactor_->is_background_compaction_running());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    compactor_->stop_background_compaction();
    EXPECT_FALSE(compactor_->is_background_compaction_running());
}

TEST_F(Sprint22DirectTest, CompactorManualCompaction) {
    compactor_ = std::make_unique<VectorFileCompactor>(*vector_store_);
    store_vectors(100);
    
    for (int i = 0; i < 50; i++) {
        vector_store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    auto stats = compactor_->compact_database(test_db_id_, true);
    EXPECT_TRUE(stats.success) << "Manual compaction failed: " << stats.error_message;
    EXPECT_GT(stats.bytes_reclaimed, 0) << "Should reclaim space";
    
    for (int i = 50; i < 100; i++) {
        auto result = vector_store_->retrieve_vector(test_db_id_, "vec_" + std::to_string(i));
        EXPECT_TRUE(result.has_value()) << "Vector vec_" << i << " missing after compaction";
    }
}

TEST_F(Sprint22DirectTest, CompactorNeedsCompaction) {
    compactor_ = std::make_unique<VectorFileCompactor>(*vector_store_);
    store_vectors(50);
    
    for (int i = 0; i < 40; i++) {
        vector_store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    bool needs = compactor_->needs_compaction(test_db_id_);
    EXPECT_TRUE(needs) << "Should need compaction after 80% deletion";
}

TEST_F(Sprint22DirectTest, BackupManagerInitialization) {
    std::string backup_dir = test_dir_ + "/backups";
    backup_manager_ = std::make_unique<IncrementalBackupManager>(*vector_store_, backup_dir);
    EXPECT_NE(backup_manager_, nullptr);
    EXPECT_TRUE(fs::exists(backup_dir));
}

TEST_F(Sprint22DirectTest, BackupManagerFullBackup) {
    std::string backup_dir = test_dir_ + "/backups";
    backup_manager_ = std::make_unique<IncrementalBackupManager>(*vector_store_, backup_dir);
    
    store_vectors(30);
    auto stats = backup_manager_->create_full_backup(test_db_id_);
    EXPECT_TRUE(stats.success) << "Backup failed: " << stats.error_message;
    EXPECT_EQ(stats.vectors_backed_up, 30);
    EXPECT_GT(stats.bytes_written, 0);
    EXPECT_FALSE(stats.metadata.backup_id.empty());
    EXPECT_TRUE(stats.metadata.is_full_backup);
}

TEST_F(Sprint22DirectTest, BackupManagerIncrementalBackup) {
    std::string backup_dir = test_dir_ + "/backups";
    backup_manager_ = std::make_unique<IncrementalBackupManager>(*vector_store_, backup_dir);
    backup_manager_->enable_change_tracking(test_db_id_);
    
    store_vectors(20);
    auto full = backup_manager_->create_full_backup(test_db_id_);
    ASSERT_TRUE(full.success);
    
    for (int i = 20; i < 30; i++) {
        std::string vid = "vec_" + std::to_string(i);
        std::vector<float> vals(dimension_, static_cast<float>(i) * 0.1f);
        vector_store_->store_vector(test_db_id_, vid, vals);
        backup_manager_->record_vector_change(test_db_id_, vid);
    }
    
    auto incr = backup_manager_->create_incremental_backup(test_db_id_);
    EXPECT_TRUE(incr.success) << "Incremental failed: " << incr.error_message;
    EXPECT_FALSE(incr.metadata.is_full_backup);
    EXPECT_EQ(incr.metadata.parent_backup_id, full.metadata.backup_id);
}

TEST_F(Sprint22DirectTest, BackupManagerRestore) {
    std::string backup_dir = test_dir_ + "/backups";
    backup_manager_ = std::make_unique<IncrementalBackupManager>(*vector_store_, backup_dir);
    
    store_vectors(25);
    auto backup = backup_manager_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup.success);
    std::string backup_id = backup.metadata.backup_id;
    
    for (int i = 0; i < 25; i++) {
        vector_store_->delete_vector(test_db_id_, "vec_" + std::to_string(i));
    }
    
    auto restore = backup_manager_->restore_from_backup(backup_id, test_db_id_);
    EXPECT_TRUE(restore.success) << "Restore failed: " << restore.error_message;
    EXPECT_EQ(restore.vectors_restored, 25);
    
    for (int i = 0; i < 25; i++) {
        auto result = vector_store_->retrieve_vector(test_db_id_, "vec_" + std::to_string(i));
        EXPECT_TRUE(result.has_value()) << "Vector vec_" << i << " not restored";
    }
}

TEST_F(Sprint22DirectTest, BackupManagerListBackups) {
    std::string backup_dir = test_dir_ + "/backups";
    backup_manager_ = std::make_unique<IncrementalBackupManager>(*vector_store_, backup_dir);
    backup_manager_->enable_change_tracking(test_db_id_);
    
    store_vectors(10);
    auto full = backup_manager_->create_full_backup(test_db_id_);
    ASSERT_TRUE(full.success);
    
    for (int i = 10; i < 15; i++) {
        std::string vid = "vec_" + std::to_string(i);
        std::vector<float> vals(dimension_, static_cast<float>(i) * 0.1f);
        vector_store_->store_vector(test_db_id_, vid, vals);
        backup_manager_->record_vector_change(test_db_id_, vid);
    }
    auto incr = backup_manager_->create_incremental_backup(test_db_id_);
    ASSERT_TRUE(incr.success);
    
    auto backups = backup_manager_->list_backups(test_db_id_);
    EXPECT_GE(backups.size(), 2);
    
    bool found_full = false, found_incr = false;
    for (const auto& b : backups) {
        if (b.is_full_backup) found_full = true;
        else found_incr = true;
    }
    EXPECT_TRUE(found_full);
    EXPECT_TRUE(found_incr);
}

} // namespace test
