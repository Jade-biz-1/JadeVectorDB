/**
 * Integration tests for Incremental Backup (Sprint 2.2)
 */

#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include "storage/incremental_backup_manager.h"
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

namespace jadevectordb {
namespace test {

class IncrementalBackupIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directories
        test_dir_ = "./test_backup_" + std::to_string(std::time(nullptr));
        backup_dir_ = test_dir_ + "/backups";
        fs::create_directories(test_dir_);
        fs::create_directories(backup_dir_);
        
        // Create vector store
        store_ = std::make_unique<MemoryMappedVectorStore>(test_dir_);
        
        // Create backup manager
        backup_mgr_ = std::make_unique<IncrementalBackupManager>(*store_, backup_dir_);
        
        // Create test database
        test_db_id_ = "test_backup_db";
        dimension_ = 128;
        store_->create_vector_file(test_db_id_, dimension_, 500);  // Larger capacity for tests
    }
    
    void TearDown() override {
        // Close all files
        store_->close_vector_file(test_db_id_, true);
        backup_mgr_.reset();
        store_.reset();
        
        // Cleanup test directories
        fs::remove_all(test_dir_);
    }
    
    // Helper: Add vectors
    void add_vectors(int count, const std::string& prefix = "vec_") {
        for (int i = 0; i < count; i++) {
            std::string vector_id = prefix + std::to_string(i);
            std::vector<float> values(dimension_, static_cast<float>(i) * 0.1f);
            ASSERT_TRUE(store_->store_vector(test_db_id_, vector_id, values));
        }
        store_->flush(test_db_id_, true);
    }
    
    // Helper: Modify vectors
    void modify_vectors(int start, int count, const std::string& prefix = "vec_") {
        for (int i = start; i < start + count; i++) {
            std::string vector_id = prefix + std::to_string(i);
            std::vector<float> values(dimension_, static_cast<float>(i) * 0.2f);  // Different values
            ASSERT_TRUE(store_->update_vector(test_db_id_, vector_id, values));
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

TEST_F(IncrementalBackupIntegrationTest, FullBackup) {
    // Add vectors
    add_vectors(50);
    
    // Create full backup
    auto stats = backup_mgr_->create_full_backup(test_db_id_);
    
    // Verify backup succeeded
    EXPECT_TRUE(stats.success) << "Error: " << stats.error_message;
    EXPECT_EQ(stats.vectors_backed_up, 50);
    EXPECT_GT(stats.bytes_written, 0);
    EXPECT_GT(stats.duration_seconds, 0.0);
    EXPECT_FALSE(stats.metadata.backup_id.empty());
    EXPECT_TRUE(stats.metadata.is_full_backup);
    EXPECT_EQ(stats.metadata.database_id, test_db_id_);
    EXPECT_EQ(stats.metadata.vector_count, 50);
    
    std::cout << "Full backup created: " << stats.metadata.backup_id << "\n";
    std::cout << "Vectors backed up: " << stats.vectors_backed_up << "\n";
    std::cout << "Bytes written: " << stats.bytes_written << "\n";
    std::cout << "Duration: " << stats.duration_seconds << "s\n";
}

TEST_F(IncrementalBackupIntegrationTest, IncrementalBackup) {
    // Add initial vectors and create full backup
    add_vectors(50);
    auto full_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(full_stats.success);
    
    // Enable change tracking
    backup_mgr_->enable_change_tracking(test_db_id_);
    EXPECT_TRUE(backup_mgr_->is_change_tracking_enabled(test_db_id_));
    
    // Modify some vectors
    modify_vectors(0, 10);
    
    // Record changes
    for (int i = 0; i < 10; i++) {
        backup_mgr_->record_vector_change(test_db_id_, "vec_" + std::to_string(i));
    }
    
    // Create incremental backup
    auto incr_stats = backup_mgr_->create_incremental_backup(test_db_id_);
    
    // Verify incremental backup
    EXPECT_TRUE(incr_stats.success) << "Error: " << incr_stats.error_message;
    EXPECT_EQ(incr_stats.vectors_backed_up, 10);
    EXPECT_FALSE(incr_stats.metadata.is_full_backup);
    EXPECT_EQ(incr_stats.metadata.parent_backup_id, full_stats.metadata.backup_id);
    EXPECT_LT(incr_stats.bytes_written, full_stats.bytes_written);  // Smaller than full backup
    
    std::cout << "Incremental backup created: " << incr_stats.metadata.backup_id << "\n";
    std::cout << "Parent backup: " << incr_stats.metadata.parent_backup_id << "\n";
    std::cout << "Vectors backed up: " << incr_stats.vectors_backed_up << "\n";
}

TEST_F(IncrementalBackupIntegrationTest, RestoreFromFullBackup) {
    // Add vectors and create backup
    add_vectors(30);
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    
    std::string backup_id = backup_stats.metadata.backup_id;
    
    // Delete database
    store_->delete_database_vectors(test_db_id_);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 0);
    
    // Restore from backup
    auto restore_stats = backup_mgr_->restore_from_backup(backup_id, test_db_id_);
    
    // Verify restore
    EXPECT_TRUE(restore_stats.success) << "Error: " << restore_stats.error_message;
    EXPECT_EQ(restore_stats.vectors_restored, 30);
    EXPECT_GT(restore_stats.duration_seconds, 0.0);
    
    // Verify vectors are restored
    size_t count = store_->get_vector_count(test_db_id_);
    EXPECT_EQ(count, 30);
    
    // Verify vector data is correct
    for (int i = 0; i < 30; i++) {
        std::string vector_id = "vec_" + std::to_string(i);
        auto values_opt = store_->retrieve_vector(test_db_id_, vector_id);
        ASSERT_TRUE(values_opt.has_value());
        EXPECT_FLOAT_EQ((*values_opt)[0], static_cast<float>(i) * 0.1f);
    }
    
    std::cout << "Restored " << restore_stats.vectors_restored << " vectors\n";
}

TEST_F(IncrementalBackupIntegrationTest, RestoreFromBackupChain) {
    // Initial vectors and full backup
    add_vectors(20);
    auto full_backup = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(full_backup.success);
    
    // Enable change tracking
    backup_mgr_->enable_change_tracking(test_db_id_);
    
    // First incremental: modify 5 vectors
    modify_vectors(0, 5);
    for (int i = 0; i < 5; i++) {
        backup_mgr_->record_vector_change(test_db_id_, "vec_" + std::to_string(i));
    }
    auto incr1 = backup_mgr_->create_incremental_backup(test_db_id_);
    ASSERT_TRUE(incr1.success);
    
    // Second incremental: modify 5 more vectors
    modify_vectors(5, 5);
    for (int i = 5; i < 10; i++) {
        backup_mgr_->record_vector_change(test_db_id_, "vec_" + std::to_string(i));
    }
    auto incr2 = backup_mgr_->create_incremental_backup(test_db_id_);
    ASSERT_TRUE(incr2.success);
    
    // Get backup chain
    auto chain = backup_mgr_->get_backup_chain(incr2.metadata.backup_id);
    EXPECT_EQ(chain.size(), 3);  // Full + 2 incrementals
    EXPECT_EQ(chain[0], full_backup.metadata.backup_id);
    EXPECT_EQ(chain[1], incr1.metadata.backup_id);
    EXPECT_EQ(chain[2], incr2.metadata.backup_id);
    
    // Delete database
    store_->delete_database_vectors(test_db_id_);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 0);
    
    // Restore from latest incremental (should restore full chain)
    auto restore_stats = backup_mgr_->restore_from_backup(incr2.metadata.backup_id, test_db_id_);
    
    EXPECT_TRUE(restore_stats.success);
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 20);
    
    // Verify modified vectors have latest values
    for (int i = 0; i < 10; i++) {
        auto values_opt = store_->retrieve_vector(test_db_id_, "vec_" + std::to_string(i));
        ASSERT_TRUE(values_opt.has_value());
        EXPECT_FLOAT_EQ((*values_opt)[0], static_cast<float>(i) * 0.2f);  // Modified values
    }
    
    std::cout << "Restored from chain of " << chain.size() << " backups\n";
}

TEST_F(IncrementalBackupIntegrationTest, ListBackups) {
    // Create multiple backups
    add_vectors(20);
    auto backup1 = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup1.success);
    
    backup_mgr_->enable_change_tracking(test_db_id_);
    modify_vectors(0, 5);
    for (int i = 0; i < 5; i++) {
        backup_mgr_->record_vector_change(test_db_id_, "vec_" + std::to_string(i));
    }
    
    auto backup2 = backup_mgr_->create_incremental_backup(test_db_id_);
    ASSERT_TRUE(backup2.success);
    
    // List all backups
    auto backups = backup_mgr_->list_backups();
    EXPECT_GE(backups.size(), 2);
    
    // List backups for specific database
    auto db_backups = backup_mgr_->list_backups(test_db_id_);
    EXPECT_EQ(db_backups.size(), 2);
    
    // Verify metadata
    bool found_full = false;
    bool found_incr = false;
    for (const auto& meta : db_backups) {
        EXPECT_EQ(meta.database_id, test_db_id_);
        if (meta.is_full_backup) found_full = true;
        else found_incr = true;
    }
    EXPECT_TRUE(found_full);
    EXPECT_TRUE(found_incr);
    
    std::cout << "Found " << db_backups.size() << " backups for database\n";
}

TEST_F(IncrementalBackupIntegrationTest, VerifyBackup) {
    // Create backup
    add_vectors(15);
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    
    std::string backup_id = backup_stats.metadata.backup_id;
    
    // Verify backup integrity
    bool valid = backup_mgr_->verify_backup(backup_id);
    EXPECT_TRUE(valid);
    
    std::cout << "Backup verification: " << (valid ? "PASSED" : "FAILED") << "\n";
}

TEST_F(IncrementalBackupIntegrationTest, DeleteBackup) {
    // Create backup
    add_vectors(10);
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    
    std::string backup_id = backup_stats.metadata.backup_id;
    
    // Verify backup exists
    auto backups_before = backup_mgr_->list_backups(test_db_id_);
    EXPECT_GE(backups_before.size(), 1);
    
    // Delete backup
    bool deleted = backup_mgr_->delete_backup(backup_id);
    EXPECT_TRUE(deleted);
    
    // Verify backup is gone
    auto backups_after = backup_mgr_->list_backups(test_db_id_);
    EXPECT_LT(backups_after.size(), backups_before.size());
    
    std::cout << "Backup deleted successfully\n";
}

TEST_F(IncrementalBackupIntegrationTest, ChangeTracking) {
    // Enable change tracking
    backup_mgr_->enable_change_tracking(test_db_id_);
    EXPECT_TRUE(backup_mgr_->is_change_tracking_enabled(test_db_id_));
    
    // Record changes
    backup_mgr_->record_vector_change(test_db_id_, "vec_1");
    backup_mgr_->record_vector_change(test_db_id_, "vec_2");
    backup_mgr_->record_vector_change(test_db_id_, "vec_3");
    
    // Clear tracking
    backup_mgr_->clear_change_tracking(test_db_id_);
    
    // Disable change tracking
    backup_mgr_->disable_change_tracking(test_db_id_);
    EXPECT_FALSE(backup_mgr_->is_change_tracking_enabled(test_db_id_));
}

TEST_F(IncrementalBackupIntegrationTest, NoChangesToBackup) {
    // Create full backup
    add_vectors(10);
    auto full_backup = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(full_backup.success);
    
    // Enable change tracking but make no changes
    backup_mgr_->enable_change_tracking(test_db_id_);
    
    // Try to create incremental backup with no changes
    auto incr_backup = backup_mgr_->create_incremental_backup(test_db_id_);
    
    // Should fail or indicate no changes
    EXPECT_FALSE(incr_backup.success);
    EXPECT_EQ(incr_backup.error_message, "No changes to backup");
}

TEST_F(IncrementalBackupIntegrationTest, RestoreToDifferentDatabase) {
    // Create backup of original database
    add_vectors(25);
    auto backup_stats = backup_mgr_->create_full_backup(test_db_id_);
    ASSERT_TRUE(backup_stats.success);
    
    // Create new database
    std::string new_db_id = "restored_db";
    store_->create_vector_file(new_db_id, dimension_, 50);
    
    // Restore to new database
    auto restore_stats = backup_mgr_->restore_from_backup(
        backup_stats.metadata.backup_id, new_db_id);
    
    EXPECT_TRUE(restore_stats.success);
    EXPECT_EQ(store_->get_vector_count(new_db_id), 25);
    
    // Verify both databases exist and have same content
    EXPECT_EQ(store_->get_vector_count(test_db_id_), 25);
    EXPECT_EQ(store_->get_vector_count(new_db_id), 25);
    
    std::cout << "Restored to different database successfully\n";
    
    // Cleanup
    store_->close_vector_file(new_db_id, true);
}

} // namespace test
} // namespace jadevectordb
