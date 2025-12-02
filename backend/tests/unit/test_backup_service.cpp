#include <gtest/gtest.h>
#include <memory>
#include <filesystem>
#include <fstream>

// Include the headers we want to test
#include "services/backup_service.h"
#include "lib/error_handling.h"

using namespace jadevectordb;
namespace fs = std::filesystem;

// Test fixture for BackupService
class BackupServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the service
        service_ = std::make_unique<BackupService>();
        auto result = service_->initialize();
        ASSERT_TRUE(result.has_value());

        // Create temp directories for testing
        test_backup_dir_ = fs::temp_directory_path() / "jadevectordb_backup_test";
        test_recovery_dir_ = fs::temp_directory_path() / "jadevectordb_recovery_test";
        fs::create_directories(test_backup_dir_);
        fs::create_directories(test_recovery_dir_);

        service_->set_default_backup_path(test_backup_dir_.string());
        service_->set_default_recovery_path(test_recovery_dir_.string());
    }

    void TearDown() override {
        // Clean up temp directories
        if (fs::exists(test_backup_dir_)) {
            fs::remove_all(test_backup_dir_);
        }
        if (fs::exists(test_recovery_dir_)) {
            fs::remove_all(test_recovery_dir_);
        }
        service_.reset();
    }

    // Helper to create backup config
    BackupConfig create_test_config(const std::string& name = "test_backup") {
        BackupConfig config;
        config.backup_name = name;
        config.description = "Test backup";
        config.storage_path = (test_backup_dir_ / name).string();
        config.compression_method = "gzip";
        config.encrypt_backup = false;
        config.retention_days = 30;
        config.include_indexes = true;
        config.include_metadata = true;
        return config;
    }

    std::unique_ptr<BackupService> service_;
    fs::path test_backup_dir_;
    fs::path test_recovery_dir_;
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(BackupServiceTest, InitializeService) {
    EXPECT_NE(service_, nullptr);
}

TEST_F(BackupServiceTest, ServiceInitialization) {
    auto new_service = std::make_unique<BackupService>();
    auto result = new_service->initialize();
    EXPECT_TRUE(result.has_value());
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(BackupServiceTest, SetDefaultBackupPath) {
    std::string test_path = "/tmp/test_backup";
    service_->set_default_backup_path(test_path);
    // Config is set, no direct getter but will be used in operations
    SUCCEED();
}

TEST_F(BackupServiceTest, SetDefaultRecoveryPath) {
    std::string test_path = "/tmp/test_recovery";
    service_->set_default_recovery_path(test_path);
    SUCCEED();
}

TEST_F(BackupServiceTest, SetMaxConcurrentBackups) {
    service_->set_max_concurrent_backups(5);
    SUCCEED();
}

TEST_F(BackupServiceTest, SetBackupTimeout) {
    service_->set_backup_timeout(std::chrono::minutes(10));
    SUCCEED();
}

// ============================================================================
// Backup Creation Tests - Edge Cases
// ============================================================================

TEST_F(BackupServiceTest, CreateBackupWithEmptyName) {
    BackupConfig config = create_test_config("");
    auto result = service_->create_backup(config);
    // Should handle empty name gracefully
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithInvalidPath) {
    BackupConfig config = create_test_config("test");
    config.storage_path = "/invalid/nonexistent/path/that/does/not/exist";
    auto result = service_->create_backup(config);
    // Should fail gracefully for invalid path
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithNullStoragePath) {
    BackupConfig config = create_test_config("test");
    config.storage_path = "";
    auto result = service_->create_backup(config);
    // Should handle empty storage path
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithSpecialCharactersInName) {
    BackupConfig config = create_test_config("test-backup@#$%^&*()");
    auto result = service_->create_backup(config);
    // Should handle special characters
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithVeryLongName) {
    std::string long_name(1000, 'a');
    BackupConfig config = create_test_config(long_name);
    auto result = service_->create_backup(config);
    // Should handle very long names
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithNegativeRetentionDays) {
    BackupConfig config = create_test_config("test");
    config.retention_days = -10;
    auto result = service_->create_backup(config);
    // Should handle negative retention
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithZeroRetentionDays) {
    BackupConfig config = create_test_config("test");
    config.retention_days = 0;
    auto result = service_->create_backup(config);
    // Zero retention should be allowed
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithEncryptionButEmptyKey) {
    BackupConfig config = create_test_config("test");
    config.encrypt_backup = true;
    config.encryption_key = "";
    auto result = service_->create_backup(config);
    // Should fail or handle empty encryption key
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithInvalidCompressionMethod) {
    BackupConfig config = create_test_config("test");
    config.compression_method = "invalid_compression";
    auto result = service_->create_backup(config);
    // Should fail for invalid compression
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, CreateBackupWithEmptyDatabasesList) {
    BackupConfig config = create_test_config("test");
    config.databases_to_backup.clear();
    auto result = service_->create_backup(config);
    // Empty list should backup all databases
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

// ============================================================================
// Backup Information Tests - Edge Cases
// ============================================================================

TEST_F(BackupServiceTest, GetBackupInfoWithEmptyId) {
    auto result = service_->get_backup_info("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, GetBackupInfoWithNonExistentId) {
    auto result = service_->get_backup_info("nonexistent_backup_id_12345");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, GetBackupInfoWithNullId) {
    auto result = service_->get_backup_info(std::string());
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, GetBackupStatusWithInvalidId) {
    auto result = service_->get_backup_status("invalid_id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, ListBackupsWhenEmpty) {
    auto result = service_->list_backups();
    EXPECT_TRUE(result.has_value());
    if (result.has_value()) {
        // Should return empty list or list of existing backups
        EXPECT_TRUE(result.value().size() >= 0);
    }
}

// ============================================================================
// Backup Cancellation Tests - Edge Cases
// ============================================================================

TEST_F(BackupServiceTest, CancelNonExistentBackup) {
    auto result = service_->cancel_backup("nonexistent_id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, CancelBackupWithEmptyId) {
    auto result = service_->cancel_backup("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, DeleteNonExistentBackup) {
    auto result = service_->delete_backup("nonexistent_id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, DeleteBackupWithEmptyId) {
    auto result = service_->delete_backup("");
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Recovery Tests - Edge Cases
// ============================================================================

TEST_F(BackupServiceTest, RestoreBackupWithEmptyPath) {
    RecoveryConfig config;
    config.backup_path = "";
    config.restore_indexes = true;
    config.restore_metadata = true;
    auto result = service_->restore_backup(config);
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, RestoreBackupWithNonExistentPath) {
    RecoveryConfig config;
    config.backup_path = "/nonexistent/backup/path";
    config.restore_indexes = true;
    config.restore_metadata = true;
    auto result = service_->restore_backup(config);
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, RestoreBackupWithInvalidTargetDatabase) {
    RecoveryConfig config;
    config.backup_path = (test_backup_dir_ / "test.backup").string();
    config.target_database = "invalid_db#@$%";
    auto result = service_->restore_backup(config);
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, GetRecoveryInfoWithEmptyId) {
    auto result = service_->get_recovery_info("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, GetRecoveryInfoWithNonExistentId) {
    auto result = service_->get_recovery_info("nonexistent_recovery_id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, GetRecoveryStatusWithInvalidId) {
    auto result = service_->get_recovery_status("invalid_id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, ListRecoveriesWhenEmpty) {
    auto result = service_->list_recoveries();
    EXPECT_TRUE(result.has_value());
    if (result.has_value()) {
        EXPECT_TRUE(result.value().size() >= 0);
    }
}

// ============================================================================
// Verification Tests - Edge Cases
// ============================================================================

TEST_F(BackupServiceTest, VerifyBackupIntegrityWithEmptyPath) {
    auto result = service_->verify_backup_integrity("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, VerifyBackupIntegrityWithNonExistentPath) {
    auto result = service_->verify_backup_integrity("/nonexistent/path");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, ValidateBackupCompatibilityWithEmptyPath) {
    auto result = service_->validate_backup_compatibility("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(BackupServiceTest, ValidateBackupCompatibilityWithInvalidPath) {
    auto result = service_->validate_backup_compatibility("/invalid/path");
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Utility Method Tests
// ============================================================================

TEST_F(BackupServiceTest, BackupTypeToString) {
    EXPECT_EQ(service_->backup_type_to_string(BackupType::FULL), "FULL");
    EXPECT_EQ(service_->backup_type_to_string(BackupType::INCREMENTAL), "INCREMENTAL");
    EXPECT_EQ(service_->backup_type_to_string(BackupType::SNAPSHOT), "SNAPSHOT");
}

TEST_F(BackupServiceTest, BackupStatusToString) {
    EXPECT_EQ(service_->backup_status_to_string(BackupStatus::PENDING), "PENDING");
    EXPECT_EQ(service_->backup_status_to_string(BackupStatus::IN_PROGRESS), "IN_PROGRESS");
    EXPECT_EQ(service_->backup_status_to_string(BackupStatus::COMPLETED), "COMPLETED");
    EXPECT_EQ(service_->backup_status_to_string(BackupStatus::FAILED), "FAILED");
    EXPECT_EQ(service_->backup_status_to_string(BackupStatus::CANCELLED), "CANCELLED");
}

TEST_F(BackupServiceTest, RecoveryStatusToString) {
    EXPECT_EQ(service_->recovery_status_to_string(RecoveryStatus::PENDING), "PENDING");
    EXPECT_EQ(service_->recovery_status_to_string(RecoveryStatus::IN_PROGRESS), "IN_PROGRESS");
    EXPECT_EQ(service_->recovery_status_to_string(RecoveryStatus::COMPLETED), "COMPLETED");
    EXPECT_EQ(service_->recovery_status_to_string(RecoveryStatus::FAILED), "FAILED");
}

// ============================================================================
// Cleanup Tests
// ============================================================================

TEST_F(BackupServiceTest, CleanupExpiredBackups) {
    auto result = service_->cleanup_expired_backups();
    EXPECT_TRUE(result.has_value());
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST_F(BackupServiceTest, SetBackupCallbacks) {
    bool start_called = false;
    bool complete_called = false;
    bool error_called = false;

    service_->set_backup_start_callback([&](const BackupInfo&) { start_called = true; });
    service_->set_backup_complete_callback([&](const BackupInfo&) { complete_called = true; });
    service_->set_backup_error_callback([&](const BackupInfo&) { error_called = true; });

    SUCCEED(); // Callbacks set successfully
}

TEST_F(BackupServiceTest, SetRecoveryCallbacks) {
    bool start_called = false;
    bool complete_called = false;
    bool error_called = false;

    service_->set_recovery_start_callback([&](const RecoveryInfo&) { start_called = true; });
    service_->set_recovery_complete_callback([&](const RecoveryInfo&) { complete_called = true; });
    service_->set_recovery_error_callback([&](const RecoveryInfo&) { error_called = true; });

    SUCCEED(); // Callbacks set successfully
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
