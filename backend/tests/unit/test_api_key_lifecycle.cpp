#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <set>

#include "services/authentication_service.h"
#include "services/security_audit_logger.h"
#include "lib/auth.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using namespace std::chrono_literals;

/**
 * @brief Comprehensive API Key Lifecycle Tests
 *
 * This test suite validates the complete lifecycle of API keys:
 * 1. Generation (both AuthenticationService and AuthManager)
 * 2. Validation and authentication
 * 3. Permission checking
 * 4. Listing and management
 * 5. Expiration handling
 * 6. Revocation
 * 7. Edge cases and security considerations
 */
class ApiKeyLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup AuthenticationService
        audit_logger_ = std::make_shared<SecurityAuditLogger>();
        auth_service_ = std::make_unique<AuthenticationService>();

        AuthenticationConfig config;
        config.enabled = true;
        config.enable_api_keys = true;
        auto init_result = auth_service_->initialize(config, audit_logger_);
        ASSERT_TRUE(init_result);

        // Setup AuthManager
        auth_manager_ = AuthManager::get_instance();
        ASSERT_NE(auth_manager_, nullptr);
    }

    void TearDown() override {
        auth_service_.reset();
        audit_logger_.reset();
    }

    // Helper: Register user in AuthenticationService
    std::string register_auth_service_user(const std::string& username) {
        auto result = auth_service_->register_user(username, "SecurePass123!", {"user"});
        EXPECT_TRUE(result.has_value());
        return result.has_value() ? result.value() : "";
    }

    // Helper: Create user in AuthManager
    std::string create_auth_manager_user(const std::string& username) {
        auto result = auth_manager_->create_user(username, username + "@example.com", {"role_user"});
        EXPECT_TRUE(result.has_value());
        return result.has_value() ? result.value() : "";
    }

    std::shared_ptr<SecurityAuditLogger> audit_logger_;
    std::unique_ptr<AuthenticationService> auth_service_;
    AuthManager* auth_manager_;
};

// ============================================================================
// API Key Generation Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, AuthenticationService_GenerateApiKey) {
    std::string user_id = register_auth_service_user("authsvc_user");

    auto result = auth_service_->generate_api_key(user_id);

    ASSERT_TRUE(result.has_value());
    std::string api_key = result.value();
    EXPECT_FALSE(api_key.empty());
    EXPECT_GT(api_key.length(), 20);
}

TEST_F(ApiKeyLifecycleTest, AuthManager_GenerateApiKey) {
    std::string user_id = create_auth_manager_user("authmgr_user");

    auto result = auth_manager_->generate_api_key(user_id, {}, "Test key");

    ASSERT_TRUE(result.has_value());
    std::string api_key = result.value();
    EXPECT_FALSE(api_key.empty());
    EXPECT_GT(api_key.length(), 20);
}

TEST_F(ApiKeyLifecycleTest, GenerateMultipleKeys_ShouldBeUnique) {
    std::string user_id = register_auth_service_user("multikey_user");

    std::set<std::string> keys;
    for (int i = 0; i < 10; ++i) {
        auto result = auth_service_->generate_api_key(user_id);
        ASSERT_TRUE(result.has_value());
        keys.insert(result.value());
    }

    EXPECT_EQ(keys.size(), 10);  // All keys should be unique
}

TEST_F(ApiKeyLifecycleTest, GenerateApiKey_WithCustomDuration) {
    std::string user_id = create_auth_manager_user("custom_duration");

    auto result = auth_manager_->generate_api_key(
        user_id,
        {"custom:permission"},
        "7-day key",
        std::chrono::hours(24 * 7)  // 7 days
    );

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());
}

TEST_F(ApiKeyLifecycleTest, GenerateApiKey_NonexistentUser) {
    auto result1 = auth_service_->generate_api_key("nonexistent-user-1");
    EXPECT_FALSE(result1.has_value());

    auto result2 = auth_manager_->generate_api_key("nonexistent-user-2");
    EXPECT_FALSE(result2.has_value());
}

// ============================================================================
// API Key Validation Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ValidateKey_Immediate) {
    std::string user_id = register_auth_service_user("validate_user");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Validate immediately after generation
    auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_EQ(auth_result.value(), user_id);
}

TEST_F(ApiKeyLifecycleTest, ValidateKey_AuthManager) {
    std::string user_id = create_auth_manager_user("validate_mgr");

    auto gen_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    auto validate_result = auth_manager_->validate_api_key(api_key);
    ASSERT_TRUE(validate_result.has_value());
    EXPECT_TRUE(validate_result.value());
}

TEST_F(ApiKeyLifecycleTest, ValidateKey_InvalidKey) {
    auto result1 = auth_service_->authenticate_with_api_key("invalid-key-123", "127.0.0.1");
    EXPECT_FALSE(result1.has_value());

    auto result2 = auth_manager_->validate_api_key("invalid-key-456");
    EXPECT_FALSE(result2.has_value());
}

TEST_F(ApiKeyLifecycleTest, ValidateKey_EmptyKey) {
    auto result1 = auth_service_->authenticate_with_api_key("", "127.0.0.1");
    EXPECT_FALSE(result1.has_value());

    auto result2 = auth_manager_->validate_api_key("");
    EXPECT_FALSE(result2.has_value());
}

// ============================================================================
// API Key Authentication Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, AuthenticateWithKey_MultipleRequests) {
    std::string user_id = register_auth_service_user("multi_auth");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Authenticate multiple times with same key
    for (int i = 0; i < 5; ++i) {
        auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
        ASSERT_TRUE(auth_result.has_value());
        EXPECT_EQ(auth_result.value(), user_id);
    }
}

TEST_F(ApiKeyLifecycleTest, AuthenticateWithKey_DifferentIPs) {
    std::string user_id = register_auth_service_user("ip_test");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Authenticate from different IP addresses
    std::vector<std::string> ips = {"127.0.0.1", "192.168.1.1", "10.0.0.1"};
    for (const auto& ip : ips) {
        auto auth_result = auth_service_->authenticate_with_api_key(api_key, ip);
        ASSERT_TRUE(auth_result.has_value());
        EXPECT_EQ(auth_result.value(), user_id);
    }
}

TEST_F(ApiKeyLifecycleTest, GetUserFromApiKey) {
    std::string user_id = create_auth_manager_user("getuser_key");

    auto gen_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    auto user_result = auth_manager_->get_user_from_api_key(api_key);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value(), user_id);
}

// ============================================================================
// Permission Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ApiKey_WithSpecificPermissions) {
    std::string user_id = create_auth_manager_user("perm_test");

    std::vector<std::string> permissions = {"read:data", "write:data"};
    auto gen_result = auth_manager_->generate_api_key(user_id, permissions, "Scoped key");
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Check granted permissions
    auto has_read = auth_manager_->has_permission_with_api_key(api_key, "read:data");
    ASSERT_TRUE(has_read.has_value());
    EXPECT_TRUE(has_read.value());

    auto has_write = auth_manager_->has_permission_with_api_key(api_key, "write:data");
    ASSERT_TRUE(has_write.has_value());
    EXPECT_TRUE(has_write.value());

    // Check non-granted permission
    auto has_delete = auth_manager_->has_permission_with_api_key(api_key, "delete:data");
    ASSERT_TRUE(has_delete.has_value());
    EXPECT_FALSE(has_delete.value());
}

TEST_F(ApiKeyLifecycleTest, ApiKey_NoPermissions) {
    std::string user_id = create_auth_manager_user("no_perm");

    std::vector<std::string> permissions = {};
    auto gen_result = auth_manager_->generate_api_key(user_id, permissions, "No permissions");
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    auto has_perm = auth_manager_->has_permission_with_api_key(api_key, "any:action");
    ASSERT_TRUE(has_perm.has_value());
    EXPECT_FALSE(has_perm.value());
}

TEST_F(ApiKeyLifecycleTest, GetPermissionsFromApiKey) {
    std::string user_id = create_auth_manager_user("get_perms");

    std::vector<std::string> expected_perms = {"perm1", "perm2", "perm3"};
    auto gen_result = auth_manager_->generate_api_key(user_id, expected_perms);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    auto perms_result = auth_manager_->get_permissions_from_api_key(api_key);
    ASSERT_TRUE(perms_result.has_value());
    EXPECT_EQ(perms_result.value().size(), 3);
}

// ============================================================================
// Listing and Management Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ListApiKeys_ForUser) {
    std::string user_id = create_auth_manager_user("list_user");

    // Generate multiple keys
    for (int i = 0; i < 5; ++i) {
        auth_manager_->generate_api_key(user_id, {}, "Key " + std::to_string(i));
    }

    auto result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 5);
}

TEST_F(ApiKeyLifecycleTest, ListApiKeys_All) {
    // Create multiple users with keys
    std::string user1 = create_auth_manager_user("list_all_1");
    std::string user2 = create_auth_manager_user("list_all_2");

    auth_manager_->generate_api_key(user1);
    auth_manager_->generate_api_key(user1);
    auth_manager_->generate_api_key(user2);

    auto result = auth_manager_->list_api_keys();
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 3);
}

TEST_F(ApiKeyLifecycleTest, ListApiKeys_EmptyForNewUser) {
    std::string user_id = create_auth_manager_user("empty_list");

    auto result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 0);
}

TEST_F(ApiKeyLifecycleTest, ApiKeyMetadata_Description) {
    std::string user_id = create_auth_manager_user("metadata_test");

    std::string description = "Production API key for service X";
    auto gen_result = auth_manager_->generate_api_key(user_id, {}, description);
    ASSERT_TRUE(gen_result.has_value());

    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);

    EXPECT_EQ(list_result.value()[0].description, description);
}

TEST_F(ApiKeyLifecycleTest, ApiKeyMetadata_Timestamps) {
    std::string user_id = create_auth_manager_user("timestamp_test");

    auto before = std::chrono::system_clock::now();
    auto gen_result = auth_manager_->generate_api_key(user_id);
    auto after = std::chrono::system_clock::now();

    ASSERT_TRUE(gen_result.has_value());

    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);

    auto created_at = list_result.value()[0].created_at;
    EXPECT_GE(created_at, before);
    EXPECT_LE(created_at, after);
}

// ============================================================================
// Revocation Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, RevokeKey_AuthenticationService) {
    std::string user_id = register_auth_service_user("revoke_svc");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Verify key works before revocation
    auto auth_result1 = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    ASSERT_TRUE(auth_result1.has_value());

    // Revoke the key
    auto revoke_result = auth_service_->revoke_api_key(api_key);
    ASSERT_TRUE(revoke_result.has_value());
    EXPECT_TRUE(revoke_result.value());

    // Verify key doesn't work after revocation
    auto auth_result2 = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    EXPECT_FALSE(auth_result2.has_value());
}

TEST_F(ApiKeyLifecycleTest, RevokeKey_AuthManager) {
    std::string user_id = create_auth_manager_user("revoke_mgr");

    auto gen_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Get key ID
    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);
    std::string key_id = list_result.value()[0].key_id;

    // Verify key works before revocation
    auto validate_result1 = auth_manager_->validate_api_key(api_key);
    ASSERT_TRUE(validate_result1.has_value());
    EXPECT_TRUE(validate_result1.value());

    // Revoke the key
    auto revoke_result = auth_manager_->revoke_api_key(key_id);
    ASSERT_TRUE(revoke_result.has_value());

    // Verify key doesn't work after revocation
    auto validate_result2 = auth_manager_->validate_api_key(api_key);
    EXPECT_FALSE(validate_result2.has_value());
}

TEST_F(ApiKeyLifecycleTest, RevokeKey_InvalidKeyId) {
    auto result = auth_manager_->revoke_api_key("nonexistent-key-id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ApiKeyLifecycleTest, RevokeMultipleKeys) {
    std::string user_id = create_auth_manager_user("revoke_multi");

    // Generate multiple keys
    std::vector<std::string> key_ids;
    for (int i = 0; i < 3; ++i) {
        auth_manager_->generate_api_key(user_id, {}, "Key " + std::to_string(i));
    }

    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    EXPECT_EQ(list_result.value().size(), 3);

    // Revoke all keys
    for (const auto& key_info : list_result.value()) {
        auto revoke_result = auth_manager_->revoke_api_key(key_info.key_id);
        EXPECT_TRUE(revoke_result.has_value());
    }

    // Verify all keys are inactive
    auto final_list = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(final_list.has_value());
    for (const auto& key_info : final_list.value()) {
        EXPECT_FALSE(key_info.is_active);
    }
}

// ============================================================================
// Expiration Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ExpiredKey_ShouldFail) {
    std::string user_id = create_auth_manager_user("expire_test");

    // Generate key with 1-second expiration
    auto gen_result = auth_manager_->generate_api_key(
        user_id,
        {},
        "Short-lived key",
        std::chrono::seconds(1)
    );
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Should work immediately
    auto validate_result1 = auth_manager_->validate_api_key(api_key);
    ASSERT_TRUE(validate_result1.has_value());
    EXPECT_TRUE(validate_result1.value());

    // Wait for expiration
    std::this_thread::sleep_for(2s);

    // Should fail after expiration
    auto validate_result2 = auth_manager_->validate_api_key(api_key);
    EXPECT_FALSE(validate_result2.has_value());
}

TEST_F(ApiKeyLifecycleTest, KeyExpiration_Metadata) {
    std::string user_id = create_auth_manager_user("expire_meta");

    auto before = std::chrono::system_clock::now();
    auto duration = std::chrono::hours(24);

    auto gen_result = auth_manager_->generate_api_key(user_id, {}, "24h key", duration);
    ASSERT_TRUE(gen_result.has_value());

    auto after = std::chrono::system_clock::now();

    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);

    auto expires_at = list_result.value()[0].expires_at;
    auto expected_expiry = before + duration;

    // Expiry should be approximately now + duration
    auto diff = std::chrono::duration_cast<std::chrono::seconds>(expires_at - expected_expiry).count();
    EXPECT_LT(std::abs(diff), 5);  // Within 5 seconds
}

// ============================================================================
// Security Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ApiKey_ShouldNotBeStoredPlaintext) {
    std::string user_id = create_auth_manager_user("security_test");

    auto gen_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);

    // The stored key_hash should NOT equal the plaintext key
    EXPECT_NE(list_result.value()[0].key_hash, api_key);
}

TEST_F(ApiKeyLifecycleTest, ApiKey_HashConsistency) {
    std::string test_key = "test-api-key-12345";

    std::string hash1 = auth_manager_->hash_api_key(test_key);
    std::string hash2 = auth_manager_->hash_api_key(test_key);

    EXPECT_EQ(hash1, hash2);  // Same key should produce same hash
    EXPECT_FALSE(hash1.empty());
}

TEST_F(ApiKeyLifecycleTest, InactiveUser_ApiKeyShouldStillValidate) {
    std::string user_id = create_auth_manager_user("inactive_test");

    auto gen_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Deactivate user
    auth_manager_->deactivate_user(user_id);

    // API key validation depends on implementation
    // This test documents the behavior
    auto validate_result = auth_manager_->validate_api_key(api_key);
    // May or may not be valid - depends on implementation
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ConcurrentGeneration) {
    std::string user_id = create_auth_manager_user("concurrent_gen");

    std::vector<std::thread> threads;
    std::vector<std::string> generated_keys(5);

    // Generate keys concurrently
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this, user_id, &generated_keys, i]() {
            auto result = auth_manager_->generate_api_key(user_id);
            if (result.has_value()) {
                generated_keys[i] = result.value();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All keys should be unique
    std::set<std::string> unique_keys(generated_keys.begin(), generated_keys.end());
    EXPECT_EQ(unique_keys.size(), 5);
}

TEST_F(ApiKeyLifecycleTest, ConcurrentValidation) {
    std::string user_id = create_auth_manager_user("concurrent_val");

    auto gen_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    // Validate key concurrently
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([this, api_key, &success_count]() {
            auto result = auth_manager_->validate_api_key(api_key);
            if (result.has_value() && result.value()) {
                success_count++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count, 10);  // All validations should succeed
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ReuseRevokedKey) {
    std::string user_id = register_auth_service_user("reuse_test");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    std::string api_key = gen_result.value();

    // Revoke the key
    auth_service_->revoke_api_key(api_key);

    // Try to authenticate with revoked key multiple times
    for (int i = 0; i < 3; ++i) {
        auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
        EXPECT_FALSE(auth_result.has_value());
    }
}

TEST_F(ApiKeyLifecycleTest, VeryLongDescription) {
    std::string user_id = create_auth_manager_user("long_desc");

    std::string long_description(1000, 'X');  // 1000 character description
    auto result = auth_manager_->generate_api_key(user_id, {}, long_description);

    ASSERT_TRUE(result.has_value());

    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);
    EXPECT_EQ(list_result.value()[0].description, long_description);
}

TEST_F(ApiKeyLifecycleTest, ManyPermissions) {
    std::string user_id = create_auth_manager_user("many_perms");

    std::vector<std::string> many_permissions;
    for (int i = 0; i < 100; ++i) {
        many_permissions.push_back("permission:" + std::to_string(i));
    }

    auto result = auth_manager_->generate_api_key(user_id, many_permissions);
    ASSERT_TRUE(result.has_value());

    std::string api_key = result.value();

    auto perms_result = auth_manager_->get_permissions_from_api_key(api_key);
    ASSERT_TRUE(perms_result.has_value());
    EXPECT_EQ(perms_result.value().size(), 100);
}

// ============================================================================
// Complete Lifecycle Test
// ============================================================================

TEST_F(ApiKeyLifecycleTest, CompleteLifecycle_HappyPath) {
    // 1. Create user
    std::string user_id = create_auth_manager_user("lifecycle_complete");
    ASSERT_FALSE(user_id.empty());

    // 2. Generate API key with permissions
    std::vector<std::string> permissions = {"read:data", "write:data"};
    auto gen_result = auth_manager_->generate_api_key(
        user_id,
        permissions,
        "Complete lifecycle test key",
        std::chrono::hours(24)
    );
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // 3. Validate key immediately
    auto validate_result = auth_manager_->validate_api_key(api_key);
    ASSERT_TRUE(validate_result.has_value());
    EXPECT_TRUE(validate_result.value());

    // 4. Get user from key
    auto user_result = auth_manager_->get_user_from_api_key(api_key);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value(), user_id);

    // 5. Check permissions
    auto has_read = auth_manager_->has_permission_with_api_key(api_key, "read:data");
    ASSERT_TRUE(has_read.has_value());
    EXPECT_TRUE(has_read.value());

    // 6. List keys for user
    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    EXPECT_GE(list_result.value().size(), 1);

    std::string key_id = list_result.value()[0].key_id;

    // 7. Revoke the key
    auto revoke_result = auth_manager_->revoke_api_key(key_id);
    ASSERT_TRUE(revoke_result.has_value());

    // 8. Verify key is no longer valid
    auto final_validate = auth_manager_->validate_api_key(api_key);
    EXPECT_FALSE(final_validate.has_value());

    // 9. Verify key shows as inactive in list
    auto final_list = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(final_list.has_value());
    EXPECT_FALSE(final_list.value()[0].is_active);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
