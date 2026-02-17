#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <set>
#include <filesystem>
#include <atomic>

#include "services/authentication_service.h"
#include "services/security_audit_logger.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using namespace std::chrono_literals;

/**
 * @brief Comprehensive API Key Lifecycle Tests
 *
 * Validates the complete lifecycle of API keys using AuthenticationService:
 * 1. Generation with description, scopes, and validity
 * 2. Authentication (in-memory fast path and DB fallback)
 * 3. Listing (all keys and per-user, returning rich APIKey objects)
 * 4. Revocation (by database key_id)
 * 5. Persistence across service restarts
 * 6. Edge cases and security
 */
class ApiKeyLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use a unique temp directory per test to avoid shared DB state
        test_data_dir_ = std::filesystem::temp_directory_path() /
            ("jade_apikey_unit_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(test_data_dir_);

        audit_logger_ = std::make_shared<SecurityAuditLogger>();
        auth_service_ = std::make_unique<AuthenticationService>(test_data_dir_.string());

        AuthenticationConfig config;
        config.enabled = true;
        config.enable_api_keys = true;
        auto init_result = auth_service_->initialize(config, audit_logger_);
        ASSERT_TRUE(init_result);
    }

    void TearDown() override {
        auth_service_.reset();
        audit_logger_.reset();
        std::filesystem::remove_all(test_data_dir_);
    }

    std::string create_user(const std::string& username) {
        auto result = auth_service_->register_user(username, "SecurePass123!", {"user"});
        EXPECT_TRUE(result.has_value());
        return result.has_value() ? result.value() : "";
    }

    std::filesystem::path test_data_dir_;
    std::shared_ptr<SecurityAuditLogger> audit_logger_;
    std::unique_ptr<AuthenticationService> auth_service_;
};

// ============================================================================
// Generation Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, GenerateApiKey_Basic) {
    std::string user_id = create_user("gen_basic");

    auto result = auth_service_->generate_api_key(user_id);

    ASSERT_TRUE(result.has_value());
    std::string api_key = result.value();
    EXPECT_FALSE(api_key.empty());
    EXPECT_GT(api_key.length(), 20u);
    // Key should start with jadevdb_ prefix
    EXPECT_EQ(api_key.substr(0, 8), "jadevdb_");
}

TEST_F(ApiKeyLifecycleTest, GenerateApiKey_WithDescription) {
    std::string user_id = create_user("gen_desc");

    auto result = auth_service_->generate_api_key(user_id, "Production key");
    ASSERT_TRUE(result.has_value());

    // Verify description is stored
    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_EQ(list_result.value().size(), 1u);
    EXPECT_EQ(list_result.value()[0].key_name, "Production key");
}

TEST_F(ApiKeyLifecycleTest, GenerateApiKey_WithScopesAndValidity) {
    std::string user_id = create_user("gen_scopes");

    std::vector<std::string> scopes = {"read", "write"};
    auto result = auth_service_->generate_api_key(user_id, "Scoped key", scopes, 30);
    ASSERT_TRUE(result.has_value());

    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_EQ(list_result.value().size(), 1u);

    const auto& key = list_result.value()[0];
    EXPECT_EQ(key.key_name, "Scoped key");
    EXPECT_TRUE(key.is_active);
    EXPECT_GT(key.expires_at, 0);
    EXPECT_GT(key.created_at, 0);
}

TEST_F(ApiKeyLifecycleTest, GenerateMultipleKeys_ShouldBeUnique) {
    std::string user_id = create_user("gen_multi");

    std::set<std::string> keys;
    for (int i = 0; i < 10; ++i) {
        auto result = auth_service_->generate_api_key(user_id);
        ASSERT_TRUE(result.has_value());
        keys.insert(result.value());
    }

    EXPECT_EQ(keys.size(), 10u);
}

TEST_F(ApiKeyLifecycleTest, GenerateApiKey_NonexistentUser) {
    auto result = auth_service_->generate_api_key("nonexistent_user");
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Authentication Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, AuthenticateWithKey_Success) {
    std::string user_id = create_user("auth_ok");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());

    auto auth_result = auth_service_->authenticate_with_api_key(gen_result.value(), "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_EQ(auth_result.value(), user_id);
}

TEST_F(ApiKeyLifecycleTest, AuthenticateWithKey_Invalid) {
    auto result = auth_service_->authenticate_with_api_key("invalid_key_12345", "127.0.0.1");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ApiKeyLifecycleTest, AuthenticateWithKey_Empty) {
    auto result = auth_service_->authenticate_with_api_key("", "127.0.0.1");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ApiKeyLifecycleTest, AuthenticateWithKey_MultipleRequests) {
    std::string user_id = create_user("auth_multi");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    for (int i = 0; i < 5; ++i) {
        auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
        ASSERT_TRUE(auth_result.has_value());
        EXPECT_EQ(auth_result.value(), user_id);
    }
}

// ============================================================================
// Listing Tests (Rich APIKey Objects)
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ListApiKeys_ForUser) {
    std::string user_id = create_user("list_user");

    for (int i = 0; i < 5; ++i) {
        auth_service_->generate_api_key(user_id, "Key " + std::to_string(i));
    }

    auto result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 5u);

    // Verify rich data is present
    for (const auto& key : result.value()) {
        EXPECT_FALSE(key.api_key_id.empty());
        EXPECT_EQ(key.user_id, user_id);
        EXPECT_FALSE(key.key_prefix.empty());
        EXPECT_TRUE(key.is_active);
        EXPECT_GT(key.created_at, 0);
    }
}

TEST_F(ApiKeyLifecycleTest, ListApiKeys_All) {
    std::string user1 = create_user("list_all_1");
    std::string user2 = create_user("list_all_2");

    auth_service_->generate_api_key(user1);
    auth_service_->generate_api_key(user1);
    auth_service_->generate_api_key(user2);

    auto result = auth_service_->list_api_keys();
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 3u);
}

TEST_F(ApiKeyLifecycleTest, ListApiKeys_EmptyForNewUser) {
    std::string user_id = create_user("list_empty");

    auto result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 0u);
}

TEST_F(ApiKeyLifecycleTest, ListApiKeys_KeyPrefix) {
    std::string user_id = create_user("list_prefix");

    auto gen_result = auth_service_->generate_api_key(user_id, "Test key");
    ASSERT_TRUE(gen_result.has_value());
    std::string raw_key = gen_result.value();

    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_EQ(list_result.value().size(), 1u);

    // key_prefix should be first 12 chars of the raw key
    EXPECT_EQ(list_result.value()[0].key_prefix, raw_key.substr(0, 12));
}

// ============================================================================
// Revocation Tests (by database key_id)
// ============================================================================

TEST_F(ApiKeyLifecycleTest, RevokeKey_ByDatabaseId) {
    std::string user_id = create_user("revoke_dbid");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // Get the database key_id from the list
    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_EQ(list_result.value().size(), 1u);
    std::string key_id = list_result.value()[0].api_key_id;

    // Verify key works before revocation
    auto auth_before = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    ASSERT_TRUE(auth_before.has_value());

    // Revoke by database ID
    auto revoke_result = auth_service_->revoke_api_key(key_id);
    ASSERT_TRUE(revoke_result.has_value());

    // Verify key no longer works
    auto auth_after = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    EXPECT_FALSE(auth_after.has_value());

    // Verify key shows as inactive in list
    auto final_list = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(final_list.has_value());
    ASSERT_EQ(final_list.value().size(), 1u);
    EXPECT_FALSE(final_list.value()[0].is_active);
}

TEST_F(ApiKeyLifecycleTest, RevokeKey_ByRawKey_Fallback) {
    std::string user_id = create_user("revoke_raw");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // Revoke using raw key (in-memory fallback)
    auto revoke_result = auth_service_->revoke_api_key(api_key);
    ASSERT_TRUE(revoke_result.has_value());

    // Verify key no longer works
    auto auth_after = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    EXPECT_FALSE(auth_after.has_value());
}

TEST_F(ApiKeyLifecycleTest, RevokeKey_InvalidId) {
    auto result = auth_service_->revoke_api_key("nonexistent_key_id");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ApiKeyLifecycleTest, RevokeMultipleKeys) {
    std::string user_id = create_user("revoke_multi");

    // Generate 3 keys
    for (int i = 0; i < 3; ++i) {
        auth_service_->generate_api_key(user_id, "Key " + std::to_string(i));
    }

    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    EXPECT_EQ(list_result.value().size(), 3u);

    // Revoke all keys by their database IDs
    for (const auto& key : list_result.value()) {
        auto revoke_result = auth_service_->revoke_api_key(key.api_key_id);
        EXPECT_TRUE(revoke_result.has_value());
    }

    // Verify all keys show as inactive
    auto final_list = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(final_list.has_value());
    for (const auto& key : final_list.value()) {
        EXPECT_FALSE(key.is_active);
    }
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, KeysPersistedAcrossRestart) {
    std::string user_id = create_user("persist_test");

    // Generate a key
    auto gen_result = auth_service_->generate_api_key(user_id, "Persistent key");
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // Destroy and recreate the service (simulates restart)
    auth_service_.reset();
    auth_service_ = std::make_unique<AuthenticationService>(test_data_dir_.string());

    AuthenticationConfig config;
    config.enabled = true;
    config.enable_api_keys = true;
    ASSERT_TRUE(auth_service_->initialize(config, audit_logger_));

    // Key should still be listed (from DB)
    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_EQ(list_result.value().size(), 1u);
    EXPECT_EQ(list_result.value()[0].key_name, "Persistent key");
    EXPECT_TRUE(list_result.value()[0].is_active);

    // Authentication should work via DB fallback
    auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_EQ(auth_result.value(), user_id);
}

// ============================================================================
// Security Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, KeyHash_NotPlaintext) {
    std::string user_id = create_user("hash_test");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0u);

    // The stored key_hash should NOT equal the plaintext key
    EXPECT_NE(list_result.value()[0].key_hash, api_key);
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ConcurrentGeneration) {
    std::string user_id = create_user("concurrent_gen");

    std::vector<std::thread> threads;
    std::vector<std::string> generated_keys(5);

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this, &user_id, &generated_keys, i]() {
            auto result = auth_service_->generate_api_key(user_id);
            if (result.has_value()) {
                generated_keys[i] = result.value();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::set<std::string> unique_keys(generated_keys.begin(), generated_keys.end());
    unique_keys.erase("");  // Remove any empty entries from failed generations
    EXPECT_EQ(unique_keys.size(), 5u);
}

TEST_F(ApiKeyLifecycleTest, ConcurrentAuthentication) {
    std::string user_id = create_user("concurrent_auth");

    auto gen_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([this, &api_key, &success_count]() {
            auto result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
            if (result.has_value()) {
                success_count++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count, 10);
}

// ============================================================================
// Complete Lifecycle Test
// ============================================================================

TEST_F(ApiKeyLifecycleTest, CompleteLifecycle) {
    // 1. Create user
    std::string user_id = create_user("lifecycle");
    ASSERT_FALSE(user_id.empty());

    // 2. Generate key with description and scopes
    std::vector<std::string> scopes = {"read", "write"};
    auto gen_result = auth_service_->generate_api_key(user_id, "Lifecycle test key", scopes, 30);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // 3. Authenticate with key
    auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_EQ(auth_result.value(), user_id);

    // 4. List keys — verify rich data
    auto list_result = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_EQ(list_result.value().size(), 1u);

    const auto& key = list_result.value()[0];
    EXPECT_EQ(key.key_name, "Lifecycle test key");
    EXPECT_EQ(key.user_id, user_id);
    EXPECT_TRUE(key.is_active);
    EXPECT_GT(key.expires_at, 0);
    EXPECT_GT(key.created_at, 0);
    std::string key_id = key.api_key_id;

    // 5. Revoke by database key_id
    auto revoke_result = auth_service_->revoke_api_key(key_id);
    ASSERT_TRUE(revoke_result.has_value());

    // 6. Verify key no longer works
    auto auth_after = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    EXPECT_FALSE(auth_after.has_value());

    // 7. Verify key shows as inactive
    auto final_list = auth_service_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(final_list.has_value());
    ASSERT_EQ(final_list.value().size(), 1u);
    EXPECT_FALSE(final_list.value()[0].is_active);
}
