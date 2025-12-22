#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>

#include "services/authentication_service.h"
#include "services/security_audit_logger.h"
#include "lib/error_handling.h"

namespace jadevectordb {

// Test fixture for API key lifecycle tests
class ApiKeyLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize security audit logger
        audit_logger_ = std::make_shared<SecurityAuditLogger>();

        // Initialize authentication service
        auth_service_ = std::make_unique<AuthenticationService>();

        AuthenticationConfig config;
        config.enable_api_keys = true;

        auth_service_->initialize(config, audit_logger_);

        // Register a test user for API key operations
        auto reg_result = auth_service_->register_user(
            "api_test_user",
            "TestPass123!",
            {"user"}
        );
        ASSERT_TRUE(reg_result.has_value());
        test_user_id_ = reg_result.value();
    }

    void TearDown() override {
        // Clean up is handled automatically
    }

    std::unique_ptr<AuthenticationService> auth_service_;
    std::shared_ptr<SecurityAuditLogger> audit_logger_;
    std::string test_user_id_;
};

// ============================================================================
// API KEY CREATION TESTS
// ============================================================================

TEST_F(ApiKeyLifecycleTest, GenerateApiKeyForUser) {
    auto result = auth_service_->generate_api_key(test_user_id_);

    ASSERT_TRUE(result.has_value()) << "API key generation should succeed";
    EXPECT_FALSE(result.value().empty()) << "API key should not be empty";
}

TEST_F(ApiKeyLifecycleTest, GenerateMultipleApiKeysForSameUser) {
    auto result1 = auth_service_->generate_api_key(test_user_id_);
    auto result2 = auth_service_->generate_api_key(test_user_id_);
    auto result3 = auth_service_->generate_api_key(test_user_id_);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
    ASSERT_TRUE(result3.has_value());

    // All keys should be unique
    EXPECT_NE(result1.value(), result2.value());
    EXPECT_NE(result2.value(), result3.value());
    EXPECT_NE(result1.value(), result3.value());
}

TEST_F(ApiKeyLifecycleTest, GenerateApiKeyForNonExistentUser) {
    auto result = auth_service_->generate_api_key("nonexistent_user_id");

    EXPECT_FALSE(result.has_value()) << "API key generation should fail for non-existent user";
}

// ============================================================================
// API KEY AUTHENTICATION TESTS
// ============================================================================

TEST_F(ApiKeyLifecycleTest, AuthenticateWithValidApiKey) {
    // Generate API key
    auto gen_result = auth_service_->generate_api_key(test_user_id_);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // Authenticate with the key
    auto auth_result = auth_service_->authenticate_with_api_key(api_key);

    ASSERT_TRUE(auth_result.has_value()) << "Authentication with valid API key should succeed";
    EXPECT_EQ(auth_result.value(), test_user_id_) << "Should return correct user ID";
}

TEST_F(ApiKeyLifecycleTest, AuthenticateWithInvalidApiKey) {
    auto auth_result = auth_service_->authenticate_with_api_key("invalid_api_key_12345");

    EXPECT_FALSE(auth_result.has_value()) << "Authentication with invalid API key should fail";
}

// ============================================================================
// API KEY REVOCATION TESTS
// ============================================================================

TEST_F(ApiKeyLifecycleTest, RevokeValidApiKey) {
    // Generate API key
    auto gen_result = auth_service_->generate_api_key(test_user_id_);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // Verify key works before revocation
    auto auth_before = auth_service_->authenticate_with_api_key(api_key);
    ASSERT_TRUE(auth_before.has_value());

    // Revoke the key
    auto revoke_result = auth_service_->revoke_api_key(api_key);
    ASSERT_TRUE(revoke_result.has_value()) << "API key revocation should succeed";

    // Verify key no longer works
    auto auth_after = auth_service_->authenticate_with_api_key(api_key);
    EXPECT_FALSE(auth_after.has_value()) << "Authentication should fail with revoked key";
}

TEST_F(ApiKeyLifecycleTest, RevokeInvalidApiKey) {
    auto revoke_result = auth_service_->revoke_api_key("nonexistent_key");

    EXPECT_FALSE(revoke_result.has_value()) << "Revocation of non-existent key should fail";
}

TEST_F(ApiKeyLifecycleTest, RevokeAlreadyRevokedKey) {
    // Generate and revoke key
    auto gen_result = auth_service_->generate_api_key(test_user_id_);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    auto revoke1 = auth_service_->revoke_api_key(api_key);
    ASSERT_TRUE(revoke1.has_value());

    // Try to revoke again
    auto revoke2 = auth_service_->revoke_api_key(api_key);
    EXPECT_FALSE(revoke2.has_value()) << "Revoking already revoked key should fail";
}

// ============================================================================
// API KEY LISTING TESTS
// ============================================================================

TEST_F(ApiKeyLifecycleTest, ListAllApiKeys) {
    // Generate some API keys
    auto key1 = auth_service_->generate_api_key(test_user_id_);
    auto key2 = auth_service_->generate_api_key(test_user_id_);

    ASSERT_TRUE(key1.has_value());
    ASSERT_TRUE(key2.has_value());

    // List all API keys
    auto list_result = auth_service_->list_api_keys();

    ASSERT_TRUE(list_result.has_value());
    EXPECT_GE(list_result.value().size(), 2) << "Should have at least 2 API keys";
}

TEST_F(ApiKeyLifecycleTest, ListApiKeysForSpecificUser) {
    // Create another user
    auto user2_result = auth_service_->register_user(
        "api_test_user2",
        "TestPass123!",
        {"user"}
    );
    ASSERT_TRUE(user2_result.has_value());
    std::string user2_id = user2_result.value();

    // Generate keys for both users
    auto key1 = auth_service_->generate_api_key(test_user_id_);
    auto key2 = auth_service_->generate_api_key(test_user_id_);
    auto key3 = auth_service_->generate_api_key(user2_id);

    ASSERT_TRUE(key1.has_value());
    ASSERT_TRUE(key2.has_value());
    ASSERT_TRUE(key3.has_value());

    // List keys for first user
    auto list_result = auth_service_->list_api_keys_for_user(test_user_id_);

    ASSERT_TRUE(list_result.has_value());
    EXPECT_GE(list_result.value().size(), 2) << "First user should have at least 2 keys";
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST_F(ApiKeyLifecycleTest, CompleteApiKeyLifecycle) {
    // 1. Generate API key
    auto gen_result = auth_service_->generate_api_key(test_user_id_);
    ASSERT_TRUE(gen_result.has_value());
    std::string api_key = gen_result.value();

    // 2. Authenticate with API key
    auto auth_result = auth_service_->authenticate_with_api_key(api_key);
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_EQ(auth_result.value(), test_user_id_);

    // 3. List API keys
    auto list_result = auth_service_->list_api_keys_for_user(test_user_id_);
    ASSERT_TRUE(list_result.has_value());
    EXPECT_GE(list_result.value().size(), 1);

    // 4. Revoke API key
    auto revoke_result = auth_service_->revoke_api_key(api_key);
    ASSERT_TRUE(revoke_result.has_value());

    // 5. Verify key is revoked
    auto auth_after_revoke = auth_service_->authenticate_with_api_key(api_key);
    EXPECT_FALSE(auth_after_revoke.has_value());
}

} // namespace jadevectordb
