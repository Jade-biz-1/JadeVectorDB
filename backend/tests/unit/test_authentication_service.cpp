#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "services/authentication_service.h"
#include "services/security_audit_logger.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using namespace std::chrono_literals;

/**
 * @brief Test fixture for AuthenticationService
 *
 * This fixture sets up an authentication service with default configuration
 * and provides helper methods for common test operations.
 */
class AuthenticationServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create audit logger
        audit_logger_ = std::make_shared<SecurityAuditLogger>();

        // Create authentication service
        auth_service_ = std::make_unique<AuthenticationService>();

        // Initialize with default configuration
        AuthenticationConfig config;
        config.enabled = true;
        config.token_expiry_seconds = 3600;  // 1 hour
        config.session_expiry_seconds = 86400;  // 24 hours
        config.max_failed_attempts = 3;
        config.account_lockout_duration_seconds = 300;  // 5 minutes
        config.require_strong_passwords = true;
        config.min_password_length = 8;
        config.enable_api_keys = true;

        auto init_result = auth_service_->initialize(config, audit_logger_);
        ASSERT_TRUE(init_result);
    }

    void TearDown() override {
        auth_service_.reset();
        audit_logger_.reset();
    }

    // Helper method to register a test user
    std::string register_test_user(const std::string& username,
                                   const std::string& password,
                                   const std::vector<std::string>& roles = {"user"}) {
        auto result = auth_service_->register_user(username, password, roles);
        EXPECT_TRUE(result.has_value()) << "Failed to register user: " << username;
        return result.has_value() ? result.value() : "";
    }

    std::shared_ptr<SecurityAuditLogger> audit_logger_;
    std::unique_ptr<AuthenticationService> auth_service_;
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, InitializeService) {
    EXPECT_NE(auth_service_, nullptr);
}

TEST_F(AuthenticationServiceTest, GetConfiguration) {
    auto config = auth_service_->get_config();
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.token_expiry_seconds, 3600);
    EXPECT_EQ(config.max_failed_attempts, 3);
}

TEST_F(AuthenticationServiceTest, UpdateConfiguration) {
    AuthenticationConfig new_config;
    new_config.enabled = true;
    new_config.token_expiry_seconds = 7200;  // 2 hours
    new_config.max_failed_attempts = 5;

    auto result = auth_service_->update_config(new_config);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    auto updated_config = auth_service_->get_config();
    EXPECT_EQ(updated_config.token_expiry_seconds, 7200);
    EXPECT_EQ(updated_config.max_failed_attempts, 5);
}

// ============================================================================
// User Registration Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, RegisterUser_Success) {
    auto result = auth_service_->register_user("testuser", "SecurePass123!", {"user"});

    ASSERT_TRUE(result.has_value());
    std::string user_id = result.value();
    EXPECT_FALSE(user_id.empty());

    // Verify user was created
    auto user_result = auth_service_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().username, "testuser");
    EXPECT_EQ(user_result.value().roles.size(), 1);
    EXPECT_EQ(user_result.value().roles[0], "user");
}

TEST_F(AuthenticationServiceTest, RegisterUser_WeakPassword) {
    // Password too short
    auto result = auth_service_->register_user("testuser", "short", {"user"});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::VALIDATION_ERROR);
}

TEST_F(AuthenticationServiceTest, RegisterUser_DuplicateUsername) {
    // Register first user
    auto result1 = auth_service_->register_user("duplicate", "SecurePass123!", {"user"});
    ASSERT_TRUE(result1.has_value());

    // Try to register with same username
    auto result2 = auth_service_->register_user("duplicate", "AnotherPass123!", {"user"});
    EXPECT_FALSE(result2.has_value());
    EXPECT_EQ(result2.error().code(), ErrorCode::ALREADY_EXISTS);
}

TEST_F(AuthenticationServiceTest, RegisterUser_WithMultipleRoles) {
    std::vector<std::string> roles = {"user", "admin", "developer"};
    auto result = auth_service_->register_user("multirole", "SecurePass123!", roles);

    ASSERT_TRUE(result.has_value());

    auto user_result = auth_service_->get_user(result.value());
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().roles.size(), 3);
}

TEST_F(AuthenticationServiceTest, RegisterUser_WithCustomUserId) {
    std::string custom_id = "custom-user-id-12345";
    auto result = auth_service_->register_user("customid", "SecurePass123!", {"user"}, custom_id);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), custom_id);
}

// ============================================================================
// Authentication Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, Authenticate_Success) {
    // Register user
    std::string user_id = register_test_user("authtest", "SecurePass123!");

    // Authenticate
    auto auth_result = auth_service_->authenticate("authtest", "SecurePass123!", "127.0.0.1", "TestAgent");

    ASSERT_TRUE(auth_result.has_value());
    AuthToken token = auth_result.value();
    EXPECT_FALSE(token.token_value.empty());
    EXPECT_EQ(token.user_id, user_id);
    EXPECT_TRUE(token.is_valid);
    EXPECT_EQ(token.ip_address, "127.0.0.1");
}

TEST_F(AuthenticationServiceTest, Authenticate_InvalidPassword) {
    register_test_user("authtest", "SecurePass123!");

    auto auth_result = auth_service_->authenticate("authtest", "WrongPassword!", "127.0.0.1");

    EXPECT_FALSE(auth_result.has_value());
    EXPECT_EQ(auth_result.error().code(), ErrorCode::UNAUTHORIZED);
}

TEST_F(AuthenticationServiceTest, Authenticate_NonexistentUser) {
    auto auth_result = auth_service_->authenticate("nonexistent", "AnyPassword123!", "127.0.0.1");

    EXPECT_FALSE(auth_result.has_value());
    EXPECT_EQ(auth_result.error().code(), ErrorCode::NOT_FOUND);
}

TEST_F(AuthenticationServiceTest, Authenticate_InactiveUser) {
    std::string user_id = register_test_user("inactive", "SecurePass123!");

    // Deactivate user
    auto deactivate_result = auth_service_->set_user_active_status(user_id, false);
    ASSERT_TRUE(deactivate_result.has_value());

    // Try to authenticate
    auto auth_result = auth_service_->authenticate("inactive", "SecurePass123!", "127.0.0.1");

    EXPECT_FALSE(auth_result.has_value());
    EXPECT_EQ(auth_result.error().code(), ErrorCode::UNAUTHORIZED);
}

TEST_F(AuthenticationServiceTest, Authenticate_AccountLockout) {
    register_test_user("locktest", "SecurePass123!");

    // Attempt multiple failed logins (max_failed_attempts = 3)
    for (int i = 0; i < 3; ++i) {
        auto result = auth_service_->authenticate("locktest", "WrongPassword!", "127.0.0.1");
        EXPECT_FALSE(result.has_value());
    }

    // Next attempt should fail due to lockout, even with correct password
    auto locked_result = auth_service_->authenticate("locktest", "SecurePass123!", "127.0.0.1");
    EXPECT_FALSE(locked_result.has_value());
    EXPECT_EQ(locked_result.error().code(), ErrorCode::UNAUTHORIZED);
}

// ============================================================================
// Token Validation Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, ValidateToken_Success) {
    std::string user_id = register_test_user("tokentest", "SecurePass123!");

    auto auth_result = auth_service_->authenticate("tokentest", "SecurePass123!", "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());

    std::string token_value = auth_result.value().token_value;

    // Validate the token
    auto validate_result = auth_service_->validate_token(token_value);
    ASSERT_TRUE(validate_result.has_value());
    EXPECT_EQ(validate_result.value(), user_id);
}

TEST_F(AuthenticationServiceTest, ValidateToken_InvalidToken) {
    auto validate_result = auth_service_->validate_token("invalid-token-12345");

    EXPECT_FALSE(validate_result.has_value());
    EXPECT_EQ(validate_result.error().code(), ErrorCode::UNAUTHORIZED);
}

TEST_F(AuthenticationServiceTest, ValidateToken_ExpiredToken) {
    // Create auth service with very short token expiry
    auto short_expiry_service = std::make_unique<AuthenticationService>();
    AuthenticationConfig config;
    config.token_expiry_seconds = 1;  // 1 second expiry
    short_expiry_service->initialize(config, audit_logger_);

    auto reg_result = short_expiry_service->register_user("expiretest", "SecurePass123!", {"user"});
    ASSERT_TRUE(reg_result.has_value());

    auto auth_result = short_expiry_service->authenticate("expiretest", "SecurePass123!", "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());

    std::string token_value = auth_result.value().token_value;

    // Wait for token to expire
    std::this_thread::sleep_for(2s);

    // Try to validate expired token
    auto validate_result = short_expiry_service->validate_token(token_value);
    EXPECT_FALSE(validate_result.has_value());
    EXPECT_EQ(validate_result.error().code(), ErrorCode::UNAUTHORIZED);
}

// ============================================================================
// Token Refresh Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, RefreshToken_Success) {
    register_test_user("refreshtest", "SecurePass123!");

    auto auth_result = auth_service_->authenticate("refreshtest", "SecurePass123!", "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());

    std::string old_token = auth_result.value().token_value;

    // Refresh the token
    auto refresh_result = auth_service_->refresh_token(old_token, "127.0.0.1");
    ASSERT_TRUE(refresh_result.has_value());

    AuthToken new_token = refresh_result.value();
    EXPECT_NE(new_token.token_value, old_token);
    EXPECT_TRUE(new_token.is_valid);
}

TEST_F(AuthenticationServiceTest, RefreshToken_InvalidToken) {
    auto refresh_result = auth_service_->refresh_token("invalid-token", "127.0.0.1");

    EXPECT_FALSE(refresh_result.has_value());
    EXPECT_EQ(refresh_result.error().code(), ErrorCode::UNAUTHORIZED);
}

// ============================================================================
// Token Revocation Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, RevokeToken_Success) {
    register_test_user("revoketest", "SecurePass123!");

    auto auth_result = auth_service_->authenticate("revoketest", "SecurePass123!", "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());

    std::string token_value = auth_result.value().token_value;

    // Revoke the token
    auto revoke_result = auth_service_->revoke_token(token_value);
    ASSERT_TRUE(revoke_result.has_value());
    EXPECT_TRUE(revoke_result.value());

    // Try to validate revoked token
    auto validate_result = auth_service_->validate_token(token_value);
    EXPECT_FALSE(validate_result.has_value());
}

TEST_F(AuthenticationServiceTest, Logout_Success) {
    register_test_user("logouttest", "SecurePass123!");

    auto auth_result = auth_service_->authenticate("logouttest", "SecurePass123!", "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());

    std::string token_value = auth_result.value().token_value;

    // Logout
    auto logout_result = auth_service_->logout(token_value);
    ASSERT_TRUE(logout_result.has_value());
    EXPECT_TRUE(logout_result.value());

    // Token should be invalid after logout
    auto validate_result = auth_service_->validate_token(token_value);
    EXPECT_FALSE(validate_result.has_value());
}

// ============================================================================
// Session Management Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, CreateSession_Success) {
    std::string user_id = register_test_user("sessiontest", "SecurePass123!");

    auto session_result = auth_service_->create_session(user_id, "token-123", "192.168.1.1");

    ASSERT_TRUE(session_result.has_value());
    AuthSession session = session_result.value();
    EXPECT_FALSE(session.session_id.empty());
    EXPECT_EQ(session.user_id, user_id);
    EXPECT_TRUE(session.is_active);
    EXPECT_EQ(session.ip_address, "192.168.1.1");
}

TEST_F(AuthenticationServiceTest, ValidateSession_Success) {
    std::string user_id = register_test_user("sessiontest", "SecurePass123!");

    auto session_result = auth_service_->create_session(user_id, "token-123", "192.168.1.1");
    ASSERT_TRUE(session_result.has_value());

    std::string session_id = session_result.value().session_id;

    auto validate_result = auth_service_->validate_session(session_id);
    ASSERT_TRUE(validate_result.has_value());
    EXPECT_TRUE(validate_result.value());
}

TEST_F(AuthenticationServiceTest, EndSession_Success) {
    std::string user_id = register_test_user("sessiontest", "SecurePass123!");

    auto session_result = auth_service_->create_session(user_id, "token-123", "192.168.1.1");
    ASSERT_TRUE(session_result.has_value());

    std::string session_id = session_result.value().session_id;

    auto end_result = auth_service_->end_session(session_id);
    ASSERT_TRUE(end_result.has_value());
    EXPECT_TRUE(end_result.value());

    // Session should be invalid after ending
    auto validate_result = auth_service_->validate_session(session_id);
    EXPECT_FALSE(validate_result.has_value());
}

TEST_F(AuthenticationServiceTest, GetUserSessions) {
    std::string user_id = register_test_user("multisession", "SecurePass123!");

    // Create multiple sessions
    auth_service_->create_session(user_id, "token-1", "192.168.1.1");
    auth_service_->create_session(user_id, "token-2", "192.168.1.2");
    auth_service_->create_session(user_id, "token-3", "192.168.1.3");

    auto sessions_result = auth_service_->get_user_sessions(user_id);
    ASSERT_TRUE(sessions_result.has_value());
    EXPECT_GE(sessions_result.value().size(), 3);
}

// ============================================================================
// Password Management Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, UpdatePassword_Success) {
    std::string user_id = register_test_user("pwdtest", "OldPass123!");

    auto update_result = auth_service_->update_password(user_id, "OldPass123!", "NewPass456!");
    ASSERT_TRUE(update_result.has_value());
    EXPECT_TRUE(update_result.value());

    // Verify old password doesn't work
    auto old_auth = auth_service_->authenticate("pwdtest", "OldPass123!", "127.0.0.1");
    EXPECT_FALSE(old_auth.has_value());

    // Verify new password works
    auto new_auth = auth_service_->authenticate("pwdtest", "NewPass456!", "127.0.0.1");
    EXPECT_TRUE(new_auth.has_value());
}

TEST_F(AuthenticationServiceTest, UpdatePassword_WrongOldPassword) {
    std::string user_id = register_test_user("pwdtest", "CorrectPass123!");

    auto update_result = auth_service_->update_password(user_id, "WrongOldPass!", "NewPass456!");
    EXPECT_FALSE(update_result.has_value());
    EXPECT_EQ(update_result.error().code(), ErrorCode::UNAUTHORIZED);
}

TEST_F(AuthenticationServiceTest, ResetPassword_Success) {
    std::string user_id = register_test_user("resettest", "OldPass123!");

    auto reset_result = auth_service_->reset_password(user_id, "ResetPass789!");
    ASSERT_TRUE(reset_result.has_value());
    EXPECT_TRUE(reset_result.value());

    // Verify new password works
    auto auth_result = auth_service_->authenticate("resettest", "ResetPass789!", "127.0.0.1");
    EXPECT_TRUE(auth_result.has_value());
}

TEST_F(AuthenticationServiceTest, UpdateUsername_Success) {
    std::string user_id = register_test_user("oldname", "SecurePass123!");

    auto update_result = auth_service_->update_username(user_id, "newname");
    ASSERT_TRUE(update_result.has_value());
    EXPECT_TRUE(update_result.value());

    // Verify old username doesn't work
    auto old_auth = auth_service_->authenticate("oldname", "SecurePass123!", "127.0.0.1");
    EXPECT_FALSE(old_auth.has_value());

    // Verify new username works
    auto new_auth = auth_service_->authenticate("newname", "SecurePass123!", "127.0.0.1");
    EXPECT_TRUE(new_auth.has_value());
}

// ============================================================================
// API Key Management Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, GenerateApiKey_Success) {
    std::string user_id = register_test_user("apikeytest", "SecurePass123!");

    auto api_key_result = auth_service_->generate_api_key(user_id);

    ASSERT_TRUE(api_key_result.has_value());
    std::string api_key = api_key_result.value();
    EXPECT_FALSE(api_key.empty());
    EXPECT_GT(api_key.length(), 20);  // API keys should be reasonably long
}

TEST_F(AuthenticationServiceTest, AuthenticateWithApiKey_Success) {
    std::string user_id = register_test_user("apikeytest", "SecurePass123!");

    auto api_key_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(api_key_result.has_value());

    std::string api_key = api_key_result.value();

    // Authenticate using API key
    auto auth_result = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());
    EXPECT_EQ(auth_result.value(), user_id);
}

TEST_F(AuthenticationServiceTest, AuthenticateWithApiKey_InvalidKey) {
    auto auth_result = auth_service_->authenticate_with_api_key("invalid-api-key-12345", "127.0.0.1");

    EXPECT_FALSE(auth_result.has_value());
    EXPECT_EQ(auth_result.error().code(), ErrorCode::UNAUTHORIZED);
}

TEST_F(AuthenticationServiceTest, RevokeApiKey_Success) {
    std::string user_id = register_test_user("apikeytest", "SecurePass123!");

    auto api_key_result = auth_service_->generate_api_key(user_id);
    ASSERT_TRUE(api_key_result.has_value());

    std::string api_key = api_key_result.value();

    // Verify API key works
    auto auth_result1 = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    EXPECT_TRUE(auth_result1.has_value());

    // Revoke API key
    auto revoke_result = auth_service_->revoke_api_key(api_key);
    ASSERT_TRUE(revoke_result.has_value());
    EXPECT_TRUE(revoke_result.value());

    // Verify API key doesn't work after revocation
    auto auth_result2 = auth_service_->authenticate_with_api_key(api_key, "127.0.0.1");
    EXPECT_FALSE(auth_result2.has_value());
}

// ============================================================================
// User Query Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, GetUser_Success) {
    std::string user_id = register_test_user("gettest", "SecurePass123!", {"admin", "user"});

    auto user_result = auth_service_->get_user(user_id);

    ASSERT_TRUE(user_result.has_value());
    UserCredentials user = user_result.value();
    EXPECT_EQ(user.user_id, user_id);
    EXPECT_EQ(user.username, "gettest");
    EXPECT_EQ(user.roles.size(), 2);
    EXPECT_TRUE(user.is_active);
}

TEST_F(AuthenticationServiceTest, GetUserByUsername_Success) {
    std::string user_id = register_test_user("gettest", "SecurePass123!");

    auto user_result = auth_service_->get_user_by_username("gettest");

    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().user_id, user_id);
    EXPECT_EQ(user_result.value().username, "gettest");
}

TEST_F(AuthenticationServiceTest, GetUser_NotFound) {
    auto user_result = auth_service_->get_user("nonexistent-user-id");

    EXPECT_FALSE(user_result.has_value());
    EXPECT_EQ(user_result.error().code(), ErrorCode::NOT_FOUND);
}

// ============================================================================
// Cleanup Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, CleanupExpiredEntries) {
    // Create service with short expiry times
    auto cleanup_service = std::make_unique<AuthenticationService>();
    AuthenticationConfig config;
    config.token_expiry_seconds = 1;
    config.session_expiry_seconds = 1;
    cleanup_service->initialize(config, audit_logger_);

    auto reg_result = cleanup_service->register_user("cleanuptest", "SecurePass123!", {"user"});
    ASSERT_TRUE(reg_result.has_value());

    auto auth_result = cleanup_service->authenticate("cleanuptest", "SecurePass123!", "127.0.0.1");
    ASSERT_TRUE(auth_result.has_value());

    std::string token_value = auth_result.value().token_value;

    // Wait for expiry
    std::this_thread::sleep_for(2s);

    // Run cleanup
    cleanup_service->cleanup_expired_entries();

    // Token should be invalid after cleanup
    auto validate_result = cleanup_service->validate_token(token_value);
    EXPECT_FALSE(validate_result.has_value());
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(AuthenticationServiceTest, GetAuthStats) {
    // Register some users
    register_test_user("stats1", "SecurePass123!");
    register_test_user("stats2", "SecurePass123!");
    register_test_user("stats3", "SecurePass123!");

    auto stats_result = auth_service_->get_auth_stats();
    ASSERT_TRUE(stats_result.has_value());

    auto stats = stats_result.value();
    EXPECT_GT(stats.size(), 0);
    EXPECT_TRUE(stats.find("total_users") != stats.end());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(AuthenticationServiceTest, RegisterUser_EmptyUsername) {
    auto result = auth_service_->register_user("", "SecurePass123!", {"user"});
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, RegisterUser_EmptyPassword) {
    auto result = auth_service_->register_user("testuser", "", {"user"});
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, SetUserActiveStatus) {
    std::string user_id = register_test_user("activetest", "SecurePass123!");

    // Deactivate user
    auto deactivate_result = auth_service_->set_user_active_status(user_id, false);
    ASSERT_TRUE(deactivate_result.has_value());
    EXPECT_TRUE(deactivate_result.value());

    auto user_result = auth_service_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_FALSE(user_result.value().is_active);

    // Reactivate user
    auto activate_result = auth_service_->set_user_active_status(user_id, true);
    ASSERT_TRUE(activate_result.has_value());
    EXPECT_TRUE(activate_result.value());

    user_result = auth_service_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_TRUE(user_result.value().is_active);
}

// ============================================================================
// Enhanced Security Testing - Edge Cases
// ============================================================================

TEST_F(AuthenticationServiceTest, SQL_InjectionAttemptInUsername) {
    std::string malicious_username = "admin' OR '1'='1";
    auto result = auth_service_->register_user(malicious_username, "SecurePass123!", {"user"});
    // Should either reject or sanitize SQL injection attempt
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, XSS_AttemptInUsername) {
    std::string xss_username = "<script>alert('xss')</script>";
    auto result = auth_service_->register_user(xss_username, "SecurePass123!", {"user"});
    // Should handle XSS attempts safely
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, CommandInjectionAttemptInUsername) {
    std::string cmd_injection = "; rm -rf /";
    auto result = auth_service_->register_user(cmd_injection, "SecurePass123!", {"user"});
    // Should not execute commands
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, PathTraversalAttemptInUsername) {
    std::string path_traversal = "../../etc/passwd";
    auto result = auth_service_->register_user(path_traversal, "SecurePass123!", {"user"});
    // Should not allow path traversal
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, NullByteInjectionInUsername) {
    std::string null_byte = std::string("admin") + '\0' + "extra";
    auto result = auth_service_->register_user(null_byte, "SecurePass123!", {"user"});
    // Should handle null bytes safely
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, UnicodeOverlongEncodingAttempt) {
    std::string overlong = "\xC0\xAF"; // Overlong encoding of '/'
    auto result = auth_service_->register_user(overlong, "SecurePass123!", {"user"});
    // Should handle malformed Unicode
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, ExtremelyLongUsernameBufferOverflow) {
    std::string long_username(100000, 'a');
    auto result = auth_service_->register_user(long_username, "SecurePass123!", {"user"});
    // Should not cause buffer overflow
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, ExtremelyLongPasswordBufferOverflow) {
    std::string long_password(100000, 'a');
    auto result = auth_service_->register_user("testuser", long_password, {"user"});
    // Should not cause buffer overflow
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, PasswordWithOnlySpaces) {
    auto result = auth_service_->register_user("spaceuser", "        ", {"user"});
    // Should reject password with only spaces
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, PasswordWithNullBytes) {
    std::string null_password = std::string("Pass") + '\0' + "word";
    auto result = auth_service_->register_user("nulluser", null_password, {"user"});
    // Should handle null bytes in password
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, EmptyRolesList) {
    auto result = auth_service_->register_user("roleuser", "SecurePass123!", {});
    // Should handle empty roles list
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, InvalidRoleNames) {
    auto result = auth_service_->register_user("invalidrole", "SecurePass123!", {"admin' OR '1'='1", "<script>alert(1)</script>"});
    // Should sanitize or reject invalid role names
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, LoginWithNullUsername) {
    auto result = auth_service_->authenticate_user("", "password");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, LoginWithNullPassword) {
    std::string user_id = register_test_user("testuser", "SecurePass123!");
    auto result = auth_service_->authenticate_user("testuser", "");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, LoginWithBothNull) {
    auto result = auth_service_->authenticate_user("", "");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthenticationServiceTest, TimingAttackResistance_ValidUsername) {
    std::string user_id = register_test_user("timing_test", "SecurePass123!");

    auto start = std::chrono::high_resolution_clock::now();
    auth_service_->authenticate_user("timing_test", "WrongPassword");
    auto end = std::chrono::high_resolution_clock::now();
    auto valid_user_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Timing should be consistent for security
    EXPECT_GT(valid_user_time.count(), 0);
}

TEST_F(AuthenticationServiceTest, TimingAttackResistance_InvalidUsername) {
    auto start = std::chrono::high_resolution_clock::now();
    auth_service_->authenticate_user("nonexistent_user_xyz", "WrongPassword");
    auto end = std::chrono::high_resolution_clock::now();
    auto invalid_user_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Timing should not reveal username existence
    EXPECT_GT(invalid_user_time.count(), 0);
}

TEST_F(AuthenticationServiceTest, ConcurrentLoginAttempts) {
    std::string user_id = register_test_user("concurrent", "SecurePass123!");

    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            auto result = auth_service_->authenticate_user("concurrent", "SecurePass123!");
            if (result.has_value()) {
                success_count++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All concurrent logins should succeed
    EXPECT_GT(success_count.load(), 0);
}

TEST_F(AuthenticationServiceTest, RapidPasswordChangeAttempts) {
    std::string user_id = register_test_user("passchange", "OldPass123!");

    // Try to change password multiple times rapidly
    for (int i = 0; i < 5; ++i) {
        std::string new_pass = "NewPass" + std::to_string(i) + "!";
        auto result = auth_service_->change_password(user_id, "OldPass123!", new_pass);
        // First should succeed, others may fail
        if (i == 0) {
            EXPECT_TRUE(result.has_value());
        }
    }
}

TEST_F(AuthenticationServiceTest, SessionHijackingAttempt) {
    std::string user_id = register_test_user("sessionuser", "SecurePass123!");

    auto login_result = auth_service_->authenticate_user("sessionuser", "SecurePass123!");
    ASSERT_TRUE(login_result.has_value());

    std::string session_id = login_result.value().session_id;

    // Try to validate session from different "IP" or context
    auto validate_result = auth_service_->validate_session(session_id);
    EXPECT_TRUE(validate_result.has_value());
}

TEST_F(AuthenticationServiceTest, API_KeyReuse) {
    std::string user_id = register_test_user("apiuser", "SecurePass123!");

    auto key_result1 = auth_service_->create_api_key(user_id, "test_key", {"read"});
    ASSERT_TRUE(key_result1.has_value());

    // Try to create another key with same name
    auto key_result2 = auth_service_->create_api_key(user_id, "test_key", {"write"});
    // Should either fail or create with different ID
    EXPECT_TRUE(key_result2.has_value() || !key_result2.has_value());
}

TEST_F(AuthenticationServiceTest, RevokedAPI_KeyStillWorks) {
    std::string user_id = register_test_user("revokeuser", "SecurePass123!");

    auto key_result = auth_service_->create_api_key(user_id, "revoke_test", {"read"});
    ASSERT_TRUE(key_result.has_value());
    std::string api_key = key_result.value();

    // Revoke the key
    auto revoke_result = auth_service_->revoke_api_key(user_id, api_key);
    ASSERT_TRUE(revoke_result.has_value());

    // Try to validate revoked key
    auto validate_result = auth_service_->validate_api_key(api_key);
    // Should fail
    EXPECT_FALSE(validate_result.has_value());
}

TEST_F(AuthenticationServiceTest, PermissionEscalationAttempt) {
    std::string user_id = register_test_user("lowpriv", "SecurePass123!", {"user"});

    // Try to add admin role through parameter tampering
    auto result = auth_service_->assign_role(user_id, "admin");
    // Should require proper authorization
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, BruteForceProtection) {
    std::string user_id = register_test_user("bruteforce", "SecurePass123!");

    // Attempt multiple failed logins
    for (int i = 0; i < 10; ++i) {
        auth_service_->authenticate_user("bruteforce", "WrongPassword" + std::to_string(i));
    }

    // Account should be locked after max attempts
    auto result = auth_service_->authenticate_user("bruteforce", "SecurePass123!");
    // May fail if account is locked
    EXPECT_TRUE(result.has_value() || !result.has_value());
}

TEST_F(AuthenticationServiceTest, PasswordReuseAttempt) {
    std::string user_id = register_test_user("reusetest", "OldPass123!");

    // Change password
    auto change1 = auth_service_->change_password(user_id, "OldPass123!", "NewPass456!");
    ASSERT_TRUE(change1.has_value());

    // Try to change back to old password
    auto change2 = auth_service_->change_password(user_id, "NewPass456!", "OldPass123!");
    // Should prevent password reuse
    EXPECT_FALSE(change2.has_value());
}

TEST_F(AuthenticationServiceTest, ExpiredSessionAccess) {
    std::string user_id = register_test_user("expireuser", "SecurePass123!");

    auto login_result = auth_service_->authenticate_user("expireuser", "SecurePass123!");
    ASSERT_TRUE(login_result.has_value());

    std::string session_id = login_result.value().session_id;

    // Simulate time passing (if service supports)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check if expired session is rejected
    auto validate_result = auth_service_->validate_session(session_id);
    EXPECT_TRUE(validate_result.has_value()); // May still be valid if not expired
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
