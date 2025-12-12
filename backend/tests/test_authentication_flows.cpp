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

// Test fixture for authentication service tests
class AuthenticationFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize security audit logger
        audit_logger_ = std::make_shared<SecurityAuditLogger>();

        // Initialize authentication service
        auth_service_ = std::make_unique<AuthenticationService>();

        AuthenticationConfig config;
        config.require_strong_passwords = true;
        config.token_expiry_seconds = 86400;  // 24 hours
        config.max_failed_attempts = 3;
        config.account_lockout_duration_seconds = 1800;  // 30 minutes
        config.session_expiry_seconds = 7200;  // 120 minutes

        auth_service_->initialize(config, audit_logger_);
    }

    void TearDown() override {
        // Clean up is handled automatically
    }

    std::unique_ptr<AuthenticationService> auth_service_;
    std::shared_ptr<SecurityAuditLogger> audit_logger_;
};

// ============================================================================
// USER REGISTRATION TESTS
// ============================================================================

TEST_F(AuthenticationFlowTest, RegisterUserWithValidCredentials) {
    auto result = auth_service_->register_user(
        "testuser",
        "SecurePassword123!",
        {"user"}
    );

    ASSERT_TRUE(result.has_value()) << "Registration should succeed with valid credentials";
    EXPECT_FALSE(result.value().empty()) << "User ID should not be empty";
}

TEST_F(AuthenticationFlowTest, RegisterUserWithMultipleRoles) {
    auto result = auth_service_->register_user(
        "admin_user",
        "AdminPass123!",
        {"admin", "developer", "user"}
    );

    ASSERT_TRUE(result.has_value()) << "Registration should succeed with multiple roles";

    // Verify user can be retrieved
    auto user_result = auth_service_->get_user(result.value());
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().username, "admin_user");
    EXPECT_EQ(user_result.value().roles.size(), 3);
}

TEST_F(AuthenticationFlowTest, RegisterUserWithDuplicateUsername) {
    // Register first user
    auto result1 = auth_service_->register_user(
        "duplicate_user",
        "Password123!",
        {"user"}
    );
    ASSERT_TRUE(result1.has_value());

    // Attempt to register with same username
    auto result2 = auth_service_->register_user(
        "duplicate_user",
        "DifferentPass123!",
        {"user"}
    );

    ASSERT_FALSE(result2.has_value()) << "Registration should fail with duplicate username";
    EXPECT_EQ(result2.error().code, ErrorCode::ALREADY_EXISTS);
}

TEST_F(AuthenticationFlowTest, RegisterUserWithWeakPassword) {
    auto result = auth_service_->register_user(
        "weakpass_user",
        "weak",
        {"user"}
    );

    ASSERT_FALSE(result.has_value()) << "Registration should fail with weak password";
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_ARGUMENT);
}

TEST_F(AuthenticationFlowTest, RegisterUserWithEmptyCredentials) {
    auto result1 = auth_service_->register_user("", "Password123!", {"user"});
    ASSERT_FALSE(result1.has_value()) << "Registration should fail with empty username";

    auto result2 = auth_service_->register_user("user", "", {"user"});
    ASSERT_FALSE(result2.has_value()) << "Registration should fail with empty password";
}

// ============================================================================
// LOGIN TESTS
// ============================================================================

TEST_F(AuthenticationFlowTest, LoginWithValidCredentials) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "login_user",
        "LoginPass123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    // Login
    auto login_result = auth_service_->authenticate(
        "login_user",
        "LoginPass123!"
    );

    ASSERT_TRUE(login_result.has_value()) << "Login should succeed with valid credentials";
    EXPECT_FALSE(login_result.value().token_value.empty()) << "Token should not be empty";
    EXPECT_EQ(login_result.value().user_id, reg_result.value());
}

TEST_F(AuthenticationFlowTest, LoginWithWrongPassword) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "wrong_pass_user",
        "CorrectPass123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    // Attempt login with wrong password
    auto login_result = auth_service_->authenticate(
        "wrong_pass_user",
        "WrongPass123!"
    );

    ASSERT_FALSE(login_result.has_value()) << "Login should fail with wrong password";
    EXPECT_EQ(login_result.error().code, ErrorCode::AUTHENTICATION_ERROR);
}

TEST_F(AuthenticationFlowTest, LoginWithNonExistentUser) {
    auto login_result = auth_service_->authenticate(
        "nonexistent_user",
        "AnyPassword123!"
    );

    ASSERT_FALSE(login_result.has_value()) << "Login should fail for non-existent user";
    EXPECT_EQ(login_result.error().code, ErrorCode::AUTHENTICATION_ERROR);
}

TEST_F(AuthenticationFlowTest, LoginWithInactiveUser) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "inactive_user",
        "Password123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    // Deactivate user
    auto deactivate_result = auth_service_->set_user_active_status(reg_result.value(), false);
    ASSERT_TRUE(deactivate_result.has_value());

    // Attempt login
    auto login_result = auth_service_->authenticate(
        "inactive_user",
        "Password123!"
    );

    ASSERT_FALSE(login_result.has_value()) << "Login should fail for inactive user";
}

TEST_F(AuthenticationFlowTest, AccountLockoutAfterFailedAttempts) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "lockout_user",
        "CorrectPass123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    // Attempt login with wrong password 3 times (max_failed_login_attempts)
    for (int i = 0; i < 3; i++) {
        auto login_result = auth_service_->authenticate(
            "lockout_user",
            "WrongPass123!"
        );
        EXPECT_FALSE(login_result.has_value());
    }

    // Fourth attempt should fail even with correct password (account locked)
    auto login_result = auth_service_->authenticate(
        "lockout_user",
        "CorrectPass123!"
    );

    ASSERT_FALSE(login_result.has_value()) << "Login should fail for locked account";
}

// ============================================================================
// LOGOUT TESTS
// ============================================================================

TEST_F(AuthenticationFlowTest, LogoutWithValidToken) {
    // Register and login
    auto reg_result = auth_service_->register_user(
        "logout_user",
        "Password123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    auto login_result = auth_service_->authenticate(
        "logout_user",
        "Password123!"
    );
    ASSERT_TRUE(login_result.has_value());

    std::string token = login_result.value().token_value;

    // Logout
    auto logout_result = auth_service_->logout(token);
    ASSERT_TRUE(logout_result.has_value()) << "Logout should succeed with valid token";

    // Verify token is no longer valid
    auto verify_result = auth_service_->validate_token(token);
    EXPECT_FALSE(verify_result.has_value()) << "Token should be invalid after logout";
}

TEST_F(AuthenticationFlowTest, LogoutWithInvalidToken) {
    auto logout_result = auth_service_->logout("invalid_token_12345");

    ASSERT_FALSE(logout_result.has_value()) << "Logout should fail with invalid token";
}

TEST_F(AuthenticationFlowTest, LogoutTwiceWithSameToken) {
    // Register and login
    auto reg_result = auth_service_->register_user(
        "double_logout_user",
        "Password123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    auto login_result = auth_service_->authenticate(
        "double_logout_user",
        "Password123!"
    );
    ASSERT_TRUE(login_result.has_value());

    std::string token = login_result.value().token_value;

    // First logout
    auto logout1 = auth_service_->logout(token);
    ASSERT_TRUE(logout1.has_value());

    // Second logout with same token
    auto logout2 = auth_service_->logout(token);
    EXPECT_FALSE(logout2.has_value()) << "Second logout should fail";
}

// ============================================================================
// PASSWORD RESET TESTS
// ============================================================================

// DISABLED: forgot_password() API does not exist
/*
TEST_F(AuthenticationFlowTest, RequestPasswordReset) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "reset_user",
        "OldPassword123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());
    std::string user_id = reg_result.value();

    // Request password reset
    auto reset_request = auth_service_->forgot_password(
        "reset_user",
        "reset_user@example.com"
    );

    ASSERT_TRUE(reset_request.has_value()) << "Password reset request should succeed";
    EXPECT_FALSE(reset_request.value().empty()) << "Reset token should not be empty";
}
*/

// DISABLED: forgot_password() and token-based reset not implemented
/*
TEST_F(AuthenticationFlowTest, ResetPasswordWithValidToken) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "reset_pass_user",
        "OldPassword123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());
    std::string user_id = reg_result.value();

    // Request password reset
    auto reset_token_result = auth_service_->forgot_password(
        "reset_pass_user",
        "user@example.com"
    );
    ASSERT_TRUE(reset_token_result.has_value());
    std::string reset_token = reset_token_result.value();

    // Reset password
    auto reset_result = auth_service_->reset_password(
        user_id,
        reset_token,
        "NewSecurePass123!"
    );

    ASSERT_TRUE(reset_result.has_value()) << "Password reset should succeed with valid token";

    // Verify new password works
    auto login_result = auth_service_->authenticate(
        "reset_pass_user",
        "NewSecurePass123!"
    );
    EXPECT_TRUE(login_result.has_value()) << "Login should succeed with new password";

    // Verify old password doesn't work
    auto old_login_result = auth_service_->authenticate(
        "reset_pass_user",
        "OldPassword123!"
    );
    EXPECT_FALSE(old_login_result.has_value()) << "Login should fail with old password";
}
*/

// DISABLED: forgot_password() and token-based reset not implemented
/*
TEST_F(AuthenticationFlowTest, ResetPasswordWithInvalidToken) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "invalid_token_user",
        "OldPassword123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());
    std::string user_id = reg_result.value();

    // Attempt reset with invalid token
    auto reset_result = auth_service_->reset_password(
        user_id,
        "invalid_reset_token",
        "NewPassword123!"
    );

    ASSERT_FALSE(reset_result.has_value()) << "Password reset should fail with invalid token";
}
*/

// DISABLED: forgot_password() and token-based reset not implemented
/*
TEST_F(AuthenticationFlowTest, ResetPasswordWithWeakPassword) {
    // Register user
    auto reg_result = auth_service_->register_user(
        "weak_reset_user",
        "OldPassword123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());
    std::string user_id = reg_result.value();

    // Request password reset
    auto reset_token_result = auth_service_->forgot_password(
        "weak_reset_user",
        "user@example.com"
    );
    ASSERT_TRUE(reset_token_result.has_value());
    std::string reset_token = reset_token_result.value();

    // Attempt reset with weak password
    auto reset_result = auth_service_->reset_password(
        user_id,
        reset_token,
        "weak"
    );

    ASSERT_FALSE(reset_result.has_value()) << "Password reset should fail with weak password";
}
*/

// ============================================================================
// TOKEN VERIFICATION TESTS
// ============================================================================

TEST_F(AuthenticationFlowTest, VerifyValidToken) {
    // Register and login
    auto reg_result = auth_service_->register_user(
        "verify_user",
        "Password123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());

    auto login_result = auth_service_->authenticate(
        "verify_user",
        "Password123!"
    );
    ASSERT_TRUE(login_result.has_value());

    std::string token = login_result.value().token_value;

    // Verify token
    auto verify_result = auth_service_->validate_token(token);
    ASSERT_TRUE(verify_result.has_value()) << "Token verification should succeed";
    EXPECT_EQ(verify_result.value(), reg_result.value());
}

TEST_F(AuthenticationFlowTest, VerifyInvalidToken) {
    auto verify_result = auth_service_->validate_token("completely_invalid_token");

    ASSERT_FALSE(verify_result.has_value()) << "Token verification should fail for invalid token";
}

TEST_F(AuthenticationFlowTest, VerifyExpiredToken) {
    // This test would require mocking time or waiting for token expiration
    // For now, we'll skip the actual expiration test
    // In a real scenario, you'd set a very short expiration time and wait
    GTEST_SKIP() << "Token expiration test requires time mocking";
}

// ============================================================================
// USER SESSION TESTS
// ============================================================================

TEST_F(AuthenticationFlowTest, GetUserSessions) {
    // Register and login multiple times
    auto reg_result = auth_service_->register_user(
        "session_user",
        "Password123!",
        {"user"}
    );
    ASSERT_TRUE(reg_result.has_value());
    std::string user_id = reg_result.value();

    // Create multiple sessions
    for (int i = 0; i < 3; i++) {
        auto login_result = auth_service_->authenticate(
            "session_user",
            "Password123!"
        );
        ASSERT_TRUE(login_result.has_value());
    }

    // Get sessions
    auto sessions_result = auth_service_->get_user_sessions(user_id);
    ASSERT_TRUE(sessions_result.has_value());
    EXPECT_GE(sessions_result.value().size(), 3) << "Should have at least 3 active sessions";
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST_F(AuthenticationFlowTest, CompleteAuthenticationFlow) {
    // 1. Register user
    auto reg_result = auth_service_->register_user(
        "flow_user",
        "InitialPass123!",
        {"user", "developer"}
    );
    ASSERT_TRUE(reg_result.has_value());
    std::string user_id = reg_result.value();

    // 2. Login
    auto login_result = auth_service_->authenticate(
        "flow_user",
        "InitialPass123!"
    );
    ASSERT_TRUE(login_result.has_value());
    std::string token = login_result.value().token_value;

    // 3. Verify token
    auto verify_result = auth_service_->validate_token(token);
    ASSERT_TRUE(verify_result.has_value());
    EXPECT_EQ(verify_result.value(), user_id);

    // 4. Logout
    auto logout_result = auth_service_->logout(token);
    ASSERT_TRUE(logout_result.has_value());

    // 5. Verify token is now invalid
    auto verify_after_logout = auth_service_->validate_token(token);
    EXPECT_FALSE(verify_after_logout.has_value());

    // 6-7. Password reset steps commented out - forgot_password API not implemented
    /*
    auto reset_token_result = auth_service_->forgot_password(
        "flow_user",
        "flow@example.com"
    );
    ASSERT_TRUE(reset_token_result.has_value());

    auto reset_result = auth_service_->reset_password(
        user_id,
        reset_token_result.value(),
        "NewPassword123!"
    );
    ASSERT_TRUE(reset_result.has_value());

    auto new_login_result = auth_service_->authenticate(
        "flow_user",
        "NewPassword123!"
    );
    ASSERT_TRUE(new_login_result.has_value());
    */

    // 6. Test password update instead (using update_password which exists)
    auto update_result = auth_service_->update_password(
        user_id,
        "InitialPass123!",  // old password
        "NewPassword123!"    // new password
    );
    ASSERT_TRUE(update_result.has_value());

    // 7. Login with new password
    auto new_login_result = auth_service_->authenticate(
        "flow_user",
        "NewPassword123!"
    );
    ASSERT_TRUE(new_login_result.has_value());
}

} // namespace jadevectordb
