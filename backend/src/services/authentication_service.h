#ifndef JADEVECTORDB_AUTHENTICATION_SERVICE_H
#define JADEVECTORDB_AUTHENTICATION_SERVICE_H

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "security_audit_logger.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <vector>

namespace jadevectordb {

// User authentication credentials
struct UserCredentials {
    std::string user_id;
    std::string username;
    std::string password_hash;
    std::string salt;
    std::vector<std::string> roles;
    bool is_active;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_login;
    int failed_login_attempts;

    UserCredentials() : is_active(true), failed_login_attempts(0) {}
};

// Authentication token
struct AuthToken {
    std::string token_id;
    std::string user_id;
    std::string token_value;
    std::chrono::system_clock::time_point issued_at;
    std::chrono::system_clock::time_point expires_at;
    bool is_valid;
    std::string ip_address;
    std::string user_agent;

    AuthToken() : is_valid(true) {}
};

// Authentication session
struct AuthSession {
    std::string session_id;
    std::string user_id;
    std::string token_id;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_activity;
    std::chrono::system_clock::time_point expires_at;
    std::string ip_address;
    bool is_active;

    AuthSession() : is_active(true) {}
};

// Authentication configuration
struct AuthenticationConfig {
    bool enabled = true;
    int token_expiry_seconds = 3600;  // 1 hour default
    int session_expiry_seconds = 86400;  // 24 hours default
    int max_failed_attempts = 5;
    int account_lockout_duration_seconds = 900;  // 15 minutes
    bool require_strong_passwords = true;
    int min_password_length = 8;
    bool enable_two_factor = false;
    bool enable_api_keys = true;
    std::string password_hash_algorithm = "bcrypt";  // bcrypt, argon2, scrypt
    int bcrypt_work_factor = 12;
    bool log_authentication_events = true;
};

/**
 * @brief Authentication Service for User Authentication and Session Management
 *
 * This service handles user authentication, token generation and validation,
 * session management, and integrates with the security audit logger.
 */
class AuthenticationService {
private:
    std::shared_ptr<logging::Logger> logger_;
    AuthenticationConfig config_;
    std::shared_ptr<SecurityAuditLogger> audit_logger_;

    // User credentials storage (in production, this would be a database)
    std::unordered_map<std::string, UserCredentials> users_;

    // Active tokens
    std::unordered_map<std::string, AuthToken> tokens_;

    // Active sessions
    std::unordered_map<std::string, AuthSession> sessions_;

    // API keys (user_id -> api_key)
    std::unordered_map<std::string, std::string> api_keys_;

    mutable std::mutex users_mutex_;
    mutable std::mutex tokens_mutex_;
    mutable std::mutex sessions_mutex_;
    mutable std::mutex api_keys_mutex_;

public:
    explicit AuthenticationService();
    ~AuthenticationService() = default;

    // Initialize authentication service
    bool initialize(const AuthenticationConfig& config,
                   std::shared_ptr<SecurityAuditLogger> audit_logger = nullptr);

    // User registration
    Result<std::string> register_user(const std::string& username,
                                     const std::string& password,
                                     const std::vector<std::string>& roles,
                                     const std::string& user_id_override = "");
    
    // Update username
    Result<bool> update_username(const std::string& user_id,
                                const std::string& new_username);

    // User authentication
    Result<AuthToken> authenticate(const std::string& username,
                                  const std::string& password,
                                  const std::string& ip_address = "",
                                  const std::string& user_agent = "");

    // Authenticate with API key
    Result<std::string> authenticate_with_api_key(const std::string& api_key,
                                                 const std::string& ip_address = "");

    // Validate token
    Result<std::string> validate_token(const std::string& token_value);

    // Refresh token
    Result<AuthToken> refresh_token(const std::string& token_value,
                                   const std::string& ip_address = "");

    // Revoke token
    Result<bool> revoke_token(const std::string& token_value);

    // Logout (revoke token and end session)
    Result<bool> logout(const std::string& token_value);

    // Create session
    Result<AuthSession> create_session(const std::string& user_id,
                                       const std::string& token_id,
                                       const std::string& ip_address);

    // Validate session
    Result<bool> validate_session(const std::string& session_id);

    // End session
    Result<bool> end_session(const std::string& session_id);

    // Update user password
    Result<bool> update_password(const std::string& user_id,
                                const std::string& old_password,
                                const std::string& new_password);

    // Reset password (admin function)
    Result<bool> reset_password(const std::string& user_id,
                               const std::string& new_password);

    // Generate API key for user
    Result<std::string> generate_api_key(const std::string& user_id);

    // Revoke API key
    Result<bool> revoke_api_key(const std::string& api_key);

    // Get user by ID
    Result<UserCredentials> get_user(const std::string& user_id) const;

    // Get user by username
    Result<UserCredentials> get_user_by_username(const std::string& username) const;

    // Update user status
    Result<bool> set_user_active_status(const std::string& user_id, bool is_active);

    // Check if user is locked out
    bool is_user_locked_out(const std::string& user_id) const;

    // Get active sessions for user
    Result<std::vector<AuthSession>> get_user_sessions(const std::string& user_id) const;

    // Cleanup expired tokens and sessions
    void cleanup_expired_entries();

    // Get authentication statistics
    Result<std::unordered_map<std::string, std::string>> get_auth_stats() const;

    // Update authentication configuration
    Result<bool> update_config(const AuthenticationConfig& new_config);

    // Get current configuration
    AuthenticationConfig get_config() const;

private:
    // Password hashing
    std::string hash_password(const std::string& password, const std::string& salt) const;

    // Generate salt
    std::string generate_salt() const;

    // Verify password
    bool verify_password(const std::string& password,
                        const std::string& password_hash,
                        const std::string& salt) const;

    // Validate password strength
    bool is_strong_password(const std::string& password) const;

    // Generate token
    std::string generate_token() const;

    // Generate session ID
    std::string generate_session_id() const;

    // Generate API key
    std::string generate_api_key_value() const;

    // Check if token is expired
    bool is_token_expired(const AuthToken& token) const;

    // Check if session is expired
    bool is_session_expired(const AuthSession& session) const;

    // Log authentication event
    void log_auth_event(SecurityEventType event_type,
                       const std::string& user_id,
                       const std::string& ip_address,
                       bool success,
                       const std::string& details = "");

    // Handle failed login attempt
    void handle_failed_login(const std::string& user_id,
                            const std::string& ip_address);

    // Reset failed login attempts
    void reset_failed_login_attempts(const std::string& user_id);

    // Validate configuration
    bool validate_config(const AuthenticationConfig& config) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_AUTHENTICATION_SERVICE_H
