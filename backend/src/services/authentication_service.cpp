#include "authentication_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <openssl/sha.h>

namespace jadevectordb {

AuthenticationService::AuthenticationService() {
    logger_ = logging::LoggerManager::get_logger("AuthenticationService");
}

bool AuthenticationService::initialize(const AuthenticationConfig& config,
                                      std::shared_ptr<SecurityAuditLogger> audit_logger) {
    try {
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid authentication configuration");
            return false;
        }

        config_ = config;
        audit_logger_ = audit_logger;

        LOG_INFO(logger_, "AuthenticationService initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize: " + std::string(e.what()));
        return false;
    }
}

Result<std::string> AuthenticationService::register_user(const std::string& username,
                                                         const std::string& password,
                                                         const std::vector<std::string>& roles,
                                                         const std::string& user_id_override) {
    try {
        if (username.empty() || password.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Username and password are required");
        }

        // Check password strength
        if (config_.require_strong_passwords && !is_strong_password(password)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT,
                        "Password does not meet strength requirements");
        }

        std::lock_guard<std::mutex> lock(users_mutex_);

        // Check if username already exists
        for (const auto& [user_id, user] : users_) {
            if (user.username == username) {
                LOG_WARN(logger_, "Attempt to register existing username: " + username);
                RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Username already exists");
            }
        }

        // Generate user ID
        std::string user_id = user_id_override;
        if (user_id.empty()) {
            user_id = "user_" + generate_token();
        }

        // Generate salt and hash password
        std::string salt = generate_salt();
        std::string password_hash = hash_password(password, salt);

        // Create user credentials
        UserCredentials credentials;
        credentials.user_id = user_id;
        credentials.username = username;
        credentials.password_hash = password_hash;
        credentials.salt = salt;
        credentials.roles = roles;
        credentials.is_active = true;
        credentials.created_at = std::chrono::system_clock::now();
        credentials.failed_login_attempts = 0;

        users_[user_id] = credentials;

        LOG_INFO(logger_, "User registered successfully: " + username + " (" + user_id + ")");
        log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id, "", true,
                      "User registration");

        return user_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in register_user: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to register user: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::update_username(const std::string& user_id,
                                                    const std::string& new_username) {
    try {
        if (user_id.empty() || new_username.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "User id and username are required");
        }

        std::lock_guard<std::mutex> lock(users_mutex_);

        auto it = users_.find(user_id);
        if (it == users_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
        }

        for (const auto& entry : users_) {
            if (entry.first != user_id && entry.second.username == new_username) {
                RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Username already exists");
            }
        }

        it->second.username = new_username;
        log_auth_event(SecurityEventType::ADMIN_OPERATION, user_id, "", true,
                       "Username updated");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_username: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update username: " + std::string(e.what()));
    }
}

Result<AuthToken> AuthenticationService::authenticate(const std::string& username,
                                                      const std::string& password,
                                                      const std::string& ip_address,
                                                      const std::string& user_agent) {
    try {
        if (username.empty() || password.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Username and password are required");
        }

        std::lock_guard<std::mutex> lock(users_mutex_);

        // Find user by username
        UserCredentials* user_creds = nullptr;
        std::string user_id;

        for (auto& [id, creds] : users_) {
            if (creds.username == username) {
                user_creds = &creds;
                user_id = id;
                break;
            }
        }

        if (!user_creds) {
            LOG_WARN(logger_, "Authentication failed: user not found - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, "", ip_address, false,
                          "User not found");
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Invalid credentials");
        }

        // Check if user is active
        if (!user_creds->is_active) {
            LOG_WARN(logger_, "Authentication failed: inactive user - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, user_id, ip_address, false,
                          "User inactive");
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "User account is inactive");
        }

        // Check if user is locked out
        if (is_user_locked_out(user_id)) {
            LOG_WARN(logger_, "Authentication failed: user locked out - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, user_id, ip_address, false,
                          "Account locked out");
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Account is temporarily locked");
        }

        // Verify password
        if (!verify_password(password, user_creds->password_hash, user_creds->salt)) {
            handle_failed_login(user_id, ip_address);
            LOG_WARN(logger_, "Authentication failed: invalid password - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, user_id, ip_address, false,
                          "Invalid password");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid credentials");
        }

        // Authentication successful
        reset_failed_login_attempts(user_id);
        user_creds->last_login = std::chrono::system_clock::now();

        // Generate token
        AuthToken token;
        token.token_id = generate_token();
        token.user_id = user_id;
        token.token_value = generate_token();
        token.issued_at = std::chrono::system_clock::now();
        token.expires_at = token.issued_at + std::chrono::seconds(config_.token_expiry_seconds);
        token.is_valid = true;
        token.ip_address = ip_address;
        token.user_agent = user_agent;

        {
            std::lock_guard<std::mutex> token_lock(tokens_mutex_);
            tokens_[token.token_value] = token;
        }

        // Create session
        create_session(user_id, token.token_id, ip_address);

        LOG_INFO(logger_, "User authenticated successfully: " + username);
        log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id, ip_address, true,
                      "Login successful");

        return token;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in authenticate: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Authentication failed: " + std::string(e.what()));
    }
}

Result<std::string> AuthenticationService::authenticate_with_api_key(const std::string& api_key,
                                                                     const std::string& ip_address) {
    try {
        if (api_key.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "API key is required");
        }

        if (!config_.enable_api_keys) {
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "API key authentication is disabled");
        }

        std::lock_guard<std::mutex> lock(api_keys_mutex_);

        // Find user by API key
        for (const auto& [user_id, key] : api_keys_) {
            if (key == api_key) {
                LOG_INFO(logger_, "API key authentication successful for user: " + user_id);
                log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id, ip_address, true,
                              "API key authentication");
                return user_id;
            }
        }

        LOG_WARN(logger_, "Invalid API key used from IP: " + ip_address);
        log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, "", ip_address, false,
                      "Invalid API key");
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid API key");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in authenticate_with_api_key: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "API key authentication failed: " + std::string(e.what()));
    }
}

Result<std::string> AuthenticationService::validate_token(const std::string& token_value) {
    try {
        if (token_value.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Token is required");
        }

        std::lock_guard<std::mutex> lock(tokens_mutex_);

        auto it = tokens_.find(token_value);
        if (it == tokens_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Invalid token");
        }

        AuthToken& token = it->second;

        if (!token.is_valid) {
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Token has been revoked");
        }

        if (is_token_expired(token)) {
            token.is_valid = false;
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Token has expired");
        }

        return token.user_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in validate_token: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Token validation failed: " + std::string(e.what()));
    }
}

Result<AuthToken> AuthenticationService::refresh_token(const std::string& token_value,
                                                       const std::string& ip_address) {
    try {
        // Validate current token
        auto user_id_result = validate_token(token_value);
        if (!user_id_result.has_value()) {
            return tl::unexpected(user_id_result.error());
        }

        std::string user_id = user_id_result.value();

        // Revoke old token
        revoke_token(token_value);

        // Generate new token
        AuthToken new_token;
        new_token.token_id = generate_token();
        new_token.user_id = user_id;
        new_token.token_value = generate_token();
        new_token.issued_at = std::chrono::system_clock::now();
        new_token.expires_at = new_token.issued_at +
                              std::chrono::seconds(config_.token_expiry_seconds);
        new_token.is_valid = true;
        new_token.ip_address = ip_address;

        {
            std::lock_guard<std::mutex> lock(tokens_mutex_);
            tokens_[new_token.token_value] = new_token;
        }

        LOG_INFO(logger_, "Token refreshed for user: " + user_id);
        return new_token;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in refresh_token: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Token refresh failed: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::revoke_token(const std::string& token_value) {
    try {
        std::lock_guard<std::mutex> lock(tokens_mutex_);

        auto it = tokens_.find(token_value);
        if (it == tokens_.end()) {
            return true;  // Token doesn't exist, consider it revoked
        }

        it->second.is_valid = false;
        LOG_INFO(logger_, "Token revoked for user: " + it->second.user_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in revoke_token: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Token revocation failed: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::logout(const std::string& token_value) {
    try {
        auto user_id_result = validate_token(token_value);
        if (!user_id_result.has_value()) {
            return tl::unexpected(user_id_result.error());
        }

        // Revoke token
        revoke_token(token_value);

        // End all sessions for this token
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        for (auto& [session_id, session] : sessions_) {
            if (session.user_id == user_id_result.value()) {
                session.is_active = false;
            }
        }

        LOG_INFO(logger_, "User logged out: " + user_id_result.value());
        log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id_result.value(), "", true,
                      "Logout successful");

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in logout: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Logout failed: " + std::string(e.what()));
    }
}

Result<AuthSession> AuthenticationService::create_session(const std::string& user_id,
                                                          const std::string& token_id,
                                                          const std::string& ip_address) {
    try {
        AuthSession session;
        session.session_id = generate_session_id();
        session.user_id = user_id;
        session.token_id = token_id;
        session.created_at = std::chrono::system_clock::now();
        session.last_activity = session.created_at;
        session.expires_at = session.created_at +
                           std::chrono::seconds(config_.session_expiry_seconds);
        session.ip_address = ip_address;
        session.is_active = true;

        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            sessions_[session.session_id] = session;
        }

        LOG_DEBUG(logger_, "Session created for user: " + user_id);
        return session;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_session: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Session creation failed: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::validate_session(const std::string& session_id) {
    try {
        std::lock_guard<std::mutex> lock(sessions_mutex_);

        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Session not found");
        }

        AuthSession& session = it->second;

        if (!session.is_active) {
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Session is inactive");
        }

        if (is_session_expired(session)) {
            session.is_active = false;
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Session has expired");
        }

        // Update last activity
        session.last_activity = std::chrono::system_clock::now();
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in validate_session: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Session validation failed: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::end_session(const std::string& session_id) {
    try {
        std::lock_guard<std::mutex> lock(sessions_mutex_);

        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            it->second.is_active = false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in end_session: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "End session failed: " + std::string(e.what()));
    }
}

Result<std::string> AuthenticationService::generate_api_key(const std::string& user_id) {
    try {
        if (!config_.enable_api_keys) {
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "API keys are disabled");
        }

        std::string api_key = generate_api_key_value();

        {
            std::lock_guard<std::mutex> lock(api_keys_mutex_);
            api_keys_[user_id] = api_key;
        }

        LOG_INFO(logger_, "API key generated for user: " + user_id);
        log_auth_event(SecurityEventType::CONFIGURATION_CHANGE, user_id, "", true,
                      "API key generated");

        return api_key;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in generate_api_key: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "API key generation failed: " + std::string(e.what()));
    }
}

void AuthenticationService::cleanup_expired_entries() {
    try {
        auto now = std::chrono::system_clock::now();

        // Cleanup expired tokens
        {
            std::lock_guard<std::mutex> lock(tokens_mutex_);
            for (auto it = tokens_.begin(); it != tokens_.end();) {
                if (!it->second.is_valid || is_token_expired(it->second)) {
                    it = tokens_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Cleanup expired sessions
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            for (auto it = sessions_.begin(); it != sessions_.end();) {
                if (!it->second.is_active || is_session_expired(it->second)) {
                    it = sessions_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        LOG_DEBUG(logger_, "Expired entries cleaned up");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cleanup_expired_entries: " + std::string(e.what()));
    }
}

// Private helper methods

std::string AuthenticationService::hash_password(const std::string& password,
                                                const std::string& salt) const {
    // Simple SHA-256 hash for demonstration (in production, use bcrypt/argon2)
    std::string salted_password = password + salt;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(salted_password.c_str()),
           salted_password.length(), hash);

    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return ss.str();
}

std::string AuthenticationService::generate_salt() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::stringstream ss;
    for (int i = 0; i < 16; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << dis(gen);
    }

    return ss.str();
}

bool AuthenticationService::verify_password(const std::string& password,
                                           const std::string& password_hash,
                                           const std::string& salt) const {
    std::string computed_hash = hash_password(password, salt);
    return computed_hash == password_hash;
}

bool AuthenticationService::is_strong_password(const std::string& password) const {
    if (password.length() < config_.min_password_length) {
        return false;
    }

    bool has_upper = false, has_lower = false, has_digit = false, has_special = false;

    for (char c : password) {
        if (std::isupper(c)) has_upper = true;
        if (std::islower(c)) has_lower = true;
        if (std::isdigit(c)) has_digit = true;
        if (std::ispunct(c)) has_special = true;
    }

    return has_upper && has_lower && has_digit && has_special;
}

std::string AuthenticationService::generate_token() const {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    std::stringstream ss;
    ss << std::hex << dis(gen) << dis(gen);

    return ss.str();
}

std::string AuthenticationService::generate_session_id() const {
    return "session_" + generate_token();
}

std::string AuthenticationService::generate_api_key_value() const {
    return "jadevdb_" + generate_token();
}

bool AuthenticationService::is_token_expired(const AuthToken& token) const {
    return std::chrono::system_clock::now() > token.expires_at;
}

bool AuthenticationService::is_session_expired(const AuthSession& session) const {
    return std::chrono::system_clock::now() > session.expires_at;
}

void AuthenticationService::log_auth_event(SecurityEventType event_type,
                                          const std::string& user_id,
                                          const std::string& ip_address,
                                          bool success,
                                          const std::string& details) {
    if (config_.log_authentication_events && audit_logger_) {
        SecurityEvent event(event_type, user_id, ip_address, "authentication",
                          "authenticate", success);
        event.details = details;
        audit_logger_->log_security_event(event);
    }
}

void AuthenticationService::handle_failed_login(const std::string& user_id,
                                               const std::string& ip_address) {
    auto it = users_.find(user_id);
    if (it != users_.end()) {
        it->second.failed_login_attempts++;
        LOG_WARN(logger_, "Failed login attempt " +
                std::to_string(it->second.failed_login_attempts) +
                " for user: " + user_id);
    }
}

void AuthenticationService::reset_failed_login_attempts(const std::string& user_id) {
    auto it = users_.find(user_id);
    if (it != users_.end()) {
        it->second.failed_login_attempts = 0;
    }
}

bool AuthenticationService::is_user_locked_out(const std::string& user_id) const {
    auto it = users_.find(user_id);
    if (it != users_.end()) {
        return it->second.failed_login_attempts >= config_.max_failed_attempts;
    }
    return false;
}

bool AuthenticationService::validate_config(const AuthenticationConfig& config) const {
    if (config.token_expiry_seconds <= 0) return false;
    if (config.session_expiry_seconds <= 0) return false;
    if (config.max_failed_attempts <= 0) return false;
    if (config.min_password_length < 4) return false;

    return true;
}

Result<bool> AuthenticationService::reset_password(const std::string& user_id,
                                                   const std::string& new_password) {
    try {
        if (user_id.empty() || new_password.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "User ID and new password are required");
        }

        // Check password strength if enabled
        if (config_.require_strong_passwords && !is_strong_password(new_password)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT,
                        "Password does not meet strength requirements");
        }

        std::lock_guard<std::mutex> lock(users_mutex_);

        auto it = users_.find(user_id);
        if (it == users_.end()) {
            LOG_WARN(logger_, "Attempt to reset password for non-existent user: " + user_id);
            RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found");
        }

        // Generate new salt and hash password
        std::string salt = generate_salt();
        std::string password_hash = hash_password(new_password, salt);

        // Update user credentials
        it->second.password_hash = password_hash;
        it->second.salt = salt;
        it->second.failed_login_attempts = 0;  // Reset failed attempts

        LOG_INFO(logger_, "Password reset successfully for user: " + user_id);
        log_auth_event(SecurityEventType::ADMIN_OPERATION, user_id, "", true,
                      "Password reset");

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in reset_password: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to reset password: " + std::string(e.what()));
    }
}

Result<UserCredentials> AuthenticationService::get_user_by_username(const std::string& username) const {
    try {
        if (username.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Username is required");
        }

        std::lock_guard<std::mutex> lock(users_mutex_);

        // Search for user by username
        for (const auto& [user_id, credentials] : users_) {
            if (credentials.username == username) {
                LOG_DEBUG(logger_, "Found user by username: " + username);
                return credentials;
            }
        }

        LOG_WARN(logger_, "User not found with username: " + username);
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_user_by_username: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get user: " + std::string(e.what()));
    }
}

Result<std::vector<AuthSession>> AuthenticationService::get_user_sessions(const std::string& user_id) const {
    try {
        if (user_id.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "User ID is required");
        }

        std::lock_guard<std::mutex> lock(sessions_mutex_);

        // Collect all sessions for the given user
        std::vector<AuthSession> user_sessions;
        for (const auto& [session_id, session] : sessions_) {
            if (session.user_id == user_id) {
                user_sessions.push_back(session);
            }
        }

        LOG_DEBUG(logger_, "Found " + std::to_string(user_sessions.size()) +
                  " sessions for user: " + user_id);

        return user_sessions;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_user_sessions: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get user sessions: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::seed_default_users() {
    try {
        // Check environment - only seed in non-production environments
        const char* env = std::getenv("JADE_ENV");
        std::string environment = env ? env : "development";

        // Convert to lowercase for comparison
        std::transform(environment.begin(), environment.end(), environment.begin(), ::tolower);

        // Only seed in development, test, or local environments
        if (environment != "development" &&
            environment != "dev" &&
            environment != "test" &&
            environment != "testing" &&
            environment != "local") {
            LOG_INFO(logger_, "Skipping default user seeding in " + environment + " environment");
            return true;  // Not an error, just skipping
        }

        LOG_INFO(logger_, "Seeding default users for " + environment + " environment");

        // Define default users
        struct DefaultUser {
            std::string username;
            std::string password;
            std::vector<std::string> roles;
            std::string user_id;
        };

        std::vector<DefaultUser> default_users = {
            {
                "admin",
                "admin123",  // Simple password for dev/test
                {"admin", "developer", "user"},
                "user_admin_default"
            },
            {
                "dev",
                "dev123",
                {"developer", "user"},
                "user_dev_default"
            },
            {
                "test",
                "test123",
                {"tester", "user"},
                "user_test_default"
            }
        };

        int created_count = 0;
        int skipped_count = 0;

        for (const auto& default_user : default_users) {
            // Check if user already exists (idempotent operation)
            bool user_exists = false;
            {
                std::lock_guard<std::mutex> lock(users_mutex_);
                for (const auto& [user_id, user] : users_) {
                    if (user.username == default_user.username) {
                        user_exists = true;
                        break;
                    }
                }
            }

            if (user_exists) {
                LOG_DEBUG(logger_, "Default user '" + default_user.username + "' already exists, skipping");
                skipped_count++;
                continue;
            }

            // Register the default user
            auto result = register_user(
                default_user.username,
                default_user.password,
                default_user.roles,
                default_user.user_id
            );

            if (result.has_value()) {
                LOG_INFO(logger_, "Created default user: " + default_user.username +
                        " with roles: [" + [&]() {
                            std::string roles_str;
                            for (size_t i = 0; i < default_user.roles.size(); ++i) {
                                if (i > 0) roles_str += ", ";
                                roles_str += default_user.roles[i];
                            }
                            return roles_str;
                        }() + "]");
                created_count++;
            } else {
                LOG_WARN(logger_, "Failed to create default user '" + default_user.username +
                        "': " + result.error().message);
            }
        }

        LOG_INFO(logger_, "Default user seeding complete: " +
                std::to_string(created_count) + " created, " +
                std::to_string(skipped_count) + " skipped");

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in seed_default_users: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to seed default users: " + std::string(e.what()));
    }
}

Result<bool> AuthenticationService::set_user_active_status(const std::string& user_id, bool is_active) {
    std::lock_guard<std::mutex> lock(users_mutex_);

    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }

    it->second.is_active = is_active;
    LOG_INFO(logger_, "User " + user_id + " active status set to " + (is_active ? "true" : "false"));

    return true;
}

Result<bool> AuthenticationService::revoke_api_key(const std::string& api_key) {
    std::lock_guard<std::mutex> lock(api_keys_mutex_);

    auto it = api_keys_.find(api_key);
    if (it == api_keys_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "API key not found");
    }

    std::string user_id = it->second;
    api_keys_.erase(it);
    LOG_INFO(logger_, "API key revoked for user: " + user_id);

    return true;
}

} // namespace jadevectordb
