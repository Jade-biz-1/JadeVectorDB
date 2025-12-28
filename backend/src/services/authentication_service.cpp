
#include "authentication_service.h"
#include "models/auth.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "metrics/prometheus_metrics.h"
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <openssl/sha.h>

namespace jadevectordb {

// TODO: Implement password update logic. This is a stub implementation to unblock builds and tests.
// The implementation should verify the old password, check password strength, update the hash and salt, and log the event.
Result<bool> AuthenticationService::update_password(const std::string& user_id,
                                                   const std::string& old_password,
                                                   const std::string& new_password) {
    // STUB IMPLEMENTATION
    // TODO: Implement full password update logic.
    // Steps should include:
    // 1. Validate input arguments.
    // 2. Lock users_mutex_.
    // 3. Find user by user_id.
    // 4. Verify old_password matches current password.
    // 5. Check new_password strength if required.
    // 6. Generate new salt and hash for new_password.
    // 7. Update user credentials and reset failed attempts.
    // 8. Log the password update event.
    // 9. Return appropriate Result<bool>.
    (void)user_id;
    (void)old_password;
    (void)new_password;
    RETURN_ERROR(ErrorCode::NOT_IMPLEMENTED, "update_password is not yet implemented");
}

AuthenticationService::AuthenticationService(const std::string& data_directory)
    : data_directory_(data_directory) {
    logger_ = logging::LoggerManager::get_logger("AuthenticationService");
    std::cout << "[DEBUG] AuthenticationService constructor - data_directory: " << data_directory << std::endl;
}

bool AuthenticationService::initialize(const AuthenticationConfig& config,
                                      std::shared_ptr<SecurityAuditLogger> audit_logger) {
    try {
        std::cout << "[DEBUG] AuthenticationService::initialize() called" << std::endl;

        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid authentication configuration");
            return false;
        }

        config_ = config;
        audit_logger_ = audit_logger;

        // Initialize SQLite persistence layer
        std::cout << "[DEBUG] Initializing SQLite persistence at: " << data_directory_ << std::endl;
        persistence_ = std::make_unique<SQLitePersistenceLayer>(data_directory_);

        auto init_result = persistence_->initialize();
        if (!init_result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize persistence layer: " +
                     ErrorHandler::format_error(init_result.error()));
            std::cout << "[DEBUG] Persistence initialization FAILED" << std::endl;
            return false;
        }

        std::cout << "[DEBUG] Persistence initialized successfully" << std::endl;
        LOG_INFO(logger_, "AuthenticationService initialized successfully with SQLite persistence");
        return true;
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] Exception in initialize: " << e.what() << std::endl;
        LOG_ERROR(logger_, "Exception in initialize: " + std::string(e.what()));
        return false;
    }
}

Result<std::string> AuthenticationService::register_user(const std::string& username,
                                                         const std::string& password,
                                                         const std::vector<std::string>& roles,
                                                         const std::string& user_id_override) {
    try {
        std::cout << "[DEBUG] register_user() called for: " << username << std::endl;

        if (username.empty() || password.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Username and password are required");
        }

        // Check password strength
        if (config_.require_strong_passwords && !is_strong_password(password)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT,
                        "Password does not meet strength requirements");
        }

        // Check if username already exists in database
        auto exists_result = persistence_->user_exists(username);
        if (!exists_result.has_value()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to check if user exists: " +
                        ErrorHandler::format_error(exists_result.error()));
        }
        if (exists_result.value()) {
            std::cout << "[DEBUG] User " << username << " already exists in database" << std::endl;
            LOG_WARN(logger_, "Attempt to register existing username: " + username);
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Username already exists");
        }

        // Generate user ID
        std::string user_id = user_id_override;
        if (user_id.empty()) {
            user_id = "user_" + generate_token();
        }

        // Generate salt and hash password
        std::string salt = generate_salt();
        std::string password_hash = hash_password(password, salt);

        std::cout << "[DEBUG] Creating user in database: " << username << " with ID: " << user_id << std::endl;

        // Create user in database
        std::string email = username + "@jadevectordb.local";  // Default email
        auto create_result = persistence_->create_user(username, email, password_hash, salt);
        if (!create_result.has_value()) {
            std::cout << "[DEBUG] Failed to create user in database: " <<
                      ErrorHandler::format_error(create_result.error()) << std::endl;
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create user: " +
                        ErrorHandler::format_error(create_result.error()));
        }

        user_id = create_result.value();  // Use the ID returned by persistence layer

        std::cout << "[DEBUG] User created successfully with ID: " << user_id << std::endl;

        // Assign roles to user
        for (const auto& role : roles) {
            std::cout << "[DEBUG] Assigning role: " << role << " to user: " << username << std::endl;
            auto role_result = persistence_->assign_role_to_user(user_id, role);
            if (!role_result.has_value()) {
                LOG_WARN(logger_, "Failed to assign role '" + role + "' to user: " +
                         ErrorHandler::format_error(role_result.error()));
            }
        }

        LOG_INFO(logger_, "User registered successfully: " + username + " (" + user_id + ")");
        log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id, "", true,
                      "User registration");

        std::cout << "[DEBUG] User " << username << " registered successfully" << std::endl;

        return user_id;
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] Exception in register_user: " << e.what() << std::endl;
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

Result<LocalAuthToken> AuthenticationService::authenticate(const std::string& username,
                                                      const std::string& password,
                                                      const std::string& ip_address,
                                                      const std::string& user_agent) {
    // Record metrics
    auto metrics = PrometheusMetricsManager::get_instance();
    auto timer = metrics->create_auth_timer("login");
    metrics->record_auth_request("login", "initiated");
    
    try {
        std::cout << "[DEBUG] authenticate() called for user: " << username << std::endl;

        if (username.empty() || password.empty()) {
            metrics->record_auth_request("login", "invalid_input");
            metrics->record_auth_error("login", "empty_credentials");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Username and password are required");
        }

        // Get user from database
        auto user_result = persistence_->get_user_by_username(username);
        if (!user_result.has_value()) {
            std::cout << "[DEBUG] User not found in database: " << username << std::endl;
            metrics->record_auth_request("login", "user_not_found");
            metrics->record_failed_login();
            LOG_WARN(logger_, "Authentication failed: user not found - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, "", ip_address, false,
                          "User not found");
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Invalid username or password");
        }

        auto& user = user_result.value();
        std::string user_id = user.user_id;

        std::cout << "[DEBUG] User found: " << username << ", user_id: " << user_id << std::endl;

        // Check if user is active
        if (!user.is_active) {
            std::cout << "[DEBUG] User is inactive: " << username << std::endl;
            metrics->record_auth_request("login", "inactive_user");
            metrics->record_failed_login();
            LOG_WARN(logger_, "Authentication failed: inactive user - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, user_id, ip_address, false,
                          "User inactive");
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "User account is inactive");
        }

        // Check if user is locked out
        if (user.account_locked_until > 0 && user.account_locked_until > std::time(nullptr)) {
            std::cout << "[DEBUG] User is locked: " << username << std::endl;
            metrics->record_auth_request("login", "locked_out");
            metrics->record_failed_login();
            LOG_WARN(logger_, "Authentication failed: user locked out - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, user_id, ip_address, false,
                          "Account locked out");
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "Account is temporarily locked");
        }

        // Verify password
        std::cout << "[DEBUG] Verifying password for user: " << username << std::endl;
        if (!verify_password(password, user.password_hash, user.salt)) {
            handle_failed_login(user_id, ip_address);
            metrics->record_auth_request("login", "invalid_password");
            metrics->record_failed_login();
            LOG_WARN(logger_, "Authentication failed: invalid password - " + username);
            log_auth_event(SecurityEventType::AUTHENTICATION_FAILURE, user_id, ip_address, false,
                          "Invalid password");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid credentials");
        }

        // Authentication successful
        std::cout << "[DEBUG] Authentication successful for user: " << username << std::endl;

        // Update last login in database
        auto update_result = persistence_->update_last_login(user_id);
        if (!update_result.has_value()) {
            LOG_WARN(logger_, "Failed to update last login for user: " + user_id);
        }

        reset_failed_login_attempts(user_id);

        // Generate token
        LocalAuthToken token;
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

        metrics->record_auth_request("login", "success");
        metrics->increment_active_sessions();
        LOG_INFO(logger_, "User authenticated successfully: " + username);
        log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id, ip_address, true,
                      "Login successful");

        return token;
    } catch (const std::exception& e) {
        metrics->record_auth_error("login", "exception");
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

        // Map is api_key -> user_id, so direct lookup
        auto it = api_keys_.find(api_key);
        if (it != api_keys_.end()) {
            std::string user_id = it->second;
            LOG_INFO(logger_, "API key authentication successful for user: " + user_id);
            log_auth_event(SecurityEventType::AUTHENTICATION_SUCCESS, user_id, ip_address, true,
                          "API key authentication");
            return user_id;
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

        LocalAuthToken& token = it->second;

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

Result<LocalAuthToken> AuthenticationService::refresh_token(const std::string& token_value,
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
        LocalAuthToken new_token;
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

Result<LocalAuthSession> AuthenticationService::create_session(const std::string& user_id,
                                                          const std::string& token_id,
                                                          const std::string& ip_address) {
    try {
        LocalAuthSession session;
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

        LocalAuthSession& session = it->second;

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

        // Validate that user exists
        {
            std::lock_guard<std::mutex> lock(users_mutex_);
            auto it = users_.find(user_id);
            if (it == users_.end()) {
                LOG_WARN(logger_, "Attempt to generate API key for non-existent user: " + user_id);
                RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
            }
        }

        std::string api_key = generate_api_key_value();

        {
            std::lock_guard<std::mutex> lock(api_keys_mutex_);
            api_keys_[api_key] = user_id;  // Map is api_key -> user_id
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

bool AuthenticationService::is_token_expired(const LocalAuthToken& token) const {
    // LocalAuthToken uses std::chrono::system_clock::time_point
    return std::chrono::system_clock::now() > token.expires_at;
}

bool AuthenticationService::is_session_expired(const LocalAuthSession& session) const {
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

Result<UserCredentials> AuthenticationService::get_user(const std::string& user_id) const {
    try {
        if (user_id.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "User ID is required");
        }

        {
            std::lock_guard<std::mutex> lock(users_mutex_);

            // Lookup user by ID in memory first
            auto it = users_.find(user_id);
            if (it != users_.end()) {
                LOG_DEBUG(logger_, "Found user in memory by ID: " + user_id);
                return it->second;
            }
        }

        // If not in memory, try database
        if (persistence_) {
            auto db_result = persistence_->get_user(user_id);
            if (db_result.has_value()) {
                LOG_DEBUG(logger_, "Found user in database by ID: " + user_id);
                // Convert User to UserCredentials
                const auto& db_user = db_result.value();
                UserCredentials creds;
                creds.user_id = db_user.user_id;
                creds.username = db_user.username;
                creds.email = db_user.email;
                creds.password_hash = db_user.password_hash;
                creds.salt = db_user.salt;
                creds.is_active = db_user.is_active;
                creds.failed_login_attempts = db_user.failed_login_attempts;

                // Convert int64_t timestamps to time_point
                creds.created_at = std::chrono::system_clock::time_point(
                    std::chrono::milliseconds(db_user.created_at));
                creds.last_login = std::chrono::system_clock::time_point(
                    std::chrono::milliseconds(db_user.last_login));

                // Get roles from database (separate query)
                auto roles_result = persistence_->get_user_roles(user_id);
                if (roles_result.has_value()) {
                    creds.roles = roles_result.value();
                }

                return creds;
            }
        }

        LOG_WARN(logger_, "User not found with ID: " + user_id);
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_user: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get user");
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

Result<std::vector<LocalAuthSession>> AuthenticationService::get_user_sessions(const std::string& user_id) const {
    try {
        if (user_id.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "User ID is required");
        }

        std::lock_guard<std::mutex> lock(sessions_mutex_);

        // Collect all sessions for the given user
        std::vector<LocalAuthSession> user_sessions;
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
        // DEBUG: Print to console to verify execution
        std::cout << "[DEBUG] seed_default_users() called" << std::endl;

        // Check environment - only seed in non-production environments
        const char* env = std::getenv("JADEVECTORDB_ENV");
        std::string environment = env ? env : "development";

        // DEBUG: Print environment variable
        std::cout << "[DEBUG] JADEVECTORDB_ENV = " << (env ? env : "NULL") << std::endl;
        std::cout << "[DEBUG] environment after default = " << environment << std::endl;

        // Convert to lowercase for comparison
        std::transform(environment.begin(), environment.end(), environment.begin(), ::tolower);

        std::cout << "[DEBUG] environment after lowercase = " << environment << std::endl;

        // Only seed in development, test, or local environments
        if (environment != "development" &&
            environment != "dev" &&
            environment != "test" &&
            environment != "testing" &&
            environment != "local") {
            std::cout << "[DEBUG] Skipping seeding - environment is: " << environment << std::endl;
            LOG_INFO(logger_, "Skipping default user seeding in " + environment + " environment");
            return true;  // Not an error, just skipping
        }

        std::cout << "[DEBUG] Proceeding with seeding for environment: " << environment << std::endl;
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

        std::cout << "[DEBUG] Starting to process " << default_users.size() << " default users" << std::endl;

        for (const auto& default_user : default_users) {
            std::cout << "[DEBUG] Processing user: " << default_user.username << std::endl;

            // Check if user already exists in database (idempotent operation)
            auto exists_result = persistence_->user_exists(default_user.username);
            if (exists_result.has_value() && exists_result.value()) {
                std::cout << "[DEBUG] User '" << default_user.username << "' already exists in database, skipping" << std::endl;
                LOG_DEBUG(logger_, "Default user '" + default_user.username + "' already exists, skipping");
                skipped_count++;
                continue;
            }

            std::cout << "[DEBUG] Attempting to register user: " << default_user.username << std::endl;

            // Register the default user (this will persist to database)
            auto result = register_user(
                default_user.username,
                default_user.password,
                default_user.roles,
                default_user.user_id
            );

            if (result.has_value()) {
                std::cout << "[DEBUG] Successfully created user: " << default_user.username << std::endl;
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
                std::cout << "[DEBUG] Failed to create user '" << default_user.username << "': " << result.error().message << std::endl;
                LOG_WARN(logger_, "Failed to create default user '" + default_user.username +
                        "': " + result.error().message);
            }
        }

        std::cout << "[DEBUG] Seeding complete: " << created_count << " created, " << skipped_count << " skipped" << std::endl;

        LOG_INFO(logger_, "Default user seeding complete: " +
                std::to_string(created_count) + " created, " +
                std::to_string(skipped_count) + " skipped");

        return true;
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] Exception in seed_default_users: " << e.what() << std::endl;
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

    // Map is api_key -> user_id, so direct lookup
    auto it = api_keys_.find(api_key);
    if (it == api_keys_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "API key not found");
    }

    std::string user_id = it->second;
    api_keys_.erase(it);
    LOG_INFO(logger_, "API key revoked for user: " + user_id);

    return true;
}

Result<std::vector<UserCredentials>> AuthenticationService::list_users() const {
    try {
        // Get users from database instead of in-memory map
        auto db_users_result = persistence_->list_users();
        if (!db_users_result.has_value()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to list users from database: " +
                        ErrorHandler::format_error(db_users_result.error()));
        }

        auto db_users = db_users_result.value();
        std::vector<UserCredentials> users;
        users.reserve(db_users.size());

        // Convert User objects to UserCredentials
        for (const auto& db_user : db_users) {
            UserCredentials user_cred;
            user_cred.user_id = db_user.user_id;
            user_cred.username = db_user.username;
            user_cred.email = db_user.email;
            user_cred.is_active = db_user.is_active;
            user_cred.password_hash = db_user.password_hash;
            user_cred.salt = db_user.salt;
            user_cred.failed_login_attempts = db_user.failed_login_attempts;

            // Convert int64_t timestamps to chrono::time_point
            user_cred.created_at = std::chrono::system_clock::from_time_t(db_user.created_at);
            user_cred.last_login = std::chrono::system_clock::from_time_t(db_user.last_login);

            // Get roles for this user
            auto roles_result = persistence_->get_user_roles(db_user.user_id);
            if (roles_result.has_value()) {
                user_cred.roles = roles_result.value();
            }

            users.push_back(user_cred);
        }

        LOG_INFO(logger_, "Listed " + std::to_string(users.size()) + " users from database");
        return users;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list_users: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to list users: " + std::string(e.what()));
    }
}

Result<size_t> AuthenticationService::get_user_count() const {
    try {
        // Get count from database instead of in-memory map
        auto users_result = persistence_->list_users();
        if (!users_result.has_value()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get user count from database: " +
                        ErrorHandler::format_error(users_result.error()));
        }
        return users_result.value().size();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_user_count: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get user count: " + std::string(e.what()));
    }
}

Result<std::vector<std::pair<std::string, std::string>>> AuthenticationService::list_api_keys() const {
    std::lock_guard<std::mutex> lock(api_keys_mutex_);

    std::vector<std::pair<std::string, std::string>> api_keys;
    api_keys.reserve(api_keys_.size());

    // Map is api_key -> user_id
    for (const auto& [api_key, user_id] : api_keys_) {
        api_keys.push_back({user_id, api_key});
    }

    LOG_INFO(logger_, "Listed " + std::to_string(api_keys.size()) + " API keys");
    return api_keys;
}

Result<std::vector<std::pair<std::string, std::string>>> AuthenticationService::list_api_keys_for_user(const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(api_keys_mutex_);

    std::vector<std::pair<std::string, std::string>> user_api_keys;

    // Map is api_key -> user_id, so iterate and filter by user_id
    for (const auto& [api_key, uid] : api_keys_) {
        if (uid == user_id) {
            user_api_keys.push_back({uid, api_key});
        }
    }

    LOG_INFO(logger_, "Listed " + std::to_string(user_api_keys.size()) + " API keys for user: " + user_id);
    return user_api_keys;
}

Result<bool> AuthenticationService::update_email(const std::string& user_id, const std::string& new_email) {
    std::lock_guard<std::mutex> lock(users_mutex_);

    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }

    it->second.email = new_email;
    LOG_INFO(logger_, "Updated email for user: " + user_id);

    return true;
}

Result<bool> AuthenticationService::update_roles(const std::string& user_id, const std::vector<std::string>& new_roles) {
    std::lock_guard<std::mutex> lock(users_mutex_);

    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }

    it->second.roles = new_roles;

    std::string roles_str = "[";
    for (size_t i = 0; i < new_roles.size(); ++i) {
        if (i > 0) roles_str += ", ";
        roles_str += new_roles[i];
    }
    roles_str += "]";

    LOG_INFO(logger_, "Updated roles for user " + user_id + " to: " + roles_str);

    return true;
}

} // namespace jadevectordb
