// Authentication handlers implementation for REST API
// This file contains T219 implementations: register, login, logout, forgot password, reset password

#include "api/rest/rest_api.h"
// REMOVED: #include "lib/auth.h" - migrated to AuthenticationService
#include "services/authentication_service.h"
#include <crow/json.h>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace jadevectordb {

// ============================================================================
// T219.1: REGISTER ENDPOINT
// POST /v1/auth/register
// ============================================================================
crow::response RestApiImpl::handle_register_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received register request");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in register request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("username") || !body_json.has("password")) {
            return crow::response(400, "{\"error\":\"Missing required fields: username, password\"}");
        }

        std::string username = body_json["username"].s();
        std::string password = body_json["password"].s();

        // Optional fields
        std::vector<std::string> roles;
        if (body_json.has("roles") && body_json["roles"].t() == crow::json::type::List) {
            for (size_t i = 0; i < body_json["roles"].size(); i++) {
                roles.push_back(body_json["roles"][i].s());
            }
        } else {
            // Default role for new users
            roles.push_back("user");
        }

        std::string user_id;
        if (body_json.has("user_id")) {
            user_id = body_json["user_id"].s();
        }

        // Register user with authentication service
        auto register_result = authentication_service_->register_user(username, password, roles, user_id);

        if (!register_result.has_value()) {
            LOG_ERROR(logger_, "Failed to register user " + username + ": " +
                     ErrorHandler::format_error(register_result.error()));

            // Check for specific error codes
            if (register_result.error().code == ErrorCode::ALREADY_EXISTS) {
                return crow::response(409, "{\"error\":\"Username already exists\"}");
            }

            crow::json::wvalue error_response;
            error_response["error"] = register_result.error().message;
            return crow::response(400, error_response);
        }

        std::string created_user_id = register_result.value();

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                username,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "User registered successfully"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["user_id"] = created_user_id;
        response["username"] = username;
        response["message"] = "User registered successfully";
        response["created_at"] = to_iso_string(std::chrono::system_clock::now());

        LOG_INFO(logger_, "Successfully registered user: " + username + " (ID: " + created_user_id + ")");

        return crow::response(201, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_register_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T219.2: LOGIN ENDPOINT
// POST /v1/auth/login
// ============================================================================
crow::response RestApiImpl::handle_login_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received login request");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in login request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("username") || !body_json.has("password")) {
            return crow::response(400, "{\"error\":\"Missing required fields: username, password\"}");
        }

        std::string username = body_json["username"].s();
        std::string password = body_json["password"].s();

        // Get client info for session
        std::string ip_address = req.get_header_value("X-Forwarded-For").empty() ?
                                req.remote_ip_address : req.get_header_value("X-Forwarded-For");
        std::string user_agent = req.get_header_value("User-Agent");

        // Authenticate with authentication service
        auto auth_result = authentication_service_->authenticate(username, password, ip_address, user_agent);

        if (!auth_result.has_value()) {
            LOG_WARN(logger_, "Failed login attempt for user: " + username);

            // Log failed authentication event
            if (security_audit_logger_) {
                security_audit_logger_->log_authentication_attempt(
                    username,
                    ip_address,
                    false,
                    "Invalid username or password"
                );
            }

            return crow::response(401, "{\"error\":\"Invalid username or password\"}");
        }

        LocalAuthToken token = auth_result.value();

        // Log successful authentication event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                username,
                ip_address,
                true,
                "Login successful"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["token"] = token.token_value;
        response["user_id"] = token.user_id;
        response["expires_at"] = to_iso_string(token.expires_at);
        response["token_type"] = "Bearer";

        LOG_INFO(logger_, "Successfully authenticated user: " + username + " (ID: " + token.user_id + ")");

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_login_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T219.3: LOGOUT ENDPOINT
// POST /v1/auth/logout
// ============================================================================
crow::response RestApiImpl::handle_logout_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received logout request");

        // Extract token from Authorization header
        auto auth_header = req.get_header_value("Authorization");
        if (auth_header.empty()) {
            return crow::response(401, "{\"error\":\"Missing Authorization header\"}");
        }

        std::string token;
        if (auth_header.substr(0, 7) == "Bearer ") {
            token = auth_header.substr(7);
        } else {
            return crow::response(401, "{\"error\":\"Invalid Authorization header format. Use 'Bearer <token>'\"}");
        }

        // Validate token first to get user_id
        auto validate_result = authentication_service_->validate_token(token);
        if (!validate_result.has_value()) {
            return crow::response(401, "{\"error\":\"Invalid or expired token\"}");
        }

        std::string user_id = validate_result.value();

        // Logout (revoke token and end session)
        auto logout_result = authentication_service_->logout(token);

        if (!logout_result.has_value()) {
            LOG_ERROR(logger_, "Failed to logout user: " + user_id);
            crow::json::wvalue error_response;
            error_response["error"] = logout_result.error().message;
            return crow::response(500, error_response);
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "User logged out successfully"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["message"] = "Successfully logged out";
        response["user_id"] = user_id;

        LOG_INFO(logger_, "Successfully logged out user: " + user_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_logout_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T219.4: FORGOT PASSWORD ENDPOINT
// POST /v1/auth/forgot-password
// ============================================================================
crow::response RestApiImpl::handle_forgot_password_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received forgot password request");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in forgot password request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("username") && !body_json.has("email")) {
            return crow::response(400, "{\"error\":\"Missing required field: username or email\"}");
        }

        std::string identifier = body_json.has("username") ?
                                body_json["username"].s() : body_json["email"].s();

        // Get user by username
        auto user_result = authentication_service_->get_user_by_username(identifier);

        if (!user_result.has_value()) {
            // For security, don't reveal if user exists or not
            // Return success even if user doesn't exist
            LOG_WARN(logger_, "Forgot password request for non-existent user: " + identifier);

            crow::json::wvalue response;
            response["message"] = "If the account exists, a password reset token will be generated";
            return crow::response(200, response);
        }

        UserCredentials user = user_result.value();

        // Generate password reset token
        std::string reset_token = generate_secure_token();

        // Store reset token with expiration (15 minutes)
        {
            std::lock_guard<std::mutex> lock(password_reset_mutex_);
            PasswordResetToken token_record;
            token_record.token = reset_token;
            token_record.user_id = user.user_id;
            token_record.expires_at = std::chrono::system_clock::now() + std::chrono::minutes(15);
            password_reset_tokens_[reset_token] = token_record;
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user.username,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "Password reset token generated"
            );
        }

        // In a real implementation, send email with reset token
        // For now, return the token directly (NOT secure for production!)
        crow::json::wvalue response;
        response["message"] = "Password reset token generated";
        response["reset_token"] = reset_token;  // In production, this would be sent via email
        response["expires_in_minutes"] = 15;
        response["note"] = "Token generated (would be sent via email in production)";

        // In a real implementation, we would send the token via email
        // send_password_reset_email(user.email, reset_token);

        LOG_INFO(logger_, "Generated password reset token for user: " + user.username);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_forgot_password_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T219.5: RESET PASSWORD ENDPOINT
// POST /v1/auth/reset-password
// ============================================================================
crow::response RestApiImpl::handle_reset_password_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received reset password request");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in reset password request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("reset_token") || !body_json.has("new_password")) {
            return crow::response(400, "{\"error\":\"Missing required fields: reset_token, new_password\"}");
        }

        std::string reset_token = body_json["reset_token"].s();
        std::string new_password = body_json["new_password"].s();

        // Verify reset token
        std::string user_id;
        {
            std::lock_guard<std::mutex> lock(password_reset_mutex_);

            auto it = password_reset_tokens_.find(reset_token);
            if (it == password_reset_tokens_.end()) {
                LOG_WARN(logger_, "Invalid password reset token");
                return crow::response(400, "{\"error\":\"Invalid or expired reset token\"}");
            }

            PasswordResetToken& token_record = it->second;

            // Check if token is expired
            if (std::chrono::system_clock::now() > token_record.expires_at) {
                LOG_WARN(logger_, "Expired password reset token for user: " + token_record.user_id);
                password_reset_tokens_.erase(it);
                return crow::response(400, "{\"error\":\"Reset token has expired\"}");
            }

            user_id = token_record.user_id;

            // Remove token after use (one-time use)
            password_reset_tokens_.erase(it);
        }

        // Reset password
        auto reset_result = authentication_service_->reset_password(user_id, new_password);

        if (!reset_result.has_value()) {
            LOG_ERROR(logger_, "Failed to reset password for user: " + user_id);
            crow::json::wvalue error_response;
            error_response["error"] = reset_result.error().message;
            return crow::response(500, error_response);
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "Password reset successfully"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["message"] = "Password reset successfully";
        response["user_id"] = user_id;

        LOG_INFO(logger_, "Successfully reset password for user: " + user_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_reset_password_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// AUTHENTICATION ROUTES REGISTRATION
// This method registers all authentication endpoints with Crow
// ============================================================================
void RestApiImpl::handle_authentication_routes() {
    LOG_INFO(logger_, "Registering authentication routes with rate limiting and IP blocking");
    
    // Helper function to extract client IP from request
    auto extract_client_ip = [](const crow::request& req) -> std::string {
        // Try X-Forwarded-For header first (for proxies)
        std::string forwarded_for = req.get_header_value("X-Forwarded-For");
        if (!forwarded_for.empty()) {
            // Take the first IP in the list
            size_t comma = forwarded_for.find(',');
            return comma != std::string::npos ? forwarded_for.substr(0, comma) : forwarded_for;
        }
        
        // Try X-Real-IP header
        std::string real_ip = req.get_header_value("X-Real-IP");
        if (!real_ip.empty()) {
            return real_ip;
        }
        
        // Fall back to remote address
        return req.remote_ip_address;
    };

    // POST /v1/auth/register - With rate limiting and IP blocking
    CROW_ROUTE((*app_), "/v1/auth/register")
        .methods(crow::HTTPMethod::POST)
        ([this, extract_client_ip](const crow::request& req) {
            std::string client_ip = extract_client_ip(req);
            
            // Check if IP is blocked
            if (ip_blocker_->is_blocked(client_ip)) {
                int remaining = ip_blocker_->remaining_block_seconds(client_ip);
                crow::json::wvalue error_response;
                error_response["error"] = "ip_blocked";
                error_response["reason"] = "too_many_failures";
                error_response["retry_after"] = remaining;
                error_response["message"] = "IP address temporarily blocked due to too many failed attempts";
                
                LOG_WARN(logger_, "Blocked registration attempt from IP: " << client_ip);
                if (security_audit_logger_) {
                    security_audit_logger_->log_authentication_attempt("", client_ip, false, "Registration blocked - IP banned");
                }
                
                crow::response resp(403, error_response);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("Retry-After", std::to_string(remaining));
                return resp;
            }
            
            // Check registration rate limit
            if (!registration_rate_limiter_->allow(client_ip)) {
                double retry_after = registration_rate_limiter_->retry_after_seconds(client_ip);
                crow::json::wvalue error_response;
                error_response["error"] = "rate_limit_exceeded";
                error_response["retry_after"] = static_cast<int>(std::ceil(retry_after));
                error_response["message"] = "Too many registration attempts. Please try again later.";
                
                LOG_WARN(logger_, "Rate limit exceeded for registration from IP: " << client_ip);
                if (security_audit_logger_) {
                    security_audit_logger_->log_authentication_attempt("", client_ip, false, "Registration rate limited");
                }
                
                crow::response resp(429, error_response);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("Retry-After", std::to_string(static_cast<int>(std::ceil(retry_after))));
                return resp;
            }
            
            // Process registration (delegate to existing handler)
            return handle_register_request(req);
        });

    // POST /v1/auth/login - With rate limiting and IP blocking
    CROW_ROUTE((*app_), "/v1/auth/login")
        .methods(crow::HTTPMethod::POST)
        ([this, extract_client_ip](const crow::request& req) {
            std::string client_ip = extract_client_ip(req);
            
            // Check if IP is blocked
            if (ip_blocker_->is_blocked(client_ip)) {
                int remaining = ip_blocker_->remaining_block_seconds(client_ip);
                crow::json::wvalue error_response;
                error_response["error"] = "ip_blocked";
                error_response["reason"] = "too_many_failed_login_attempts";
                error_response["retry_after"] = remaining;
                error_response["message"] = "IP address temporarily blocked due to too many failed login attempts";
                
                LOG_WARN(logger_, "Blocked login attempt from IP: " << client_ip);
                if (security_audit_logger_) {
                    security_audit_logger_->log_authentication_attempt("", client_ip, false, "Login blocked - IP banned");
                }
                
                crow::response resp(403, error_response);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("Retry-After", std::to_string(remaining));
                return resp;
            }
            
            // Check login rate limit
            if (!login_rate_limiter_->allow(client_ip)) {
                double retry_after = login_rate_limiter_->retry_after_seconds(client_ip);
                crow::json::wvalue error_response;
                error_response["error"] = "rate_limit_exceeded";
                error_response["retry_after"] = static_cast<int>(std::ceil(retry_after));
                error_response["message"] = "Too many login attempts. Please try again later.";
                
                LOG_WARN(logger_, "Rate limit exceeded for login from IP: " << client_ip);
                if (security_audit_logger_) {
                    security_audit_logger_->log_authentication_attempt("", client_ip, false, "Login rate limited");
                }
                
                crow::response resp(429, error_response);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("Retry-After", std::to_string(static_cast<int>(std::ceil(retry_after))));
                return resp;
            }
            
            // Process login
            auto login_response = handle_login_request(req);
            
            // Check if login was successful or failed
            if (login_response.code == 200) {
                // Successful login - clear failure history
                ip_blocker_->record_success(client_ip);
                LOG_INFO(logger_, "Successful login from IP: " << client_ip);
            } else {
                // Failed login - record failure
                bool now_blocked = ip_blocker_->record_failure(client_ip, "invalid_credentials");
                if (now_blocked) {
                    LOG_WARN(logger_, "IP address blocked after failed login: " << client_ip);
                    if (security_audit_logger_) {
                        security_audit_logger_->log_authentication_attempt("", client_ip, false, "IP auto-blocked after max failures");
                    }
                }
            }
            
            return login_response;
        });

    // POST /v1/auth/logout
    CROW_ROUTE((*app_), "/v1/auth/logout")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            return handle_logout_request(req);
        });

    // POST /v1/auth/forgot-password - With rate limiting and IP blocking
    CROW_ROUTE((*app_), "/v1/auth/forgot-password")
        .methods(crow::HTTPMethod::POST)
        ([this, extract_client_ip](const crow::request& req) {
            std::string client_ip = extract_client_ip(req);
            
            // Check if IP is blocked
            if (ip_blocker_->is_blocked(client_ip)) {
                int remaining = ip_blocker_->remaining_block_seconds(client_ip);
                crow::json::wvalue error_response;
                error_response["error"] = "ip_blocked";
                error_response["reason"] = "too_many_failures";
                error_response["retry_after"] = remaining;
                error_response["message"] = "IP address temporarily blocked";
                
                crow::response resp(403, error_response);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("Retry-After", std::to_string(remaining));
                return resp;
            }
            
            // Extract user ID from request to use for rate limiting
            std::string user_id;
            try {
                auto body = crow::json::load(req.body);
                if (body && body.has("username")) {
                    user_id = body["username"].s();
                } else if (body && body.has("email")) {
                    user_id = body["email"].s();
                }
            } catch (...) {
                // Use IP if we can't get user ID
                user_id = client_ip;
            }
            
            // Check password reset rate limit (per user)
            if (!password_reset_rate_limiter_->allow(user_id)) {
                double retry_after = password_reset_rate_limiter_->retry_after_seconds(user_id);
                crow::json::wvalue error_response;
                error_response["error"] = "rate_limit_exceeded";
                error_response["retry_after"] = static_cast<int>(std::ceil(retry_after));
                error_response["message"] = "Too many password reset requests. Please try again later.";
                
                LOG_WARN(logger_, "Rate limit exceeded for password reset from user: " << user_id);
                
                crow::response resp(429, error_response);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("Retry-After", std::to_string(static_cast<int>(std::ceil(retry_after))));
                return resp;
            }
            
            return handle_forgot_password_request(req);
        });

    // POST /v1/auth/reset-password
    CROW_ROUTE((*app_), "/v1/auth/reset-password")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            return handle_reset_password_request(req);
        });

    LOG_INFO(logger_, "Authentication routes registered successfully with security middleware");
}

} // namespace jadevectordb
