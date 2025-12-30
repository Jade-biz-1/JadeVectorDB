// User management handlers implementation for REST API
// This file contains T220 implementations: create, list, get, update, delete users

#include "api/rest/rest_api.h"
#include "services/authentication_service.h"
#include <crow/json.h>
#include <chrono>

namespace jadevectordb {

// ============================================================================
// T220.1: CREATE USER ENDPOINT
// POST /v1/users
// ============================================================================
crow::response RestApiImpl::handle_create_user_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received create user request");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in create user request");
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

        std::string email;
        if (body_json.has("email")) {
            email = body_json["email"].s();
        }

        std::string user_id_override;
        if (body_json.has("user_id")) {
            user_id_override = body_json["user_id"].s();
        }

        // Create user via authentication service
        auto create_result = authentication_service_->register_user(username, password, roles, user_id_override);

        if (!create_result.has_value()) {
            LOG_ERROR(logger_, "Failed to create user " + username + ": " +
                     ErrorHandler::format_error(create_result.error()));

            // Check for specific error codes
            if (create_result.error().code == ErrorCode::ALREADY_EXISTS) {
                return crow::response(409, "{\"error\":\"Username already exists\"}");
            }

            crow::json::wvalue error_response;
            error_response["error"] = create_result.error().message;
            return crow::response(400, error_response);
        }

        std::string created_user_id = create_result.value();

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                username,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "User created via API"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["user_id"] = created_user_id;
        response["username"] = username;
        if (!email.empty()) {
            response["email"] = email;
        }
        response["roles"] = crow::json::wvalue::list();
        for (size_t i = 0; i < roles.size(); i++) {
            response["roles"][i] = roles[i];
        }
        response["message"] = "User created successfully";
        response["created_at"] = to_iso_string(std::chrono::system_clock::now());

        LOG_INFO(logger_, "Successfully created user: " + username + " (ID: " + created_user_id + ")");

        return crow::response(201, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_create_user_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T220.2: LIST USERS ENDPOINT
// GET /v1/users
// ============================================================================
crow::response RestApiImpl::handle_list_users_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received list users request");

        // Get list of users from AuthenticationService
        auto users_result = authentication_service_->list_users();

        if (!users_result.has_value()) {
            LOG_ERROR(logger_, "Failed to list users: " +
                     ErrorHandler::format_error(users_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = users_result.error().message;
            return crow::response(500, error_response);
        }

        auto users = users_result.value();

        // Build response
        crow::json::wvalue response;
        response["users"] = crow::json::wvalue::list();
        response["count"] = users.size();

        for (size_t i = 0; i < users.size(); i++) {
            const auto& user = users[i];
            crow::json::wvalue user_obj;
            user_obj["user_id"] = user.user_id;
            user_obj["username"] = user.username;
            user_obj["email"] = user.email;
            user_obj["is_active"] = user.is_active;
            user_obj["created_at"] = to_iso_string(user.created_at);
            user_obj["last_login"] = to_iso_string(user.last_login);

            // Add roles
            user_obj["roles"] = crow::json::wvalue::list();
            for (size_t j = 0; j < user.roles.size(); j++) {
                user_obj["roles"][j] = user.roles[j];
            }

            response["users"][i] = std::move(user_obj);
        }

        LOG_INFO(logger_, "Successfully retrieved " + std::to_string(users.size()) + " users");

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_list_users_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T220.3: GET USER ENDPOINT
// GET /v1/users/{id}
// ============================================================================
crow::response RestApiImpl::handle_get_user_request(const crow::request& req, const std::string& user_id) {
    try {
        LOG_INFO(logger_, "Received get user request for ID: " + user_id);

        // Get user from AuthenticationService
        auto user_result = authentication_service_->get_user(user_id);

        if (!user_result.has_value()) {
            LOG_WARN(logger_, "User not found: " + user_id);
            return crow::response(404, "{\"error\":\"User not found\"}");
        }

        auto user = user_result.value();

        // Build response
        crow::json::wvalue response;
        response["user_id"] = user.user_id;
        response["username"] = user.username;
        response["email"] = user.email;
        response["is_active"] = user.is_active;
        response["created_at"] = to_iso_string(user.created_at);
        response["last_login"] = to_iso_string(user.last_login);

        // Add roles
        response["roles"] = crow::json::wvalue::list();
        for (size_t i = 0; i < user.roles.size(); i++) {
            response["roles"][i] = user.roles[i];
        }

        LOG_INFO(logger_, "Successfully retrieved user: " + user_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_get_user_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T220.4: UPDATE USER ENDPOINT
// PUT /v1/users/{id}
// ============================================================================
crow::response RestApiImpl::handle_update_user_request(const crow::request& req, const std::string& user_id) {
    try {
        LOG_INFO(logger_, "Received update user request for ID: " + user_id);

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in update user request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Check if user exists
        auto user_result = authentication_service_->get_user(user_id);
        if (!user_result.has_value()) {
            LOG_WARN(logger_, "User not found for update: " + user_id);
            return crow::response(404, "{\"error\":\"User not found\"}");
        }

        // Update username if provided
        if (body_json.has("username")) {
            std::string new_username = body_json["username"].s();
            auto update_result = authentication_service_->update_username(user_id, new_username);
            if (!update_result.has_value()) {
                LOG_ERROR(logger_, "Failed to update username: " +
                         ErrorHandler::format_error(update_result.error()));
                crow::json::wvalue error_response;
                error_response["error"] = update_result.error().message;
                return crow::response(400, error_response);
            }
        }

        // Update email if provided
        if (body_json.has("email")) {
            std::string new_email = body_json["email"].s();
            auto update_result = authentication_service_->update_email(user_id, new_email);
            if (!update_result.has_value()) {
                LOG_ERROR(logger_, "Failed to update email: " +
                         ErrorHandler::format_error(update_result.error()));
                crow::json::wvalue error_response;
                error_response["error"] = update_result.error().message;
                return crow::response(400, error_response);
            }
        }

        // Update roles if provided
        if (body_json.has("roles") && body_json["roles"].t() == crow::json::type::List) {
            std::vector<std::string> new_roles;
            for (size_t i = 0; i < body_json["roles"].size(); i++) {
                new_roles.push_back(body_json["roles"][i].s());
            }
            auto update_result = authentication_service_->update_roles(user_id, new_roles);
            if (!update_result.has_value()) {
                LOG_ERROR(logger_, "Failed to update roles: " +
                         ErrorHandler::format_error(update_result.error()));
                crow::json::wvalue error_response;
                error_response["error"] = update_result.error().message;
                return crow::response(400, error_response);
            }
        }

        // Update active status if provided
        if (body_json.has("is_active")) {
            bool is_active = body_json["is_active"].b();
            auto status_result = authentication_service_->set_user_active_status(user_id, is_active);
            if (!status_result.has_value()) {
                LOG_ERROR(logger_, "Failed to update active status: " +
                         ErrorHandler::format_error(status_result.error()));
                crow::json::wvalue error_response;
                error_response["error"] = status_result.error().message;
                return crow::response(400, error_response);
            }
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "User updated via API"
            );
        }

        // Get updated user
        auto updated_user_result = authentication_service_->get_user(user_id);
        if (!updated_user_result.has_value()) {
            return crow::response(500, "{\"error\":\"Failed to retrieve updated user\"}");
        }

        auto user = updated_user_result.value();

        // Build response
        crow::json::wvalue response;
        response["user_id"] = user.user_id;
        response["username"] = user.username;
        response["email"] = user.email;
        response["is_active"] = user.is_active;
        response["message"] = "User updated successfully";

        LOG_INFO(logger_, "Successfully updated user: " + user_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_update_user_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T220.5: DELETE USER ENDPOINT
// DELETE /v1/users/{id}
// ============================================================================
crow::response RestApiImpl::handle_delete_user_request(const crow::request& req, const std::string& user_id) {
    try {
        LOG_INFO(logger_, "Received delete user request for ID: " + user_id);

        // Check if user exists
        auto user_result = authentication_service_->get_user(user_id);
        if (!user_result.has_value()) {
            LOG_WARN(logger_, "User not found for deletion: " + user_id);
            return crow::response(404, "{\"error\":\"User not found\"}");
        }

        std::string username = user_result.value().username;

        // Delete user (soft delete by deactivating)
        auto delete_result = authentication_service_->set_user_active_status(user_id, false);
        if (!delete_result.has_value()) {
            LOG_ERROR(logger_, "Failed to delete user " + user_id + ": " +
                     ErrorHandler::format_error(delete_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = delete_result.error().message;
            return crow::response(500, error_response);
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "User deleted via API"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["user_id"] = user_id;
        response["username"] = username;
        response["message"] = "User deleted successfully";

        LOG_INFO(logger_, "Successfully deleted user: " + user_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_delete_user_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// USER MANAGEMENT ROUTES REGISTRATION
// This method registers all user management endpoints with Crow
// ============================================================================
void RestApiImpl::handle_user_management_routes() {
    LOG_INFO(logger_, "Registering user management routes");

    // POST /v1/users - Create user
    CROW_ROUTE((*app_), "/v1/users")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            return handle_create_user_request(req);
        });

    // GET /v1/users - List users
    CROW_ROUTE((*app_), "/v1/users")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_list_users_request(req);
        });

    // GET /v1/users/{id} - Get user
    CROW_ROUTE((*app_), "/v1/users/<string>")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req, const std::string& user_id) {
            return handle_get_user_request(req, user_id);
        });

    // PUT /v1/users/{id} - Update user
    CROW_ROUTE((*app_), "/v1/users/<string>")
        .methods(crow::HTTPMethod::PUT)
        ([this](const crow::request& req, const std::string& user_id) {
            return handle_update_user_request(req, user_id);
        });

    // DELETE /v1/users/{id} - Delete user
    CROW_ROUTE((*app_), "/v1/users/<string>")
        .methods(crow::HTTPMethod::DELETE)
        ([this](const crow::request& req, const std::string& user_id) {
            return handle_delete_user_request(req, user_id);
        });

    // PUT /v1/users/{id}/password - Change user password (self-service)
    CROW_ROUTE((*app_), "/v1/users/<string>/password")
        .methods(crow::HTTPMethod::PUT)
        ([this](const crow::request& req, const std::string& user_id) {
            return handle_change_user_password_request(req, user_id);
        });

    // POST /v1/admin/users/{id}/reset-password - Admin reset user password
    CROW_ROUTE((*app_), "/v1/admin/users/<string>/reset-password")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req, const std::string& user_id) {
            return handle_admin_reset_user_password_request(req, user_id);
        });

    LOG_INFO(logger_, "User management routes registered successfully");
}

// ============================================================================
// PASSWORD CHANGE ENDPOINT (Self-Service)
// PUT /v1/users/{id}/password
// ============================================================================
crow::response RestApiImpl::handle_change_user_password_request(const crow::request& req, const std::string& user_id) {
    try {
        LOG_INFO(logger_, "Received password change request for user: " + user_id);

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in password change request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("old_password") || !body_json.has("new_password")) {
            return crow::response(400, "{\"error\":\"Missing required fields: old_password, new_password\"}");
        }

        std::string old_password = body_json["old_password"].s();
        std::string new_password = body_json["new_password"].s();

        // Call authentication service to update password
        auto result = authentication_service_->update_password(user_id, old_password, new_password);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to update password: " +
                     ErrorHandler::format_error(result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = result.error().message;
            error_response["code"] = static_cast<int>(result.error().code);

            // Return appropriate HTTP status based on error code
            int status_code = 500;
            if (result.error().code == ErrorCode::UNAUTHENTICATED) {
                status_code = 401;  // Incorrect old password
            } else if (result.error().code == ErrorCode::INVALID_ARGUMENT) {
                status_code = 400;  // Invalid password format
            } else if (result.error().code == ErrorCode::NOT_FOUND) {
                status_code = 404;  // User not found
            }

            return crow::response(status_code, error_response);
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "Password changed successfully"
            );
        }

        // Success response
        crow::json::wvalue response;
        response["success"] = true;
        response["message"] = "Password updated successfully";

        LOG_INFO(logger_, "Password changed successfully for user: " + user_id);
        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_change_user_password_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// ADMIN PASSWORD RESET ENDPOINT
// POST /v1/admin/users/{id}/reset-password
// ============================================================================
crow::response RestApiImpl::handle_admin_reset_user_password_request(const crow::request& req, const std::string& user_id) {
    try {
        LOG_INFO(logger_, "Received admin password reset request for user: " + user_id);

        // TODO: Add admin authorization check here
        // For now, assuming the request is authenticated and authorized

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in admin password reset request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("new_password")) {
            return crow::response(400, "{\"error\":\"Missing required field: new_password\"}");
        }

        std::string new_password = body_json["new_password"].s();

        // Call authentication service to reset password
        auto result = authentication_service_->reset_password(user_id, new_password);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to reset password: " +
                     ErrorHandler::format_error(result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = result.error().message;
            error_response["code"] = static_cast<int>(result.error().code);

            // Return appropriate HTTP status based on error code
            int status_code = 500;
            if (result.error().code == ErrorCode::INVALID_ARGUMENT) {
                status_code = 400;  // Invalid password format
            } else if (result.error().code == ErrorCode::NOT_FOUND) {
                status_code = 404;  // User not found
            }

            return crow::response(status_code, error_response);
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "Password reset by admin - user must change on next login"
            );
        }

        // Success response
        crow::json::wvalue response;
        response["success"] = true;
        response["message"] = "Password reset successfully - user must change password on next login";
        response["must_change_password"] = true;

        LOG_INFO(logger_, "Password reset by admin for user: " + user_id);
        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_admin_reset_user_password_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

} // namespace jadevectordb
