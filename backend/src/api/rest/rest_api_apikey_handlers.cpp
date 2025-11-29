// API key management handlers implementation for REST API
// This file contains T221 implementations: list, create, revoke API keys

#include "api/rest/rest_api.h"
#include "lib/auth.h"
#include <crow/json.h>
#include <chrono>

namespace jadevectordb {

// ============================================================================
// T221.1: CREATE API KEY ENDPOINT
// POST /v1/api-keys
// ============================================================================
crow::response RestApiImpl::handle_create_api_key_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received create API key request");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in create API key request");
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        // Validate required fields
        if (!body_json.has("user_id")) {
            return crow::response(400, "{\"error\":\"Missing required field: user_id\"}");
        }

        std::string user_id = body_json["user_id"].s();

        // Optional fields
        std::vector<std::string> permissions;
        if (body_json.has("permissions") && body_json["permissions"].t() == crow::json::type::List) {
            for (size_t i = 0; i < body_json["permissions"].size(); i++) {
                permissions.push_back(body_json["permissions"][i].s());
            }
        }

        std::string description;
        if (body_json.has("description")) {
            description = body_json["description"].s();
        }

        // Validity duration (default 30 days)
        int validity_days = 30;
        if (body_json.has("validity_days")) {
            validity_days = body_json["validity_days"].i();
        }
        std::chrono::hours validity_duration = std::chrono::hours(24 * validity_days);

        // Generate API key
        auto api_key_result = auth_manager_->generate_api_key(
            user_id, permissions, description, validity_duration);

        if (!api_key_result.has_value()) {
            LOG_ERROR(logger_, "Failed to generate API key for user " + user_id + ": " +
                     ErrorHandler::format_error(api_key_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = api_key_result.error().message;
            return crow::response(400, error_response);
        }

        std::string api_key = api_key_result.value();

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                user_id,
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "API key created"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["api_key"] = api_key;
        response["user_id"] = user_id;
        response["description"] = description;
        response["validity_days"] = validity_days;
        response["permissions"] = crow::json::wvalue::list();
        for (size_t i = 0; i < permissions.size(); i++) {
            response["permissions"][i] = permissions[i];
        }
        response["message"] = "API key created successfully";
        response["created_at"] = to_iso_string(std::chrono::system_clock::now());

        LOG_INFO(logger_, "Successfully created API key for user: " + user_id);

        return crow::response(201, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_create_api_key_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T221.2: LIST API KEYS ENDPOINT
// GET /v1/api-keys
// ============================================================================
crow::response RestApiImpl::handle_list_api_keys_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received list API keys request");

        // Check if filtering by user_id
        std::string user_id_filter = req.url_params.get("user_id");

        // Get list of API keys
        auto keys_result = user_id_filter.empty() ?
            auth_manager_->list_api_keys() :
            auth_manager_->list_api_keys_for_user(user_id_filter);

        if (!keys_result.has_value()) {
            LOG_ERROR(logger_, "Failed to list API keys: " +
                     ErrorHandler::format_error(keys_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = keys_result.error().message;
            return crow::response(500, error_response);
        }

        auto keys = keys_result.value();

        // Build response
        crow::json::wvalue response;
        response["api_keys"] = crow::json::wvalue::list();
        response["count"] = keys.size();

        for (size_t i = 0; i < keys.size(); i++) {
            const auto& key = keys[i];
            crow::json::wvalue key_obj;
            key_obj["key_id"] = key.key_id;
            key_obj["user_id"] = key.user_id;
            key_obj["description"] = key.description;
            key_obj["is_active"] = key.is_active;
            key_obj["created_at"] = to_iso_string(key.created_at);
            key_obj["expires_at"] = to_iso_string(key.expires_at);

            // Add permissions
            key_obj["permissions"] = crow::json::wvalue::list();
            for (size_t j = 0; j < key.permissions.size(); j++) {
                key_obj["permissions"][j] = key.permissions[j];
            }

            // Don't include the actual key hash for security
            response["api_keys"][i] = std::move(key_obj);
        }

        LOG_INFO(logger_, "Successfully retrieved " + std::to_string(keys.size()) + " API keys");

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_list_api_keys_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T221.3: REVOKE API KEY ENDPOINT
// DELETE /v1/api-keys/{id}
// ============================================================================
crow::response RestApiImpl::handle_revoke_api_key_request(const crow::request& req, const std::string& key_id) {
    try {
        LOG_INFO(logger_, "Received revoke API key request for ID: " + key_id);

        // Revoke the API key
        auto revoke_result = auth_manager_->revoke_api_key(key_id);

        if (!revoke_result.has_value()) {
            LOG_ERROR(logger_, "Failed to revoke API key " + key_id + ": " +
                     ErrorHandler::format_error(revoke_result.error()));

            // Check for specific error codes
            if (revoke_result.error().code == ErrorCode::NOT_FOUND) {
                return crow::response(404, "{\"error\":\"API key not found\"}");
            }

            crow::json::wvalue error_response;
            error_response["error"] = revoke_result.error().message;
            return crow::response(400, error_response);
        }

        // Log audit event
        if (security_audit_logger_) {
            security_audit_logger_->log_authentication_attempt(
                key_id,  // Use key_id as identifier
                req.get_header_value("X-Forwarded-For").empty() ?
                    req.remote_ip_address : req.get_header_value("X-Forwarded-For"),
                true,
                "API key revoked"
            );
        }

        // Build success response
        crow::json::wvalue response;
        response["key_id"] = key_id;
        response["message"] = "API key revoked successfully";

        LOG_INFO(logger_, "Successfully revoked API key: " + key_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_revoke_api_key_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// API KEY MANAGEMENT ROUTES REGISTRATION
// This method registers all API key management endpoints with Crow
// ============================================================================
void RestApiImpl::handle_api_key_routes() {
    LOG_INFO(logger_, "Registering API key management routes");

    // POST /v1/api-keys - Create API key
    CROW_ROUTE((*app_), "/v1/api-keys")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            return handle_create_api_key_request(req);
        });

    // GET /v1/api-keys - List API keys
    CROW_ROUTE((*app_), "/v1/api-keys")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_list_api_keys_request(req);
        });

    // DELETE /v1/api-keys/{id} - Revoke API key
    CROW_ROUTE((*app_), "/v1/api-keys/<string>")
        .methods(crow::HTTPMethod::DELETE)
        ([this](const crow::request& req, const std::string& key_id) {
            return handle_revoke_api_key_request(req, key_id);
        });

    LOG_INFO(logger_, "API key management routes registered successfully");
}

} // namespace jadevectordb
