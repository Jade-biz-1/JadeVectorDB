// Security audit and session management handlers for REST API
// This file contains T222 implementations: audit log and session routes

#include "api/rest/rest_api.h"
#include "services/security_audit_logger.h"
#include "services/authentication_service.h"
#include <crow/json.h>
#include <chrono>

namespace jadevectordb {

// Helper to convert SecurityEventType to string
std::string event_type_to_string(SecurityEventType type) {
    switch (type) {
        case SecurityEventType::AUTHENTICATION_ATTEMPT: return "AUTHENTICATION_ATTEMPT";
        case SecurityEventType::AUTHENTICATION_SUCCESS: return "AUTHENTICATION_SUCCESS";
        case SecurityEventType::AUTHENTICATION_FAILURE: return "AUTHENTICATION_FAILURE";
        case SecurityEventType::AUTHORIZATION_CHECK: return "AUTHORIZATION_CHECK";
        case SecurityEventType::AUTHORIZATION_GRANTED: return "AUTHORIZATION_GRANTED";
        case SecurityEventType::AUTHORIZATION_DENIED: return "AUTHORIZATION_DENIED";
        case SecurityEventType::DATA_ACCESS: return "DATA_ACCESS";
        case SecurityEventType::DATA_MODIFICATION: return "DATA_MODIFICATION";
        case SecurityEventType::DATA_DELETION: return "DATA_DELETION";
        case SecurityEventType::CONFIGURATION_CHANGE: return "CONFIGURATION_CHANGE";
        case SecurityEventType::ADMIN_OPERATION: return "ADMIN_OPERATION";
        case SecurityEventType::SECURITY_POLICY_VIOLATION: return "SECURITY_POLICY_VIOLATION";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// T222.1: GET AUDIT LOG ENDPOINT
// GET /v1/security/audit-log
// ============================================================================
crow::response RestApiImpl::handle_get_audit_log_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received get audit log request");

        if (!security_audit_logger_) {
            return crow::response(501, "{\"error\":\"Audit logging not configured\"}");
        }

        // Parse query parameters
        std::string user_id = req.url_params.get("user_id") ? req.url_params.get("user_id") : "";
        std::string event_type_str = req.url_params.get("event_type") ? req.url_params.get("event_type") : "";
        int limit = req.url_params.get("limit") ? std::stoi(req.url_params.get("limit")) : 100;

        // Default to AUTHENTICATION_ATTEMPT if not specified
        SecurityEventType event_type = SecurityEventType::AUTHENTICATION_ATTEMPT;

        // Parse event_type if provided
        if (!event_type_str.empty()) {
            if (event_type_str == "AUTHENTICATION_SUCCESS") event_type = SecurityEventType::AUTHENTICATION_SUCCESS;
            else if (event_type_str == "AUTHENTICATION_FAILURE") event_type = SecurityEventType::AUTHENTICATION_FAILURE;
            else if (event_type_str == "AUTHORIZATION_CHECK") event_type = SecurityEventType::AUTHORIZATION_CHECK;
            else if (event_type_str == "AUTHORIZATION_GRANTED") event_type = SecurityEventType::AUTHORIZATION_GRANTED;
            else if (event_type_str == "AUTHORIZATION_DENIED") event_type = SecurityEventType::AUTHORIZATION_DENIED;
            else if (event_type_str == "DATA_ACCESS") event_type = SecurityEventType::DATA_ACCESS;
            else if (event_type_str == "DATA_MODIFICATION") event_type = SecurityEventType::DATA_MODIFICATION;
            else if (event_type_str == "DATA_DELETION") event_type = SecurityEventType::DATA_DELETION;
            else if (event_type_str == "CONFIGURATION_CHANGE") event_type = SecurityEventType::CONFIGURATION_CHANGE;
            else if (event_type_str == "ADMIN_OPERATION") event_type = SecurityEventType::ADMIN_OPERATION;
            else if (event_type_str == "SECURITY_POLICY_VIOLATION") event_type = SecurityEventType::SECURITY_POLICY_VIOLATION;
        }

        // Search audit logs
        auto events_result = security_audit_logger_->search_audit_log(
            event_type, user_id, {}, {}, limit);

        if (!events_result.has_value()) {
            LOG_ERROR(logger_, "Failed to search audit log: " +
                     ErrorHandler::format_error(events_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = events_result.error().message;
            return crow::response(500, error_response);
        }

        auto events = events_result.value();

        // Build response
        crow::json::wvalue response;
        response["events"] = crow::json::wvalue::list();
        response["count"] = events.size();

        for (size_t i = 0; i < events.size(); i++) {
            const auto& event = events[i];
            crow::json::wvalue event_obj;
            event_obj["event_id"] = event.event_id;
            event_obj["event_type"] = event_type_to_string(event.event_type);
            event_obj["user_id"] = event.user_id;
            event_obj["ip_address"] = event.ip_address;
            event_obj["resource_accessed"] = event.resource_accessed;
            event_obj["operation"] = event.operation;
            event_obj["success"] = event.success;
            event_obj["details"] = event.details;
            event_obj["session_id"] = event.session_id;
            event_obj["client_info"] = event.client_info;
            event_obj["timestamp"] = to_iso_string(event.timestamp);

            response["events"][i] = std::move(event_obj);
        }

        LOG_INFO(logger_, "Successfully retrieved " + std::to_string(events.size()) + " audit log events");

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_get_audit_log_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T222.2: GET SESSIONS ENDPOINT
// GET /v1/security/sessions
// ============================================================================
crow::response RestApiImpl::handle_get_sessions_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received get sessions request");

        if (!authentication_service_) {
            return crow::response(501, "{\"error\":\"Authentication service not configured\"}");
        }

        // Parse query parameters
        std::string user_id = req.url_params.get("user_id") ? req.url_params.get("user_id") : "";

        if (user_id.empty()) {
            return crow::response(400, "{\"error\":\"Missing required parameter: user_id\"}");
        }

        // Get user sessions
        auto sessions_result = authentication_service_->get_user_sessions(user_id);

        if (!sessions_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get sessions for user " + user_id + ": " +
                     ErrorHandler::format_error(sessions_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = sessions_result.error().message;
            return crow::response(500, error_response);
        }

        auto sessions = sessions_result.value();

        // Build response
        crow::json::wvalue response;
        response["sessions"] = crow::json::wvalue::list();
        response["count"] = sessions.size();
        response["user_id"] = user_id;

        for (size_t i = 0; i < sessions.size(); i++) {
            const auto& session = sessions[i];
            crow::json::wvalue session_obj;
            session_obj["session_id"] = session.session_id;
            session_obj["user_id"] = session.user_id;
            session_obj["token_id"] = session.token_id;
            session_obj["ip_address"] = session.ip_address;
            session_obj["is_active"] = session.is_active;
            session_obj["created_at"] = to_iso_string(session.created_at);
            session_obj["last_activity"] = to_iso_string(session.last_activity);
            session_obj["expires_at"] = to_iso_string(session.expires_at);

            response["sessions"][i] = std::move(session_obj);
        }

        LOG_INFO(logger_, "Successfully retrieved " + std::to_string(sessions.size()) +
                 " sessions for user: " + user_id);

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_get_sessions_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// T222.3: GET AUDIT STATS ENDPOINT
// GET /v1/security/audit-stats
// ============================================================================
crow::response RestApiImpl::handle_get_audit_stats_request(const crow::request& req) {
    try {
        LOG_INFO(logger_, "Received get audit stats request");

        if (!security_audit_logger_) {
            return crow::response(501, "{\"error\":\"Audit logging not configured\"}");
        }

        // Get audit statistics
        auto stats_result = security_audit_logger_->get_audit_stats();

        if (!stats_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get audit stats: " +
                     ErrorHandler::format_error(stats_result.error()));
            crow::json::wvalue error_response;
            error_response["error"] = stats_result.error().message;
            return crow::response(500, error_response);
        }

        auto stats = stats_result.value();

        // Build response
        crow::json::wvalue response;
        for (const auto& [key, value] : stats) {
            response[key] = value;
        }

        LOG_INFO(logger_, "Successfully retrieved audit statistics");

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_get_audit_stats_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// ============================================================================
// SECURITY ROUTES REGISTRATION
// This method registers all security-related endpoints with Crow
// ============================================================================
void RestApiImpl::handle_security_routes() {
    LOG_INFO(logger_, "Registering security audit routes");

    // GET /v1/security/audit-log - Get audit log events
    CROW_ROUTE((*app_), "/v1/security/audit-log")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_get_audit_log_request(req);
        });

    // GET /v1/security/sessions - Get active sessions for a user
    CROW_ROUTE((*app_), "/v1/security/sessions")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_get_sessions_request(req);
        });

    // GET /v1/security/audit-stats - Get audit log statistics
    CROW_ROUTE((*app_), "/v1/security/audit-stats")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_get_audit_stats_request(req);
        });

    LOG_INFO(logger_, "Security audit routes registered successfully");
}

} // namespace jadevectordb
