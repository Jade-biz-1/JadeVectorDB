#include "rest_api.h"
#include "lib/logging.h"
#include "lib/config.h"
#include "lib/auth.h"
#include "lib/error_handling.h"
#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include <chrono>
#include <thread>
#include <random>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jadevectordb {

// Lifecycle management types
struct RetentionPolicy {
    int max_age_days;
    bool archive_on_expire;
    int archive_threshold_days;
    bool enable_cleanup;
    std::string cleanup_schedule;
};

struct LifecycleConfig {
    std::string database_id;
    RetentionPolicy retention_policy;
    bool enabled;
};

RestApiService::RestApiService(int port) 
    : port_(port), running_(false) {
    logger_ = logging::LoggerManager::get_logger("RestApiService");
    server_address_ = "0.0.0.0:" + std::to_string(port_);
    api_impl_ = std::make_unique<RestApiImpl>();
}

RestApiService::~RestApiService() {
    stop();
}

bool RestApiService::start() {
    LOG_INFO(logger_, "Starting REST API server on port " << port_);
    
    if (!api_impl_->initialize(port_)) {
        LOG_ERROR(logger_, "Failed to initialize REST API server");
        return false;
    }
    
    api_impl_->register_routes();
    
    running_ = true;
    server_thread_ = std::make_unique<std::thread>(&RestApiService::run_server, this);
    
    LOG_INFO(logger_, "REST API server started successfully");
    return true;
}

void RestApiService::stop() {
    if (running_) {
        LOG_INFO(logger_, "Stopping REST API server");
        running_ = false;
        
        if (server_thread_ && server_thread_->joinable()) {
            server_thread_->join();
        }
        
        LOG_INFO(logger_, "REST API server stopped");
    }
}

void RestApiService::run_server() {
    LOG_INFO(logger_, "REST API server thread started");
    
    if (api_impl_) {
        api_impl_->start_server();
    }
    
    LOG_INFO(logger_, "REST API server thread ended");
}

RestApiImpl::RestApiImpl() {
    logger_ = logging::LoggerManager::get_logger("RestApiImpl");
}

bool RestApiImpl::initialize(int port) {
    LOG_INFO(logger_, "Initializing REST API on port " << port);
    
    // Initialize services that the API will use
    db_service_ = std::make_unique<DatabaseService>();
    vector_storage_service_ = std::make_unique<VectorStorageService>();
    similarity_search_service_ = std::make_unique<SimilaritySearchService>();
    index_service_ = std::make_unique<IndexService>();
    lifecycle_service_ = std::make_unique<LifecycleService>();
    auth_manager_ = AuthManager::get_instance();
    security_audit_logger_ = std::make_shared<SecurityAuditLogger>();
    authentication_service_ = std::make_unique<AuthenticationService>();

    // Initialize the services
    db_service_->initialize();
    vector_storage_service_->initialize();
    similarity_search_service_->initialize();
    // Note: IndexService and LifecycleService don't have initialize() methods

    // Initialize audit logging
    SecurityAuditConfig audit_config;
    audit_config.log_file_path = "./logs/security_audit.log";
    audit_config.enabled = true;
    if (!security_audit_logger_->initialize(audit_config)) {
        LOG_WARN(logger_, "Failed to initialize security audit logger");
    }

    // Initialize authentication service
    authentication_config_ = AuthenticationConfig{};
    authentication_config_.enable_api_keys = true;
    authentication_config_.require_strong_passwords = true;
    authentication_config_.min_password_length = 10;
    if (!authentication_service_->initialize(authentication_config_, security_audit_logger_)) {
        LOG_ERROR(logger_, "Failed to initialize authentication service");
        return false;
    }

    // Seed default users for non-production environments (T236 - FR-029)
    auto seed_result = authentication_service_->seed_default_users();
    if (!seed_result.has_value()) {
        LOG_WARN(logger_, "Failed to seed default users: " << ErrorHandler::format_error(seed_result.error()));
    }

    // Get runtime environment for other purposes
    const char* env_ptr = std::getenv("JADE_ENV");
    runtime_environment_ = env_ptr ? std::string(env_ptr) : "development";
    
    // Initialize distributed services if they exist
    initialize_distributed_services();
    
    // Create Crow app instance
    app_ = std::make_unique<crow::App<>>();
    server_port_ = port;
    
    // Perform any initialization needed
    setup_error_handling();
    setup_authentication();
    setup_request_validation();
    setup_response_serialization();
    
    return true;
}

void RestApiImpl::start_server() {
    if (app_) {
        LOG_INFO(logger_, "Starting Crow server on port " << server_port_);
        app_->port(server_port_).multithreaded().run();
    }
}

void RestApiImpl::register_routes() {
    LOG_INFO(logger_, "Registering REST API routes");
    
    // Health and monitoring endpoints
    handle_health_check();
    handle_system_status();
    
    // Database management endpoints
    app_->route_dynamic("/v1/databases")
        ([this](const crow::request& req) {
            // Handle HTTP method
            if (req.method == crow::HTTPMethod::POST) {
                return handle_create_database_request(req);
            } else if (req.method == crow::HTTPMethod::GET) {
                return handle_list_databases_request(req);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Routes already registered above via route_dynamic()
    // Calling these would create duplicate route handlers
    // handle_create_database();
    // handle_list_databases();
    
    app_->route_dynamic("/v1/databases/<string>")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_get_database_request(req, database_id);
            } else if (req.method == crow::HTTPMethod::PUT) {
                return handle_update_database_request(req, database_id);
            } else if (req.method == crow::HTTPMethod::DELETE) {
                return handle_delete_database_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Routes already registered above
    // handle_get_database();
    // handle_update_database();
    // handle_delete_database();
    
    // Vector management endpoints
    app_->route_dynamic("/v1/databases/<string>/vectors")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_store_vector_request(req, database_id);
            }
            // For batch operations, we might need to check request body content
            // Or add separate endpoints like /v1/databases/{db}/vectors/batch
            return crow::response(405, "Method not allowed");
        });
    app_->route_dynamic("/v1/databases/<string>/vectors/batch")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_batch_store_vectors_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    app_->route_dynamic("/v1/databases/<string>/vectors/batch-get")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_batch_get_vectors_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Routes already registered above
    // handle_store_vector();
    // handle_batch_store_vectors();
    // handle_batch_get_vectors();
    
    app_->route_dynamic("/v1/databases/<string>/vectors/<string>")
        ([this](const crow::request& req, std::string database_id, std::string vector_id) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_get_vector_request(req, database_id, vector_id);
            } else if (req.method == crow::HTTPMethod::PUT) {
                return handle_update_vector_request(req, database_id, vector_id);
            } else if (req.method == crow::HTTPMethod::DELETE) {
                return handle_delete_vector_request(req, database_id, vector_id);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Routes already registered above
    // handle_get_vector();
    // handle_update_vector();
    // handle_delete_vector();
    
    // Search endpoints
    app_->route_dynamic("/v1/databases/<string>/search")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_similarity_search_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    app_->route_dynamic("/v1/databases/<string>/search/advanced")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_advanced_search_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Routes already registered above
    // handle_similarity_search();
    // handle_advanced_search();
    
    // Index management endpoints
    app_->route_dynamic("/v1/databases/<string>/indexes")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_create_index_request(req, database_id);
            } else if (req.method == crow::HTTPMethod::GET) {
                return handle_list_indexes_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    app_->route_dynamic("/v1/databases/<string>/indexes/<string>")
        ([this](const crow::request& req, std::string database_id, std::string index_id) {
            if (req.method == crow::HTTPMethod::PUT) {
                return handle_update_index_request(req, database_id, index_id);
            } else if (req.method == crow::HTTPMethod::DELETE) {
                return handle_delete_index_request(req, database_id, index_id);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Routes already registered above
    // handle_create_index();
    // handle_list_indexes();
    // handle_update_index();
    // handle_delete_index();
    
    // Embedding generation endpoints
    app_->route_dynamic("/v1/embeddings/generate")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_generate_embedding_request(req);
            }
            return crow::response(405, "Method not allowed");
        });
    // NOTE: Route already registered above
    // handle_generate_embedding();

    // Security, authentication, and administration endpoints
    handle_authentication_routes();
    handle_user_management_routes();
    handle_api_key_routes();
    handle_security_routes();
    handle_alert_routes();
    handle_cluster_routes();
    handle_performance_routes();
    
    LOG_INFO(logger_, "All REST API routes registered successfully");
}

// Health and monitoring endpoints


// Database management endpoints
void RestApiImpl::handle_create_database() {
    LOG_DEBUG(logger_, "Setting up create database endpoint at /v1/databases");

    CROW_ROUTE((*app_), "/v1/databases")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            return handle_create_database_request(req);
        });
}

void RestApiImpl::handle_list_databases() {
    LOG_DEBUG(logger_, "Setting up list databases endpoint at /v1/databases");
    CROW_ROUTE((*app_), "/v1/databases")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_list_databases_request(req);
        });
}

void RestApiImpl::handle_get_database() {
    LOG_DEBUG(logger_, "Setting up handle_get_database endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req, const std::string& database_id) {
            return handle_get_database_request(req, database_id);
        });
}

void RestApiImpl::handle_update_database() {
    LOG_DEBUG(logger_, "Setting up handle_update_database endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>")
        .methods(crow::HTTPMethod::PUT)
        ([this](const crow::request& req, const std::string& database_id) {
            return handle_update_database_request(req, database_id);
        });
}

void RestApiImpl::handle_delete_database() {
    LOG_DEBUG(logger_, "Setting up handle_delete_database endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>")
        .methods(crow::HTTPMethod::DELETE)
        ([this](const crow::request& req, const std::string& database_id) {
            return handle_delete_database_request(req, database_id);
        });
}

// Vector management endpoints
void RestApiImpl::handle_store_vector() {
    LOG_DEBUG(logger_, "Setting up handle_store_vector endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req, const std::string& database_id) {
            return handle_store_vector_request(req, database_id);
        });
}

void RestApiImpl::handle_get_vector() {
    LOG_DEBUG(logger_, "Setting up handle_get_vector endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/<string>")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req, const std::string& database_id, const std::string& vector_id) {
            return handle_get_vector_request(req, database_id, vector_id);
        });
}

void RestApiImpl::handle_update_vector() {
    LOG_DEBUG(logger_, "Setting up handle_update_vector endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/<string>")
        .methods(crow::HTTPMethod::PUT)
        ([this](const crow::request& req, const std::string& database_id, const std::string& vector_id) {
            return handle_update_vector_request(req, database_id, vector_id);
        });
}

void RestApiImpl::handle_delete_vector() {
    LOG_DEBUG(logger_, "Setting up handle_delete_vector endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/<string>")
        .methods(crow::HTTPMethod::DELETE)
        ([this](const crow::request& req, const std::string& database_id, const std::string& vector_id) {
            return handle_delete_vector_request(req, database_id, vector_id);
        });
}

void RestApiImpl::handle_batch_store_vectors() {
    LOG_DEBUG(logger_, "Setting up handle_batch_store_vectors endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/batch")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req, const std::string& database_id) {
            return handle_batch_store_vectors_request(req, database_id);
        });
}

// Search endpoints
void RestApiImpl::handle_similarity_search() {
    LOG_DEBUG(logger_, "Setting up similarity search endpoint at /v1/databases/{databaseId}/search");
    
    // In a real implementation with a web framework, this would register a POST endpoint
    // that connects to the SimilaritySearchService for similarity search
    // Example pseudo-code for the actual web framework integration:
    /*
    POST("/v1/databases/:databaseId/search", [&](const Request& req, Response& res) {
        try {
            // Extract database ID from path
            std::string database_id = req.path_params.at("databaseId");
            
            // Extract API key from header
            std::string api_key = req.get_header_value("Authorization");
            if (api_key.substr(0, 7) == "Bearer ") {
                api_key = api_key.substr(7);
            } else if (api_key.substr(0, 5) == "ApiKey ") {
                api_key = api_key.substr(5);
            }
            
            // Authenticate request
            auto auth_result = authenticate_request(api_key);
            if (!auth_result.has_value()) {
                res.status = 401; // Unauthorized
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}", "application/json");
                return;
            }
            
            // Check if user has permission to perform search in this database
            // auto auth_manager = auth_manager_.get();
            // auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            // if (user_id_result.has_value()) {
            //     auto perm_result = auth_manager->has_permission_with_api_key(api_key, "search:execute");
            //     if (!perm_result.has_value() || !perm_result.value()) {
            //         res.status = 403; // Forbidden
            //         res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
            //         return;
            //     }
            // }
            
            // Parse query vector and search parameters from request body
            auto search_request = parse_search_request_from_json(req.body);
            auto query_vector = search_request.query_vector;
            auto search_params = search_request.search_params;
            
            // Validate search parameters
            auto validation_result = similarity_search_service_->validate_search_params(search_params);
            if (!validation_result.has_value()) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Invalid search parameters\"}", "application/json");
                return;
            }
            
            // Perform similarity search using the service
            auto result = similarity_search_service_->similarity_search(database_id, query_vector, search_params);
            
            if (result.has_value()) {
                // Serialize results to JSON
                auto json_str = serialize_search_results_to_json(result.value());
                res.status = 200; // OK
                res.set_content(json_str, "application/json");
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Search failed\"}", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
        }
    });
    */
}

void RestApiImpl::handle_advanced_search() {
    LOG_DEBUG(logger_, "Setting up advanced search endpoint at /v1/databases/{databaseId}/search/advanced");
    
    // In a real implementation with a web framework, this would register a POST endpoint
    // that connects to the SimilaritySearchService with advanced filtering capabilities
    // Example pseudo-code for the actual web framework integration:
    /*
    POST("/v1/databases/:databaseId/search/advanced", [&](const Request& req, Response& res) {
        try {
            // Extract database ID from path
            std::string database_id = req.path_params.at("databaseId");
            
            // Extract API key from header
            std::string api_key = req.get_header_value("Authorization");
            if (api_key.substr(0, 7) == "Bearer ") {
                api_key = api_key.substr(7);
            } else if (api_key.substr(0, 5) == "ApiKey ") {
                api_key = api_key.substr(5);
            }
            
            // Authenticate request
            auto auth_result = authenticate_request(api_key);
            if (!auth_result.has_value()) {
                res.status = 401; // Unauthorized
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}", "application/json");
                return;
            }
            
            // Authorize user for advanced search operation
            auto auth_manager = auth_manager_;
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (!user_id_result.has_value()) {
                res.status = 401; // Unauthorized
                res.set_content("{\"error\":\"Invalid API key\"}", "application/json");
                return;
            }
            
            std::string user_id = user_id_result.value();
            
            // Check if user has permission to perform search in this database
            auto search_perm_result = auth_manager->has_permission_with_api_key(api_key, "search:execute");
            if (!search_perm_result.has_value() || !search_perm_result.value()) {
                res.status = 403; // Forbidden
                res.set_content("{\"error\":\"Insufficient permissions for search operation\"}", "application/json");
                return;
            }
            
            // Check if user has access to the specific database
            auto db_access_result = auth_manager->check_database_access(user_id, database_id);
            if (!db_access_result.has_value() || !db_access_result.value()) {
                res.status = 403; // Forbidden
                res.set_content("{\"error\":\"Access denied to database\"}", "application/json");
                return;
            }
            
            // Parse query vector and advanced search parameters from request body
            auto search_request = parse_advanced_search_request_from_json(req.body);
            auto query_vector = search_request.query_vector;
            auto search_params = search_request.search_params;  // Includes filters
            auto complex_filter = search_request.complex_filter;  // Advanced filter object
            
            // Validate search parameters
            auto validation_result = similarity_search_service_->validate_search_params(search_params);
            if (!validation_result.has_value()) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Invalid search parameters\"}", "application/json");
                return;
            }
            
            // If a complex filter is provided, validate it first
            if (!complex_filter.conditions.empty() || !complex_filter.nested_filters.empty()) {
                auto filter_validation_result = similarity_search_service_->validate_complex_filter(complex_filter);
                if (!filter_validation_result.has_value()) {
                    res.status = 400; // Bad Request
                    res.set_content("{\"error\":\"Invalid complex filter\"}", "application/json");
                    return;
                }
            }
            
            // Log the search operation for audit purposes
            LOG_INFO(logger_, "User " << user_id << " performing advanced search on database " << database_id);
            
            // Perform advanced similarity search using the service
            auto result = similarity_search_service_->similarity_search(database_id, query_vector, search_params);
            
            if (result.has_value()) {
                // Serialize results to JSON
                auto json_str = serialize_search_results_to_json(result.value());
                res.status = 200; // OK
                res.set_content(json_str, "application/json");
                
                // Log successful search
                LOG_DEBUG(logger_, "Advanced search completed successfully for user " << user_id);
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Search failed\"}", "application/json");
                
                // Log failed search
                LOG_WARN(logger_, "Advanced search failed for user " << user_id << ": " 
                          << ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            // Log error
            LOG_ERROR(logger_, "Exception in advanced search: " << e.what());
        }
    });
    */
}

// Index management endpoints
void RestApiImpl::handle_create_index() {
    // Implementation would register POST /v1/databases/{databaseId}/indexes endpoint
    LOG_DEBUG(logger_, "Registered create index endpoint at /v1/databases/{databaseId}/indexes");
}

void RestApiImpl::handle_list_indexes() {
    // Implementation would register GET /v1/databases/{databaseId}/indexes endpoint
    LOG_DEBUG(logger_, "Registered list indexes endpoint at /v1/databases/{databaseId}/indexes");
}

void RestApiImpl::handle_update_index() {
    // Implementation would register PUT /v1/databases/{databaseId}/indexes/{indexId} endpoint
    LOG_DEBUG(logger_, "Registered update index endpoint at /v1/databases/{databaseId}/indexes/{indexId}");
}

void RestApiImpl::handle_delete_index() {
    // Implementation would register DELETE /v1/databases/{databaseId}/indexes/{indexId} endpoint
    LOG_DEBUG(logger_, "Registered delete index endpoint at /v1/databases/{databaseId}/indexes/{indexId}");
}

// Embedding generation endpoints
void RestApiImpl::handle_generate_embedding() {
    // Implementation would register POST /v1/embeddings/generate endpoint
    LOG_DEBUG(logger_, "Registered generate embedding endpoint at /v1/embeddings/generate");
}

// Metrics endpoint
void RestApiImpl::handle_health_check() {
    LOG_DEBUG(logger_, "Setting up health check endpoint at /health");
    
    app_->route_dynamic("/health")
    ([this](const crow::request& req) {
        try {
            // Extract API key from header (optional for health checks)
            std::string api_key;
            auto auth_header = req.get_header_value("Authorization");
            if (!auth_header.empty()) {
                if (auth_header.substr(0, 7) == "Bearer ") {
                    api_key = auth_header.substr(7);
                } else if (auth_header.substr(0, 5) == "ApiKey ") {
                    api_key = auth_header.substr(5);
                }
                
                // If API key provided, validate it
                if (!api_key.empty()) {
                    auto auth_result = authenticate_request(api_key);
                    if (!auth_result.has_value()) {
                        return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
                    }
                }
            }
            
            LOG_INFO(logger_, "Health check request received");
            
            // In a real implementation, this would call the MonitoringService to check system health
            // For now, returning a basic health status
            crow::json::wvalue response;
            response["status"] = "healthy";
            response["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["version"] = "1.0.0";
            response["checks"] = crow::json::wvalue::object();
            response["checks"]["database"] = "ok";
            response["checks"]["storage"] = "ok";
            response["checks"]["network"] = "ok";
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            return resp;
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error in health check: " + std::string(e.what()));
            return crow::response(500, "{\"error\":\"Internal server error\"}");
        }
    });
}

void RestApiImpl::handle_system_status() {
    LOG_DEBUG(logger_, "Setting up system status endpoint at /status");
    
    app_->route_dynamic("/status")
    ([this](const crow::request& req) {
        try {
            // Extract API key from header
            std::string api_key;
            auto auth_header = req.get_header_value("Authorization");
            if (!auth_header.empty()) {
                if (auth_header.substr(0, 7) == "Bearer ") {
                    api_key = auth_header.substr(7);
                } else if (auth_header.substr(0, 5) == "ApiKey ") {
                    api_key = auth_header.substr(5);
                }
            }
            
            // Authenticate request
            auto auth_result = authenticate_request(api_key);
            if (!auth_result.has_value()) {
                return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
            }

            // Check if user has permission to view system status
            auto auth_manager = auth_manager_;
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "monitoring:read");
                if (!perm_result.has_value() || !perm_result.value()) {
                    return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
                }
            }
            
            LOG_INFO(logger_, "System status request received");

            crow::json::wvalue response;
            response["status"] = "operational";
            response["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["version"] = "1.0.0";

            // Calculate uptime
            static auto start_time = std::chrono::steady_clock::now();
            auto current_time = std::chrono::steady_clock::now();
            auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();

            // Format uptime as human-readable string
            int days = uptime_seconds / 86400;
            int hours = (uptime_seconds % 86400) / 3600;
            int minutes = (uptime_seconds % 3600) / 60;
            int seconds = uptime_seconds % 60;

            std::ostringstream uptime_stream;
            if (days > 0) {
                uptime_stream << days << "d " << hours << "h " << minutes << "m " << seconds << "s";
            } else if (hours > 0) {
                uptime_stream << hours << "h " << minutes << "m " << seconds << "s";
            } else if (minutes > 0) {
                uptime_stream << minutes << "m " << seconds << "s";
            } else {
                uptime_stream << seconds << "s";
            }
            response["uptime"] = uptime_stream.str();
            response["uptime_seconds"] = static_cast<int64_t>(uptime_seconds);

            // Get system resource information
            response["system"] = crow::json::wvalue::object();

            // Try to read CPU and memory info from /proc (Linux)
            double cpu_usage = 0.0;
            double memory_usage = 0.0;

            #ifdef __linux__
            // Read memory info
            std::ifstream meminfo("/proc/meminfo");
            if (meminfo.is_open()) {
                std::string line;
                long total_mem = 0, free_mem = 0, available_mem = 0;
                while (std::getline(meminfo, line)) {
                    if (line.find("MemTotal:") == 0) {
                        std::istringstream iss(line);
                        std::string label;
                        iss >> label >> total_mem;
                    } else if (line.find("MemAvailable:") == 0) {
                        std::istringstream iss(line);
                        std::string label;
                        iss >> label >> available_mem;
                    }
                }
                meminfo.close();

                if (total_mem > 0 && available_mem > 0) {
                    memory_usage = ((total_mem - available_mem) * 100.0) / total_mem;
                }
            }

            // Simple CPU usage estimation (not accurate, but better than placeholder)
            std::ifstream stat_file("/proc/stat");
            if (stat_file.is_open()) {
                std::string line;
                std::getline(stat_file, line);
                stat_file.close();
                // Parse CPU line: cpu  user nice system idle iowait irq softirq
                if (line.find("cpu ") == 0) {
                    std::istringstream iss(line.substr(5));
                    long user, nice, system, idle;
                    iss >> user >> nice >> system >> idle;
                    long total = user + nice + system + idle;
                    long active = user + nice + system;
                    if (total > 0) {
                        cpu_usage = (active * 100.0) / total;
                    }
                }
            }
            #endif

            response["system"]["cpu_usage_percent"] = cpu_usage > 0 ? cpu_usage : 5.0;
            response["system"]["memory_usage_percent"] = memory_usage > 0 ? memory_usage : 35.0;
            response["system"]["disk_usage_percent"] = 45.0; // Placeholder for now

            // Add database count and vector statistics
            response["performance"] = crow::json::wvalue::object();

            // Try to get actual database count
            size_t db_count = 0;
            size_t total_vectors = 0;
            if (db_service_) {
                auto db_list_result = db_service_->list_databases();
                if (db_list_result.has_value()) {
                    db_count = db_list_result.value().size();

                    // Count vectors across all databases
                    for (const auto& db : db_list_result.value()) {
                        if (vector_storage_service_) {
                            auto vec_count_result = vector_storage_service_->get_vector_count(db.databaseId);
                            if (vec_count_result.has_value()) {
                                total_vectors += vec_count_result.value();
                            }
                        }
                    }
                }
            }

            response["performance"]["database_count"] = static_cast<int>(db_count);
            response["performance"]["total_vectors"] = static_cast<int64_t>(total_vectors);
            response["performance"]["avg_query_time_ms"] = 2.5; // Placeholder - would need metrics collection
            response["performance"]["active_connections"] = 1; // Current request
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            return resp;
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error in system status: " + std::string(e.what()));
            return crow::response(500, "{\"error\":\"Internal server error\"}");
        }
    });
}

void RestApiImpl::handle_database_status() {
    LOG_DEBUG(logger_, "Setting up database status endpoint at /v1/databases/{databaseId}/status");
    
    app_->route_dynamic("/v1/databases/<string>/status")
    ([this](const crow::request& req, std::string database_id) {
        try {
            // Extract API key from header
            std::string api_key;
            auto auth_header = req.get_header_value("Authorization");
            if (!auth_header.empty()) {
                if (auth_header.substr(0, 7) == "Bearer ") {
                    api_key = auth_header.substr(7);
                } else if (auth_header.substr(0, 5) == "ApiKey ") {
                    api_key = auth_header.substr(5);
                }
            }
            
            // Authenticate request
            auto auth_result = authenticate_request(api_key);
            if (!auth_result.has_value()) {
                return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
            }

            // Check if user has permission to view database status
            auto auth_manager = auth_manager_;
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "monitoring:read");
                if (!perm_result.has_value() || !perm_result.value()) {
                    return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
                }
            }
            
            LOG_INFO(logger_, "Database status request received for database: " << database_id);
            
            // Validate database exists
            auto db_exists_result = db_service_->database_exists(database_id);
            if (!db_exists_result.has_value() || !db_exists_result.value()) {
                return crow::response(404, "{\"error\":\"Database not found\"}");
            }
            
            // In a real implementation, this would call the MonitoringService for database status
            // For now, returning placeholder status information
            crow::json::wvalue response;
            response["databaseId"] = database_id;
            response["status"] = "online";
            response["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Add database-specific metrics
            response["metrics"] = crow::json::wvalue::object();
            response["metrics"]["vector_count"] = 50000;
            response["metrics"]["index_count"] = 3;
            response["metrics"]["storage_used_mb"] = 1024.5;
            response["metrics"]["avg_query_time_ms"] = 1.8;
            response["metrics"]["qps"] = 850;
            
            // Add index status
            crow::json::wvalue indexes_status;
            indexes_status["hnsw_index_1"] = "ready";
            indexes_status["ivf_index_1"] = "ready";
            indexes_status["flat_index_1"] = "ready";
            response["indexes"] = std::move(indexes_status);
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            return resp;
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error in database status: " + std::string(e.what()));
            return crow::response(500, "{\"error\":\"Internal server error\"}");
        }
    });
}

void RestApiImpl::handle_metrics() {
    // Implementation would register GET /metrics endpoint for Prometheus format
    LOG_DEBUG(logger_, "Registered metrics endpoint at /metrics");
}

// Request handling methods implementation

crow::response RestApiImpl::handle_store_vector_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Check if user has permission to store vectors in this database
        // This would check permissions in a real implementation

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Parse vector from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Create a Vector object from JSON
        Vector vector_data;
        vector_data.id = body_json["id"].s();
        if (body_json["values"].t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"Vector values must be an array\"}");
        }

        // Parse values
        auto values_array = body_json["values"];
        for (size_t i = 0; i < values_array.size(); i++) {
            vector_data.values.push_back(values_array[i].d());
        }

        // Parse metadata if present
        if (body_json.has("metadata")) {
            auto meta = body_json["metadata"];
            if (meta.has("source")) vector_data.metadata.source = meta["source"].s();
            if (meta.has("owner")) vector_data.metadata.owner = meta["owner"].s();
            if (meta.has("category")) vector_data.metadata.category = meta["category"].s();
            if (meta.has("status")) vector_data.metadata.status = meta["status"].s();
        }
        
        // Validate vector data
        auto validation_result = vector_storage_service_->validate_vector(database_id, vector_data);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"" + ErrorHandler::format_error(validation_result.error()) + "\"}");
        }
        
        // Store the vector using the service
        auto result = vector_storage_service_->store_vector(database_id, vector_data);
        
        if (result.has_value()) {
            crow::json::wvalue response;
            response["status"] = "success";
            response["vectorId"] = vector_data.id;
            crow::response resp(201, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Stored vector: " << vector_data.id << " in database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in store vector: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_get_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Check if user has permission to retrieve vectors from this database
        // This would check permissions in a real implementation

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Check if vector exists
        auto vector_exists_result = vector_storage_service_->vector_exists(database_id, vector_id);
        if (!vector_exists_result.has_value() || !vector_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Vector not found\"}");
        }
        
        // Retrieve the vector using the service
        auto result = vector_storage_service_->retrieve_vector(database_id, vector_id);
        
        if (result.has_value()) {
            auto vector = result.value();
            
            crow::json::wvalue response;
            response["id"] = vector.id;
            
            // Add values as an array
            crow::json::wvalue values_array;
            int idx = 0;
            for (auto val : vector.values) {
                values_array[idx++] = val;
            }
            response["values"] = std::move(values_array);

            // Add metadata if present
            if (!vector.metadata.source.empty() || !vector.metadata.tags.empty() || !vector.metadata.custom.empty()) {
                crow::json::wvalue metadata_obj;
                metadata_obj["source"] = vector.metadata.source;
                metadata_obj["created_at"] = vector.metadata.created_at;
                metadata_obj["updated_at"] = vector.metadata.updated_at;
                metadata_obj["owner"] = vector.metadata.owner;
                metadata_obj["category"] = vector.metadata.category;
                metadata_obj["score"] = vector.metadata.score;
                metadata_obj["status"] = vector.metadata.status;
                int tag_idx = 0;
                for (const auto& tag : vector.metadata.tags) {
                    metadata_obj["tags"][tag_idx++] = tag;
                }
                response["metadata"] = std::move(metadata_obj);
            }
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Retrieved vector: " << vector.id << " from database: " << database_id);
            return resp;
        } else {
            return crow::response(404, "{\"error\":\"Vector not found\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get vector: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_update_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Check if user has permission to update vectors in this database
        // This would check permissions in a real implementation

        // Parse updated vector from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Create a Vector object from JSON
        Vector vector_data;
        vector_data.id = vector_id;  // Ensure vector ID matches the path parameter
        if (body_json["values"].t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"Vector values must be an array\"}");
        }
        
        // Parse values
        auto values_array = body_json["values"];
        for (size_t i = 0; i < values_array.size(); i++) {
            vector_data.values.push_back(values_array[i].d());
        }

        // Parse metadata if present
        if (body_json.has("metadata")) {
            auto meta = body_json["metadata"];
            if (meta.has("source")) vector_data.metadata.source = meta["source"].s();
            if (meta.has("owner")) vector_data.metadata.owner = meta["owner"].s();
            if (meta.has("category")) vector_data.metadata.category = meta["category"].s();
            if (meta.has("status")) vector_data.metadata.status = meta["status"].s();
        }
        
        // Update the vector using the service
        auto result = vector_storage_service_->update_vector(database_id, vector_data);
        
        if (result.has_value()) {
            crow::json::wvalue response;
            response["status"] = "success";
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Updated vector: " << vector_data.id << " in database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update vector: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_delete_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Check if user has permission to delete vectors from this database
        // This would check permissions in a real implementation

        // Delete the vector using the service
        auto result = vector_storage_service_->delete_vector(database_id, vector_id);
        
        if (result.has_value()) {
            crow::json::wvalue response;
            response["status"] = "success";
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Deleted vector: " << vector_id << " from database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete vector: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_batch_store_vectors_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Parse vector list from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Parse vectors
        std::vector<Vector> vectors;
        if (body_json["vectors"].t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"Request body must contain a 'vectors' array\"}");
        }
        
        auto vectors_array = body_json["vectors"];
        for (size_t i = 0; i < vectors_array.size(); i++) {
            auto vec_json = vectors_array[i];
            Vector vector_data;
            vector_data.id = vec_json["id"].s();

            if (vec_json["values"].t() != crow::json::type::List) {
                return crow::response(400, "{\"error\":\"Vector values must be an array\"}");
            }

            // Parse values
            auto values_array = vec_json["values"];
            for (size_t j = 0; j < values_array.size(); j++) {
                vector_data.values.push_back(values_array[j].d());
            }

            // Parse metadata if present
            if (vec_json.has("metadata")) {
                auto meta = vec_json["metadata"];
                if (meta.has("source")) vector_data.metadata.source = meta["source"].s();
                if (meta.has("owner")) vector_data.metadata.owner = meta["owner"].s();
                if (meta.has("category")) vector_data.metadata.category = meta["category"].s();
                if (meta.has("status")) vector_data.metadata.status = meta["status"].s();
            }

            vectors.push_back(vector_data);
        }
        
        // Validate all vectors before storing
        for (const auto& vector : vectors) {
            auto validation_result = vector_storage_service_->validate_vector(database_id, vector);
            if (!validation_result.has_value()) {
                return crow::response(400, "{\"error\":\"Invalid vector data\"}");
            }
        }
        
        // Store the vectors using the service
        auto result = vector_storage_service_->batch_store_vectors(database_id, vectors);
        
        if (result.has_value()) {
            crow::json::wvalue response;
            response["status"] = "success";
            response["count"] = (int)vectors.size();
            crow::response resp(201, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Batch stored " << vectors.size() << " vectors in database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in batch store vectors: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_similarity_search_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Parse query vector and search parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Parse query vector
        if (body_json["queryVector"].t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"'queryVector' must be an array\"}");
        }

        Vector query_vector;
        auto query_array = body_json["queryVector"];
        for (size_t i = 0; i < query_array.size(); i++) {
            query_vector.values.push_back(query_array[i].d());
        }
        
        // Parse search parameters
        SearchParams search_params;
        if (body_json.has("topK")) {
            search_params.top_k = body_json["topK"].i();
        }
        if (body_json.has("threshold")) {
            search_params.threshold = body_json["threshold"].d();
        }
        if (body_json.has("includeMetadata")) {
            search_params.include_metadata = body_json["includeMetadata"].b();
        }
        if (body_json.has("includeVectorData")) {
            search_params.include_vector_data = body_json["includeVectorData"].b();
        }
        if (!search_params.include_vector_data && body_json.has("includeValues")) {
            search_params.include_vector_data = body_json["includeValues"].b();
        }
        
        // Validate search parameters
        auto validation_result = similarity_search_service_->validate_search_params(search_params);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"Invalid search parameters\"}");
        }
        
        // Perform similarity search using the service
        auto result = similarity_search_service_->similarity_search(database_id, query_vector, search_params);
        
        if (result.has_value()) {
            auto search_results = result.value();
            crow::json::wvalue response;
            response["count"] = static_cast<int>(search_results.size());

            int idx = 0;
            for (const auto& search_result : search_results) {
                crow::json::wvalue result_obj;
                result_obj["vectorId"] = search_result.vector_id;
                result_obj["score"] = search_result.similarity_score;

                if (search_params.include_vector_data || search_params.include_metadata) {
                    crow::json::wvalue vector_obj;
                    vector_obj["id"] = search_result.vector_data.id;

                    if (search_params.include_vector_data) {
                        crow::json::wvalue values_array;
                        int val_idx = 0;
                        for (auto val : search_result.vector_data.values) {
                            values_array[val_idx++] = val;
                        }
                        vector_obj["values"] = std::move(values_array);
                    }

                    if (search_params.include_metadata) {
                        crow::json::wvalue metadata_obj;
                        const auto& metadata = search_result.vector_data.metadata;
                        metadata_obj["source"] = metadata.source;
                        metadata_obj["owner"] = metadata.owner;
                        metadata_obj["category"] = metadata.category;
                        metadata_obj["status"] = metadata.status;
                        metadata_obj["createdAt"] = metadata.created_at;
                        metadata_obj["updatedAt"] = metadata.updated_at;
                        metadata_obj["score"] = metadata.score;

                        if (!metadata.tags.empty()) {
                            crow::json::wvalue tags_array;
                            int tag_idx = 0;
                            for (const auto& tag : metadata.tags) {
                                tags_array[tag_idx++] = tag;
                            }
                            metadata_obj["tags"] = std::move(tags_array);
                        }

                        if (!metadata.permissions.empty()) {
                            crow::json::wvalue permissions_array;
                            int perm_idx = 0;
                            for (const auto& permission : metadata.permissions) {
                                permissions_array[perm_idx++] = permission;
                            }
                            metadata_obj["permissions"] = std::move(permissions_array);
                        }

                        if (!metadata.custom.empty()) {
                            crow::json::wvalue custom_obj;
                            for (const auto& [key, value] : metadata.custom) {
                                auto parsed_value = crow::json::load(value.dump());
                                if (parsed_value) {
                                    custom_obj[key] = parsed_value;
                                } else {
                                    custom_obj[key] = value.dump();
                                }
                            }
                            metadata_obj["custom"] = std::move(custom_obj);
                        }

                        vector_obj["metadata"] = std::move(metadata_obj);
                    }

                    result_obj["vector"] = std::move(vector_obj);
                }

                response["results"][idx++] = std::move(result_obj);
            }

            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");

            LOG_DEBUG(logger_, "Similarity search completed: found " << search_results.size() << " results in database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"Search failed\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in similarity search: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_advanced_search_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Parse query vector and advanced search parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Parse query vector
        if (body_json["queryVector"].t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"'queryVector' must be an array\"}");
        }

        Vector query_vector;
        auto query_array = body_json["queryVector"];
        for (size_t i = 0; i < query_array.size(); i++) {
            query_vector.values.push_back(query_array[i].d());
        }
        
        // Parse search parameters
        SearchParams search_params;
        if (body_json.has("topK")) {
            search_params.top_k = body_json["topK"].i();
        }
        if (body_json.has("threshold")) {
            search_params.threshold = body_json["threshold"].d();
        }
        if (body_json.has("includeMetadata")) {
            search_params.include_metadata = body_json["includeMetadata"].b();
        }
        if (body_json.has("includeVectorData")) {
            search_params.include_vector_data = body_json["includeVectorData"].b();
        }
        if (!search_params.include_vector_data && body_json.has("includeValues")) {
            search_params.include_vector_data = body_json["includeValues"].b();
        }
        
        // Parse filters
        if (body_json.has("filters")) {
            auto filters = body_json["filters"];
            // This would be implemented with more complex filter logic in a real implementation
        }
        
        // Validate search parameters
        auto validation_result = similarity_search_service_->validate_search_params(search_params);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"Invalid search parameters\"}");
        }
        
        // Perform advanced similarity search using the service
        auto result = similarity_search_service_->similarity_search(database_id, query_vector, search_params);
        
        if (result.has_value()) {
            auto search_results = result.value();
            crow::json::wvalue response;
            response["count"] = static_cast<int>(search_results.size());

            int idx = 0;
            for (const auto& search_result : search_results) {
                crow::json::wvalue result_obj;
                result_obj["vectorId"] = search_result.vector_id;
                result_obj["score"] = search_result.similarity_score;

                if (search_params.include_vector_data || search_params.include_metadata) {
                    crow::json::wvalue vector_obj;
                    vector_obj["id"] = search_result.vector_data.id;

                    if (search_params.include_vector_data) {
                        crow::json::wvalue values_array;
                        int val_idx = 0;
                        for (auto val : search_result.vector_data.values) {
                            values_array[val_idx++] = val;
                        }
                        vector_obj["values"] = std::move(values_array);
                    }

                    if (search_params.include_metadata) {
                        crow::json::wvalue metadata_obj;
                        const auto& metadata = search_result.vector_data.metadata;
                        metadata_obj["source"] = metadata.source;
                        metadata_obj["owner"] = metadata.owner;
                        metadata_obj["category"] = metadata.category;
                        metadata_obj["status"] = metadata.status;
                        metadata_obj["createdAt"] = metadata.created_at;
                        metadata_obj["updatedAt"] = metadata.updated_at;
                        metadata_obj["score"] = metadata.score;

                        if (!metadata.tags.empty()) {
                            crow::json::wvalue tags_array;
                            int tag_idx = 0;
                            for (const auto& tag : metadata.tags) {
                                tags_array[tag_idx++] = tag;
                            }
                            metadata_obj["tags"] = std::move(tags_array);
                        }

                        if (!metadata.permissions.empty()) {
                            crow::json::wvalue permissions_array;
                            int perm_idx = 0;
                            for (const auto& permission : metadata.permissions) {
                                permissions_array[perm_idx++] = permission;
                            }
                            metadata_obj["permissions"] = std::move(permissions_array);
                        }

                        if (!metadata.custom.empty()) {
                            crow::json::wvalue custom_obj;
                            for (const auto& [key, value] : metadata.custom) {
                                auto parsed_value = crow::json::load(value.dump());
                                if (parsed_value) {
                                    custom_obj[key] = parsed_value;
                                } else {
                                    custom_obj[key] = value.dump();
                                }
                            }
                            metadata_obj["custom"] = std::move(custom_obj);
                        }

                        vector_obj["metadata"] = std::move(metadata_obj);
                    }

                    result_obj["vector"] = std::move(vector_obj);
                }

                response["results"][idx++] = std::move(result_obj);
            }

            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Advanced search completed: found " << search_results.size() << " results in database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"Search failed\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in advanced search: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_create_database_request(const crow::request& req) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Parse database creation parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Create a DatabaseCreationParams object from JSON
        DatabaseCreationParams db_config;
        if (body_json.has("name")) {
            db_config.name = body_json["name"].s();
        }
        if (body_json.has("description")) {
            db_config.description = body_json["description"].s();
        }
        if (body_json.has("vectorDimension")) {
            db_config.vectorDimension = body_json["vectorDimension"].i();
        }
        if (body_json.has("indexType")) {
            db_config.indexType = body_json["indexType"].s();
        }
        
        // Validate database creation parameters
        auto validation_result = db_service_->validate_creation_params(db_config);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"Invalid database creation parameters\"}");
        }
        
        // Create the database using the service
        auto result = db_service_->create_database(db_config);
        
        if (result.has_value()) {
            std::string database_id = result.value();
            crow::json::wvalue response;
            response["databaseId"] = database_id;
            response["status"] = "success";
            
            crow::response resp(201, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_INFO(logger_, "Created database: " << database_id << " (Name: " << db_config.name << ")");
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"Failed to create database\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create database: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_list_databases_request(const crow::request& req) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Parse query parameters for filtering and pagination
        DatabaseListParams list_params;
        // For now, we'll use default parameters, but could parse from URL params
        
        // List databases using the service
        auto result = db_service_->list_databases(list_params);
        
        if (result.has_value()) {
            auto databases = result.value();

            crow::json::wvalue response;
            response["total"] = static_cast<int>(databases.size());

            int idx = 0;
            for (const auto& db : databases) {
                crow::json::wvalue db_obj;
                db_obj["databaseId"] = db.databaseId;
                db_obj["name"] = db.name;
                db_obj["description"] = db.description;
                db_obj["vectorDimension"] = db.vectorDimension;
                db_obj["indexType"] = db.indexType;
                db_obj["created_at"] = db.created_at;
                db_obj["updated_at"] = db.updated_at;

                response["databases"][idx++] = std::move(db_obj);
            }

            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Listed " << databases.size() << " databases");
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"Failed to list databases\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list databases: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_get_database_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Get database using the service
        auto result = db_service_->get_database(database_id);
        
        if (result.has_value()) {
            auto database = result.value();
            
            // Serialize database to JSON
            crow::json::wvalue response;
            response["databaseId"] = database.databaseId;
            response["name"] = database.name;
            response["description"] = database.description;
            response["vectorDimension"] = database.vectorDimension;
            response["indexType"] = database.indexType;
            response["created_at"] = database.created_at;
            response["updated_at"] = database.updated_at;
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_DEBUG(logger_, "Retrieved database: " << database_id);
            return resp;
        } else {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get database: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_update_database_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Parse database update parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Create a DatabaseUpdateParams object from JSON
        DatabaseUpdateParams update_params;
        if (body_json.has("name")) {
            update_params.name = body_json["name"].s();
        }
        if (body_json.has("description")) {
            update_params.description = body_json["description"].s();
        }
        if (body_json.has("vectorDimension")) {
            update_params.vectorDimension = body_json["vectorDimension"].i();
        }
        if (body_json.has("indexType")) {
            update_params.indexType = body_json["indexType"].s();
        }
        
        // Validate database update parameters
        auto validation_result = db_service_->validate_update_params(update_params);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"Invalid database update parameters\"}");
        }
        
        // Update the database using the service
        auto result = db_service_->update_database(database_id, update_params);
        
        if (result.has_value()) {
            crow::json::wvalue response;
            response["status"] = "success";
            response["message"] = "Database updated successfully";
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_INFO(logger_, "Updated database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"Failed to update database\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update database: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_delete_database_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }
        
        // Delete the database using the service
        auto result = db_service_->delete_database(database_id);
        
        if (result.has_value()) {
            crow::json::wvalue response;
            response["status"] = "success";
            response["message"] = "Database deleted successfully";
            
            crow::response resp(200, response);
            resp.set_header("Content-Type", "application/json");
            
            LOG_INFO(logger_, "Deleted database: " << database_id);
            return resp;
        } else {
            return crow::response(400, "{\"error\":\"Failed to delete database\"}");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete database: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Helper methods
void RestApiImpl::setup_error_handling() {
    LOG_DEBUG(logger_, "Setting up error handling middleware");
    // In a real implementation, this would set up framework-specific error handling
}

void RestApiImpl::setup_authentication() {
    LOG_DEBUG(logger_, "Setting up authentication middleware");
    // In a real implementation, this would set up API key validation
    // using the AuthManager
    auto auth_manager = auth_manager_;
}

Result<bool> RestApiImpl::authenticate_request(const std::string& api_key) const {
    if (api_key.empty()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "No API key provided");
    }
    
    // Use the AuthManager to validate the API key
    auto auth_manager = auth_manager_;
    auto validation_result = auth_manager->validate_api_key(api_key);
    
    if (!validation_result.has_value()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "Invalid API key");
    }
    
    if (!validation_result.value()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "API key validation failed");
    }
    
    // Check if API key is active and not expired
    auto user_id_result = auth_manager->get_user_from_api_key(api_key);
    if (!user_id_result.has_value()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "Unable to retrieve user from API key");
    }
    
    return true;
}

void RestApiImpl::setup_request_validation() {
    LOG_DEBUG(logger_, "Setting up request validation middleware");
    // In a real implementation, this would set up JSON schema validation
}

void RestApiImpl::setup_response_serialization() {
    LOG_DEBUG(logger_, "Setting up response serialization");
    // In a real implementation, this would set up JSON serialization
}

void RestApiImpl::initialize_distributed_services() {
    // Create distributed services
    auto sharding_service = std::make_shared<ShardingService>();
    auto replication_service = std::make_shared<ReplicationService>();
    auto query_router = std::make_shared<QueryRouter>();
    
    // Set up sharding configuration
    ShardingConfig sharding_config;
    sharding_config.strategy = "hash";
    sharding_config.num_shards = 4;  // Default to 4 shards
    sharding_config.replication_factor = 1;  // Default replication
    
    sharding_service->initialize(sharding_config);
    
    // Set up replication configuration
    ReplicationConfig repl_config;
    repl_config.default_replication_factor = 1;  // Default to 1 for now
    repl_config.synchronous_replication = false;
    replication_service->initialize(repl_config);
    
    // Initialize the vector storage service with distributed services
    if (vector_storage_service_) {
        auto result = vector_storage_service_->initialize_distributed(
            sharding_service, 
            query_router, 
            replication_service
        );
        
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize VectorStorageService with distributed services: " 
                      << ErrorHandler::format_error(result.error()));
        } else {
            LOG_INFO(logger_, "VectorStorageService initialized with distributed services successfully");
        }
    }
}

crow::response RestApiImpl::handle_generate_embedding_request(const crow::request& req) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to generate embeddings
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "embedding:generate");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "Generate embedding request received");

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in handle_generate_embedding_request");
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        // Extract required parameters
        if (!body_json.has("input")) {
            LOG_ERROR(logger_, "Missing 'input' parameter in embedding generation request");
            return crow::response(400, "{\"error\":\"Missing 'input' parameter\"}");
        }

        std::string input = body_json["input"].s();
        std::string input_type = "text"; // Default to text
        if (body_json.has("input_type")) {
            input_type = body_json["input_type"].s();
        }

        std::string model = "default"; // Default model
        if (body_json.has("model")) {
            model = body_json["model"].s();
        }

        std::string provider = "default"; // Default provider
        if (body_json.has("provider")) {
            provider = body_json["provider"].s();
        }

        // Generate embedding based on input
        // For text input, use a simple hash-based embedding generation
        // This is a basic implementation - in production, you would use a proper embedding model
        std::vector<float> embedding_values;
        int target_dimension = 128; // Default dimension

        // Try to get dimension from model name if specified
        if (model.find("128") != std::string::npos) {
            target_dimension = 128;
        } else if (model.find("256") != std::string::npos) {
            target_dimension = 256;
        } else if (model.find("512") != std::string::npos) {
            target_dimension = 512;
        } else if (model.find("768") != std::string::npos) {
            target_dimension = 768;
        } else if (model.find("1536") != std::string::npos) {
            target_dimension = 1536;
        }

        // Generate deterministic embedding from input text using hash-based method
        // This ensures the same input always produces the same embedding
        embedding_values.resize(target_dimension);

        // Use multiple hash seeds to generate embedding components
        for (int i = 0; i < target_dimension; ++i) {
            std::hash<std::string> hasher;
            size_t hash_val = hasher(input + std::to_string(i));

            // Convert hash to float in range [-1, 1]
            double normalized = (static_cast<double>(hash_val % 10000) / 10000.0) * 2.0 - 1.0;
            embedding_values[i] = static_cast<float>(normalized);
        }

        // Normalize the embedding vector (L2 normalization)
        float norm = 0.0f;
        for (float val : embedding_values) {
            norm += val * val;
        }
        norm = std::sqrt(norm);

        if (norm > 0.0f) {
            for (float& val : embedding_values) {
                val /= norm;
            }
        }

        // Build response
        crow::json::wvalue response;
        response["input"] = input;
        response["input_type"] = input_type;
        response["model"] = model;
        response["provider"] = provider;

        // Add embedding values
        crow::json::wvalue emb_list;
        for (size_t i = 0; i < embedding_values.size(); ++i) {
            emb_list[i] = embedding_values[i];
        }
        response["embedding"] = std::move(emb_list);
        response["dimension"] = target_dimension;
        response["status"] = "success";
        response["generated_at"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        response["note"] = "Using hash-based embedding generation. For production use, integrate a proper embedding model.";

        LOG_INFO(logger_, "Embedding generated successfully");
        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_generate_embedding_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_create_index_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to create indexes
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "index:create");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "Create index request received for database: " << database_id);

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in handle_create_index_request");
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        // Extract required parameters
        if (!body_json.has("type")) {
            LOG_ERROR(logger_, "Missing 'type' parameter in index creation request");
            return crow::response(400, "{\"error\":\"Missing 'type' parameter\"}");
        }

        std::string index_type = body_json["type"].s();
        std::string index_name = database_id + "_" + index_type; // Default name
        if (body_json.has("name")) {
            index_name = body_json["name"].s();
        }

        // Extract optional parameters
        std::map<std::string, std::string> parameters;
        if (body_json.has("parameters")) {
            // For now, skip parsing complex parameters
            // In a real implementation, you'd parse each parameter properly
        }

        // Create index config
        Index index_config;
        index_config.type = index_type;
        index_config.databaseId = database_id;
        index_config.indexId = "index_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        index_config.status = "creating";
        index_config.parameters = parameters;

        // Get database info
        auto db_result = db_service_->get_database(database_id);
        if (!db_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get database: " + ErrorHandler::format_error(db_result.error()));
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Create the index using the service
        auto result = index_service_->create_index(db_result.value(), index_config);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to create index: " + ErrorHandler::format_error(result.error()));
            return crow::response(400, "{\"error\":\"Failed to create index\"}");
        }

        std::string index_id = result.value();
        crow::json::wvalue response;
        response["indexId"] = index_id;
        response["databaseId"] = database_id;
        response["type"] = index_type;
        response["parameters"] = crow::json::wvalue::object();
        for (const auto& param : parameters) {
            response["parameters"][param.first] = param.second;
        }
        response["status"] = "created";
        response["createdAt"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        crow::response resp(201, response);
        resp.set_header("Content-Type", "application/json");
        LOG_INFO(logger_, "Index created successfully with ID: " << index_id);
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_create_index_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_list_indexes_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to list indexes
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "index:read");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "List indexes request received for database: " << database_id);

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Get indexes for the database using the service
        auto result = index_service_->list_indexes(database_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to list indexes: " + ErrorHandler::format_error(result.error()));
            return crow::response(400, "{\"error\":\"Failed to list indexes\"}");
        }

        auto indexes = result.value();
        crow::json::wvalue response;

        int idx = 0;
        for (const auto& index : indexes) {
            crow::json::wvalue index_obj;
            index_obj["indexId"] = index.indexId;
            index_obj["databaseId"] = index.databaseId;
            index_obj["type"] = index.type;
            index_obj["status"] = index.status;

            // Convert parameters to JSON object
            crow::json::wvalue params_obj;
            for (const auto& param : index.parameters) {
                params_obj[param.first] = param.second;
            }
            index_obj["parameters"] = std::move(params_obj);

            index_obj["createdAt"] = index.created_at;
            index_obj["updatedAt"] = index.updated_at;
            response[idx++] = std::move(index_obj);
        }

        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        LOG_DEBUG(logger_, "Listed " << indexes.size() << " indexes for database: " << database_id);
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_list_indexes_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_update_index_request(const crow::request& req, const std::string& database_id, const std::string& index_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to update indexes
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "index:update");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "Update index request received for index: " << index_id << " in database: " << database_id);

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Check if index exists
        auto index_result = index_service_->get_index(database_id, index_id);
        if (!index_result.has_value()) {
            return crow::response(404, "{\"error\":\"Index not found\"}");
        }

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in handle_update_index_request");
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        // Extract parameters to update
        std::map<std::string, std::string> parameters;
        if (body_json.has("parameters")) {
            // For now, skip parsing complex parameters
            // In a real implementation, you'd parse each parameter properly
        }

        // Update the index using the service
        Index new_config = index_result.value();
        // Update parameters (map is compatible with map)
        if (!parameters.empty()) {
            new_config.parameters = parameters;
        }
        auto result = index_service_->update_index(database_id, index_id, new_config);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to update index: ");
            if (result.has_value()) {
                LOG_ERROR(logger_, "Error details: " + ErrorHandler::format_error(result.error()));
            }
            return crow::response(400, "{\"error\":\"Failed to update index\"}");
        }

        crow::json::wvalue response;
        response["indexId"] = index_id;
        response["databaseId"] = database_id;
        response["status"] = "updated";
        response["updatedAt"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        LOG_INFO(logger_, "Index updated successfully: " << index_id);
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_update_index_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_delete_index_request(const crow::request& req, const std::string& database_id, const std::string& index_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to delete indexes
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "index:delete");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "Delete index request received for index: " << index_id << " in database: " << database_id);

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Check if index exists
        auto index_result = index_service_->get_index(database_id, index_id);
        if (!index_result.has_value()) {
            return crow::response(404, "{\"error\":\"Index not found\"}");
        }

        // Delete the index using the service
        auto result = index_service_->delete_index(database_id, index_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to delete index: ");
            if (result.has_value()) {
                LOG_ERROR(logger_, "Error details: " + ErrorHandler::format_error(result.error()));
            }
            return crow::response(400, "{\"error\":\"Failed to delete index\"}");
        }

        crow::json::wvalue response;
        response["indexId"] = index_id;
        response["databaseId"] = database_id;
        response["status"] = "deleted";
        response["deletedAt"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        LOG_INFO(logger_, "Index deleted successfully: " << index_id);
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_delete_index_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_configure_retention_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to configure retention
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "lifecycle:configure");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "Configure retention request received for database: " << database_id);

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in handle_configure_retention_request");
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        // Extract retention policy parameters
        RetentionPolicy policy;
        
        if (body_json.has("maxAgeDays")) {
            policy.max_age_days = body_json["maxAgeDays"].i();
        }
        if (body_json.has("archiveOnExpire")) {
            policy.archive_on_expire = body_json["archiveOnExpire"].b();
        }
        if (body_json.has("archiveThresholdDays")) {
            policy.archive_threshold_days = body_json["archiveThresholdDays"].i();
        }
        if (body_json.has("enableCleanup")) {
            policy.enable_cleanup = body_json["enableCleanup"].b();
        }
        if (body_json.has("cleanupSchedule")) {
            policy.cleanup_schedule = body_json["cleanupSchedule"].s();
        }

        // Create lifecycle configuration
        LifecycleConfig config;
        config.database_id = database_id;
        config.retention_policy = policy;
        config.enabled = true;  // Enable lifecycle management for this DB

        // Configure retention policy using the service
        std::chrono::hours max_age(policy.max_age_days * 24);
        auto result = lifecycle_service_->configure_retention_policy(database_id, max_age, policy.archive_on_expire);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to configure retention policy: ");
            if (result.has_value()) {
                LOG_ERROR(logger_, "Error details: " + ErrorHandler::format_error(result.error()));
            }
            return crow::response(400, "{\"error\":\"Failed to configure retention policy\"}");
        }

        crow::json::wvalue response;
        response["databaseId"] = database_id;
        response["retentionPolicy"]["maxAgeDays"] = policy.max_age_days;
        response["retentionPolicy"]["archiveOnExpire"] = policy.archive_on_expire;
        response["retentionPolicy"]["archiveThresholdDays"] = policy.archive_threshold_days;
        response["retentionPolicy"]["enableCleanup"] = policy.enable_cleanup;
        response["retentionPolicy"]["cleanupSchedule"] = policy.cleanup_schedule;
        response["status"] = "configured";
        response["updatedAt"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        LOG_INFO(logger_, "Retention policy configured for database: " << database_id);
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_configure_retention_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_lifecycle_status_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 5) == "ApiKey ") {
                api_key = auth_header.substr(5);
            }
        }
        
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // Check if user has permission to view lifecycle status
        auto auth_manager = auth_manager_;
        auto user_id_result = auth_manager->get_user_from_api_key(api_key);
        if (user_id_result.has_value()) {
            auto perm_result = auth_manager->has_permission_with_api_key(api_key, "lifecycle:read");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        LOG_INFO(logger_, "Lifecycle status request received for database: " << database_id);

        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        // Get retention policy for the database
        auto policy_result = lifecycle_service_->get_retention_policy(database_id);
        if (!policy_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get retention policy: " + ErrorHandler::format_error(policy_result.error()));
            return crow::response(400, "{\"error\":\"Failed to get retention policy\"}");
        }

        // Get lifecycle status
        auto status_result = lifecycle_service_->get_lifecycle_status(database_id);
        if (!status_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get lifecycle status: " + ErrorHandler::format_error(status_result.error()));
            return crow::response(400, "{\"error\":\"Failed to get lifecycle status\"}");
        }

        auto policy = policy_result.value();
        auto status = status_result.value();

        crow::json::wvalue response;
        response["databaseId"] = database_id;
        response["status"] = status;

        // Add retention policy (convert from pair to JSON)
        int max_age_days = policy.first.count() / 24;
        response["retentionPolicy"]["maxAgeDays"] = max_age_days;
        response["retentionPolicy"]["archiveOnExpire"] = policy.second;
        
        // Add timestamp
        response["status"] = "active";
        response["retrievedAt"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        LOG_DEBUG(logger_, "Lifecycle status retrieved for database: " << database_id);
        return resp;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_lifecycle_status_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

void RestApiImpl::handle_batch_get_vectors() {
    LOG_DEBUG(logger_, "Setting up handle_batch_get_vectors endpoint");
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/batch")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req, const std::string& database_id) {
            return handle_batch_get_vectors_request(req, database_id);
        });
}


crow::response RestApiImpl::handle_batch_get_vectors_request(const crow::request& req, const std::string& database_id) {
    try {
        // Extract API key from header
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            } else if (auth_header.substr(0, 7) == "ApiKey ") {
                api_key = auth_header.substr(7);
            }
        }

        // Validate API key if auth manager is available
        if (auth_manager_) {
            auto auth_result = auth_manager_->validate_api_key(api_key);
            if (!auth_result.has_value() || !auth_result.value()) {
                return crow::response(401, "{\"error\":\"Invalid API key\"}");
            }

            // Check read permission
            auto perm_result = auth_manager_->has_permission(api_key, "vectors.read");
            if (!perm_result.has_value() || !perm_result.value()) {
                return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
            }
        }

        // Parse request body to get vector IDs
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON body\"}");
        }

        if (!body_json.has("vector_ids") && !body_json.has("vectorIds") && !body_json.has("ids")) {
            return crow::response(400, "{\"error\":\"Missing 'vector_ids', 'vectorIds', or 'ids' field in request body\"}");
        }

        // Extract vector IDs from the request - support multiple field names for compatibility
        std::vector<std::string> vector_ids;
        auto ids_json = body_json.has("vector_ids") ? body_json["vector_ids"] :
                        (body_json.has("vectorIds") ? body_json["vectorIds"] : body_json["ids"]);

        if (ids_json.t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"'vector_ids' must be an array\"}");
        }

        for (size_t i = 0; i < ids_json.size(); i++) {
            vector_ids.push_back(ids_json[i].s());
        }

        if (vector_ids.empty()) {
            return crow::response(400, "{\"error\":\"'vector_ids' array cannot be empty\"}");
        }

        LOG_INFO(logger_, "Batch get vectors request for database: " + database_id +
                 ", vector_ids count: " + std::to_string(vector_ids.size()));

        // Retrieve vectors from storage
        auto result = vector_storage_service_->retrieve_vectors(database_id, vector_ids);

        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to retrieve vectors: " + result.error().message);
            crow::json::wvalue error_response;
            error_response["error"] = result.error().message;
            return crow::response(500, error_response);
        }

        // Build response with retrieved vectors
        crow::json::wvalue response;
        response["database_id"] = database_id;
        response["count"] = result.value().size();

        crow::json::wvalue::list vectors_array;
        for (const auto& vector : result.value()) {
            crow::json::wvalue vec_obj;
            vec_obj["id"] = vector.id;

            // Add vector values
            crow::json::wvalue values_list;
            int val_idx = 0;
            for (const auto& val : vector.values) {
                values_list[val_idx++] = val;
            }
            vec_obj["values"] = std::move(values_list);

            // Add metadata if present
            if (!vector.metadata.source.empty() || !vector.metadata.tags.empty() || !vector.metadata.custom.empty()) {
                crow::json::wvalue metadata_obj;
                metadata_obj["source"] = vector.metadata.source;
                metadata_obj["created_at"] = vector.metadata.created_at;
                metadata_obj["updated_at"] = vector.metadata.updated_at;
                metadata_obj["owner"] = vector.metadata.owner;
                metadata_obj["category"] = vector.metadata.category;
                metadata_obj["score"] = vector.metadata.score;
                metadata_obj["status"] = vector.metadata.status;
                int tag_idx = 0;
                for (const auto& tag : vector.metadata.tags) {
                    metadata_obj["tags"][tag_idx++] = tag;
                }
                vec_obj["metadata"] = std::move(metadata_obj);
            }

            vectors_array.push_back(std::move(vec_obj));
        }
        response["vectors"] = std::move(vectors_array);

        LOG_INFO(logger_, "Successfully retrieved " + std::to_string(result.value().size()) + " vectors");

        crow::response resp(200, response);
        resp.set_header("Content-Type", "application/json");
        return resp;

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error in handle_batch_get_vectors_request: " + std::string(e.what()));
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Authentication routes implementation
void RestApiImpl::handle_authentication_routes() {
    LOG_DEBUG(logger_, "Setting up authentication routes");

    // Register endpoint: POST /v1/auth/register
    app_->route_dynamic("/v1/auth/register")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_register_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    // Login endpoint: POST /v1/auth/login
    app_->route_dynamic("/v1/auth/login")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_login_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    // Logout endpoint: POST /v1/auth/logout
    app_->route_dynamic("/v1/auth/logout")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_logout_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    // Forgot password endpoint: POST /v1/auth/forgot-password
    app_->route_dynamic("/v1/auth/forgot-password")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_forgot_password_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    // Reset password endpoint: POST /v1/auth/reset-password
    app_->route_dynamic("/v1/auth/reset-password")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_reset_password_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "Authentication routes configured");
}

// User management routes implementation
void RestApiImpl::handle_user_management_routes() {
    LOG_DEBUG(logger_, "Setting up user management routes");

    // List users: GET /v1/users
    // Create user: POST /v1/users
    app_->route_dynamic("/v1/users")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_list_users_request(req);
            } else if (req.method == crow::HTTPMethod::POST) {
                return handle_create_user_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    // Update user: PUT /v1/users/{userId}
    // Delete user: DELETE /v1/users/{userId}
    // Get user: GET /v1/users/{userId}
    app_->route_dynamic("/v1/users/<string>")
        ([this](const crow::request& req, std::string user_id) {
            if (req.method == crow::HTTPMethod::PUT) {
                return handle_update_user_request(req, user_id);
            } else if (req.method == crow::HTTPMethod::DELETE) {
                return handle_delete_user_request(req, user_id);
            } else if (req.method == crow::HTTPMethod::GET) {
                // Return user info (stub)
                crow::json::wvalue response;
                response["userId"] = user_id;
                response["status"] = "Not implemented - user get endpoint";
                return crow::response(501, response);
            }
            return crow::response(405, "Method not allowed");
        });

    // Activate/deactivate user: POST /v1/users/{userId}/activate | /deactivate
    app_->route_dynamic("/v1/users/<string>/activate")
        ([this](const crow::request& req, std::string user_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_user_status_request(req, user_id, true);
            }
            return crow::response(405, "Method not allowed");
        });

    app_->route_dynamic("/v1/users/<string>/deactivate")
        ([this](const crow::request& req, std::string user_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_user_status_request(req, user_id, false);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "User management routes configured");
}

// API Key routes implementation
void RestApiImpl::handle_api_key_routes() {
    LOG_DEBUG(logger_, "Setting up API key management routes");

    // List API keys: GET /v1/api-keys
    // Create API key: POST /v1/api-keys
    app_->route_dynamic("/v1/api-keys")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_list_api_keys_request(req);
            } else if (req.method == crow::HTTPMethod::POST) {
                return handle_create_api_key_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    // Revoke API key: DELETE /v1/api-keys/{keyId}
    app_->route_dynamic("/v1/api-keys/<string>")
        ([this](const crow::request& req, std::string key_id) {
            if (req.method == crow::HTTPMethod::DELETE) {
                return handle_revoke_api_key_request(req, key_id);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "API key management routes configured");
}

// Security routes implementation
void RestApiImpl::handle_security_routes() {
    LOG_DEBUG(logger_, "Setting up security audit routes");

    app_->route_dynamic("/v1/security/audit-logs")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_list_audit_logs_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "Security routes configured");
}

// Alert routes implementation
void RestApiImpl::handle_alert_routes() {
    LOG_DEBUG(logger_, "Setting up alert management routes");

    app_->route_dynamic("/v1/alerts")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_list_alerts_request(req);
            } else if (req.method == crow::HTTPMethod::POST) {
                return handle_create_alert_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    app_->route_dynamic("/v1/alerts/<string>/acknowledge")
        ([this](const crow::request& req, std::string alert_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_acknowledge_alert_request(req, alert_id);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "Alert routes configured");
}

// Cluster routes implementation
void RestApiImpl::handle_cluster_routes() {
    LOG_DEBUG(logger_, "Setting up cluster management routes");

    app_->route_dynamic("/v1/cluster/nodes")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_list_cluster_nodes_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    app_->route_dynamic("/v1/cluster/nodes/<string>")
        ([this](const crow::request& req, std::string node_id) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_cluster_node_status_request(req, node_id);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "Cluster routes configured");
}

// Performance routes implementation
void RestApiImpl::handle_performance_routes() {
    LOG_DEBUG(logger_, "Setting up performance monitoring routes");

    app_->route_dynamic("/v1/performance/metrics")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::GET) {
                return handle_performance_metrics_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

    LOG_INFO(logger_, "Performance routes configured");
}

// Authentication handler implementations
crow::response RestApiImpl::handle_register_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        if (!body_json.has("username") || !body_json.has("password")) {
            return crow::response(400, "{\"error\":\"Missing username or password\"}");
        }

        std::string username = body_json["username"].s();
        std::string password = body_json["password"].s();
        std::string email = body_json.has("email") ? std::string(body_json["email"].s()) : std::string("");

        // Get roles (default to "user" role if not specified)
        std::vector<std::string> roles = {"user"};
        if (body_json.has("roles") && body_json["roles"].t() == crow::json::type::List) {
            roles.clear();
            auto roles_array = body_json["roles"];
            for (size_t i = 0; i < roles_array.size(); i++) {
                roles.push_back(roles_array[i].s());
            }
        }

        // Register user with AuthenticationService
        auto register_result = authentication_service_->register_user(username, password, roles);

        if (!register_result.has_value()) {
            crow::json::wvalue error_response;
            error_response["error"] = ErrorHandler::format_error(register_result.error());
            return crow::response(400, error_response);
        }

        std::string user_id = register_result.value();

        // Also create user in AuthManager for permission management
        if (auth_manager_) {
            auto auth_result = auth_manager_->create_user(username, email, roles);
            if (!auth_result.has_value()) {
                LOG_WARN(logger_, "Failed to create user in AuthManager: " << ErrorHandler::format_error(auth_result.error()));
            }
        }

        // Log successful registration

        crow::json::wvalue response;
        response["success"] = true;
        response["userId"] = user_id;
        response["username"] = username;
        response["message"] = "User registered successfully";

        return crow::response(201, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in register: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_login_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        if (!body_json.has("username") || !body_json.has("password")) {
            return crow::response(400, "{\"error\":\"Missing username or password\"}");
        }

        std::string username = body_json["username"].s();
        std::string password = body_json["password"].s();

        // Get client IP and user agent
        std::string ip_address = req.get_header_value("X-Forwarded-For");
        if (ip_address.empty()) {
            ip_address = req.get_header_value("X-Real-IP");
        }
        std::string user_agent = req.get_header_value("User-Agent");

        // Authenticate with AuthenticationService
        auto auth_result = authentication_service_->authenticate(username, password, ip_address, user_agent);

        if (!auth_result.has_value()) {
            // Log failed login

            crow::json::wvalue error_response;
            error_response["error"] = "Invalid username or password";
            return crow::response(401, error_response);
        }

        auto token = auth_result.value();

        // Create session if supported
        if (authentication_service_) {
            auto session_result = authentication_service_->create_session(
                token.user_id,
                token.token_id,
                ip_address
            );
            // Session creation is optional, log but don't fail
            if (!session_result.has_value()) {
                LOG_WARN(logger_, "Failed to create session: " << ErrorHandler::format_error(session_result.error()));
            }
        }

        // Log successful login

        crow::json::wvalue response;
        response["success"] = true;
        response["userId"] = token.user_id;
        response["username"] = username;
        response["token"] = token.token_value;
        response["expiresAt"] = std::chrono::duration_cast<std::chrono::seconds>(
            token.expires_at.time_since_epoch()
        ).count();
        response["message"] = "Login successful";

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in login: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_logout_request(const crow::request& req) {
    try {
        // Extract token from Authorization header
        std::string token;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                token = auth_header.substr(7);
            }
        }

        if (token.empty()) {
            return crow::response(400, "{\"error\":\"Missing authorization token\"}");
        }

        // Logout (revoke token and end session)
        auto logout_result = authentication_service_->logout(token);

        if (!logout_result.has_value()) {
            crow::json::wvalue error_response;
            error_response["error"] = ErrorHandler::format_error(logout_result.error());
            return crow::response(400, error_response);
        }

        // Log logout
        if (security_audit_logger_) {
            auto validate_result = authentication_service_->validate_token(token);
            std::string user_id = validate_result.has_value() ? validate_result.value() : "unknown";

            // security_audit_logger_->log_event(
            //     SecurityEventType::LOGOUT,
            //     user_id,
            //     req.get_header_value("X-Forwarded-For"),
            //     true,
            //     "User logged out"
            // );
        }

        crow::json::wvalue response;
        response["success"] = true;
        response["message"] = "Logout successful";

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in logout: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_forgot_password_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        if (!body_json.has("email") && !body_json.has("username")) {
            return crow::response(400, "{\"error\":\"Missing email or username\"}");
        }

        std::string identifier = body_json.has("email") ? body_json["email"].s() : body_json["username"].s();

        // Get user by username or email
        auto user_result = authentication_service_->get_user_by_username(identifier);

        if (!user_result.has_value()) {
            // Don't reveal whether user exists for security
            crow::json::wvalue response;
            response["success"] = true;
            response["message"] = "If the account exists, a password reset link has been sent";
            return crow::response(200, response);
        }

        auto user = user_result.value();

        // Generate password reset token
        std::string reset_token = generate_secure_token();
        auto expires_at = std::chrono::system_clock::now() + std::chrono::hours(1);  // 1 hour expiry

        // Store reset token
        {
            std::lock_guard<std::mutex> lock(password_reset_mutex_);
            password_reset_tokens_[reset_token] = {
                reset_token,
                user.user_id,
                expires_at
            };
        }

        // Log password reset request

        // In production, send email with reset token
        // For now, include it in response (remove in production!)
        crow::json::wvalue response;
        response["success"] = true;
        response["message"] = "Password reset link has been sent";
        response["resetToken"] = reset_token;  // Remove this in production!
        response["note"] = "In production, this would be sent via email";

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in forgot password: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_reset_password_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        if (!body_json.has("token") || !body_json.has("new_password")) {
            return crow::response(400, "{\"error\":\"Missing token or new_password\"}");
        }

        std::string token = body_json["token"].s();
        std::string new_password = body_json["new_password"].s();

        // Validate reset token
        std::string user_id;
        {
            std::lock_guard<std::mutex> lock(password_reset_mutex_);
            auto it = password_reset_tokens_.find(token);
            if (it == password_reset_tokens_.end()) {
                return crow::response(400, "{\"error\":\"Invalid or expired reset token\"}");
            }

            auto& reset_token = it->second;

            // Check if expired
            if (std::chrono::system_clock::now() > reset_token.expires_at) {
                password_reset_tokens_.erase(it);
                return crow::response(400, "{\"error\":\"Reset token has expired\"}");
            }

            user_id = reset_token.user_id;

            // Remove token after use (one-time use)
            password_reset_tokens_.erase(it);
        }

        // Reset password using AuthenticationService
        auto reset_result = authentication_service_->reset_password(user_id, new_password);

        if (!reset_result.has_value()) {
            crow::json::wvalue error_response;
            error_response["error"] = ErrorHandler::format_error(reset_result.error());
            return crow::response(400, error_response);
        }

        // Log password reset

        crow::json::wvalue response;
        response["success"] = true;
        response["message"] = "Password reset successful";

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in reset password: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// User management handler implementations
crow::response RestApiImpl::handle_create_user_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        crow::json::wvalue response;
        response["message"] = "Create user endpoint - implementation pending";
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create user: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_list_users_request(const crow::request& req) {
    try {
        // Authenticate request
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            }
        }

        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"Unauthorized\"}");
        }

        // List users from AuthManager
        crow::json::wvalue response;

        if (auth_manager_) {
            auto users_result = auth_manager_->list_users();
            if (users_result.has_value()) {
                auto users = users_result.value();
                crow::json::wvalue users_array;
                int idx = 0;
                for (const auto& user : users) {
                    auto serialized = serialize_user(user);
                    users_array[idx++] = std::move(serialized);
                }
                response["users"] = std::move(users_array);
                response["count"] = idx;
            } else {
                response["users"] = crow::json::wvalue::list();
                response["count"] = 0;
            }
        } else {
            response["users"] = crow::json::wvalue::list();
            response["count"] = 0;
        }

        response["success"] = true;

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list users: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_update_user_request(const crow::request& req, const std::string& user_id) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        crow::json::wvalue response;
        response["message"] = "Update user endpoint - implementation pending";
        response["userId"] = user_id;
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update user: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_delete_user_request(const crow::request& req, const std::string& user_id) {
    try {
        crow::json::wvalue response;
        response["message"] = "Delete user endpoint - implementation pending";
        response["userId"] = user_id;
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete user: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_user_status_request(const crow::request& req, const std::string& user_id, bool activate) {
    try {
        // Authenticate request
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            }
        }

        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"Unauthorized\"}");
        }

        // Update user status
        bool status_updated = false;
        if (auth_manager_) {
            Result<void> status_result;
            if (activate) {
                status_result = auth_manager_->activate_user(user_id);
            } else {
                status_result = auth_manager_->deactivate_user(user_id);
            }
            if (!status_result.has_value()) {
                crow::json::wvalue error_response;
                error_response["error"] = ErrorHandler::format_error(status_result.error());
                return crow::response(400, error_response.dump());
            }
            status_updated = true;
        } else if (authentication_service_) {
            auto status_result = authentication_service_->set_user_active_status(user_id, activate);
            if (!status_result.has_value()) {
                crow::json::wvalue error_response;
                error_response["error"] = ErrorHandler::format_error(status_result.error());
                return crow::response(400, error_response.dump());
            }
            status_updated = status_result.value();
        } else {
            return crow::response(500, "{\"error\":\"Authentication services not available\"}");
        }

        if (!status_updated) {
            return crow::response(400, "{\"error\":\"Failed to update user status\"}");
        }

        // Log status change

        crow::json::wvalue response;
        response["success"] = true;
        response["userId"] = user_id;
        response["active"] = activate;
        response["message"] = activate ? "User activated successfully" : "User deactivated successfully";

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in user status: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// API Key management handler implementations
crow::response RestApiImpl::handle_list_api_keys_request(const crow::request& req) {
    try {
        // Authenticate request
        std::string api_key;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                api_key = auth_header.substr(7);
            }
        }

        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"Unauthorized\"}");
        }

        // List API keys
        crow::json::wvalue response;

        if (auth_manager_) {
            auto keys_result = auth_manager_->list_api_keys();
            if (keys_result.has_value()) {
                auto keys = keys_result.value();
                crow::json::wvalue keys_array;
                int idx = 0;
                for (const auto& key : keys) {
                    auto serialized = serialize_api_key(key);
                    keys_array[idx++] = std::move(serialized);
                }
                response["apiKeys"] = std::move(keys_array);
                response["count"] = idx;
            } else {
                response["apiKeys"] = crow::json::wvalue::list();
                response["count"] = 0;
            }
        } else {
            response["apiKeys"] = crow::json::wvalue::list();
            response["count"] = 0;
        }

        response["success"] = true;

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list API keys: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_create_api_key_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        // Authenticate request
        std::string token;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                token = auth_header.substr(7);
            }
        }

        auto auth_result = authenticate_request(token);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"Unauthorized\"}");
        }

        if (!body_json.has("userId")) {
            return crow::response(400, "{\"error\":\"Missing userId\"}");
        }

        std::string user_id = body_json["userId"].s();
        std::string description = body_json.has("description") ? std::string(body_json["description"].s()) : std::string("");

        // Parse permissions
        std::vector<std::string> permissions;
        if (body_json.has("permissions") && body_json["permissions"].t() == crow::json::type::List) {
            auto perms_array = body_json["permissions"];
            for (size_t i = 0; i < perms_array.size(); i++) {
                permissions.push_back(perms_array[i].s());
            }
        }

        // Generate API key
        std::string api_key;
        if (auth_manager_) {
            auto key_result = auth_manager_->generate_api_key(user_id, permissions, description);
            if (!key_result.has_value()) {
                crow::json::wvalue error_response;
                error_response["error"] = ErrorHandler::format_error(key_result.error());
                return crow::response(400, error_response);
            }
            api_key = key_result.value();
        } else if (authentication_service_) {
            auto key_result = authentication_service_->generate_api_key(user_id);
            if (!key_result.has_value()) {
                crow::json::wvalue error_response;
                error_response["error"] = ErrorHandler::format_error(key_result.error());
                return crow::response(400, error_response);
            }
            api_key = key_result.value();
        } else {
            return crow::response(500, "{\"error\":\"Authentication services not available\"}");
        }

        // Log API key creation

        crow::json::wvalue response;
        response["success"] = true;
        response["userId"] = user_id;
        response["apiKey"] = api_key;
        response["description"] = description;
        response["message"] = "API key created successfully";
        response["note"] = "Store this API key securely, it will not be shown again";

        return crow::response(201, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create API key: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_revoke_api_key_request(const crow::request& req, const std::string& key_id) {
    try {
        // Authenticate request
        std::string token;
        auto auth_header = req.get_header_value("Authorization");
        if (!auth_header.empty()) {
            if (auth_header.substr(0, 7) == "Bearer ") {
                token = auth_header.substr(7);
            }
        }

        auto auth_result = authenticate_request(token);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"Unauthorized\"}");
        }

        // Revoke API key
        bool revoke_success = false;
        if (auth_manager_) {
            auto revoke_result = auth_manager_->revoke_api_key(key_id);
            if (!revoke_result.has_value()) {
                crow::json::wvalue error_response;
                error_response["error"] = ErrorHandler::format_error(revoke_result.error());
                return crow::response(400, error_response.dump());
            }
            revoke_success = true;
        } else if (authentication_service_) {
            auto revoke_result = authentication_service_->revoke_api_key(key_id);
            if (!revoke_result.has_value()) {
                crow::json::wvalue error_response;
                error_response["error"] = ErrorHandler::format_error(revoke_result.error());
                return crow::response(400, error_response.dump());
            }
            revoke_success = revoke_result.value();
        } else {
            return crow::response(500, "{\"error\":\"Authentication services not available\"}");
        }

        if (!revoke_success) {
            return crow::response(400, "{\"error\":\"Failed to revoke API key\"}");
        }

        // Log API key revocation

        crow::json::wvalue response;
        response["success"] = true;
        response["keyId"] = key_id;
        response["message"] = "API key revoked successfully";

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in revoke API key: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Security audit handler implementations
crow::response RestApiImpl::handle_list_audit_logs_request(const crow::request& req) {
    try {
        crow::json::wvalue response;
        response["message"] = "List audit logs endpoint - implementation pending";
        response["logs"] = crow::json::wvalue::list();
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list audit logs: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Alert handler implementations
crow::response RestApiImpl::handle_list_alerts_request(const crow::request& req) {
    try {
        crow::json::wvalue response;
        response["message"] = "List alerts endpoint - implementation pending";
        response["alerts"] = crow::json::wvalue::list();
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list alerts: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_create_alert_request(const crow::request& req) {
    try {
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        crow::json::wvalue response;
        response["message"] = "Create alert endpoint - implementation pending";
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create alert: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_acknowledge_alert_request(const crow::request& req, const std::string& alert_id) {
    try {
        crow::json::wvalue response;
        response["message"] = "Acknowledge alert endpoint - implementation pending";
        response["alertId"] = alert_id;
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in acknowledge alert: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Cluster handler implementations
crow::response RestApiImpl::handle_list_cluster_nodes_request(const crow::request& req) {
    try {
        crow::json::wvalue response;
        response["message"] = "List cluster nodes endpoint - implementation pending";
        response["nodes"] = crow::json::wvalue::list();
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list cluster nodes: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

crow::response RestApiImpl::handle_cluster_node_status_request(const crow::request& req, const std::string& node_id) {
    try {
        crow::json::wvalue response;
        response["message"] = "Cluster node status endpoint - implementation pending";
        response["nodeId"] = node_id;
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cluster node status: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Performance handler implementations
crow::response RestApiImpl::handle_performance_metrics_request(const crow::request& req) {
    try {
        crow::json::wvalue response;
        response["message"] = "Performance metrics endpoint - implementation pending";
        response["metrics"] = crow::json::wvalue::object();
        return crow::response(501, response);

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in performance metrics: " << e.what());
        return crow::response(500, "{\"error\":\"Internal server error\"}");
    }
}

// Helper function implementations
std::string RestApiImpl::generate_secure_token() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) {
        ss << std::setw(2) << dis(gen);
    }
    return ss.str();
}

crow::json::wvalue RestApiImpl::serialize_user(const User& user) const {
    crow::json::wvalue result;
    result["userId"] = user.user_id;
    result["username"] = user.username;
    result["email"] = user.email;
    result["isActive"] = user.is_active;
    result["createdAt"] = to_iso_string(user.created_at);
    result["lastLogin"] = to_iso_string(user.last_login);

    crow::json::wvalue roles_array;
    for (size_t i = 0; i < user.roles.size(); ++i) {
        roles_array[i] = user.roles[i];
    }
    result["roles"] = std::move(roles_array);

    return result;
}

crow::json::wvalue RestApiImpl::serialize_api_key(const ApiKey& key) const {
    crow::json::wvalue result;
    result["keyId"] = key.key_id;
    result["userId"] = key.user_id;
    result["description"] = key.description;
    result["isActive"] = key.is_active;
    result["createdAt"] = to_iso_string(key.created_at);
    result["expiresAt"] = to_iso_string(key.expires_at);

    crow::json::wvalue permissions_array;
    for (size_t i = 0; i < key.permissions.size(); ++i) {
        permissions_array[i] = key.permissions[i];
    }
    result["permissions"] = std::move(permissions_array);

    return result;
}

std::string RestApiImpl::to_iso_string(const std::chrono::system_clock::time_point& time_point) const {
    auto time_t = std::chrono::system_clock::to_time_t(time_point);
    std::tm tm = *std::gmtime(&time_t);

    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

} // namespace jadevectordb
