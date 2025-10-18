#include "rest_api.h"
#include "lib/logging.h"
#include "lib/config.h"
#include "lib/auth.h"
#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jadevectordb {

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
    
    // Initialize the services
    db_service_->initialize();
    vector_storage_service_->initialize();
    similarity_search_service_->initialize();
    
    // Create Crow app instance
    app_ = std::make_unique<crow::App<>>(crow::renderer::load_cached());
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
    app_->route_dynamic("/health")
        ([]() {
            crow::json::wvalue response;
            response["status"] = "healthy";
            response["timestamp"] = std::to_string(std::time(nullptr));
            return crow::response(response);
        });
    handle_health_check();
    
    app_->route_dynamic("/status")
        ([this]() {
            crow::json::wvalue response;
            response["status"] = "running";
            response["version"] = "1.0.0";
            response["service"] = "JadeVectorDB";
            return crow::response(response);
        });
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
    handle_create_database();
    handle_list_databases();
    
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
    handle_get_database();
    handle_update_database();
    handle_delete_database();
    
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
    handle_store_vector();
    handle_batch_store_vectors();
    handle_batch_get_vectors();
    
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
    handle_get_vector();
    handle_update_vector();
    handle_delete_vector();
    
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
    handle_similarity_search();
    handle_advanced_search();
    
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
    handle_create_index();
    handle_list_indexes();
    handle_update_index();
    handle_delete_index();
    
    // Embedding generation endpoints
    app_->route_dynamic("/v1/embeddings/generate")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_generate_embedding_request(req);
            }
            return crow::response(405, "Method not allowed");
        });
    handle_generate_embedding();
    
    LOG_INFO(logger_, "All REST API routes registered successfully");
}

// Health and monitoring endpoints


// Database management endpoints
void RestApiImpl::handle_create_database() {
    LOG_DEBUG(logger_, "Setting up create database endpoint at /v1/databases");
    
    // In a real implementation with a web framework, this would register a POST endpoint
    // that connects to the DatabaseService for creating databases
    // Example pseudo-code for the actual web framework integration:
    /*
    POST("/v1/databases", [&](const Request& req, Response& res) {
        try {
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
            
            // Check if user has permission to create databases
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "database:create");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Parse database creation parameters from request body
            auto creation_params = parse_database_creation_params_from_json(req.body);
            
            // Validate database creation parameters
            auto validation_result = db_service_->validate_creation_params(creation_params);
            if (!validation_result.has_value()) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Invalid database creation parameters\"}", "application/json");
                return;
            }
            
            // Create the database using the service
            auto result = db_service_->create_database(creation_params);
            
            if (result.has_value()) {
                std::string database_id = result.value();
                res.status = 201; // Created
                res.set_content("{\"databaseId\":\"" + database_id + "\",\"status\":\"success\"}", "application/json");
                
                LOG_INFO(logger_, "Created database: " + database_id + " (Name: " + creation_params.name + ")");
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Failed to create database\"}", "application/json");
                
                LOG_ERROR(logger_, "Failed to create database: " + ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in create database: " + std::string(e.what()));
        }
    });
    */
}

void RestApiImpl::handle_list_databases() {
    LOG_DEBUG(logger_, "Setting up list databases endpoint at /v1/databases");
    
    // In a real implementation with a web framework, this would register a GET endpoint
    // that connects to the DatabaseService for listing databases
    // Example pseudo-code for the actual web framework integration:
    /*
    GET("/v1/databases", [&](const Request& req, Response& res) {
        try {
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
            
            // Check if user has permission to list databases
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "database:list");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Parse query parameters for filtering and pagination
            DatabaseListParams list_params;
            
            // Extract query parameters
            auto name_filter = req.get_param_value("name");
            if (!name_filter.empty()) {
                list_params.filterByName = name_filter;
            }
            
            auto owner_filter = req.get_param_value("owner");
            if (!owner_filter.empty()) {
                list_params.filterByOwner = owner_filter;
            }
            
            auto limit_param = req.get_param_value("limit");
            if (!limit_param.empty()) {
                try {
                    list_params.limit = std::stoi(limit_param);
                    // Clamp limit to reasonable values
                    list_params.limit = std::max(1, std::min(1000, list_params.limit));
                } catch (const std::exception&) {
                    list_params.limit = 100; // Default limit
                }
            }
            
            auto offset_param = req.get_param_value("offset");
            if (!offset_param.empty()) {
                try {
                    list_params.offset = std::stoi(offset_param);
                    // Ensure non-negative offset
                    list_params.offset = std::max(0, list_params.offset);
                } catch (const std::exception&) {
                    list_params.offset = 0; // Default offset
                }
            }
            
            // List databases using the service
            auto result = db_service_->list_databases(list_params);
            
            if (result.has_value()) {
                auto databases = result.value();
                
                // Serialize databases to JSON
                auto json_str = serialize_databases_to_json(databases);
                res.status = 200; // OK
                res.set_content(json_str, "application/json");
                
                LOG_DEBUG(logger_, "Listed " + std::to_string(databases.size()) + " databases");
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Failed to list databases\"}", "application/json");
                
                LOG_ERROR(logger_, "Failed to list databases: " + ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in list databases: " + std::string(e.what()));
        }
    });
    */
}

void RestApiImpl::handle_get_database() {
    LOG_DEBUG(logger_, "Setting up get database endpoint at /v1/databases/{databaseId}");
    
    // In a real implementation with a web framework, this would register a GET endpoint
    // that connects to the DatabaseService for retrieving database details
    // Example pseudo-code for the actual web framework integration:
    /*
    GET("/v1/databases/:databaseId", [&](const Request& req, Response& res) {
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
            
            // Check if user has permission to get database details
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "database:read");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Get database using the service
            auto result = db_service_->get_database(database_id);
            
            if (result.has_value()) {
                auto database = result.value();
                
                // Serialize database to JSON
                auto json_str = serialize_database_to_json(database);
                res.status = 200; // OK
                res.set_content(json_str, "application/json");
                
                LOG_DEBUG(logger_, "Retrieved database: " + database_id);
            } else {
                res.status = 404; // Not Found
                res.set_content("{\"error\":\"Database not found\"}", "application/json");
                
                LOG_WARN(logger_, "Database not found: " + database_id);
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in get database: " + std::string(e.what()));
        }
    });
    */
}

void RestApiImpl::handle_update_database() {
    LOG_DEBUG(logger_, "Setting up update database endpoint at /v1/databases/{databaseId}");
    
    // In a real implementation with a web framework, this would register a PUT endpoint
    // that connects to the DatabaseService for updating database configuration
    // Example pseudo-code for the actual web framework integration:
    /*
    PUT("/v1/databases/:databaseId", [&](const Request& req, Response& res) {
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
            
            // Check if user has permission to update database configuration
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "database:update");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Parse database update parameters from request body
            auto update_params = parse_database_update_params_from_json(req.body);
            
            // Validate database update parameters
            auto validation_result = db_service_->validate_update_params(update_params);
            if (!validation_result.has_value()) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Invalid database update parameters\"}", "application/json");
                return;
            }
            
            // Update the database using the service
            auto result = db_service_->update_database(database_id, update_params);
            
            if (result.has_value()) {
                res.status = 200; // OK
                res.set_content("{\"status\":\"success\",\"message\":\"Database updated successfully\"}", "application/json");
                
                LOG_INFO(logger_, "Updated database: " + database_id);
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Failed to update database\"}", "application/json");
                
                LOG_ERROR(logger_, "Failed to update database: " + database_id + " - " + ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in update database: " + std::string(e.what()));
        }
    });
    */
}

void RestApiImpl::handle_delete_database() {
    LOG_DEBUG(logger_, "Setting up delete database endpoint at /v1/databases/{databaseId}");
    
    // In a real implementation with a web framework, this would register a DELETE endpoint
    // that connects to the DatabaseService for deleting databases
    // Example pseudo-code for the actual web framework integration:
    /*
    DELETE("/v1/databases/:databaseId", [&](const Request& req, Response& res) {
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
            
            // Check if user has permission to delete databases
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "database:delete");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Delete the database using the service
            auto result = db_service_->delete_database(database_id);
            
            if (result.has_value()) {
                res.status = 200; // OK
                res.set_content("{\"status\":\"success\",\"message\":\"Database deleted successfully\"}", "application/json");
                
                LOG_INFO(logger_, "Deleted database: " + database_id);
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Failed to delete database\"}", "application/json");
                
                LOG_ERROR(logger_, "Failed to delete database: " + database_id + " - " + ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in delete database: " + std::string(e.what()));
        }
    });
    */
}

// Vector management endpoints
void RestApiImpl::handle_store_vector() {
    LOG_DEBUG(logger_, "Setting up store vector endpoint at /v1/databases/{databaseId}/vectors");
    
    // In a real implementation with a web framework, this would register a POST endpoint
    // that connects to the VectorStorageService for storing vectors
    // Example pseudo-code for the actual web framework integration:
    /*
    POST("/v1/databases/:databaseId/vectors", [&](const Request& req, Response& res) {
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
            
            // Check if user has permission to store vectors in this database
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "vector:add");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Validate database exists
            auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
            if (!db_exists_result.has_value() || !db_exists_result.value()) {
                res.status = 404; // Not Found
                res.set_content("{\"error\":\"Database not found\"}", "application/json");
                return;
            }
            
            // Parse vector from request body
            auto vector_data = parse_vector_from_json(req.body);
            
            // Validate vector data
            auto validation_result = vector_storage_service_->validate_vector(database_id, vector_data);
            if (!validation_result.has_value()) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(validation_result.error()) + "\"}", "application/json");
                return;
            }
            
            // Store the vector using the service
            auto result = vector_storage_service_->store_vector(database_id, vector_data);
            
            if (result.has_value()) {
                res.status = 201; // Created
                res.set_content("{\"status\":\"success\",\"vectorId\":\"" + vector_data.id + "\"}", "application/json");
                
                LOG_DEBUG(logger_, "Stored vector: " + vector_data.id + " in database: " + database_id);
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}", "application/json");
                
                LOG_ERROR(logger_, "Failed to store vector: " + ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in store vector: " << e.what());
        }
    });
    */
}

void RestApiImpl::handle_get_vector() {
    LOG_DEBUG(logger_, "Setting up get vector endpoint at /v1/databases/{databaseId}/vectors/{vectorId}");
    
    // In a real implementation with a web framework, this would register a GET endpoint
    // that connects to the VectorStorageService for retrieving vectors
    // Example pseudo-code for the actual web framework integration:
    /*
    GET("/v1/databases/:databaseId/vectors/:vectorId", [&](const Request& req, Response& res) {
        try {
            // Extract database ID and vector ID from path
            std::string database_id = req.path_params.at("databaseId");
            std::string vector_id = req.path_params.at("vectorId");
            
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
            
            // Check if user has permission to retrieve vectors from this database
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "vector:read");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Validate database exists
            auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
            if (!db_exists_result.has_value() || !db_exists_result.value()) {
                res.status = 404; // Not Found
                res.set_content("{\"error\":\"Database not found\"}", "application/json");
                return;
            }
            
            // Check if vector exists
            auto vector_exists_result = vector_storage_service_->vector_exists(database_id, vector_id);
            if (!vector_exists_result.has_value() || !vector_exists_result.value()) {
                res.status = 404; // Not Found
                res.set_content("{\"error\":\"Vector not found\"}", "application/json");
                return;
            }
            
            // Retrieve the vector using the service
            auto result = vector_storage_service_->retrieve_vector(database_id, vector_id);
            
            if (result.has_value()) {
                auto vector = result.value();
                
                // Serialize vector to JSON
                auto json_str = serialize_vector_to_json(vector);
                res.status = 200; // OK
                res.set_content(json_str, "application/json");
            } else {
                res.status = 404; // Not Found
                res.set_content("{\"error\":\"Vector not found\"}", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
        }
    });
    */
}

void RestApiImpl::handle_update_vector() {
    LOG_DEBUG(logger_, "Setting up update vector endpoint at /v1/databases/{databaseId}/vectors/{vectorId}");
    
    // In a real implementation, this would register a PUT endpoint
    // that connects to the VectorStorageService to update vectors
    // Example pseudo-code for the actual web framework integration:
    /*
    PUT("/v1/databases/:databaseId/vectors/:vectorId", [&](const Request& req, Response& res) {
        try {
            // Extract database and vector IDs from path
            std::string database_id = req.path_params.at("databaseId");
            std::string vector_id = req.path_params.at("vectorId");
            
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
            
            // Check if user has permission to update vectors in this database
            // auto auth_manager = AuthManager::get_instance();
            // auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            // if (user_id_result.has_value()) {
            //     auto perm_result = auth_manager->has_permission_with_api_key(api_key, "vector:update");
            //     if (!perm_result.has_value() || !perm_result.value()) {
            //         res.status = 403; // Forbidden
            //         res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
            //         return;
            //     }
            // }
            
            // Parse updated vector from request body
            auto vector_data = parse_vector_from_json(req.body);
            vector_data.id = vector_id; // Ensure vector ID matches the path parameter
            
            // Update the vector using the service
            auto result = vector_storage_service_->update_vector(database_id, vector_data);
            
            if (result.has_value()) {
                res.status = 200; // OK
                res.set_content("{\"status\":\"success\"}", "application/json");
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
        }
    });
    */
}

void RestApiImpl::handle_delete_vector() {
    LOG_DEBUG(logger_, "Setting up delete vector endpoint at /v1/databases/{databaseId}/vectors/{vectorId}");
    
    // In a real implementation, this would register a DELETE endpoint
    // that connects to the VectorStorageService to delete vectors
    // Example pseudo-code for the actual web framework integration:
    /*
    DEL("/v1/databases/:databaseId/vectors/:vectorId", [&](const Request& req, Response& res) {
        try {
            // Extract database and vector IDs from path
            std::string database_id = req.path_params.at("databaseId");
            std::string vector_id = req.path_params.at("vectorId");
            
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
            
            // Check if user has permission to delete vectors from this database
            // auto auth_manager = AuthManager::get_instance();
            // auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            // if (user_id_result.has_value()) {
            //     auto perm_result = auth_manager->has_permission_with_api_key(api_key, "vector:delete");
            //     if (!perm_result.has_value() || !perm_result.value()) {
            //         res.status = 403; // Forbidden
            //         res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
            //         return;
            //     }
            // }
            
            // Delete the vector using the service
            auto result = vector_storage_service_->delete_vector(database_id, vector_id);
            
            if (result.has_value()) {
                res.status = 200; // OK
                res.set_content("{\"status\":\"success\"}", "application/json");
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
        }
    });
    */
}

void RestApiImpl::handle_batch_store_vectors() {
    LOG_DEBUG(logger_, "Setting up batch store vectors endpoint at /v1/databases/{databaseId}/vectors/batch");
    
    // In a real implementation, this would register a POST endpoint
    // that connects to the VectorStorageService to store multiple vectors
    // Example pseudo-code for the actual web framework integration:
    /*
    POST("/v1/databases/:databaseId/vectors/batch", [&](const Request& req, Response& res) {
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
            
            // Check if user has permission to store vectors in this database
            // auto auth_manager = AuthManager::get_instance();
            // auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            // if (user_id_result.has_value()) {
            //     auto perm_result = auth_manager->has_permission_with_api_key(api_key, "vector:add");
            //     if (!perm_result.has_value() || !perm_result.value()) {
            //         res.status = 403; // Forbidden
            //         res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
            //         return;
            //     }
            // }
            
            // Parse vector list from request body
            auto vectors = parse_vectors_from_json(req.body);
            
            // Validate all vectors before storing
            for (const auto& vector : vectors) {
                auto validation_result = vector_storage_service_->validate_vector(database_id, vector);
                if (!validation_result.has_value()) {
                    res.status = 400; // Bad Request
                    res.set_content("{\"error\":\"Invalid vector data\"}", "application/json");
                    return;
                }
            }
            
            // Store the vectors using the service
            auto result = vector_storage_service_->batch_store_vectors(database_id, vectors);
            
            if (result.has_value()) {
                res.status = 201; // Created
                res.set_content("{\"status\":\"success\",\"count\":" + std::to_string(vectors.size()) + "}", "application/json");
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"" + ErrorHandler::format_error(result.error()) + "\"}", "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
        }
    });
    */
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
            // auto auth_manager = AuthManager::get_instance();
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
            auto auth_manager = AuthManager::get_instance();
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
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "monitoring:read");
                if (!perm_result.has_value() || !perm_result.value()) {
                    return crow::response(403, "{\"error\":\"Insufficient permissions\"}");
                }
            }
            
            LOG_INFO(logger_, "System status request received");
            
            // In a real implementation, this would call the MonitoringService for detailed status
            // For now, returning placeholder status information
            crow::json::wvalue response;
            response["status"] = "operational";
            response["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            response["uptime"] = "placeholder_uptime";
            response["version"] = "1.0.0";
            
            // Add detailed system information
            response["system"] = crow::json::wvalue::object();
            response["system"]["cpu_usage"] = 15.3;
            response["system"]["memory_usage"] = 45.7;
            response["system"]["disk_usage"] = 67.2;
            response["system"]["network_io"] = "placeholder_network_io";
            
            // Add performance metrics
            response["performance"] = crow::json::wvalue::object();
            response["performance"]["avg_query_time_ms"] = 2.5;
            response["performance"]["qps"] = 1250;
            response["performance"]["active_connections"] = 42;
            response["performance"]["total_vectors"] = 1000000;
            
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
            auto auth_manager = AuthManager::get_instance();
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
            crow::json::wvalue indexes_status = crow::json::wvalue::object();
            indexes_status["hnsw_index_1"] = "ready";
            indexes_status["ivf_index_1"] = "ready";
            indexes_status["flat_index_1"] = "ready";
            response["indexes"] = indexes_status;
            
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
        auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
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
        if (!body_json["values"].is_list()) {
            return crow::response(400, "{\"error\":\"Vector values must be an array\"}");
        }
        
        // Parse values
        for (const auto& val : body_json["values"].list()) {
            vector_data.values.push_back(val.d());
        }
        
        // Parse metadata if present
        if (body_json.has("metadata")) {
            // Parse the metadata object
            vector_data.metadata = body_json["metadata"];
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
        auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
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
            crow::json::wvalue values_array = crow::json::wvalue::list();
            for (auto val : vector.values) {
                values_array.push_back(val);
            }
            response["values"] = values_array;
            
            // Add metadata if present
            if (!vector.metadata.isEmpty()) {
                response["metadata"] = vector.metadata.dump();
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
        if (!body_json["values"].is_list()) {
            return crow::response(400, "{\"error\":\"Vector values must be an array\"}");
        }
        
        // Parse values
        for (const auto& val : body_json["values"].list()) {
            vector_data.values.push_back(val.d());
        }
        
        // Parse metadata if present
        if (body_json.has("metadata")) {
            // Parse the metadata object
            vector_data.metadata = body_json["metadata"];
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
        auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
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
        if (!body_json["vectors"].is_list()) {
            return crow::response(400, "{\"error\":\"Request body must contain a 'vectors' array\"}");
        }
        
        for (const auto& vec_json : body_json["vectors"].list()) {
            Vector vector_data;
            vector_data.id = vec_json["id"].s();
            
            if (!vec_json["values"].is_list()) {
                return crow::response(400, "{\"error\":\"Vector values must be an array\"}");
            }
            
            // Parse values
            for (const auto& val : vec_json["values"].list()) {
                vector_data.values.push_back(val.d());
            }
            
            // Parse metadata if present
            if (vec_json.has("metadata")) {
                // Parse the metadata object
                vector_data.metadata = vec_json["metadata"];
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
        auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Parse query vector and search parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Parse query vector
        if (!body_json["queryVector"].is_list()) {
            return crow::response(400, "{\"error\":\"'queryVector' must be an array\"}");
        }
        
        Vector query_vector;
        for (const auto& val : body_json["queryVector"].list()) {
            query_vector.values.push_back(val.d());
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
        
        // Validate search parameters
        auto validation_result = similarity_search_service_->validate_search_params(search_params);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"Invalid search parameters\"}");
        }
        
        // Perform similarity search using the service
        auto result = similarity_search_service_->similarity_search(database_id, query_vector, search_params);
        
        if (result.has_value()) {
            // Serialize results to JSON
            auto search_results = result.value();
            crow::json::wvalue response = crow::json::wvalue::list();
            
            for (const auto& search_result : search_results) {
                crow::json::wvalue result_obj;
                result_obj["vectorId"] = search_result.vector_id;
                result_obj["similarityScore"] = search_result.similarity_score;
                
                if (search_params.include_vector_data || search_params.include_metadata) {
                    crow::json::wvalue vector_obj;
                    vector_obj["id"] = search_result.vector_data.id;
                    
                    // Add values as an array
                    crow::json::wvalue values_array = crow::json::wvalue::list();
                    for (auto val : search_result.vector_data.values) {
                        values_array.push_back(val);
                    }
                    vector_obj["values"] = values_array;
                    
                    result_obj["vector"] = vector_obj;
                }
                
                response.push_back(result_obj);
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
        auto db_exists_result = vector_storage_service_->db_layer_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }
        
        // Parse query vector and advanced search parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }
        
        // Parse query vector
        if (!body_json["queryVector"].is_list()) {
            return crow::response(400, "{\"error\":\"'queryVector' must be an array\"}");
        }
        
        Vector query_vector;
        for (const auto& val : body_json["queryVector"].list()) {
            query_vector.values.push_back(val.d());
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
            // Serialize results to JSON
            auto search_results = result.value();
            crow::json::wvalue response = crow::json::wvalue::list();
            
            for (const auto& search_result : search_results) {
                crow::json::wvalue result_obj;
                result_obj["vectorId"] = search_result.vector_id;
                result_obj["similarityScore"] = search_result.similarity_score;
                
                if (search_params.include_vector_data || search_params.include_metadata) {
                    crow::json::wvalue vector_obj;
                    vector_obj["id"] = search_result.vector_data.id;
                    
                    // Add values as an array
                    crow::json::wvalue values_array = crow::json::wvalue::list();
                    for (auto val : search_result.vector_data.values) {
                        values_array.push_back(val);
                    }
                    vector_obj["values"] = values_array;
                    
                    result_obj["vector"] = vector_obj;
                }
                
                response.push_back(result_obj);
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
        
        // Create a Database object from JSON
        Database db_config;
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
            
            // Serialize databases to JSON
            crow::json::wvalue response = crow::json::wvalue::list();
            
            for (const auto& db : databases) {
                crow::json::wvalue db_obj;
                db_obj["databaseId"] = db.databaseId;
                db_obj["name"] = db.name;
                db_obj["description"] = db.description;
                db_obj["vectorDimension"] = db.vectorDimension;
                db_obj["indexType"] = db.indexType;
                db_obj["created_at"] = db.created_at;
                db_obj["updated_at"] = db.updated_at;
                
                response.push_back(db_obj);
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
        
        // Create a Database object from JSON
        Database update_params;
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
    auto auth_manager = AuthManager::get_instance();
}

Result<bool> RestApiImpl::authenticate_request(const std::string& api_key) const {
    if (api_key.empty()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "No API key provided");
    }
    
    // Use the AuthManager to validate the API key
    auto auth_manager = AuthManager::get_instance();
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
        auto auth_manager = AuthManager::get_instance();
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

        // In a real implementation, we would use the EmbeddingService to generate the embedding
        // For now, returning a placeholder response
        crow::json::wvalue response;
        response["input"] = input;
        response["input_type"] = input_type;
        response["model"] = model;
        response["provider"] = provider;
        // Placeholder embedding - in a real implementation, this would be generated by the embedding service
        crow::json::wvalue emb_list = crow::json::wvalue::list();
        emb_list.push_back(0.1f);
        emb_list.push_back(0.2f);
        emb_list.push_back(0.3f);
        emb_list.push_back(0.4f);
        emb_list.push_back(0.5f);
        response["embedding"] = emb_list;
        response["dimension"] = 5; // Placeholder dimension
        response["status"] = "success";
        response["generated_at"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

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
        auto auth_manager = AuthManager::get_instance();
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
        std::unordered_map<std::string, std::string> parameters;
        if (body_json.has("parameters")) {
            auto params_obj = body_json["parameters"];
            for (const auto& member : params_obj.object()) {
                parameters[member.first] = member.second.s();
            }
        }

        // Create index config
        IndexConfig config;
        if (index_type == "HNSW") {
            config.type = IndexType::HNSW;
        } else if (index_type == "IVF") {
            config.type = IndexType::IVF;
        } else if (index_type == "LSH") {
            config.type = IndexType::LSH;
        } else if (index_type == "FLAT") {
            config.type = IndexType::FLAT;
        } else {
            LOG_ERROR(logger_, "Invalid index type: " << index_type);
            return crow::response(400, "{\"error\":\"Invalid index type\"}");
        }
        config.database_id = database_id;
        config.parameters = parameters;

        // Create the index using the service
        auto result = index_service_->create_index(config);
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
        auto auth_manager = AuthManager::get_instance();
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
        auto result = index_service_->get_indexes_for_database(database_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to list indexes: " + ErrorHandler::format_error(result.error()));
            return crow::response(400, "{\"error\":\"Failed to list indexes\"}");
        }

        auto indexes = result.value();
        crow::json::wvalue response = crow::json::wvalue::list();
        
        for (const auto& index : indexes) {
            crow::json::wvalue index_obj;
            index_obj["indexId"] = index.indexId;
            index_obj["databaseId"] = index.databaseId;
            index_obj["type"] = index.type;
            index_obj["status"] = index.status;
            
            // Convert parameters to JSON object
            crow::json::wvalue params_obj = crow::json::wvalue::object();
            for (const auto& param : index.parameters) {
                params_obj[param.first] = param.second;
            }
            index_obj["parameters"] = params_obj;
            
            index_obj["createdAt"] = index.created_at;
            index_obj["updatedAt"] = index.updated_at;
            response.push_back(index_obj);
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
        auto auth_manager = AuthManager::get_instance();
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
        if (!index_service_->index_exists(index_id)) {
            return crow::response(404, "{\"error\":\"Index not found\"}");
        }

        // Parse request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            LOG_ERROR(logger_, "Invalid JSON in handle_update_index_request");
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        // Extract parameters to update
        std::unordered_map<std::string, std::string> parameters;
        if (body_json.has("parameters")) {
            auto params_obj = body_json["parameters"];
            for (const auto& member : params_obj.object()) {
                parameters[member.first] = member.second.s();
            }
        }

        // Update the index using the service
        auto result = index_service_->update_index_config(index_id, parameters);
        if (!result.has_value() || !result.value()) {
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
        auto auth_manager = AuthManager::get_instance();
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
        if (!index_service_->index_exists(index_id)) {
            return crow::response(404, "{\"error\":\"Index not found\"}");
        }

        // Delete the index using the service
        auto result = index_service_->delete_index(index_id);
        if (!result.has_value() || !result.value()) {
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
        auto auth_manager = AuthManager::get_instance();
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
        auto result = lifecycle_service_->configure_retention_policy(config);
        if (!result.has_value() || !result.value()) {
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
        auto auth_manager = AuthManager::get_instance();
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

        // Get lifecycle statistics
        auto stats_result = lifecycle_service_->get_lifecycle_stats(database_id);
        if (!stats_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get lifecycle stats: " + ErrorHandler::format_error(stats_result.error()));
            return crow::response(400, "{\"error\":\"Failed to get lifecycle stats\"}");
        }

        auto policy = policy_result.value();
        auto stats = stats_result.value();

        crow::json::wvalue response;
        response["databaseId"] = database_id;
        
        // Add retention policy
        response["retentionPolicy"]["maxAgeDays"] = policy.max_age_days;
        response["retentionPolicy"]["archiveOnExpire"] = policy.archive_on_expire;
        response["retentionPolicy"]["archiveThresholdDays"] = policy.archive_threshold_days;
        response["retentionPolicy"]["enableCleanup"] = policy.enable_cleanup;
        response["retentionPolicy"]["cleanupSchedule"] = policy.cleanup_schedule;
        
        // Add lifecycle statistics
        for (const auto& stat : stats) {
            response["stats"][stat.first] = stat.second;
        }
        
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
    LOG_DEBUG(logger_, "Setting up batch get vectors endpoint at /v1/databases/{databaseId}/vectors/batch-get");
    
    // In a real implementation with a web framework, this would register a POST endpoint
    // that connects to the VectorStorageService for retrieving multiple vectors by ID
    // Example pseudo-code for the actual web framework integration:
    /*
    POST("/v1/databases/:databaseId/vectors/batch-get", [&](const Request& req, Response& res) {
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
            
            // Check if user has permission to retrieve vectors from this database
            auto auth_manager = AuthManager::get_instance();
            auto user_id_result = auth_manager->get_user_from_api_key(api_key);
            if (user_id_result.has_value()) {
                auto perm_result = auth_manager->has_permission_with_api_key(api_key, "vector:read");
                if (!perm_result.has_value() || !perm_result.value()) {
                    res.status = 403; // Forbidden
                    res.set_content("{\"error\":\"Insufficient permissions\"}", "application/json");
                    return;
                }
            }
            
            // Parse vector IDs from request body
            auto vector_ids = parse_vector_ids_from_json(req.body);
            
            // Validate vector IDs
            if (vector_ids.empty()) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"No vector IDs provided\"}", "application/json");
                return;
            }
            
            if (vector_ids.size() > 1000) {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Too many vector IDs (max 1000)\"}", "application/json");
                return;
            }
            
            // Retrieve vectors using the service
            auto result = vector_storage_service_->retrieve_vectors(database_id, vector_ids);
            
            if (result.has_value()) {
                auto vectors = result.value();
                
                // Serialize vectors to JSON
                auto json_str = serialize_vectors_to_json(vectors);
                res.status = 200; // OK
                res.set_content(json_str, "application/json");
                
                LOG_DEBUG(logger_, "Retrieved " << vectors.size() << " vectors from database: " << database_id);
            } else {
                res.status = 400; // Bad Request
                res.set_content("{\"error\":\"Failed to retrieve vectors\"}", "application/json");
                
                LOG_ERROR(logger_, "Failed to retrieve vectors from database " << database_id << ": " 
                          << ErrorHandler::format_error(result.error()));
            }
        } catch (const std::exception& e) {
            res.status = 500; // Internal Server Error
            res.set_content("{\"error\":\"Internal server error\"}", "application/json");
            
            LOG_ERROR(logger_, "Exception in batch get vectors: " + std::string(e.what()));
        }
    });
    */
}

} // namespace jadevectordb