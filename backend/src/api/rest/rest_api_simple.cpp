/*
 * ===========================================================================
 * DEPRECATION NOTICE - DO NOT USE THIS FILE FOR NEW DEVELOPMENT
 * ===========================================================================
 *
 * This file (rest_api_simple.cpp) is DEPRECATED as of 2025-11-18.
 *
 * REASON: This simplified API implementation lacks critical production features:
 *   - No authentication system (JWT, login, register)
 *   - No user management (CRUD, roles, permissions)
 *   - No API key management
 *   - No security audit logging
 *   - No monitoring endpoints (alerts, cluster, performance)
 *
 * REPLACEMENT: Use rest_api.cpp instead
 *   rest_api.cpp has all endpoints from this file PLUS comprehensive
 *   authentication, security, and monitoring features.
 *
 * MIGRATION: See REST_API_SIMPLE_DEPRECATED.md for migration guide
 *
 * This file will remain for reference but should not be used in production.
 * ===========================================================================
 */

#include "rest_api.h"
#include "lib/logging.h"
#include "lib/config.h"
#include "lib/auth.h"
#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/metadata_filter.h"
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
    vector_service_ = std::make_unique<VectorStorageService>();
    search_service_ = std::make_unique<SimilaritySearchService>();
    metadata_filter_ = std::make_unique<MetadataFilter>();
    
    // Initialize the services
    db_service_->initialize();
    vector_service_->initialize();
    search_service_->initialize();
    
    // Create Crow app instance
    app_ = std::make_unique<crow::SimpleApp>();
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
    CROW_ROUTE((*app_), "/health").methods("GET"_method)
    ([](){
        json response;
        response["status"] = "healthy";
        response["timestamp"] = std::to_string(std::time(nullptr));
        return crow::response(200, response.dump());
    });
    
    CROW_ROUTE((*app_), "/status").methods("GET"_method)
    ([this](){
        json response;
        response["status"] = "running";
        response["version"] = "1.0.0";
        response["service"] = "JadeVectorDB";
        return crow::response(200, response.dump());
    });
    
    // Database management endpoints
    CROW_ROUTE((*app_), "/v1/databases").methods("POST"_method)
    ([this](const crow::request& req){
        return handle_create_database_request(req);
    });
    
    CROW_ROUTE((*app_), "/v1/databases").methods("GET"_method)
    ([this](const crow::request& req){
        return handle_list_databases_request(req);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>").methods("GET"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_get_database_request(req, database_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>").methods("PUT"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_update_database_request(req, database_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>").methods("DELETE"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_delete_database_request(req, database_id);
    });
    
    // Vector management endpoints
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors").methods("POST"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_store_vector_request(req, database_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/<string>").methods("GET"_method)
    ([this](const crow::request& req, std::string database_id, std::string vector_id){
        return handle_get_vector_request(req, database_id, vector_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/<string>").methods("PUT"_method)
    ([this](const crow::request& req, std::string database_id, std::string vector_id){
        return handle_update_vector_request(req, database_id, vector_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/<string>").methods("DELETE"_method)
    ([this](const crow::request& req, std::string database_id, std::string vector_id){
        return handle_delete_vector_request(req, database_id, vector_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors/batch").methods("POST"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_batch_store_vectors_request(req, database_id);
    });
    
    // Search endpoints
    CROW_ROUTE((*app_), "/v1/databases/<string>/search").methods("POST"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_similarity_search_request(req, database_id);
    });
    
    CROW_ROUTE((*app_), "/v1/databases/<string>/search/advanced").methods("POST"_method)
    ([this](const crow::request& req, std::string database_id){
        return handle_advanced_search_request(req, database_id);
    });
    
    LOG_INFO(logger_, "All REST API routes registered successfully");
}

// Request handling methods
crow::response RestApiImpl::handle_create_database_request(const crow::request& req) {
    try {
        // Parse database creation parameters from request body
        auto body_json = json::parse(req.body);
        
        Database db_config;
        db_config.name = body_json.value("name", "");
        db_config.description = body_json.value("description", "");
        db_config.vectorDimension = body_json.value("vectorDimension", 128);
        db_config.indexType = body_json.value("indexType", "HNSW");
        
        // Validate database creation parameters
        auto validation_result = db_service_->validate_creation_params(db_config);
        if (!validation_result.has_value()) {
            json response;
            response["error"] = "Invalid database creation parameters";
            return crow::response(400, response.dump());
        }
        
        // Create the database using the service
        auto result = db_service_->create_database(db_config);
        
        if (result.has_value()) {
            std::string database_id = result.value();
            json response;
            response["databaseId"] = database_id;
            response["status"] = "success";
            return crow::response(201, response.dump());
        } else {
            json response;
            response["error"] = "Failed to create database";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_list_databases_request(const crow::request& req) {
    try {
        // List databases using the service
        auto result = db_service_->list_databases();
        
        if (result.has_value()) {
            auto databases = result.value();
            
            json response = json::array();
            for (const auto& db : databases) {
                json db_obj;
                db_obj["databaseId"] = db.databaseId;
                db_obj["name"] = db.name;
                db_obj["description"] = db.description;
                db_obj["vectorDimension"] = db.vectorDimension;
                db_obj["indexType"] = db.indexType;
                db_obj["created_at"] = db.created_at;
                db_obj["updated_at"] = db.updated_at;
                response.push_back(db_obj);
            }
            
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Failed to list databases";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_get_database_request(const crow::request& req, const std::string& database_id) {
    try {
        // Get database using the service
        auto result = db_service_->get_database(database_id);
        
        if (result.has_value()) {
            auto database = result.value();
            
            json response;
            response["databaseId"] = database.databaseId;
            response["name"] = database.name;
            response["description"] = database.description;
            response["vectorDimension"] = database.vectorDimension;
            response["indexType"] = database.indexType;
            response["created_at"] = database.created_at;
            response["updated_at"] = database.updated_at;
            
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Database not found";
            return crow::response(404, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_update_database_request(const crow::request& req, const std::string& database_id) {
    try {
        // Parse database update parameters from request body
        auto body_json = json::parse(req.body);
        
        Database update_params;
        if (body_json.contains("name")) {
            update_params.name = body_json["name"];
        }
        if (body_json.contains("description")) {
            update_params.description = body_json["description"];
        }
        if (body_json.contains("vectorDimension")) {
            update_params.vectorDimension = body_json["vectorDimension"];
        }
        if (body_json.contains("indexType")) {
            update_params.indexType = body_json["indexType"];
        }
        
        // Validate database update parameters
        auto validation_result = db_service_->validate_update_params(update_params);
        if (!validation_result.has_value()) {
            json response;
            response["error"] = "Invalid database update parameters";
            return crow::response(400, response.dump());
        }
        
        // Update the database using the service
        auto result = db_service_->update_database(database_id, update_params);
        
        if (result.has_value()) {
            json response;
            response["status"] = "success";
            response["message"] = "Database updated successfully";
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Failed to update database";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_delete_database_request(const crow::request& req, const std::string& database_id) {
    try {
        // Delete the database using the service
        auto result = db_service_->delete_database(database_id);
        
        if (result.has_value()) {
            json response;
            response["status"] = "success";
            response["message"] = "Database deleted successfully";
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Failed to delete database";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_store_vector_request(const crow::request& req, const std::string& database_id) {
    try {
        // Parse vector from request body
        auto body_json = json::parse(req.body);
        
        Vector vector_data;
        vector_data.id = body_json.value("id", "");
        if (body_json.contains("values") && body_json["values"].is_array()) {
            for (const auto& val : body_json["values"]) {
                vector_data.values.push_back(val.get<float>());
            }
        }
        
        // Parse metadata if present
        if (body_json.contains("metadata")) {
            // For simplicity, we'll just store the metadata as a string
            vector_data.metadata = body_json["metadata"];
        }
        
        // Validate vector data
        auto validation_result = vector_service_->validate_vector(database_id, vector_data);
        if (!validation_result.has_value()) {
            json response;
            response["error"] = "Invalid vector data";
            return crow::response(400, response.dump());
        }
        
        // Store the vector using the service
        auto result = vector_service_->store_vector(database_id, vector_data);
        
        if (result.has_value()) {
            json response;
            response["status"] = "success";
            response["vectorId"] = vector_data.id;
            return crow::response(201, response.dump());
        } else {
            json response;
            response["error"] = "Failed to store vector";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_get_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id) {
    try {
        // Retrieve the vector using the service
        auto result = vector_service_->retrieve_vector(database_id, vector_id);
        
        if (result.has_value()) {
            auto vector = result.value();
            
            json response;
            response["id"] = vector.id;
            
            // Add values as an array
            json values_array = json::array();
            for (auto val : vector.values) {
                values_array.push_back(val);
            }
            response["values"] = values_array;
            
            // Add metadata if present
            if (!vector.metadata.empty()) {
                response["metadata"] = vector.metadata;
            }
            
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Vector not found";
            return crow::response(404, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_update_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id) {
    try {
        // Parse updated vector from request body
        auto body_json = json::parse(req.body);
        
        Vector vector_data;
        vector_data.id = vector_id;  // Ensure vector ID matches the path parameter
        if (body_json.contains("values") && body_json["values"].is_array()) {
            for (const auto& val : body_json["values"]) {
                vector_data.values.push_back(val.get<float>());
            }
        }
        
        // Parse metadata if present
        if (body_json.contains("metadata")) {
            vector_data.metadata = body_json["metadata"];
        }
        
        // Update the vector using the service
        auto result = vector_service_->update_vector(database_id, vector_data);
        
        if (result.has_value()) {
            json response;
            response["status"] = "success";
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Failed to update vector";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_delete_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id) {
    try {
        // Delete the vector using the service
        auto result = vector_service_->delete_vector(database_id, vector_id);
        
        if (result.has_value()) {
            json response;
            response["status"] = "success";
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Failed to delete vector";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_batch_store_vectors_request(const crow::request& req, const std::string& database_id) {
    try {
        // Parse vector list from request body
        auto body_json = json::parse(req.body);
        
        std::vector<Vector> vectors;
        if (body_json.contains("vectors") && body_json["vectors"].is_array()) {
            for (const auto& vec_json : body_json["vectors"]) {
                Vector vector_data;
                vector_data.id = vec_json.value("id", "");
                
                if (vec_json.contains("values") && vec_json["values"].is_array()) {
                    for (const auto& val : vec_json["values"]) {
                        vector_data.values.push_back(val.get<float>());
                    }
                }
                
                // Parse metadata if present
                if (vec_json.contains("metadata")) {
                    vector_data.metadata = vec_json["metadata"];
                }
                
                vectors.push_back(vector_data);
            }
        }
        
        // Validate all vectors before storing
        for (const auto& vector : vectors) {
            auto validation_result = vector_service_->validate_vector(database_id, vector);
            if (!validation_result.has_value()) {
                json response;
                response["error"] = "Invalid vector data";
                return crow::response(400, response.dump());
            }
        }
        
        // Store the vectors using the service
        auto result = vector_service_->batch_store_vectors(database_id, vectors);
        
        if (result.has_value()) {
            json response;
            response["status"] = "success";
            response["count"] = static_cast<int>(vectors.size());
            return crow::response(201, response.dump());
        } else {
            json response;
            response["error"] = "Failed to batch store vectors";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_similarity_search_request(const crow::request& req, const std::string& database_id) {
    try {
        // Parse query vector and search parameters from request body
        auto body_json = json::parse(req.body);
        
        // Parse query vector
        Vector query_vector;
        if (body_json.contains("queryVector") && body_json["queryVector"].is_array()) {
            for (const auto& val : body_json["queryVector"]) {
                query_vector.values.push_back(val.get<float>());
            }
        }
        
        // Parse search parameters
        SearchParams search_params;
        if (body_json.contains("topK")) {
            search_params.top_k = body_json["topK"];
        }
        if (body_json.contains("threshold")) {
            search_params.threshold = body_json["threshold"];
        }
        if (body_json.contains("includeMetadata")) {
            search_params.include_metadata = body_json["includeMetadata"];
        }
        if (body_json.contains("includeVectorData")) {
            search_params.include_vector_data = body_json["includeVectorData"];
        }
        if (!search_params.include_vector_data && body_json.contains("includeValues")) {
            search_params.include_vector_data = body_json["includeValues"];
        }
        
        // Perform similarity search using the service
        auto result = search_service_->similarity_search(database_id, query_vector, search_params);
        
        if (result.has_value()) {
            auto search_results = result.value();

            json response;
            response["count"] = search_results.size();
            json results_array = json::array();
            for (const auto& search_result : search_results) {
                json result_obj;
                result_obj["vectorId"] = search_result.vector_id;
                result_obj["score"] = search_result.similarity_score;

                if (search_params.include_vector_data || search_params.include_metadata) {
                    json vector_obj;
                    vector_obj["id"] = search_result.vector_data.id;

                    if (search_params.include_vector_data) {
                        json values_array = json::array();
                        for (auto val : search_result.vector_data.values) {
                            values_array.push_back(val);
                        }
                        vector_obj["values"] = values_array;
                    }

                    if (search_params.include_metadata) {
                        json metadata_obj;
                        const auto& metadata = search_result.vector_data.metadata;
                        metadata_obj["source"] = metadata.source;
                        metadata_obj["owner"] = metadata.owner;
                        metadata_obj["category"] = metadata.category;
                        metadata_obj["status"] = metadata.status;
                        metadata_obj["createdAt"] = metadata.created_at;
                        metadata_obj["updatedAt"] = metadata.updated_at;
                        metadata_obj["score"] = metadata.score;

                        if (!metadata.tags.empty()) {
                            json tags_array = json::array();
                            for (const auto& tag : metadata.tags) {
                                tags_array.push_back(tag);
                            }
                            metadata_obj["tags"] = tags_array;
                        }

                        if (!metadata.permissions.empty()) {
                            json permissions_array = json::array();
                            for (const auto& permission : metadata.permissions) {
                                permissions_array.push_back(permission);
                            }
                            metadata_obj["permissions"] = permissions_array;
                        }

                        if (!metadata.custom.empty()) {
                            json custom_obj;
                            for (const auto& [key, value] : metadata.custom) {
                                custom_obj[key] = value;
                            }
                            metadata_obj["custom"] = custom_obj;
                        }

                        vector_obj["metadata"] = metadata_obj;
                    }

                    result_obj["vector"] = vector_obj;
                }

                results_array.push_back(result_obj);
            }

            response["results"] = results_array;
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Search failed";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
    }
}

crow::response RestApiImpl::handle_advanced_search_request(const crow::request& req, const std::string& database_id) {
    try {
        // Parse query vector and advanced search parameters from request body
        auto body_json = json::parse(req.body);
        
        // Parse query vector
        Vector query_vector;
        if (body_json.contains("queryVector") && body_json["queryVector"].is_array()) {
            for (const auto& val : body_json["queryVector"]) {
                query_vector.values.push_back(val.get<float>());
            }
        }
        
        // Parse search parameters
        SearchParams search_params;
        if (body_json.contains("topK")) {
            search_params.top_k = body_json["topK"];
        }
        if (body_json.contains("threshold")) {
            search_params.threshold = body_json["threshold"];
        }
        if (body_json.contains("includeMetadata")) {
            search_params.include_metadata = body_json["includeMetadata"];
        }
        if (body_json.contains("includeVectorData")) {
            search_params.include_vector_data = body_json["includeVectorData"];
        }
        if (!search_params.include_vector_data && body_json.contains("includeValues")) {
            search_params.include_vector_data = body_json["includeValues"];
        }
        
        // Perform advanced similarity search using the service
        auto result = search_service_->similarity_search(database_id, query_vector, search_params);
        
        if (result.has_value()) {
            auto search_results = result.value();

            json response;
            response["count"] = search_results.size();
            json results_array = json::array();
            for (const auto& search_result : search_results) {
                json result_obj;
                result_obj["vectorId"] = search_result.vector_id;
                result_obj["score"] = search_result.similarity_score;

                if (search_params.include_vector_data || search_params.include_metadata) {
                    json vector_obj;
                    vector_obj["id"] = search_result.vector_data.id;

                    if (search_params.include_vector_data) {
                        json values_array = json::array();
                        for (auto val : search_result.vector_data.values) {
                            values_array.push_back(val);
                        }
                        vector_obj["values"] = values_array;
                    }

                    if (search_params.include_metadata) {
                        json metadata_obj;
                        const auto& metadata = search_result.vector_data.metadata;
                        metadata_obj["source"] = metadata.source;
                        metadata_obj["owner"] = metadata.owner;
                        metadata_obj["category"] = metadata.category;
                        metadata_obj["status"] = metadata.status;
                        metadata_obj["createdAt"] = metadata.created_at;
                        metadata_obj["updatedAt"] = metadata.updated_at;
                        metadata_obj["score"] = metadata.score;

                        if (!metadata.tags.empty()) {
                            json tags_array = json::array();
                            for (const auto& tag : metadata.tags) {
                                tags_array.push_back(tag);
                            }
                            metadata_obj["tags"] = tags_array;
                        }

                        if (!metadata.permissions.empty()) {
                            json permissions_array = json::array();
                            for (const auto& permission : metadata.permissions) {
                                permissions_array.push_back(permission);
                            }
                            metadata_obj["permissions"] = permissions_array;
                        }

                        if (!metadata.custom.empty()) {
                            json custom_obj;
                            for (const auto& [key, value] : metadata.custom) {
                                custom_obj[key] = value;
                            }
                            metadata_obj["custom"] = custom_obj;
                        }

                        vector_obj["metadata"] = metadata_obj;
                    }

                    result_obj["vector"] = vector_obj;
                }

                results_array.push_back(result_obj);
            }

            response["results"] = results_array;
            return crow::response(200, response.dump());
        } else {
            json response;
            response["error"] = "Advanced search failed";
            return crow::response(400, response.dump());
        }
    } catch (const std::exception& e) {
        json response;
        response["error"] = "Internal server error";
        return crow::response(500, response.dump());
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

} // namespace jadevectordb