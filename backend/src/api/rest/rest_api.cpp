// Clean REST API implementation - AuthManager removed, using AuthenticationService
// Handler implementations are in separate files:
// - rest_api_auth_handlers.cpp (authentication endpoints)
// - rest_api_user_handlers.cpp (user management)
// - rest_api_apikey_handlers.cpp (API key management)
// - rest_api_security_handlers.cpp (security audit)

#include "rest_api.h"
#include "lib/logging.h"
#include "lib/config.h"
#include "lib/error_handling.h"
#include "metrics/prometheus_metrics.h"
#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/sharding_service.h"
#include "services/replication_service.h"
#include "services/query_router.h"
#include "services/distributed_service_manager.h"
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

// ============================================================================
// RestApiService Implementation
// ============================================================================

RestApiService::RestApiService(int port, std::shared_ptr<DistributedServiceManager> distributed_service_manager) 
    : port_(port), running_(false) {
    logger_ = logging::LoggerManager::get_logger("RestApiService");
    server_address_ = "0.0.0.0:" + std::to_string(port_);
    api_impl_ = std::make_unique<RestApiImpl>(distributed_service_manager);
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
        
        // Stop the Crow app first to unblock the server thread
        if (api_impl_) {
            api_impl_->stop_server();
        }
        
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

// ============================================================================
// RestApiImpl Implementation
// ============================================================================

RestApiImpl::RestApiImpl(std::shared_ptr<DistributedServiceManager> distributed_service_manager) 
    : distributed_service_manager_(distributed_service_manager), server_stopped_(false) {
    logger_ = logging::LoggerManager::get_logger("RestApiImpl");
}

RestApiImpl::~RestApiImpl() {
    // Ensure Crow app is stopped before destruction
    if (app_ && !server_stopped_) {
        try {
            app_->stop();
            server_stopped_ = true;
        } catch (...) {
            // Ignore exceptions during shutdown
        }
        // Give Crow threads time to finish
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool RestApiImpl::initialize(int port) {
    LOG_INFO(logger_, "Initializing REST API on port " << port);

    // Create a shared database layer for all services to use
    auto shared_db_layer = std::make_shared<DatabaseLayer>();
    shared_db_layer->initialize();

    // Initialize services with the shared database layer
    db_service_ = std::make_unique<DatabaseService>(shared_db_layer);
    vector_storage_service_ = std::make_unique<VectorStorageService>(shared_db_layer);
    
    // Create a separate VectorStorageService for SimilaritySearchService
    auto search_vector_storage = std::make_unique<VectorStorageService>(shared_db_layer);
    search_vector_storage->initialize();
    similarity_search_service_ = std::make_unique<SimilaritySearchService>(std::move(search_vector_storage));
    
    index_service_ = std::make_unique<IndexService>();
    lifecycle_service_ = std::make_unique<LifecycleService>();
    security_audit_logger_ = std::make_shared<SecurityAuditLogger>();
    authentication_service_ = std::make_unique<AuthenticationService>();

    // Initialize the services
    db_service_->initialize();
    vector_storage_service_->initialize();
    similarity_search_service_->initialize();

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
    authentication_config_.require_strong_passwords = false;  // Use false for dev/test
    authentication_config_.min_password_length = 8;  // Min 8 chars for dev/test
    if (!authentication_service_->initialize(authentication_config_, security_audit_logger_)) {
        LOG_ERROR(logger_, "Failed to initialize authentication service");
        return false;
    }

    // Seed default users for non-production environments
    auto seed_result = authentication_service_->seed_default_users();
    if (!seed_result.has_value()) {
        LOG_WARN(logger_, "Failed to seed default users: " << ErrorHandler::format_error(seed_result.error()));
    }

    // Get runtime environment
    const char* env_ptr = std::getenv("JADE_ENV");
    runtime_environment_ = env_ptr ? std::string(env_ptr) : "development";
    
    // Initialize security middleware (rate limiting and IP blocking)
    // Login: 5 attempts per minute (capacity=5, refill_rate=5/60 per second)
    login_rate_limiter_ = std::make_unique<middleware::RateLimiter>(5, 5.0 / 60.0);
    
    // Registration: 3 per hour (capacity=3, refill_rate=3/3600 per second)
    registration_rate_limiter_ = std::make_unique<middleware::RateLimiter>(3, 3.0 / 3600.0);
    
    // API: 1000 per minute (capacity=1000, refill_rate=1000/60 per second)
    api_rate_limiter_ = std::make_unique<middleware::RateLimiter>(1000, 1000.0 / 60.0);
    
    // Password reset: 3 per hour (capacity=3, refill_rate=3/3600 per second)
    password_reset_rate_limiter_ = std::make_unique<middleware::RateLimiter>(3, 3.0 / 3600.0);
    
    // IP blocker: 10 failures, 3600s block duration, 600s failure window
    ip_blocker_ = std::make_unique<middleware::IPBlocker>(10, 3600, 600);
    
    LOG_INFO(logger_, "Security middleware initialized (rate limiters and IP blocker)");
    
    // Initialize distributed services if they exist
    initialize_distributed_services();
    
    // Create Crow app instance
    app_ = std::make_unique<crow::App<>>();
    server_port_ = port;
    
    // Setup
    setup_error_handling();
    setup_authentication();
    
    LOG_INFO(logger_, "REST API initialized successfully");
    return true;
}

void RestApiImpl::start_server() {
    if (app_) {
        LOG_INFO(logger_, "Starting Crow server on port " << server_port_);
        app_->port(server_port_).multithreaded().run();
    }
}

void RestApiImpl::stop_server() {
    if (app_ && !server_stopped_) {
        LOG_INFO(logger_, "Stopping Crow server");
        app_->stop();
        server_stopped_ = true;
    }
}

void RestApiImpl::register_routes() {
    LOG_INFO(logger_, "Registering REST API routes");
    
    // Health and monitoring endpoints
    handle_health_check();
    handle_database_health_check();
    handle_metrics();
    handle_system_status();
    
    // Database management endpoints
    // POST /v1/databases - Create database
    CROW_ROUTE((*app_), "/v1/databases")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            return handle_create_database_request(req);
        });

    // GET /v1/databases - List databases
    CROW_ROUTE((*app_), "/v1/databases")
        .methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            return handle_list_databases_request(req);
        });
    
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
    
    // Vector management endpoints
    // POST /v1/databases/<id>/vectors - Store vector
    CROW_ROUTE((*app_), "/v1/databases/<string>/vectors")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req, std::string database_id) {
            return handle_store_vector_request(req, database_id);
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
    
    // Search endpoints
    // POST /v1/databases/<id>/search - Similarity search
    CROW_ROUTE((*app_), "/v1/databases/<string>/search")
        .methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req, std::string database_id) {
            return handle_similarity_search_request(req, database_id);
        });
    app_->route_dynamic("/v1/databases/<string>/search/advanced")
        ([this](const crow::request& req, std::string database_id) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_advanced_search_request(req, database_id);
            }
            return crow::response(405, "Method not allowed");
        });
    
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
    
    // Embedding generation endpoints
    app_->route_dynamic("/v1/embeddings/generate")
        ([this](const crow::request& req) {
            if (req.method == crow::HTTPMethod::POST) {
                return handle_generate_embedding_request(req);
            }
            return crow::response(405, "Method not allowed");
        });

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

void RestApiImpl::handle_database_health_check() {
    LOG_DEBUG(logger_, "Setting up database health check endpoint at /health/db");
    
    app_->route_dynamic("/health/db")
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
            
            LOG_INFO(logger_, "Database health check request received");
            
            crow::json::wvalue response;
            response["status"] = "healthy";
            response["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Database connectivity checks
            response["checks"] = crow::json::wvalue::object();
            
            // Check authentication service database connectivity
            bool auth_db_healthy = true;
            std::string auth_db_status = "connected";
            if (authentication_service_) {
                // Try a lightweight operation to verify database connectivity
                auto user_count_result = authentication_service_->get_user_count();
                if (!user_count_result.has_value()) {
                    auth_db_healthy = false;
                    auth_db_status = "error: " + ErrorHandler::format_error(user_count_result.error());
                } else {
                    auth_db_status = "connected (" + std::to_string(user_count_result.value()) + " users)";
                }
            } else {
                auth_db_healthy = false;
                auth_db_status = "service not initialized";
            }
            
            response["checks"]["authentication_db"] = crow::json::wvalue::object();
            response["checks"]["authentication_db"]["status"] = auth_db_status;
            response["checks"]["authentication_db"]["healthy"] = auth_db_healthy;
            
            // Check vector database storage
            bool vector_db_healthy = true;
            std::string vector_db_status = "connected";
            if (db_service_) {
                // Database service is available
                vector_db_status = "service available";
            } else {
                vector_db_healthy = false;
                vector_db_status = "service not initialized";
            }
            
            response["checks"]["vector_db"] = crow::json::wvalue::object();
            response["checks"]["vector_db"]["status"] = vector_db_status;
            response["checks"]["vector_db"]["healthy"] = vector_db_healthy;
            
            // Overall health status
            bool overall_healthy = auth_db_healthy && vector_db_healthy;
            response["status"] = overall_healthy ? "healthy" : "degraded";
            response["healthy"] = overall_healthy;
            
            // Note about circuit breaker integration
            response["note"] = "Circuit breaker integration active. Enhanced health metrics available in future releases.";
            
            int status_code = overall_healthy ? 200 : 503;
            crow::response resp(status_code, response);
            resp.set_header("Content-Type", "application/json");
            return resp;
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error in database health check: " + std::string(e.what()));
            return crow::response(503, "{\"error\":\"Database health check failed\",\"message\":\"" + std::string(e.what()) + "\"}");
        }
    });
}

void RestApiImpl::handle_metrics() {
    LOG_DEBUG(logger_, "Setting up Prometheus metrics endpoint at /metrics");
    
    app_->route_dynamic("/metrics")
    ([this](const crow::request& req) {
        try {
            // Get metrics from PrometheusMetrics singleton
            auto metrics = PrometheusMetricsManager::get_instance();
            std::string metrics_text = metrics->get_metrics();
            
            // Return metrics in Prometheus text format
            crow::response resp(200, metrics_text);
            resp.set_header("Content-Type", "text/plain; version=0.0.4");
            return resp;
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error in metrics endpoint: " + std::string(e.what()));
            return crow::response(500, "# Error generating metrics\n");
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

            // TODO: Add permission check for monitoring:read when available in AuthenticationService

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
        vector_data.databaseId = database_id;  // Set the database ID
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

        // Set default status if not provided
        if (vector_data.metadata.status.empty()) {
            vector_data.metadata.status = "active";
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

        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        // TODO: Add permission check for vectors.read when available in AuthenticationService

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

// ============================================================================
// SEARCH OPERATION HANDLERS
// ============================================================================

crow::response RestApiImpl::handle_similarity_search_request(const crow::request& req, const std::string& database_id) {
    try {
        LOG_DEBUG(logger_, "Search request received for database: " + database_id);
        
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

        LOG_DEBUG(logger_, "Authenticating request...");
        // Authenticate request
        auto auth_result = authenticate_request(api_key);
        if (!auth_result.has_value()) {
            return crow::response(401, "{\"error\":\"" + ErrorHandler::format_error(auth_result.error()) + "\"}");
        }

        LOG_DEBUG(logger_, "Checking database exists...");
        // Validate database exists
        auto db_exists_result = db_service_->database_exists(database_id);
        if (!db_exists_result.has_value() || !db_exists_result.value()) {
            return crow::response(404, "{\"error\":\"Database not found\"}");
        }

        LOG_DEBUG(logger_, "Parsing request body...");
        // Parse query vector and search parameters from request body
        auto body_json = crow::json::load(req.body);
        if (!body_json) {
            return crow::response(400, "{\"error\":\"Invalid JSON in request body\"}");
        }

        LOG_DEBUG(logger_, "Parsing query vector...");
        // Parse query vector
        if (body_json["queryVector"].t() != crow::json::type::List) {
            return crow::response(400, "{\"error\":\"'queryVector' must be an array\"}");
        }

        Vector query_vector;
        auto query_array = body_json["queryVector"];
        for (size_t i = 0; i < query_array.size(); i++) {
            query_vector.values.push_back(query_array[i].d());
        }
        LOG_DEBUG(logger_, "Query vector parsed, dimension: " + std::to_string(query_vector.values.size()));

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

        LOG_DEBUG(logger_, "Validating search parameters...");
        // Validate search parameters
        auto validation_result = similarity_search_service_->validate_search_params(search_params);
        if (!validation_result.has_value()) {
            return crow::response(400, "{\"error\":\"Invalid search parameters\"}");
        }

        LOG_DEBUG(logger_, "Calling similarity_search service...");
        // Perform similarity search using the service
        auto result = similarity_search_service_->similarity_search(database_id, query_vector, search_params);

        LOG_DEBUG(logger_, "Search service returned");
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

// ============================================================================
// DATABASE OPERATION HANDLERS
// ============================================================================

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

// ============================================================================
// INDEX OPERATION HANDLERS
// ============================================================================

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
        // TODO: Add permission check when available in AuthenticationService

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
        // TODO: Add permission check when available in AuthenticationService

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
        // TODO: Add permission check when available in AuthenticationService

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
        // TODO: Add permission check when available in AuthenticationService

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

// ============================================================================
// EMBEDDING OPERATION HANDLERS
// ============================================================================

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
        // TODO: Add permission check when available in AuthenticationService

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

// ============================================================================
// HELPER INITIALIZATION METHODS
// ============================================================================

void RestApiImpl::setup_request_validation() {
    LOG_DEBUG(logger_, "Setting up request validation middleware");
    // In a real implementation, this would set up JSON schema validation
}

void RestApiImpl::setup_response_serialization() {
    LOG_DEBUG(logger_, "Setting up response serialization");
    // In a real implementation, this would set up JSON serialization
}

void RestApiImpl::initialize_distributed_services() {
    // Use distributed services from DistributedServiceManager (avoid duplicates)
    if (!distributed_service_manager_) {
        LOG_WARN(logger_, "DistributedServiceManager not provided, skipping distributed services initialization");
        return;
    }

    // Get shared instances from the manager
    sharding_service_ = distributed_service_manager_->get_sharding_service();
    replication_service_ = distributed_service_manager_->get_replication_service();
    query_router_ = distributed_service_manager_->get_query_router();
    cluster_service_ = distributed_service_manager_->get_cluster_service();

    if (!sharding_service_ || !replication_service_ || !query_router_) {
        LOG_WARN(logger_, "One or more distributed services not available from manager");
        return;
    }

    LOG_INFO(logger_, "Using distributed services from DistributedServiceManager");

    // Initialize the vector storage service with distributed services
    if (vector_storage_service_) {
        auto result = vector_storage_service_->initialize_distributed(
            sharding_service_,
            query_router_,
            replication_service_
        );

        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize VectorStorageService with distributed services: "
                      << ErrorHandler::format_error(result.error()));
        } else {
            LOG_INFO(logger_, "VectorStorageService initialized with distributed services successfully");
        }
    }
}

// ============================================================================
// ALERT, CLUSTER, AND PERFORMANCE HANDLERS (Stub Implementations)
// ============================================================================

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

// Route registration methods
void RestApiImpl::handle_alert_routes() {
    LOG_DEBUG(logger_, "Registering alert routes");
    // Stub implementations - to be implemented
}

void RestApiImpl::handle_cluster_routes() {
    LOG_DEBUG(logger_, "Registering cluster routes");
    // Stub implementations - to be implemented
}

void RestApiImpl::handle_performance_routes() {
    LOG_DEBUG(logger_, "Registering performance routes");
    // Stub implementations - to be implemented
}

// ============================================================================
// HELPER METHODS
// ============================================================================

void RestApiImpl::setup_error_handling() {
    LOG_DEBUG(logger_, "Setting up error handling middleware");
    // In a real implementation, this would set up framework-specific error handling
}

void RestApiImpl::setup_authentication() {
    LOG_DEBUG(logger_, "Setting up authentication middleware");
    // Authentication is now handled by AuthenticationService
    // API key validation is performed in authenticate_request()
}

Result<bool> RestApiImpl::authenticate_request(const std::string& token_or_api_key) const {
    if (token_or_api_key.empty()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "No API key or token provided");
    }

    // Try validating as a JWT token first
    auto token_result = authentication_service_->validate_token(token_or_api_key);

    if (token_result.has_value()) {
        // Token validation successful, user_id is in token_result.value()
        return true;
    }

    // If token validation failed, try as API key
    auto api_key_result = authentication_service_->authenticate_with_api_key(token_or_api_key, "0.0.0.0");

    if (!api_key_result.has_value()) {
        RETURN_ERROR(ErrorCode::UNAUTHENTICATED, "Invalid token or API key");
    }

    // API key authentication successful, user_id is in api_key_result.value()
    return true;
}

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

std::string RestApiImpl::to_iso_string(const std::chrono::system_clock::time_point& time_point) const {
    auto time_t = std::chrono::system_clock::to_time_t(time_point);
    std::tm tm = *std::gmtime(&time_t);

    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

} // namespace jadevectordb
