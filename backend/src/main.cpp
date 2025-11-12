#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "lib/config.h"
#include "lib/thread_pool.h"
#include "lib/auth.h"
#include "lib/metrics.h"
#include "models/vector.h"
#include "models/database.h"
#include "models/index.h"
#include "models/embedding_model.h"
#include "services/database_layer.h"
#include "services/sharding_service.h"
#include "services/replication_service.h"
#include "services/query_router.h"
#include "services/cluster_service.h"
#include "services/distributed_service_manager.h"
#include "api/rest/rest_api.h"
#include "api/grpc/grpc_service.h"

namespace jadevectordb {

// Forward declarations for services
class DatabaseService;
class VectorStorageService;
class SimilaritySearchService;
class IndexService;

// Main application class
class JadeVectorDBApp {
private:
    std::unique_ptr<logging::Logger> logger_;
    std::unique_ptr<ConfigManager> config_mgr_;
    std::unique_ptr<ThreadPool> thread_pool_;
    AuthManager* auth_mgr_;  // Singleton - not owned
    std::unique_ptr<MetricsRegistry> metrics_registry_;
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DistributedServiceManager> distributed_service_manager_;
    std::unique_ptr<RestApiService> rest_api_service_;
    std::unique_ptr<VectorDatabaseService> grpc_service_;
    
    bool running_;

public:
    JadeVectorDBApp() : running_(false) {
        // Initialize logging
        logging::LoggerManager::initialize(logging::LogLevel::INFO);
        logger_ = std::make_unique<logging::Logger>("JadeVectorDBApp");
        
        LOG_INFO(logger_, "JadeVectorDB Application initializing...");
    }

    ~JadeVectorDBApp() {
        if (running_) {
            shutdown();
        }
        LOG_INFO(logger_, "JadeVectorDB Application terminated");
    }

    Result<void> initialize() {
        LOG_INFO(logger_, "Initializing JadeVectorDB services...");
        
        // Get configuration
        config_mgr_ = std::unique_ptr<ConfigManager>(ConfigManager::get_instance());
        
        // Load configuration from environment or default
        config_mgr_->load_from_env();
        
        // Validate configuration
        if (!config_mgr_->validate_config()) {
            RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Invalid configuration");
        }
        
        auto config = config_mgr_->get_config();
        
        // Initialize thread pool
        thread_pool_ = std::make_unique<ThreadPool>(config.thread_pool_size);
        
        // Initialize auth manager
        auth_mgr_ = AuthManager::get_instance();

        // --- Default User Creation for Local/Dev/Test Environments ---
        // Detect environment (simple heuristic: ENV=local|dev|test, or host=localhost, or port=8080)
        std::string env = std::getenv("JADEVECTORDB_ENV") ? std::getenv("JADEVECTORDB_ENV") : "";
        bool is_local_env = (env == "local" || env == "dev" || env == "test" || config.host == "localhost" || config.host == "127.0.0.1" || config.port == 8080);

        if (is_local_env) {
            LOG_INFO(logger_, "Creating default users for local/dev/test environment...");
            // Create admin user
            auto admin_result = auth_mgr_->create_user("admin", "admin@jadevectordb.local", {"role_admin"});
            if (!admin_result) {
                LOG_WARN(logger_, "Default admin user creation: " << ErrorHandler::format_error(admin_result.error()));
            }
            // Create dev user
            auto dev_result = auth_mgr_->create_user("dev", "dev@jadevectordb.local", {"role_user"});
            if (!dev_result) {
                LOG_WARN(logger_, "Default dev user creation: " << ErrorHandler::format_error(dev_result.error()));
            }
            // Create test user
            auto test_result = auth_mgr_->create_user("test", "test@jadevectordb.local", {"role_reader"});
            if (!test_result) {
                LOG_WARN(logger_, "Default test user creation: " << ErrorHandler::format_error(test_result.error()));
            }
            // Ensure all are active
            if (admin_result) auth_mgr_->activate_user(admin_result.value());
            if (dev_result) auth_mgr_->activate_user(dev_result.value());
            if (test_result) auth_mgr_->activate_user(test_result.value());
        } else {
            LOG_INFO(logger_, "Production environment detected: default users will NOT be created or enabled.");
            // If any default users exist, deactivate them
            for (const std::string& uname : {"admin", "dev", "test"}) {
                // Try to find user by username
                auto users_result = auth_mgr_->list_users();
                if (users_result) {
                    for (const auto& user : users_result.value()) {
                        if (user.username == uname) {
                            auth_mgr_->deactivate_user(user.user_id);
                            LOG_INFO(logger_, "Deactivated default user: " << uname);
                        }
                    }
                }
            }
        }
        
        // Initialize metrics registry
        metrics_registry_ = std::unique_ptr<MetricsRegistry>(MetricsManager::get_registry());
        
        // Initialize database layer
        db_layer_ = std::make_unique<DatabaseLayer>();
        auto db_result = db_layer_->initialize();
        if (!db_result) {
            LOG_ERROR(logger_, "Failed to initialize database layer: " << 
                     ErrorHandler::format_error(db_result.error()));
            return db_result;
        }

            // --- Default Database Creation for Local/Dev/Test Environments ---
            if (is_local_env) {
                LOG_INFO(logger_, "Creating default database for local/dev/test environment...");
                Database default_db;
                default_db.name = "defaultdb";
                default_db.description = "Default database for local development and testing.";
                default_db.owner = "admin";
                default_db.vectorDimension = 128;
                default_db.indexType = "HNSW";
                default_db.indexParameters = { {"M", "16"}, {"efConstruction", "200"}, {"efSearch", "64"} };
                default_db.sharding = {"hash", 1};
                default_db.replication = {1, true};
                default_db.created_at = "2025-11-12T00:00:00Z";
                default_db.updated_at = "2025-11-12T00:00:00Z";
                default_db.accessControl = { {"admin", "user", "reader"}, {"read", "search"} };
                // Add a basic embedding model
                Database::EmbeddingModel emb_model;
                emb_model.name = "BERT";
                emb_model.version = "base-uncased";
                emb_model.provider = "huggingface";
                emb_model.inputType = "text";
                emb_model.outputDimension = 128;
                emb_model.parameters = { {"maxTokens", "512"}, {"normalize", "true"} };
                emb_model.status = "active";
                default_db.embeddingModels.push_back(emb_model);
                // Metadata schema
                default_db.metadataSchema = { {"owner", "string"}, {"tags", "array<string>"}, {"category", "string"}, {"score", "float"}, {"status", "enum:active|archived|deleted"} };
                // Retention policy
                default_db.retentionPolicy = std::make_unique<Database::RetentionPolicy>(Database::RetentionPolicy{365, true});

                auto create_db_result = db_layer_->create_database(default_db);
                if (!create_db_result) {
                    LOG_WARN(logger_, "Default database creation: " << ErrorHandler::format_error(create_db_result.error()));
                } else {
                    LOG_INFO(logger_, "Default database 'defaultdb' created successfully.");
                }
            }

            // --- Basic Verification Steps After Deployment ---
            LOG_INFO(logger_, "Running basic deployment verification checks...");
            // Check default users
            bool admin_ok = false, dev_ok = false, test_ok = false;
            auto users_result = auth_mgr_->list_users();
            if (users_result) {
                for (const auto& user : users_result.value()) {
                    if (user.username == "admin" && user.is_active) admin_ok = true;
                    if (user.username == "dev" && user.is_active) dev_ok = true;
                    if (user.username == "test" && user.is_active) test_ok = true;
                }
            }
            if (admin_ok && dev_ok && test_ok) {
                LOG_INFO(logger_, "Default users (admin, dev, test) are active.");
            } else {
                LOG_WARN(logger_, "One or more default users are missing or inactive.");
            }
            // Check default database
            auto db_list_result = db_layer_->list_databases();
            bool defaultdb_ok = false;
            if (db_list_result) {
                for (const auto& db : db_list_result.value()) {
                    if (db.name == "defaultdb") defaultdb_ok = true;
                }
            }
            if (defaultdb_ok) {
                LOG_INFO(logger_, "Default database 'defaultdb' is available.");
            } else {
                LOG_WARN(logger_, "Default database 'defaultdb' is missing.");
            }
            // Final deployment status
            if (admin_ok && dev_ok && test_ok && defaultdb_ok) {
                LOG_INFO(logger_, "JadeVectorDB deployment verification PASSED: All basic checks succeeded.");
            } else {
                LOG_WARN(logger_, "JadeVectorDB deployment verification FAILED: Please check logs for details.");
            }
        
        // Initialize distributed services
        distributed_service_manager_ = std::make_unique<DistributedServiceManager>();
        DistributedConfig dist_config;
        
        // Configure sharding
        dist_config.sharding_config.strategy = "hash";
        dist_config.sharding_config.num_shards = 4;
        dist_config.sharding_config.replication_factor = 3;
        
        // Configure replication
        dist_config.replication_config.default_replication_factor = 3;
        dist_config.replication_config.synchronous_replication = false;
        dist_config.replication_config.replication_timeout_ms = 5000;
        dist_config.replication_config.replication_strategy = "simple";
        
        // Configure routing
        dist_config.routing_config.strategy = "round_robin";
        dist_config.routing_config.max_route_cache_size = 1000;
        dist_config.routing_config.route_ttl_seconds = 300;
        dist_config.routing_config.enable_adaptive_routing = true;
        
        // Configure clustering
        dist_config.cluster_host = config.host;
        dist_config.cluster_port = config.port + 1000; // Use different port for cluster communication
        dist_config.seed_nodes = {}; // Empty for standalone mode
        
        // Enable all distributed features
        dist_config.enable_sharding = true;
        dist_config.enable_replication = true;
        dist_config.enable_clustering = true;
        
        auto dist_result = distributed_service_manager_->initialize(dist_config);
        if (!dist_result.has_value()) {
            LOG_WARN(logger_, "Failed to initialize distributed services: " << 
                     ErrorHandler::format_error(dist_result.error()));
            // Continue without distributed services in standalone mode
        } else {
            LOG_INFO(logger_, "Distributed services initialized successfully");
            
            // Start distributed services
            auto start_result = distributed_service_manager_->start();
            if (!start_result.has_value()) {
                LOG_WARN(logger_, "Failed to start distributed services: " << 
                         ErrorHandler::format_error(start_result.error()));
            } else {
                LOG_INFO(logger_, "Distributed services started successfully");
            }
        }
        
        // Initialize REST API service
        rest_api_service_ = std::make_unique<RestApiService>(config.port);
        
        // Initialize gRPC service
        grpc_service_ = std::make_unique<VectorDatabaseService>(
            config.host + ":" + std::to_string(config.grpc_port));
        
        LOG_INFO(logger_, "All services initialized successfully");
        return Result<void>{};
    }

    Result<void> start() {
        LOG_INFO(logger_, "Starting JadeVectorDB application...");
        
        auto result = initialize();
        if (!result) {
            LOG_ERROR(logger_, "Failed to initialize application: " << ErrorHandler::format_error(result.error()));
            return result;
        }
        
        running_ = true;
        
        // Start REST API service
        if (!rest_api_service_->start()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to start REST API service");
        }
        
        // Start gRPC service
        if (!grpc_service_->start()) {
            LOG_WARN(logger_, "Failed to start gRPC service - continuing without gRPC support");
            // Don't fail the application if gRPC is not available
        }
        
        LOG_INFO(logger_, "JadeVectorDB application started successfully");
        
        // Main application loop would be here in real implementation
        // For now, just run for a few seconds to demonstrate startup
        int count = 0;
        while (running_ && count < 10) { // Simple loop for demo purposes
            std::this_thread::sleep_for(std::chrono::seconds(1));
            LOG_DEBUG(logger_, "Application running...");
            count++;
        }
        
        return Result<void>{};
    }

    Result<void> shutdown() {
        LOG_INFO(logger_, "Shutting down JadeVectorDB application...");
        
        running_ = false;
        
        // Stop distributed services
        if (distributed_service_manager_) {
            auto stop_result = distributed_service_manager_->stop();
            if (!stop_result.has_value()) {
                LOG_WARN(logger_, "Failed to stop distributed services: " << 
                         ErrorHandler::format_error(stop_result.error()));
            }
        }
        
        // Stop services in reverse order
        if (grpc_service_) {
            grpc_service_->stop();
        }
        
        if (rest_api_service_) {
            rest_api_service_->stop();
        }
        
        // Wait for thread pool to finish
        thread_pool_.reset();
        
        logging::LoggerManager::shutdown();
        
        LOG_INFO(logger_, "JadeVectorDB application shutdown complete");
        return Result<void>{};
    }
    
    bool is_running() const { return running_; }
};

} // namespace jadevectordb

int main(int argc, char* argv[]) {
    std::cout << "Starting JadeVectorDB..." << std::endl;
    
    jadevectordb::JadeVectorDBApp app;
    
    auto result = app.start();
    
    if (!result) {
        std::cerr << "Application failed to start: " << 
                     jadevectordb::ErrorHandler::format_error(result.error()) << std::endl;
        return 1;
    }
    
    std::cout << "JadeVectorDB application completed successfully" << std::endl;
    return 0;
}