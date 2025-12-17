#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "lib/config.h"
#include "lib/thread_pool.h"
// REMOVED: #include "lib/auth.h" - migrated to AuthenticationService
#include "lib/metrics.h"
#include "config/config_loader.h"
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
    ConfigManager* config_mgr_;  // Raw pointer - singleton, not owned
    AppConfig app_config_;  // New: Application configuration from ConfigLoader
    std::unique_ptr<ThreadPool> thread_pool_;
    // REMOVED: AuthManager* auth_mgr_ - migrated to AuthenticationService
    MetricsRegistry* metrics_registry_;  // Raw pointer - singleton, not owned
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::shared_ptr<DistributedServiceManager> distributed_service_manager_;
    std::unique_ptr<RestApiService> rest_api_service_;
    std::unique_ptr<VectorDatabaseService> grpc_service_;
    
    bool running_;

public:
    JadeVectorDBApp() : config_mgr_(nullptr), metrics_registry_(nullptr), running_(false) {
        // Initialize logging
        logging::LoggerManager::initialize(logging::LogLevel::INFO);
        logger_ = std::make_unique<logging::Logger>("JadeVectorDBApp");
        
        LOG_INFO(logger_, "JadeVectorDB Application initializing...");
    }

    ~JadeVectorDBApp() {
        if (running_) {
            auto shutdown_result = shutdown();
            // Log but don't throw in destructor
            if (!shutdown_result) {
                // Error already logged by shutdown()
            }
        }
        // Don't log after shutdown - LoggerManager is already shutdown
        // LOG_INFO removed to prevent use-after-free
    }

    Result<void> initialize() {
        LOG_INFO(logger_, "Initializing JadeVectorDB services...");
        
        // Load application configuration from JSON files + environment variables
        ConfigLoader config_loader;
        auto config_result = config_loader.load_config("./config");
        if (!config_result.has_value()) {
            LOG_ERROR(logger_, "Failed to load configuration: " << 
                     ErrorHandler::format_error(config_result.error()));
            return tl::unexpected(config_result.error());
        }
        app_config_ = config_result.value();
        LOG_INFO(logger_, "Configuration loaded for environment: " << 
                ConfigLoader::environment_to_string(app_config_.environment));
        
        // Get configuration (singleton - not owned, don't delete)
        config_mgr_ = ConfigManager::get_instance();
        
        // Load configuration from environment or default
        config_mgr_->load_from_env();
        
        // Validate configuration
        if (!config_mgr_->validate_config()) {
            RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Invalid configuration");
        }
        
        auto config = config_mgr_->get_config();
        
        // Initialize thread pool
        thread_pool_ = std::make_unique<ThreadPool>(config.thread_pool_size);

        // REMOVED: AuthManager initialization and default user creation
        // Default users are now created by AuthenticationService.seed_default_users()
        // which is called during REST API initialization (rest_api.cpp line 132)

        // Initialize metrics registry (singleton - not owned, don't delete)
        metrics_registry_ = MetricsManager::get_registry();
        
        // Initialize database layer
        db_layer_ = std::make_unique<DatabaseLayer>();
        auto db_result = db_layer_->initialize();
        if (!db_result) {
            LOG_ERROR(logger_, "Failed to initialize database layer: " << 
                     ErrorHandler::format_error(db_result.error()));
            return db_result;
        }

        // REMOVED: Default database creation and verification code
        // This was using AuthManager which has been deprecated
        // Default database creation can be added back if needed using proper service layer

        // Initialize distributed services
        distributed_service_manager_ = std::make_shared<DistributedServiceManager>();
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
        // NOTE: Sharding is currently disabled to fix vector storage issues
        // When enabled, it requires creating shard databases which is not yet implemented
        dist_config.enable_sharding = false;
        dist_config.enable_replication = false;
        dist_config.enable_clustering = false;
        
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
        
        // Initialize REST API service with distributed service manager
        rest_api_service_ = std::make_unique<RestApiService>(config.port, distributed_service_manager_);
        
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

        // Main application loop - run until shutdown is requested
        // The server will run continuously until SIGINT (Ctrl+C) or SIGTERM is received
        LOG_INFO(logger_, "Server is running. Press Ctrl+C to stop.");
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            // Server continues running, processing requests via REST API and gRPC
        }

        // Explicitly shutdown before returning
        auto shutdown_result = shutdown();
        
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
            distributed_service_manager_.reset();  // Explicitly destroy
        }
        
        // Stop services in reverse order
        if (grpc_service_) {
            grpc_service_->stop();
            grpc_service_.reset();  // Explicitly destroy
        }
        
        if (rest_api_service_) {
            rest_api_service_->stop();
            rest_api_service_.reset();  // Explicitly destroy
        }
        
        // Clean up database layer
        db_layer_.reset();
        
        // Wait for thread pool to finish
        thread_pool_.reset();
        
        LOG_INFO(logger_, "JadeVectorDB application shutdown complete");
        
        // Shutdown logging LAST, after all log statements
        logging::LoggerManager::shutdown();
        
        return Result<void>{};
    }
    
    bool is_running() const { return running_; }

    void request_shutdown() {
        running_ = false;
    }
};

} // namespace jadevectordb

// Global flag for signal handling
static std::atomic<bool> g_shutdown_requested(false);
static jadevectordb::JadeVectorDBApp* g_app_instance = nullptr;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nReceived shutdown signal. Shutting down gracefully..." << std::endl;
        g_shutdown_requested = true;
        if (g_app_instance) {
            g_app_instance->request_shutdown();
        }
    }
}

int main(int argc, char* argv[]) {
    (void)argc; // Unused - future: parse command line args
    (void)argv; // Unused - future: parse command line args
    std::cout << "Starting JadeVectorDB..." << std::endl;

    // Register signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    jadevectordb::JadeVectorDBApp app;
    g_app_instance = &app;

    auto result = app.start();

    g_app_instance = nullptr;
    
    if (!result) {
        std::cerr << "Application failed to start: " << 
                     jadevectordb::ErrorHandler::format_error(result.error()) << std::endl;
        return 1;
    }
    
    std::cout << "JadeVectorDB application completed successfully" << std::endl;
    return 0;
}