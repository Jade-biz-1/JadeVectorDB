#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "lib/config.h"
#include "lib/thread_pool.h"
// REMOVED: #include "lib/auth.h" - migrated to AuthenticationService
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
    ConfigManager* config_mgr_;  // Raw pointer - singleton, not owned
    std::unique_ptr<ThreadPool> thread_pool_;
    // REMOVED: AuthManager* auth_mgr_ - migrated to AuthenticationService
    MetricsRegistry* metrics_registry_;  // Raw pointer - singleton, not owned
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<DistributedServiceManager> distributed_service_manager_;
    std::unique_ptr<RestApiService> rest_api_service_;
    std::unique_ptr<VectorDatabaseService> grpc_service_;
    
    bool running_;

public:
    JadeVectorDBApp() : running_(false), config_mgr_(nullptr), metrics_registry_(nullptr) {
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