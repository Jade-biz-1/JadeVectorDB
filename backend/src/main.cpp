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
    std::unique_ptr<AuthManager> auth_mgr_;
    std::unique_ptr<MetricsRegistry> metrics_registry_;
    std::unique_ptr<DatabaseLayer> db_layer_;
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
        auth_mgr_ = std::make_unique<AuthManager>();
        
        // Initialize metrics registry
        metrics_registry_ = std::unique_ptr<MetricsRegistry>(MetricsManager::get_registry());
        
        // Initialize database layer
        db_layer_ = std::make_unique<DatabaseLayer>();
        auto db_result = db_layer_->initialize();
        if (!db_result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize database layer: " << 
                     ErrorHandler::format_error(db_result.error()));
            return db_result;
        }
        
        // Initialize REST API service
        rest_api_service_ = std::make_unique<RestApiService>(config.port);
        
        // Initialize gRPC service
        grpc_service_ = std::make_unique<VectorDatabaseService>(
            config.host + ":" + std::to_string(config.grpc_port));
        
        LOG_INFO(logger_, "All services initialized successfully");
        return std::expected<std::monostate, ErrorInfo>{};
    }

    Result<void> start() {
        LOG_INFO(logger_, "Starting JadeVectorDB application...");
        
        auto result = initialize();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize application: " << format_error(result.error()));
            return result;
        }
        
        running_ = true;
        
        // Start REST API service
        if (!rest_api_service_->start()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to start REST API service");
        }
        
        // Start gRPC service
        if (!grpc_service_->start()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to start gRPC service");
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
        
        return std::expected<std::monostate, ErrorInfo>{};
    }

    Result<void> shutdown() {
        LOG_INFO(logger_, "Shutting down JadeVectorDB application...");
        
        running_ = false;
        
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
        return std::expected<std::monostate, ErrorInfo>{};
    }
    
    bool is_running() const { return running_; }
};

} // namespace jadevectordb

int main(int argc, char* argv[]) {
    std::cout << "Starting JadeVectorDB..." << std::endl;
    
    jadevectordb::JadeVectorDBApp app;
    
    auto result = app.start();
    
    if (!result.has_value()) {
        std::cerr << "Application failed to start: " << 
                     jadevectordb::ErrorHandler::format_error(result.error()) << std::endl;
        return 1;
    }
    
    std::cout << "JadeVectorDB application completed successfully" << std::endl;
    return 0;
}