#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <ctime>
#include <cstring>   // for strlen
#ifdef _WIN32
#include <io.h>      // for _write on Windows
#include <windows.h>
#define STDOUT_FILENO _fileno(stdout)
#else
#include <unistd.h>  // for write, STDOUT_FILENO (async-signal-safe I/O)
#endif

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

// Global flag for signal handling
static std::atomic<bool> g_shutdown_requested(false);

// Forward declaration for JadeVectorDBApp
namespace jadevectordb {
class JadeVectorDBApp;
}
static jadevectordb::JadeVectorDBApp* g_app_instance = nullptr;

// Forward declare signal handler (defined after class)
void shutdown_signal_handler(int signal);

namespace jadevectordb {

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
    std::mutex shutdown_mutex_;
    std::condition_variable shutdown_cv_;
    
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
        
        // Initialize database layer with persistent storage
        // Use PersistentDatabasePersistence for Sprint 2.1 persistence features
        std::string vector_storage_path = "./data/jadevectordb/databases";
        auto persistent_storage = std::make_unique<PersistentDatabasePersistence>(
            vector_storage_path,
            nullptr,  // sharding_service - not used in standalone mode
            nullptr,  // query_router - not used in standalone mode
            nullptr   // replication_service - not used in standalone mode
        );
        
        db_layer_ = std::make_unique<DatabaseLayer>(
            std::move(persistent_storage),
            nullptr,  // sharding_service
            nullptr,  // query_router
            nullptr   // replication_service
        );
        
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
        
        // Load distributed configuration from app_config (JSON files + environment variables)
        // Can be overridden via:
        // - backend/config/production.json or development.json ("distributed" section)
        // - Environment variables: JADEVECTORDB_ENABLE_SHARDING, JADEVECTORDB_ENABLE_REPLICATION, etc.
        // See docs/distributed_services_api.md for details
        
        // Configure sharding from app_config
        dist_config.sharding_config.strategy = app_config_.distributed.sharding_strategy;
        dist_config.sharding_config.num_shards = app_config_.distributed.num_shards;
        dist_config.sharding_config.replication_factor = app_config_.distributed.replication_factor;
        
        // Configure replication from app_config
        dist_config.replication_config.default_replication_factor = app_config_.distributed.default_replication_factor;
        dist_config.replication_config.synchronous_replication = app_config_.distributed.synchronous_replication;
        dist_config.replication_config.replication_timeout_ms = app_config_.distributed.replication_timeout_ms;
        dist_config.replication_config.replication_strategy = app_config_.distributed.replication_strategy;
        
        // Configure routing from app_config
        dist_config.routing_config.strategy = app_config_.distributed.routing_strategy;
        dist_config.routing_config.max_route_cache_size = app_config_.distributed.max_route_cache_size;
        dist_config.routing_config.route_ttl_seconds = app_config_.distributed.route_ttl_seconds;
        dist_config.routing_config.enable_adaptive_routing = app_config_.distributed.enable_adaptive_routing;
        
        // Configure clustering from app_config
        dist_config.cluster_host = app_config_.distributed.cluster_host.empty() ? config.host : app_config_.distributed.cluster_host;
        dist_config.cluster_port = app_config_.distributed.cluster_port;
        dist_config.seed_nodes = app_config_.distributed.seed_nodes;
        
        // Enable/disable distributed features from app_config
        // NOTE: Distributed features fully implemented (12,259+ lines) but disabled by default for Phase 1
        // All distributed services (sharding, replication, clustering) are coded and unit-tested
        // Default: disabled for validated single-node deployment (Phase 1)
        // To enable: Set in config JSON or use environment variables
        // Phase 2 will production-test distributed features in multi-node environment
        // See docs/distributed_services_api.md for architecture details
        dist_config.enable_sharding = app_config_.distributed.enable_sharding;
        dist_config.enable_replication = app_config_.distributed.enable_replication;
        dist_config.enable_clustering = app_config_.distributed.enable_clustering;
        
        LOG_INFO(logger_, "Distributed config: sharding=" << dist_config.enable_sharding << 
                 ", replication=" << dist_config.enable_replication << 
                 ", clustering=" << dist_config.enable_clustering);
        
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
        // Pass the database layer to REST API so it uses persistent storage
        auto shared_db_layer = std::shared_ptr<DatabaseLayer>(db_layer_.get(), [](DatabaseLayer*){/* Don't delete, owned by unique_ptr */});
        rest_api_service_ = std::make_unique<RestApiService>(config.port, distributed_service_manager_, shared_db_layer);
        
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

        // Set shutdown callback for /admin/shutdown endpoint
        rest_api_service_->set_shutdown_callback([this]() {
            LOG_INFO(logger_, "Shutdown requested via REST API endpoint");
            request_shutdown();
        });

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
        while (running_ && !g_shutdown_requested) {
            // Check for shutdown every 100ms instead of using condition variable
            // This makes the loop more responsive to signals
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Explicitly shutdown before returning
        auto shutdown_result = shutdown();
        
        return Result<void>{};
    }

    // Public method to stop the application (called from signal handler)
    void stop() {
        running_ = false;
        shutdown_cv_.notify_all();
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
        std::unique_lock<std::mutex> lock(shutdown_mutex_);
        running_ = false;
        shutdown_cv_.notify_all();
    }
};

} // namespace jadevectordb

// Signal handler for graceful shutdown (defined after class to access methods)
// IMPORTANT: This must be async-signal-safe (no I/O, no heap allocation, etc.)
void shutdown_signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        // Use write() instead of cout (async-signal-safe)
        const char* msg = "\nReceived shutdown signal. Shutting down gracefully...\n";
        (void)write(STDOUT_FILENO, msg, strlen(msg));  // Cast to void to ignore return value

        // Set atomic flag
        g_shutdown_requested.store(true, std::memory_order_release);

        // Trigger application shutdown (this calls rest_api_service_->stop())
        if (g_app_instance) {
            g_app_instance->request_shutdown();
        }
    }
}

int main(int argc, char* argv[]) {
    (void)argc; // Unused - future: parse command line args
    (void)argv; // Unused - future: parse command line args
    std::cout << "Starting JadeVectorDB..." << std::endl;
    std::cout << "[DEBUG] Build timestamp: " << __DATE__ << " " << __TIME__ << std::endl;

    const char* env = std::getenv("JADEVECTORDB_ENV");
    std::cout << "[DEBUG] JADEVECTORDB_ENV at startup = " << (env ? env : "NULL (not set)") << std::endl;

    // Register signal handlers for graceful shutdown using sigaction (more reliable)
    struct sigaction sa;
    sa.sa_handler = shutdown_signal_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    
    if (sigaction(SIGINT, &sa, nullptr) == -1) {
        std::cerr << "Failed to install SIGINT handler" << std::endl;
    } else {
        std::cout << "[DEBUG] SIGINT handler installed successfully" << std::endl;
    }
    if (sigaction(SIGTERM, &sa, nullptr) == -1) {
        std::cerr << "Failed to install SIGTERM handler" << std::endl;
    } else {
        std::cout << "[DEBUG] SIGTERM handler installed successfully" << std::endl;
    }

    jadevectordb::JadeVectorDBApp app;
    g_app_instance = &app;

    auto result = app.start();

    if (!result) {
        std::cerr << "Application failed to start: " << 
                     jadevectordb::ErrorHandler::format_error(result.error()) << std::endl;
        g_app_instance = nullptr;
        return 1;
    }
    
    std::cout << "JadeVectorDB application completed successfully" << std::endl;
    g_app_instance = nullptr;
    return 0;
}