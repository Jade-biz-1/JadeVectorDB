#ifndef JADEVECTORDB_REST_API_H
#define JADEVECTORDB_REST_API_H

#include <string>
#include <memory>
#include <thread>
#include <future>
#include <mutex>
#include <unordered_map>
#include <chrono>

#include "lib/logging.h"
#include "lib/config.h"

// Include Crow web framework
#include <crow.h>

#include "services/index_service.h"
#include "services/lifecycle_service.h"
#include "services/sharding_service.h"
#include "services/replication_service.h"
#include "services/query_router.h"
#include "services/authentication_service.h"
#include "services/security_audit_logger.h"
#include "middleware/rate_limiter.h"
#include "middleware/ip_blocker.h"

// Forward declarations for services
namespace jadevectordb {
    class DatabaseService;
    class VectorStorageService;
    class SimilaritySearchService;
    class IndexService;
    class LifecycleService;
    // class AuthManager;  // REMOVED: Migrated to AuthenticationService
    class ShardingService;
    class ReplicationService;
    class QueryRouter;
    // struct User;  // REMOVED: Part of AuthManager
    // struct ApiKey;  // REMOVED: Part of AuthManager
    struct SecurityEvent;
}

// For now, we'll define a basic interface structure
// The actual implementation would use a web framework like Crow, Pistache, or similar
namespace jadevectordb {

// Forward declarations
class RestApiImpl;
class DistributedServiceManager;

class RestApiService {
private:
    std::unique_ptr<RestApiImpl> api_impl_;
    std::unique_ptr<std::thread> server_thread_;
    std::string server_address_;
    int port_;
    bool running_;
    
    std::shared_ptr<logging::Logger> logger_;

public:
    explicit RestApiService(int port = 8080, 
                           std::shared_ptr<DistributedServiceManager> distributed_service_manager = nullptr,
                           std::shared_ptr<DatabaseLayer> db_layer = nullptr);
    ~RestApiService();
    
    // Start the REST API server
    bool start();
    
    // Stop the REST API server
    void stop();
    
    // Check if the server is running
    bool is_running() const { return running_; }
    
    // Get the server address
    std::string get_server_address() const { return server_address_; }

    /**
     * @brief Registers callback function for admin shutdown endpoint
     *
     * This callback is invoked when an authorized admin user calls the
     * POST /admin/shutdown endpoint. The callback should perform graceful
     * shutdown of the application (close connections, save state, exit).
     *
     * @param callback Function to execute when shutdown is requested
     *                 Typically the main application's request_shutdown()
     *
     * @note Callback is executed in a detached thread with 500ms delay
     *       to allow HTTP response to be sent before shutdown begins
     *
     * @example
     * rest_api->set_shutdown_callback([this]() {
     *     LOG_INFO("Shutdown requested via API");
     *     request_shutdown();
     * });
     *
     * @see handle_shutdown_request() in rest_api.cpp
     * @see docs/admin_endpoints.md for complete documentation
     */
    void set_shutdown_callback(std::function<void()> callback);

private:
    void run_server();
};

// Forward declarations
class DistributedServiceManager;
class ShardingService;
class ReplicationService;
class QueryRouter;
class ClusterService;

// The actual implementation class for REST API
class RestApiImpl {
private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Database layer (shared with main application)
    std::shared_ptr<DatabaseLayer> db_layer_;
    
    // Services that the API will use
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
    std::unique_ptr<IndexService> index_service_;
    std::unique_ptr<LifecycleService> lifecycle_service_;
    // AuthManager* auth_manager_;  // REMOVED: Migrated to AuthenticationService
    std::unique_ptr<AuthenticationService> authentication_service_;
    std::shared_ptr<SecurityAuditLogger> security_audit_logger_;
    AuthenticationConfig authentication_config_;
    
    // Distributed services (shared from DistributedServiceManager)
    std::shared_ptr<DistributedServiceManager> distributed_service_manager_;
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<ReplicationService> replication_service_;
    std::shared_ptr<QueryRouter> query_router_;
    ClusterService* cluster_service_;  // Raw pointer from manager

    struct PasswordResetToken {
        std::string token;
        std::string user_id;
        std::chrono::system_clock::time_point expires_at;
    };

    struct AlertRecord {
        std::string alert_id;
        std::string type;
        std::string severity;
        std::string message;
        bool acknowledged;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point acknowledged_at;
    };

    std::mutex password_reset_mutex_;
    std::unordered_map<std::string, PasswordResetToken> password_reset_tokens_;

    std::mutex alert_mutex_;
    std::unordered_map<std::string, AlertRecord> alert_store_;
    std::string runtime_environment_;
    
    // Security middleware
    std::unique_ptr<middleware::RateLimiter> login_rate_limiter_;
    std::unique_ptr<middleware::RateLimiter> registration_rate_limiter_;
    std::unique_ptr<middleware::RateLimiter> api_rate_limiter_;
    std::unique_ptr<middleware::RateLimiter> password_reset_rate_limiter_;
    std::unique_ptr<middleware::IPBlocker> ip_blocker_;
    
    // Crow app instance
    std::unique_ptr<crow::App<>> app_;
    int server_port_;
    bool server_stopped_;  // Track if stop() has been called

    // Shutdown callback for admin endpoint
    std::function<void()> shutdown_callback_;

public:
    explicit RestApiImpl(std::shared_ptr<DistributedServiceManager> distributed_service_manager = nullptr,
                        std::shared_ptr<DatabaseLayer> db_layer = nullptr);
    ~RestApiImpl();
    
    // Initialize the web server framework
    bool initialize(int port);
    
    // Register all API routes
    void register_routes();
    
    // Start the server
    void start_server();
    
    // Stop the server
    void stop_server();

    // Set shutdown callback (called by admin shutdown endpoint)
    void set_shutdown_callback(std::function<void()> callback);

    // Individual route handlers
    void handle_health_check();           // GET /health
    void handle_database_health_check();  // GET /health/db
    void handle_metrics();                // GET /metrics (Prometheus)
    void handle_system_status();          // GET /status
    void handle_shutdown();               // POST /shutdown
    void handle_database_status();        // GET /v1/databases/{databaseId}/status
    
    // Request handling methods
    crow::response handle_create_database_request(const crow::request& req);
    crow::response handle_list_databases_request(const crow::request& req);
    crow::response handle_get_database_request(const crow::request& req, const std::string& database_id);
    crow::response handle_update_database_request(const crow::request& req, const std::string& database_id);
    crow::response handle_delete_database_request(const crow::request& req, const std::string& database_id);
    
    crow::response handle_list_vectors_request(const crow::request& req, const std::string& database_id);
    crow::response handle_store_vector_request(const crow::request& req, const std::string& database_id);
    crow::response handle_get_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id);
    crow::response handle_update_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id);
    crow::response handle_delete_vector_request(const crow::request& req, const std::string& database_id, const std::string& vector_id);
    crow::response handle_batch_store_vectors_request(const crow::request& req, const std::string& database_id);
    crow::response handle_batch_get_vectors_request(const crow::request& req, const std::string& database_id);
    
    crow::response handle_similarity_search_request(const crow::request& req, const std::string& database_id);
    crow::response handle_advanced_search_request(const crow::request& req, const std::string& database_id);
    
    crow::response handle_create_index_request(const crow::request& req, const std::string& database_id);
    crow::response handle_list_indexes_request(const crow::request& req, const std::string& database_id);
    crow::response handle_update_index_request(const crow::request& req, const std::string& database_id, const std::string& index_id);
    crow::response handle_delete_index_request(const crow::request& req, const std::string& database_id, const std::string& index_id);
    
    crow::response handle_generate_embedding_request(const crow::request& req);
    
    // Lifecycle management request handlers
    crow::response handle_configure_retention_request(const crow::request& req, const std::string& database_id);
    crow::response handle_lifecycle_status_request(const crow::request& req, const std::string& database_id);
    
    // Database management routes
    void handle_create_database();        // POST /v1/databases
    void handle_list_databases();         // GET /v1/databases
    void handle_get_database();           // GET /v1/databases/{databaseId}
    void handle_update_database();        // PUT /v1/databases/{databaseId}
    void handle_delete_database();        // DELETE /v1/databases/{databaseId}
    
    // Vector management routes
    void handle_store_vector();           // POST /v1/databases/{databaseId}/vectors
    void handle_get_vector();             // GET /v1/databases/{databaseId}/vectors/{vectorId}
    void handle_update_vector();          // PUT /v1/databases/{databaseId}/vectors/{vectorId}
    void handle_delete_vector();          // DELETE /v1/databases/{databaseId}/vectors/{vectorId}
    void handle_batch_store_vectors();    // POST /v1/databases/{databaseId}/vectors/batch
    void handle_batch_get_vectors();       // POST /v1/databases/{databaseId}/vectors/batch-get
    
    // Search routes
    void handle_similarity_search();      // POST /v1/databases/{databaseId}/search
    void handle_advanced_search();        // POST /v1/databases/{databaseId}/search/advanced
    
    // Index management routes
    void handle_create_index();           // POST /v1/databases/{databaseId}/indexes
    void handle_list_indexes();           // GET /v1/databases/{databaseId}/indexes
    void handle_update_index();           // PUT /v1/databases/{databaseId}/indexes/{indexId}
    void handle_delete_index();           // DELETE /v1/databases/{databaseId}/indexes/{indexId}
    
    // Embedding generation routes
    void handle_generate_embedding();     // POST /v1/embeddings/generate
    
    // Lifecycle management routes
    void handle_configure_retention();    // PUT /v1/databases/{databaseId}/lifecycle
    void handle_lifecycle_status();       // GET /v1/databases/{databaseId}/lifecycle/status
    
    // Authentication and user management routes
    void handle_authentication_routes();
    void handle_user_management_routes();
    void handle_api_key_routes();
    void handle_security_routes();
    void handle_alert_routes();
    void handle_cluster_routes();
    void handle_performance_routes();

    crow::response handle_register_request(const crow::request& req);
    crow::response handle_login_request(const crow::request& req);
    crow::response handle_logout_request(const crow::request& req);
    crow::response handle_forgot_password_request(const crow::request& req);
    crow::response handle_reset_password_request(const crow::request& req);

    crow::response handle_create_user_request(const crow::request& req);
    crow::response handle_list_users_request(const crow::request& req);
    crow::response handle_get_user_request(const crow::request& req, const std::string& user_id);
    crow::response handle_update_user_request(const crow::request& req, const std::string& user_id);
    crow::response handle_delete_user_request(const crow::request& req, const std::string& user_id);
    crow::response handle_user_status_request(const crow::request& req, const std::string& user_id, bool activate);

    crow::response handle_list_api_keys_request(const crow::request& req);
    crow::response handle_create_api_key_request(const crow::request& req);
    crow::response handle_revoke_api_key_request(const crow::request& req, const std::string& key_id);

    crow::response handle_list_audit_logs_request(const crow::request& req);
    crow::response handle_get_audit_log_request(const crow::request& req);
    crow::response handle_get_sessions_request(const crow::request& req);
    crow::response handle_get_audit_stats_request(const crow::request& req);

    crow::response handle_list_alerts_request(const crow::request& req);
    crow::response handle_create_alert_request(const crow::request& req);
    crow::response handle_acknowledge_alert_request(const crow::request& req, const std::string& alert_id);

    crow::response handle_list_cluster_nodes_request(const crow::request& req);
    crow::response handle_cluster_node_status_request(const crow::request& req, const std::string& node_id);

    crow::response handle_performance_metrics_request(const crow::request& req);

    // ============================================================================
    // Admin Endpoints
    // ============================================================================
    // These methods handle administrative operations that require elevated
    // privileges (admin role). All methods enforce authentication and authorization.

    /**
     * @brief Handles graceful server shutdown (POST /admin/shutdown)
     *
     * Requires admin role. Initiates graceful server shutdown after sending
     * HTTP response. See docs/admin_endpoints.md for details.
     *
     * @param req HTTP request with Authorization header
     * @return HTTP 200 on success, 401 if unauthorized, 500 on error
     */
    crow::response handle_shutdown_request(const crow::request& req);

    // Helper methods
    void setup_error_handling();
    void setup_authentication();
    void setup_request_validation();
    void setup_response_serialization();
    
    // Authentication helper
    Result<bool> authenticate_request(const std::string& api_key) const;
    
    // Initialize distributed services
    void initialize_distributed_services();
    std::string generate_secure_token() const;
    // REMOVED: serialize_user and serialize_api_key (used AuthManager types)
    crow::json::wvalue serialize_alert(const AlertRecord& record) const;
    crow::json::wvalue serialize_audit_event(const SecurityEvent& event) const;
    std::string to_iso_string(const std::chrono::system_clock::time_point& time_point) const;

    /**
     * @brief Extracts JWT token or API key from Authorization header
     *
     * Supports "Bearer <token>" and "ApiKey <key>" formats.
     *
     * @param req HTTP request
     * @return Extracted token/key or empty string if not found
     */
    std::string extract_api_key(const crow::request& req) const;

    /**
     * @brief Authorizes request by validating token and checking role
     *
     * Validates JWT token, retrieves user from database, and verifies the
     * user has the required role/permission.
     *
     * @param req HTTP request with Authorization header
     * @param permission Required role (e.g., "admin", "developer", "user")
     *                   Empty string skips permission check
     * @return Result containing user_id on success, error on failure
     */
    Result<std::string> authorize_api_key(const crow::request& req, const std::string& permission = "") const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_REST_API_H