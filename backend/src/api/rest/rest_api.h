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

class RestApiService {
private:
    std::unique_ptr<RestApiImpl> api_impl_;
    std::unique_ptr<std::thread> server_thread_;
    std::string server_address_;
    int port_;
    bool running_;
    
    std::shared_ptr<logging::Logger> logger_;

public:
    explicit RestApiService(int port = 8080);
    ~RestApiService();
    
    // Start the REST API server
    bool start();
    
    // Stop the REST API server
    void stop();
    
    // Check if the server is running
    bool is_running() const { return running_; }
    
    // Get the server address
    std::string get_server_address() const { return server_address_; }
    
private:
    void run_server();
};

// The actual implementation class for REST API
class RestApiImpl {
private:
    std::shared_ptr<logging::Logger> logger_;
    
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
    
    // Crow app instance
    std::unique_ptr<crow::App<>> app_;
    int server_port_;
    
public:
    explicit RestApiImpl();
    ~RestApiImpl() = default;
    
    // Initialize the web server framework
    bool initialize(int port);
    
    // Register all API routes
    void register_routes();
    
    // Start the server
    void start_server();
    
    // Individual route handlers
    void handle_health_check();           // GET /health
    void handle_system_status();          // GET /status
    void handle_database_status();        // GET /v1/databases/{databaseId}/status
    
    // Request handling methods
    crow::response handle_create_database_request(const crow::request& req);
    crow::response handle_list_databases_request(const crow::request& req);
    crow::response handle_get_database_request(const crow::request& req, const std::string& database_id);
    crow::response handle_update_database_request(const crow::request& req, const std::string& database_id);
    crow::response handle_delete_database_request(const crow::request& req, const std::string& database_id);
    
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
    
    // Metrics and monitoring routes
    void handle_metrics();                // GET /metrics (Prometheus format)
    
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
    std::string extract_api_key(const crow::request& req) const;
    Result<std::string> authorize_api_key(const crow::request& req, const std::string& permission = "") const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_REST_API_H