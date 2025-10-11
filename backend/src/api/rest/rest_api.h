#ifndef JADEVECTORDB_REST_API_H
#define JADEVECTORDB_REST_API_H

#include <string>
#include <memory>
#include <thread>
#include <future>

#include "lib/logging.h"
#include "lib/config.h"

// Forward declarations for services
namespace jadevectordb {
    class DatabaseService;
    class VectorStorageService;
    class SimilaritySearchService;
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
    
public:
    explicit RestApiImpl();
    ~RestApiImpl() = default;
    
    // Initialize the web server framework
    bool initialize(int port);
    
    // Register all API routes
    void register_routes();
    
    // Individual route handlers
    void handle_health_check();           // GET /health
    void handle_system_status();          // GET /status
    void handle_database_status();        // GET /v1/databases/{databaseId}/status
    
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
    
    // Helper methods
    void setup_error_handling();
    void setup_authentication();
    void setup_request_validation();
    void setup_response_serialization();
    
    // Authentication helper
    Result<bool> authenticate_request(const std::string& api_key) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_REST_API_H