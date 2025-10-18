#ifndef JADEVECTORDB_CONFIG_H
#define JADEVECTORDB_CONFIG_H

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace jadevectordb {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int grpc_port = 50051;
    std::string data_directory = "./data";
    size_t max_connections = 1000;
    size_t thread_pool_size = 16;
    size_t max_request_size = 10 * 1024 * 1024; // 10MB
    std::string log_level = "INFO";
    std::string log_file = "./logs/jadevectordb.log";
    size_t log_rotation_size = 10 * 1024 * 1024; // 10MB
    int log_backup_count = 5;
    
    // Vector database specific configurations
    size_t default_vector_dimension = 768;  // Default size for BERT-based embeddings
    std::string default_index_type = "HNSW";
    int default_shard_count = 1;
    int default_replication_factor = 1;
    bool enable_ssl = false;
    std::string ssl_cert_path = "";
    std::string ssl_key_path = "";
};

class ConfigManager {
private:
    ServerConfig config_;
    mutable std::mutex config_mutex_;
    static std::unique_ptr<ConfigManager> instance_;
    static std::once_flag once_flag_;

public:
    // Singleton pattern
    static ConfigManager* get_instance();
    
    // Load configuration from file
    bool load_from_file(const std::string& config_path);
    
    // Load configuration from environment variables
    void load_from_env();
    
    // Load configuration from command line arguments
    void load_from_args(int argc, char* argv[]);
    
    // Get the current configuration
    ServerConfig get_config() const;
    
    // Update configuration value
    template<typename T>
    void set_config_value(const std::string& key, const T& value);
    
    // Validate configuration
    bool validate_config() const;
    
private:
    ConfigManager() = default;
public:
    ~ConfigManager() = default;
private:
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_CONFIG_H