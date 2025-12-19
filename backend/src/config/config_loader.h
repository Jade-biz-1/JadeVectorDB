#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "lib/error_handling.h"

namespace jadevectordb {

/**
 * @brief Environment type for configuration loading
 */
enum class Environment {
    DEVELOPMENT,
    PRODUCTION,
    TESTING
};

/**
 * @brief Authentication configuration
 */
struct AuthConfig {
    bool require_strong_passwords = true;
    int min_password_length = 12;
    int password_expiry_days = 90;
    bool enable_two_factor = false;
    int session_timeout_seconds = 3600;
};

/**
 * @brief Security configuration
 */
struct SecurityConfig {
    bool enable_rate_limiting = true;
    bool enable_ip_blocking = true;
    int max_failed_logins = 5;
    int block_duration_seconds = 3600;
};

/**
 * @brief Database configuration
 */
struct DatabaseConfig {
    int connection_pool_size = 20;
    int query_timeout_seconds = 30;
    int max_retries = 3;
    std::string db_path = "./data/jadevectordb.db";
    std::string auth_db_path = "./data/auth.db";
};

/**
 * @brief Cache configuration
 */
struct CacheConfig {
    size_t permission_cache_size = 100000;
    int permission_cache_ttl_seconds = 300;
    bool enable_query_cache = true;
};

/**
 * @brief Logging configuration
 */
struct LoggingConfig {
    std::string level = "info";           // debug, info, warn, error
    std::string format = "json";          // json, text
    std::string output = "file";          // file, stdout, both
    std::string file_path = "/var/log/jadevectordb/app.log";
    int max_file_size_mb = 100;
    int max_files = 10;
    bool log_sql_queries = false;
};

/**
 * @brief Distributed system configuration
 */
struct DistributedConfigSettings {
    bool enable_sharding = false;
    bool enable_replication = false;
    bool enable_clustering = false;
    
    // Sharding settings
    std::string sharding_strategy = "hash";  // hash, range, vector, auto
    int num_shards = 4;
    int replication_factor = 3;
    
    // Replication settings
    int default_replication_factor = 3;
    bool synchronous_replication = false;
    int replication_timeout_ms = 5000;
    std::string replication_strategy = "simple";
    
    // Routing settings
    std::string routing_strategy = "round_robin";
    int max_route_cache_size = 1000;
    int route_ttl_seconds = 300;
    bool enable_adaptive_routing = true;
    
    // Cluster settings
    std::string cluster_host = "0.0.0.0";
    int cluster_port = 9080;  // Default: server_port + 1000
    std::vector<std::string> seed_nodes;  // Empty for standalone
};

/**
 * @brief Complete application configuration
 */
struct AppConfig {
    Environment environment = Environment::DEVELOPMENT;
    AuthConfig auth;
    SecurityConfig security;
    DatabaseConfig database;
    CacheConfig cache;
    LoggingConfig logging;
    DistributedConfigSettings distributed;
    
    // Runtime settings
    int server_port = 8080;
    std::string server_host = "0.0.0.0";
    
    // Secrets (loaded from env vars or Docker secrets)
    std::string db_password;
    std::string api_secret_key;
    std::string jwt_secret;
};

/**
 * @brief Configuration loader and manager
 * 
 * Loads configuration from multiple sources with precedence:
 * 1. Environment variables (highest priority)
 * 2. Docker secrets
 * 3. JSON config files
 * 4. Default values (lowest priority)
 */
class ConfigLoader {
public:
    ConfigLoader();
    ~ConfigLoader() = default;
    
    /**
     * @brief Load configuration for the application
     * 
     * Loads configs in order:
     * 1. Base config (production.json or development.json)
     * 2. Performance config (performance.json)
     * 3. Logging config (logging.json)
     * 4. Environment variable overrides
     * 5. Docker secrets
     * 
     * @param config_dir Directory containing config files
     * @return Result<AppConfig> Loaded and validated configuration
     */
    Result<AppConfig> load_config(const std::string& config_dir = "./config");
    
    /**
     * @brief Get current environment from JADEVECTORDB_ENV
     * 
     * @return Environment Current environment (default: DEVELOPMENT)
     */
    static Environment get_environment();
    
    /**
     * @brief Get environment name as string
     * 
     * @param env Environment enum
     * @return std::string Environment name
     */
    static std::string environment_to_string(Environment env);
    
private:
    /**
     * @brief Load JSON file
     * 
     * @param file_path Path to JSON file
     * @return Result<nlohmann::json> Parsed JSON
     */
    Result<nlohmann::json> load_json_file(const std::string& file_path);
    
    /**
     * @brief Apply environment variable overrides
     * 
     * Checks for environment variables and overrides config values:
     * - JADEVECTORDB_PORT
     * - JADEVECTORDB_HOST
     * - JADEVECTORDB_DB_PATH
     * - JADEVECTORDB_AUTH_DB_PATH
     * - JADEVECTORDB_LOG_LEVEL
     * - JADEVECTORDB_LOG_FILE
     * 
     * @param config Configuration to update
     */
    void apply_env_overrides(AppConfig& config);
    
    /**
     * @brief Load secrets from environment variables or Docker secrets
     * 
     * Checks in order:
     * 1. Environment variables (DB_PASSWORD, API_SECRET_KEY, JWT_SECRET)
     * 2. Docker secrets (/run/secrets/db_password, etc.)
     * 
     * @param config Configuration to update
     * @return Result<void> Success or error
     */
    Result<void> load_secrets(AppConfig& config);
    
    /**
     * @brief Read Docker secret file
     * 
     * @param secret_name Name of secret (e.g., "db_password")
     * @return std::string Secret value (empty if not found)
     */
    std::string read_docker_secret(const std::string& secret_name);
    
    /**
     * @brief Validate configuration
     * 
     * Ensures all required fields are set and valid
     * 
     * @param config Configuration to validate
     * @return Result<void> Success or error with details
     */
    Result<void> validate_config(const AppConfig& config);
    
    /**
     * @brief Merge JSON into AppConfig
     * 
     * @param config Configuration to update
     * @param json JSON data to merge
     * @param section Section name (for error messages)
     */
    void merge_json_into_config(AppConfig& config, const nlohmann::json& json, const std::string& section);
};

} // namespace jadevectordb
