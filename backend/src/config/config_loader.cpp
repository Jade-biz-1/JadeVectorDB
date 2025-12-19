#include "config/config_loader.h"
#include "lib/logging.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

namespace jadevectordb {

ConfigLoader::ConfigLoader() {
}

Result<AppConfig> ConfigLoader::load_config(const std::string& config_dir) {
    auto logger = logging::LoggerManager::get_logger("ConfigLoader");
    
    AppConfig config;
    config.environment = get_environment();
    
    LOG_INFO(logger, "Loading configuration for environment: " << environment_to_string(config.environment));
    
    // Determine base config file based on environment
    std::string base_config_file;
    if (config.environment == Environment::PRODUCTION) {
        base_config_file = config_dir + "/production.json";
    } else if (config.environment == Environment::TESTING) {
        base_config_file = config_dir + "/testing.json";
    } else {
        base_config_file = config_dir + "/development.json";
    }
    
    // Load base configuration
    if (std::filesystem::exists(base_config_file)) {
        LOG_INFO(logger, "Loading base config from: " << base_config_file);
        auto base_json_result = load_json_file(base_config_file);
        if (!base_json_result.has_value()) {
            LOG_WARN(logger, "Failed to load base config: " << ErrorHandler::format_error(base_json_result.error()));
            // Continue with defaults
        } else {
            merge_json_into_config(config, base_json_result.value(), "base");
        }
    } else {
        LOG_INFO(logger, "Base config file not found, using defaults: " << base_config_file);
    }
    
    // Load performance configuration
    std::string perf_config_file = config_dir + "/performance.json";
    if (std::filesystem::exists(perf_config_file)) {
        LOG_INFO(logger, "Loading performance config from: " << perf_config_file);
        auto perf_json_result = load_json_file(perf_config_file);
        if (perf_json_result.has_value()) {
            merge_json_into_config(config, perf_json_result.value(), "performance");
        }
    }
    
    // Load logging configuration
    std::string log_config_file = config_dir + "/logging.json";
    if (std::filesystem::exists(log_config_file)) {
        LOG_INFO(logger, "Loading logging config from: " << log_config_file);
        auto log_json_result = load_json_file(log_config_file);
        if (log_json_result.has_value()) {
            merge_json_into_config(config, log_json_result.value(), "logging");
        }
    }
    
    // Apply environment variable overrides
    apply_env_overrides(config);
    
    // Load secrets (environment variables and Docker secrets)
    auto secrets_result = load_secrets(config);
    if (!secrets_result.has_value()) {
        LOG_WARN(logger, "Failed to load some secrets: " << ErrorHandler::format_error(secrets_result.error()));
        // Continue - secrets may not be required in development
    }
    
    // Validate configuration
    auto validation_result = validate_config(config);
    if (!validation_result.has_value()) {
        LOG_ERROR(logger, "Configuration validation failed: " << ErrorHandler::format_error(validation_result.error()));
        return tl::unexpected(validation_result.error());
    }
    
    LOG_INFO(logger, "Configuration loaded successfully");
    return config;
}

Environment ConfigLoader::get_environment() {
    const char* env_str = std::getenv("JADEVECTORDB_ENV");
    if (!env_str) {
        return Environment::DEVELOPMENT;
    }
    
    std::string env(env_str);
    if (env == "production" || env == "PRODUCTION" || env == "prod") {
        return Environment::PRODUCTION;
    } else if (env == "testing" || env == "TESTING" || env == "test") {
        return Environment::TESTING;
    }
    
    return Environment::DEVELOPMENT;
}

std::string ConfigLoader::environment_to_string(Environment env) {
    switch (env) {
        case Environment::PRODUCTION:
            return "production";
        case Environment::TESTING:
            return "testing";
        case Environment::DEVELOPMENT:
        default:
            return "development";
    }
}

Result<nlohmann::json> ConfigLoader::load_json_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Failed to open config file: " + file_path);
    }
    
    try {
        nlohmann::json json;
        file >> json;
        return json;
    } catch (const nlohmann::json::exception& e) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                    "Failed to parse JSON from " + file_path + ": " + e.what());
    }
}

void ConfigLoader::apply_env_overrides(AppConfig& config) {
    // Distributed configuration overrides
    const char* enable_sharding = std::getenv("JADEVECTORDB_ENABLE_SHARDING");
    if (enable_sharding) {
        config.distributed.enable_sharding = (std::string(enable_sharding) == "true" || std::string(enable_sharding) == "1");
    }
    
    const char* enable_replication = std::getenv("JADEVECTORDB_ENABLE_REPLICATION");
    if (enable_replication) {
        config.distributed.enable_replication = (std::string(enable_replication) == "true" || std::string(enable_replication) == "1");
    }
    
    const char* enable_clustering = std::getenv("JADEVECTORDB_ENABLE_CLUSTERING");
    if (enable_clustering) {
        config.distributed.enable_clustering = (std::string(enable_clustering) == "true" || std::string(enable_clustering) == "1");
    }
    
    const char* num_shards = std::getenv("JADEVECTORDB_NUM_SHARDS");
    if (num_shards) {
        config.distributed.num_shards = std::stoi(num_shards);
    }
    
    const char* replication_factor = std::getenv("JADEVECTORDB_REPLICATION_FACTOR");
    if (replication_factor) {
        config.distributed.replication_factor = std::stoi(replication_factor);
    }
    
    const char* cluster_host = std::getenv("JADEVECTORDB_CLUSTER_HOST");
    if (cluster_host) {
        config.distributed.cluster_host = cluster_host;
    }
    
    const char* cluster_port = std::getenv("JADEVECTORDB_CLUSTER_PORT");
    if (cluster_port) {
        config.distributed.cluster_port = std::stoi(cluster_port);
    }
    
    const char* seed_nodes = std::getenv("JADEVECTORDB_SEED_NODES");
    if (seed_nodes) {
        // Parse comma-separated seed nodes
        std::string nodes_str(seed_nodes);
        config.distributed.seed_nodes.clear();
        size_t start = 0;
        size_t end = nodes_str.find(',');
        while (end != std::string::npos) {
            config.distributed.seed_nodes.push_back(nodes_str.substr(start, end - start));
            start = end + 1;
            end = nodes_str.find(',', start);
        }
        if (start < nodes_str.length()) {
            config.distributed.seed_nodes.push_back(nodes_str.substr(start));
        }
    }
    
    // Server configuration
    const char* port = std::getenv("JADEVECTORDB_PORT");
    if (port) {
        config.server_port = std::stoi(port);
    }
    
    const char* host = std::getenv("JADEVECTORDB_HOST");
    if (host) {
        config.server_host = host;
    }
    
    // Database paths
    const char* db_path = std::getenv("JADEVECTORDB_DB_PATH");
    if (db_path) {
        config.database.db_path = db_path;
    }
    
    const char* auth_db_path = std::getenv("JADEVECTORDB_AUTH_DB_PATH");
    if (auth_db_path) {
        config.database.auth_db_path = auth_db_path;
    }
    
    // Logging
    const char* log_level = std::getenv("JADEVECTORDB_LOG_LEVEL");
    if (log_level) {
        config.logging.level = log_level;
    }
    
    const char* log_file = std::getenv("JADEVECTORDB_LOG_FILE");
    if (log_file) {
        config.logging.file_path = log_file;
    }
    
    // Cache configuration
    const char* cache_size = std::getenv("JADEVECTORDB_CACHE_SIZE");
    if (cache_size) {
        config.cache.permission_cache_size = std::stoull(cache_size);
    }
    
    const char* cache_ttl = std::getenv("JADEVECTORDB_CACHE_TTL");
    if (cache_ttl) {
        config.cache.permission_cache_ttl_seconds = std::stoi(cache_ttl);
    }
}

Result<void> ConfigLoader::load_secrets(AppConfig& config) {
    auto logger = logging::LoggerManager::get_logger("ConfigLoader");
    
    // Load DB password
    const char* db_password = std::getenv("DB_PASSWORD");
    if (db_password) {
        config.db_password = db_password;
        LOG_INFO(logger, "Loaded DB_PASSWORD from environment variable");
    } else {
        std::string secret = read_docker_secret("db_password");
        if (!secret.empty()) {
            config.db_password = secret;
            LOG_INFO(logger, "Loaded db_password from Docker secret");
        }
    }
    
    // Load API secret key
    const char* api_secret = std::getenv("API_SECRET_KEY");
    if (api_secret) {
        config.api_secret_key = api_secret;
        LOG_INFO(logger, "Loaded API_SECRET_KEY from environment variable");
    } else {
        std::string secret = read_docker_secret("api_secret_key");
        if (!secret.empty()) {
            config.api_secret_key = secret;
            LOG_INFO(logger, "Loaded api_secret_key from Docker secret");
        }
    }
    
    // Load JWT secret
    const char* jwt_secret = std::getenv("JWT_SECRET");
    if (jwt_secret) {
        config.jwt_secret = jwt_secret;
        LOG_INFO(logger, "Loaded JWT_SECRET from environment variable");
    } else {
        std::string secret = read_docker_secret("jwt_secret");
        if (!secret.empty()) {
            config.jwt_secret = secret;
            LOG_INFO(logger, "Loaded jwt_secret from Docker secret");
        }
    }
    
    return {};
}

std::string ConfigLoader::read_docker_secret(const std::string& secret_name) {
    std::string secret_path = "/run/secrets/" + secret_name;
    
    std::ifstream file(secret_path);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string secret = buffer.str();
    
    // Trim trailing newline if present
    if (!secret.empty() && secret.back() == '\n') {
        secret.pop_back();
    }
    
    return secret;
}

Result<void> ConfigLoader::validate_config(const AppConfig& config) {
    // Validate server configuration
    if (config.server_port < 1 || config.server_port > 65535) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid server port: " + std::to_string(config.server_port));
    }
    
    // Validate authentication configuration
    if (config.auth.min_password_length < 8) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                    "Minimum password length must be at least 8, got: " + std::to_string(config.auth.min_password_length));
    }
    
    if (config.auth.session_timeout_seconds < 60) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                    "Session timeout must be at least 60 seconds, got: " + std::to_string(config.auth.session_timeout_seconds));
    }
    
    // Validate database configuration
    if (config.database.connection_pool_size < 1) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                    "Connection pool size must be at least 1, got: " + std::to_string(config.database.connection_pool_size));
    }
    
    if (config.database.max_retries < 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                    "Max retries cannot be negative, got: " + std::to_string(config.database.max_retries));
    }
    
    // Validate cache configuration
    if (config.cache.permission_cache_size == 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Permission cache size cannot be zero");
    }
    
    if (config.cache.permission_cache_ttl_seconds < 1) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                    "Cache TTL must be at least 1 second, got: " + std::to_string(config.cache.permission_cache_ttl_seconds));
    }
    
    // Validate logging configuration
    if (config.logging.level != "debug" && config.logging.level != "info" && 
        config.logging.level != "warn" && config.logging.level != "error") {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid log level: " + config.logging.level);
    }
    
    if (config.logging.format != "json" && config.logging.format != "text") {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid log format: " + config.logging.format);
    }
    
    // Warn if secrets are missing in production
    if (config.environment == Environment::PRODUCTION) {
        if (config.jwt_secret.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "JWT_SECRET is required in production environment");
        }
    }
    
    return {};
}

void ConfigLoader::merge_json_into_config(AppConfig& config, const nlohmann::json& json, const std::string& section) {
    try {
        // Authentication configuration
        if (json.contains("authentication")) {
            const auto& auth = json["authentication"];
            if (auth.contains("require_strong_passwords")) {
                config.auth.require_strong_passwords = auth["require_strong_passwords"].get<bool>();
            }
            if (auth.contains("min_password_length")) {
                config.auth.min_password_length = auth["min_password_length"].get<int>();
            }
            if (auth.contains("password_expiry_days")) {
                config.auth.password_expiry_days = auth["password_expiry_days"].get<int>();
            }
            if (auth.contains("enable_two_factor")) {
                config.auth.enable_two_factor = auth["enable_two_factor"].get<bool>();
            }
            if (auth.contains("session_timeout_seconds")) {
                config.auth.session_timeout_seconds = auth["session_timeout_seconds"].get<int>();
            }
        }
        
        // Security configuration
        if (json.contains("security")) {
            const auto& security = json["security"];
            if (security.contains("enable_rate_limiting")) {
                config.security.enable_rate_limiting = security["enable_rate_limiting"].get<bool>();
            }
            if (security.contains("enable_ip_blocking")) {
                config.security.enable_ip_blocking = security["enable_ip_blocking"].get<bool>();
            }
            if (security.contains("max_failed_logins")) {
                config.security.max_failed_logins = security["max_failed_logins"].get<int>();
            }
            if (security.contains("block_duration_seconds")) {
                config.security.block_duration_seconds = security["block_duration_seconds"].get<int>();
            }
        }
        
        // Database configuration
        if (json.contains("database")) {
            const auto& db = json["database"];
            if (db.contains("connection_pool_size")) {
                config.database.connection_pool_size = db["connection_pool_size"].get<int>();
            }
            if (db.contains("query_timeout_seconds")) {
                config.database.query_timeout_seconds = db["query_timeout_seconds"].get<int>();
            }
            if (db.contains("max_retries")) {
                config.database.max_retries = db["max_retries"].get<int>();
            }
            if (db.contains("db_path")) {
                config.database.db_path = db["db_path"].get<std::string>();
            }
            if (db.contains("auth_db_path")) {
                config.database.auth_db_path = db["auth_db_path"].get<std::string>();
            }
        }
        
        // Cache configuration
        if (json.contains("cache")) {
            const auto& cache = json["cache"];
            if (cache.contains("permission_cache_size")) {
                config.cache.permission_cache_size = cache["permission_cache_size"].get<size_t>();
            }
            if (cache.contains("permission_cache_ttl_seconds")) {
                config.cache.permission_cache_ttl_seconds = cache["permission_cache_ttl_seconds"].get<int>();
            }
            if (cache.contains("enable_query_cache")) {
                config.cache.enable_query_cache = cache["enable_query_cache"].get<bool>();
            }
        }
        
        // Logging configuration
        if (json.contains("logging")) {
            const auto& logging = json["logging"];
            if (logging.contains("level")) {
                config.logging.level = logging["level"].get<std::string>();
            }
            if (logging.contains("format")) {
                config.logging.format = logging["format"].get<std::string>();
            }
            if (logging.contains("output")) {
                config.logging.output = logging["output"].get<std::string>();
            }
            if (logging.contains("file_path")) {
                config.logging.file_path = logging["file_path"].get<std::string>();
            }
            if (logging.contains("max_file_size_mb")) {
                config.logging.max_file_size_mb = logging["max_file_size_mb"].get<int>();
            }
            if (logging.contains("max_files")) {
                config.logging.max_files = logging["max_files"].get<int>();
            }
            if (logging.contains("log_sql_queries")) {
                config.logging.log_sql_queries = logging["log_sql_queries"].get<bool>();
            }
        }
        
        // Server configuration
        if (json.contains("server")) {
            const auto& server = json["server"];
            if (server.contains("port")) {
                config.server_port = server["port"].get<int>();
            }
            if (server.contains("host")) {
                config.server_host = server["host"].get<std::string>();
            }
        }
        
        // Distributed configuration
        if (json.contains("distributed")) {
            const auto& dist = json["distributed"];
            if (dist.contains("enable_sharding")) {
                config.distributed.enable_sharding = dist["enable_sharding"].get<bool>();
            }
            if (dist.contains("enable_replication")) {
                config.distributed.enable_replication = dist["enable_replication"].get<bool>();
            }
            if (dist.contains("enable_clustering")) {
                config.distributed.enable_clustering = dist["enable_clustering"].get<bool>();
            }
            if (dist.contains("sharding_strategy")) {
                config.distributed.sharding_strategy = dist["sharding_strategy"].get<std::string>();
            }
            if (dist.contains("num_shards")) {
                config.distributed.num_shards = dist["num_shards"].get<int>();
            }
            if (dist.contains("replication_factor")) {
                config.distributed.replication_factor = dist["replication_factor"].get<int>();
            }
            if (dist.contains("default_replication_factor")) {
                config.distributed.default_replication_factor = dist["default_replication_factor"].get<int>();
            }
            if (dist.contains("synchronous_replication")) {
                config.distributed.synchronous_replication = dist["synchronous_replication"].get<bool>();
            }
            if (dist.contains("replication_timeout_ms")) {
                config.distributed.replication_timeout_ms = dist["replication_timeout_ms"].get<int>();
            }
            if (dist.contains("replication_strategy")) {
                config.distributed.replication_strategy = dist["replication_strategy"].get<std::string>();
            }
            if (dist.contains("routing_strategy")) {
                config.distributed.routing_strategy = dist["routing_strategy"].get<std::string>();
            }
            if (dist.contains("max_route_cache_size")) {
                config.distributed.max_route_cache_size = dist["max_route_cache_size"].get<int>();
            }
            if (dist.contains("route_ttl_seconds")) {
                config.distributed.route_ttl_seconds = dist["route_ttl_seconds"].get<int>();
            }
            if (dist.contains("enable_adaptive_routing")) {
                config.distributed.enable_adaptive_routing = dist["enable_adaptive_routing"].get<bool>();
            }
            if (dist.contains("cluster_host")) {
                config.distributed.cluster_host = dist["cluster_host"].get<std::string>();
            }
            if (dist.contains("cluster_port")) {
                config.distributed.cluster_port = dist["cluster_port"].get<int>();
            }
            if (dist.contains("seed_nodes")) {
                config.distributed.seed_nodes = dist["seed_nodes"].get<std::vector<std::string>>();
            }
        }
        
    } catch (const nlohmann::json::exception& e) {
        auto logger = logging::LoggerManager::get_logger("ConfigLoader");
        LOG_WARN(logger, "Error parsing " << section << " config: " << e.what());
    }
}

} // namespace jadevectordb
