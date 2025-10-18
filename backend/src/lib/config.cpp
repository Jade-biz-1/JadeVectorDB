#include "config.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace jadevectordb {

std::unique_ptr<ConfigManager> ConfigManager::instance_ = nullptr;
std::once_flag ConfigManager::once_flag_;

ConfigManager* ConfigManager::get_instance() {
    std::call_once(once_flag_, []() {
        instance_ = std::unique_ptr<ConfigManager>(new ConfigManager());
    });
    return instance_.get();
}

bool ConfigManager::load_from_file(const std::string& config_path) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::ifstream file(config_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Find the '=' delimiter
        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) {
            continue;
        }
        
        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);
        
        // Convert key to lowercase for case-insensitive matching
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        
        // Set configuration values based on the key
        if (key == "host") {
            config_.host = value;
        } else if (key == "port") {
            try {
                config_.port = std::stoi(value);
            } catch (const std::exception&) {
                return false; // Invalid port value
            }
        } else if (key == "grpc_port") {
            try {
                config_.grpc_port = std::stoi(value);
            } catch (const std::exception&) {
                return false; // Invalid GRPC port value
            }
        } else if (key == "data_directory") {
            config_.data_directory = value;
        } else if (key == "max_connections") {
            try {
                config_.max_connections = std::stoull(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "thread_pool_size") {
            try {
                config_.thread_pool_size = std::stoull(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "max_request_size") {
            try {
                config_.max_request_size = std::stoull(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "log_level") {
            config_.log_level = value;
        } else if (key == "log_file") {
            config_.log_file = value;
        } else if (key == "log_rotation_size") {
            try {
                config_.log_rotation_size = std::stoull(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "log_backup_count") {
            try {
                config_.log_backup_count = std::stoi(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "default_vector_dimension") {
            try {
                config_.default_vector_dimension = std::stoull(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "default_index_type") {
            config_.default_index_type = value;
        } else if (key == "default_shard_count") {
            try {
                config_.default_shard_count = std::stoi(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "default_replication_factor") {
            try {
                config_.default_replication_factor = std::stoi(value);
            } catch (const std::exception&) {
                return false;
            }
        } else if (key == "enable_ssl") {
            config_.enable_ssl = (value == "true" || value == "1" || value == "yes");
        } else if (key == "ssl_cert_path") {
            config_.ssl_cert_path = value;
        } else if (key == "ssl_key_path") {
            config_.ssl_key_path = value;
        }
    }
    
    return validate_config();
}

void ConfigManager::load_from_env() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    const char* host = std::getenv("JDB_HOST");
    if (host) config_.host = std::string(host);
    
    const char* port = std::getenv("JDB_PORT");
    if (port) {
        try {
            config_.port = std::stoi(std::string(port));
        } catch (const std::exception&) {}
    }
    
    const char* grpc_port = std::getenv("JDB_GRPC_PORT");
    if (grpc_port) {
        try {
            config_.grpc_port = std::stoi(std::string(grpc_port));
        } catch (const std::exception&) {}
    }
    
    const char* data_dir = std::getenv("JDB_DATA_DIR");
    if (data_dir) config_.data_directory = std::string(data_dir);
    
    const char* max_conn = std::getenv("JDB_MAX_CONNECTIONS");
    if (max_conn) {
        try {
            config_.max_connections = std::stoull(std::string(max_conn));
        } catch (const std::exception&) {}
    }
    
    const char* thread_pool_size = std::getenv("JDB_THREAD_POOL_SIZE");
    if (thread_pool_size) {
        try {
            config_.thread_pool_size = std::stoull(std::string(thread_pool_size));
        } catch (const std::exception&) {}
    }
    
    const char* max_request_size = std::getenv("JDB_MAX_REQUEST_SIZE");
    if (max_request_size) {
        try {
            config_.max_request_size = std::stoull(std::string(max_request_size));
        } catch (const std::exception&) {}
    }
    
    const char* log_level = std::getenv("JDB_LOG_LEVEL");
    if (log_level) config_.log_level = std::string(log_level);
    
    const char* log_file = std::getenv("JDB_LOG_FILE");
    if (log_file) config_.log_file = std::string(log_file);
    
    const char* log_rotation_size = std::getenv("JDB_LOG_ROTATION_SIZE");
    if (log_rotation_size) {
        try {
            config_.log_rotation_size = std::stoull(std::string(log_rotation_size));
        } catch (const std::exception&) {}
    }
    
    const char* log_backup_count = std::getenv("JDB_LOG_BACKUP_COUNT");
    if (log_backup_count) {
        try {
            config_.log_backup_count = std::stoi(std::string(log_backup_count));
        } catch (const std::exception&) {}
    }
    
    const char* default_vector_dim = std::getenv("JDB_DEFAULT_VECTOR_DIM");
    if (default_vector_dim) {
        try {
            config_.default_vector_dimension = std::stoull(std::string(default_vector_dim));
        } catch (const std::exception&) {}
    }
    
    const char* default_index_type = std::getenv("JDB_DEFAULT_INDEX_TYPE");
    if (default_index_type) config_.default_index_type = std::string(default_index_type);
    
    const char* default_shard_count = std::getenv("JDB_DEFAULT_SHARD_COUNT");
    if (default_shard_count) {
        try {
            config_.default_shard_count = std::stoi(std::string(default_shard_count));
        } catch (const std::exception&) {}
    }
    
    const char* default_replication_factor = std::getenv("JDB_DEFAULT_REPLICATION_FACTOR");
    if (default_replication_factor) {
        try {
            config_.default_replication_factor = std::stoi(std::string(default_replication_factor));
        } catch (const std::exception&) {}
    }
    
    const char* enable_ssl = std::getenv("JDB_ENABLE_SSL");
    if (enable_ssl) config_.enable_ssl = (std::string(enable_ssl) == "true" || 
                                          std::string(enable_ssl) == "1" || 
                                          std::string(enable_ssl) == "yes");
    
    const char* ssl_cert_path = std::getenv("JDB_SSL_CERT_PATH");
    if (ssl_cert_path) config_.ssl_cert_path = std::string(ssl_cert_path);
    
    const char* ssl_key_path = std::getenv("JDB_SSL_KEY_PATH");
    if (ssl_key_path) config_.ssl_key_path = std::string(ssl_key_path);
}

void ConfigManager::load_from_args(int argc, char* argv[]) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        
        if (i + 1 >= argc) {
            break; // No value for this argument
        }
        
        std::string value = argv[i + 1];
        
        if (arg == "--host") {
            config_.host = value;
        } else if (arg == "--port") {
            try {
                config_.port = std::stoi(value);
            } catch (const std::exception&) {}
        } else if (arg == "--grpc-port") {
            try {
                config_.grpc_port = std::stoi(value);
            } catch (const std::exception&) {}
        } else if (arg == "--data-dir") {
            config_.data_directory = value;
        } else if (arg == "--max-connections") {
            try {
                config_.max_connections = std::stoull(value);
            } catch (const std::exception&) {}
        } else if (arg == "--thread-pool-size") {
            try {
                config_.thread_pool_size = std::stoull(value);
            } catch (const std::exception&) {}
        } else if (arg == "--max-request-size") {
            try {
                config_.max_request_size = std::stoull(value);
            } catch (const std::exception&) {}
        } else if (arg == "--log-level") {
            config_.log_level = value;
        } else if (arg == "--log-file") {
            config_.log_file = value;
        } else if (arg == "--log-rotation-size") {
            try {
                config_.log_rotation_size = std::stoull(value);
            } catch (const std::exception&) {}
        } else if (arg == "--log-backup-count") {
            try {
                config_.log_backup_count = std::stoi(value);
            } catch (const std::exception&) {}
        } else if (arg == "--default-vector-dim") {
            try {
                config_.default_vector_dimension = std::stoull(value);
            } catch (const std::exception&) {}
        } else if (arg == "--default-index-type") {
            config_.default_index_type = value;
        } else if (arg == "--default-shard-count") {
            try {
                config_.default_shard_count = std::stoi(value);
            } catch (const std::exception&) {}
        } else if (arg == "--default-replication-factor") {
            try {
                config_.default_replication_factor = std::stoi(value);
            } catch (const std::exception&) {}
        } else if (arg == "--enable-ssl") {
            config_.enable_ssl = (value == "true" || value == "1" || value == "yes");
        } else if (arg == "--ssl-cert-path") {
            config_.ssl_cert_path = value;
        } else if (arg == "--ssl-key-path") {
            config_.ssl_key_path = value;
        }
    }
}

ServerConfig ConfigManager::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

template<typename T>
void ConfigManager::set_config_value(const std::string& key, const T& value) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    // This is a simplified implementation - in a real system, you would need to
    // handle different types properly, possibly with a more sophisticated approach
    if constexpr (std::is_same_v<T, std::string>) {
        if (key == "host") {
            config_.host = value;
        } else if (key == "data_directory") {
            config_.data_directory = value;
        } else if (key == "log_level") {
            config_.log_level = value;
        } else if (key == "log_file") {
            config_.log_file = value;
        } else if (key == "default_index_type") {
            config_.default_index_type = value;
        } else if (key == "ssl_cert_path") {
            config_.ssl_cert_path = value;
        } else if (key == "ssl_key_path") {
            config_.ssl_key_path = value;
        }
    } else if constexpr (std::is_same_v<T, int>) {
        if (key == "port") {
            config_.port = value;
        } else if (key == "grpc_port") {
            config_.grpc_port = value;
        } else if (key == "log_backup_count") {
            config_.log_backup_count = value;
        } else if (key == "default_shard_count") {
            config_.default_shard_count = value;
        } else if (key == "default_replication_factor") {
            config_.default_replication_factor = value;
        }
    } else if constexpr (std::is_same_v<T, size_t>) {
        if (key == "max_connections") {
            config_.max_connections = value;
        } else if (key == "thread_pool_size") {
            config_.thread_pool_size = value;
        } else if (key == "max_request_size") {
            config_.max_request_size = value;
        } else if (key == "log_rotation_size") {
            config_.log_rotation_size = value;
        } else if (key == "default_vector_dimension") {
            config_.default_vector_dimension = value;
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (key == "enable_ssl") {
            config_.enable_ssl = value;
        }
    }
    // Additional type-specific assignments would go here
}

bool ConfigManager::validate_config() const {
    // Validate port ranges
    if (config_.port < 1 || config_.port > 65535) {
        return false;
    }
    
    if (config_.grpc_port < 1 || config_.grpc_port > 65535) {
        return false;
    }
    
    // Validate thread pool size
    if (config_.thread_pool_size == 0) {
        return false;
    }
    
    // Validate vector dimension is positive
    if (config_.default_vector_dimension == 0) {
        return false;
    }
    
    // Validate shard count is positive
    if (config_.default_shard_count < 1) {
        return false;
    }
    
    // Validate replication factor is positive
    if (config_.default_replication_factor < 1) {
        return false;
    }
    
    // Validate log backup count is non-negative
    if (config_.log_backup_count < 0) {
        return false;
    }
    
    // Validate that SSL settings are coherent
    if (config_.enable_ssl && (config_.ssl_cert_path.empty() || config_.ssl_key_path.empty())) {
        return false;
    }
    
    return true;
}

} // namespace jadevectordb