#include <iostream>
#include "config/config_loader.h"
#include "lib/logging.h"

using namespace jadevectordb;

int main() {
    // Initialize logging
    logging::LoggerManager::initialize(logging::LogLevel::DEBUG);
    
    std::cout << "=== Testing ConfigLoader ===" << std::endl;
    
    ConfigLoader loader;
    
    // Test 1: Load configuration
    std::cout << "\nTest 1: Loading configuration..." << std::endl;
    auto result = loader.load_config("./config");
    
    if (!result.has_value()) {
        std::cerr << "ERROR: Failed to load config: " 
                  << ErrorHandler::format_error(result.error()) << std::endl;
        return 1;
    }
    
    AppConfig config = result.value();
    std::cout << "SUCCESS: Configuration loaded" << std::endl;
    
    // Test 2: Print configuration
    std::cout << "\n=== Configuration Summary ===" << std::endl;
    std::cout << "Environment: " << ConfigLoader::environment_to_string(config.environment) << std::endl;
    std::cout << "\nServer:" << std::endl;
    std::cout << "  Host: " << config.server_host << std::endl;
    std::cout << "  Port: " << config.server_port << std::endl;
    
    std::cout << "\nAuthentication:" << std::endl;
    std::cout << "  Require strong passwords: " << (config.auth.require_strong_passwords ? "yes" : "no") << std::endl;
    std::cout << "  Min password length: " << config.auth.min_password_length << std::endl;
    std::cout << "  Enable 2FA: " << (config.auth.enable_two_factor ? "yes" : "no") << std::endl;
    std::cout << "  Session timeout: " << config.auth.session_timeout_seconds << "s" << std::endl;
    
    std::cout << "\nSecurity:" << std::endl;
    std::cout << "  Rate limiting: " << (config.security.enable_rate_limiting ? "enabled" : "disabled") << std::endl;
    std::cout << "  IP blocking: " << (config.security.enable_ip_blocking ? "enabled" : "disabled") << std::endl;
    std::cout << "  Max failed logins: " << config.security.max_failed_logins << std::endl;
    std::cout << "  Block duration: " << config.security.block_duration_seconds << "s" << std::endl;
    
    std::cout << "\nDatabase:" << std::endl;
    std::cout << "  Connection pool size: " << config.database.connection_pool_size << std::endl;
    std::cout << "  Query timeout: " << config.database.query_timeout_seconds << "s" << std::endl;
    std::cout << "  DB path: " << config.database.db_path << std::endl;
    std::cout << "  Auth DB path: " << config.database.auth_db_path << std::endl;
    
    std::cout << "\nCache:" << std::endl;
    std::cout << "  Permission cache size: " << config.cache.permission_cache_size << std::endl;
    std::cout << "  Cache TTL: " << config.cache.permission_cache_ttl_seconds << "s" << std::endl;
    std::cout << "  Query cache: " << (config.cache.enable_query_cache ? "enabled" : "disabled") << std::endl;
    
    std::cout << "\nLogging:" << std::endl;
    std::cout << "  Level: " << config.logging.level << std::endl;
    std::cout << "  Format: " << config.logging.format << std::endl;
    std::cout << "  Output: " << config.logging.output << std::endl;
    std::cout << "  File path: " << config.logging.file_path << std::endl;
    std::cout << "  Log SQL queries: " << (config.logging.log_sql_queries ? "yes" : "no") << std::endl;
    
    std::cout << "\nSecrets:" << std::endl;
    std::cout << "  JWT secret: " << (config.jwt_secret.empty() ? "NOT SET" : "SET (" + std::to_string(config.jwt_secret.length()) + " chars)") << std::endl;
    std::cout << "  API secret: " << (config.api_secret_key.empty() ? "NOT SET" : "SET (" + std::to_string(config.api_secret_key.length()) + " chars)") << std::endl;
    std::cout << "  DB password: " << (config.db_password.empty() ? "NOT SET" : "SET (" + std::to_string(config.db_password.length()) + " chars)") << std::endl;
    
    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
