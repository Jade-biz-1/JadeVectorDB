#include "distributed_service_manager.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

DistributedServiceManager::DistributedServiceManager() 
    : initialized_(false), running_(false) {
    logger_ = logging::LoggerManager::get_logger("DistributedServiceManager");
}

DistributedServiceManager::~DistributedServiceManager() {
    if (running_) {
        stop().has_value(); // Ignore result in destructor
    }
}

Result<bool> DistributedServiceManager::initialize(const DistributedConfig& config) {
    try {
        LOG_INFO(logger_, "Initializing DistributedServiceManager");
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid distributed configuration provided");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid distributed configuration");
        }
        
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            config_ = config;
        }
        
        // Initialize individual services
        Result<bool> result;
        
        if (config_.enable_sharding) {
            result = initialize_sharding_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to initialize sharding service: " + 
                         ErrorHandler::format_error(result.error()));
                return result;
            }
        }
        
        if (config_.enable_replication) {
            result = initialize_replication_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to initialize replication service: " + 
                         ErrorHandler::format_error(result.error()));
                return result;
            }
        }
        
        if (config_.enable_clustering) {
            result = initialize_cluster_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to initialize cluster service: " + 
                         ErrorHandler::format_error(result.error()));
                return result;
            }
        }
        
        // Initialize query router last as it depends on other services
        result = initialize_query_router();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize query router: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        // Initialize security audit logger
        security_audit_logger_ = std::make_shared<SecurityAuditLogger>();
        SecurityAuditConfig audit_config;
        audit_config.log_file_path = "./distributed_security_audit.log";
        if (!security_audit_logger_->initialize(audit_config)) {
            LOG_WARN(logger_, "Failed to initialize security audit logger, continuing without it");
            // Don't fail initialization if audit logger fails
        }
        
        // Initialize performance benchmark service
        performance_benchmark_ = std::make_shared<PerformanceBenchmark>();
        
        initialized_ = true;
        LOG_INFO(logger_, "DistributedServiceManager initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in DistributedServiceManager::initialize: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize distributed services: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::start() {
    try {
        if (!initialized_) {
            LOG_ERROR(logger_, "DistributedServiceManager not initialized");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "DistributedServiceManager not initialized");
        }
        
        if (running_) {
            LOG_WARN(logger_, "DistributedServiceManager is already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting DistributedServiceManager");
        
        Result<bool> result;
        
        // Start cluster service first
        if (config_.enable_clustering && cluster_service_) {
            result = start_cluster_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to start cluster service: " + 
                         ErrorHandler::format_error(result.error()));
                
                // Attempt recovery
                LOG_WARN(logger_, "Attempting cluster service recovery...");
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Brief delay before retry
                result = start_cluster_service();
                
                if (!result.has_value()) {
                    LOG_ERROR(logger_, "Cluster service recovery failed: " + 
                             ErrorHandler::format_error(result.error()));
                    return result;
                } else {
                    LOG_INFO(logger_, "Cluster service recovery succeeded");
                }
            }
        }
        
        // Start sharding service
        if (config_.enable_sharding && sharding_service_) {
            result = start_sharding_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to start sharding service: " + 
                         ErrorHandler::format_error(result.error()));
                
                // Attempt recovery
                LOG_WARN(logger_, "Attempting sharding service recovery...");
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Brief delay before retry
                result = start_sharding_service();
                
                if (!result.has_value()) {
                    LOG_ERROR(logger_, "Sharding service recovery failed: " + 
                             ErrorHandler::format_error(result.error()));
                    return result;
                } else {
                    LOG_INFO(logger_, "Sharding service recovery succeeded");
                }
            }
        }
        
        // Start replication service
        if (config_.enable_replication && replication_service_) {
            result = start_replication_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to start replication service: " + 
                         ErrorHandler::format_error(result.error()));
                
                // Attempt recovery
                LOG_WARN(logger_, "Attempting replication service recovery...");
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Brief delay before retry
                result = start_replication_service();
                
                if (!result.has_value()) {
                    LOG_ERROR(logger_, "Replication service recovery failed: " + 
                             ErrorHandler::format_error(result.error()));
                    return result;
                } else {
                    LOG_INFO(logger_, "Replication service recovery succeeded");
                }
            }
        }
        
        // Start query router last
        if (query_router_) {
            result = start_query_router();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to start query router: " + 
                         ErrorHandler::format_error(result.error()));
                
                // Attempt recovery
                LOG_WARN(logger_, "Attempting query router recovery...");
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Brief delay before retry
                result = start_query_router();
                
                if (!result.has_value()) {
                    LOG_ERROR(logger_, "Query router recovery failed: " + 
                             ErrorHandler::format_error(result.error()));
                    return result;
                } else {
                    LOG_INFO(logger_, "Query router recovery succeeded");
                }
            }
        }
        
        // Coordinate services
        result = coordinate_services();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to coordinate services: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        running_ = true;
        LOG_INFO(logger_, "DistributedServiceManager started successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in DistributedServiceManager::start: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start distributed services: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::stop() {
    try {
        if (!running_) {
            LOG_WARN(logger_, "DistributedServiceManager is not running");
            return true;
        }
        
        LOG_INFO(logger_, "Stopping DistributedServiceManager");
        
        Result<bool> result;
        bool all_stopped_successfully = true;
        
        // Stop services in reverse order
        if (query_router_) {
            result = stop_query_router();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to stop query router: " + 
                         ErrorHandler::format_error(result.error()));
                all_stopped_successfully = false;
                
                // Log error but continue to stop other services
            }
        }
        
        if (config_.enable_replication && replication_service_) {
            result = stop_replication_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to stop replication service: " + 
                         ErrorHandler::format_error(result.error()));
                all_stopped_successfully = false;
                
                // Log error but continue to stop other services
            }
        }
        
        if (config_.enable_sharding && sharding_service_) {
            result = stop_sharding_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to stop sharding service: " + 
                         ErrorHandler::format_error(result.error()));
                all_stopped_successfully = false;
                
                // Log error but continue to stop other services
            }
        }
        
        if (config_.enable_clustering && cluster_service_) {
            result = stop_cluster_service();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to stop cluster service: " + 
                         ErrorHandler::format_error(result.error()));
                all_stopped_successfully = false;
                
                // Log error but continue
            }
        }
        
        running_ = false;
        
        if (all_stopped_successfully) {
            LOG_INFO(logger_, "DistributedServiceManager stopped successfully");
        } else {
            LOG_WARN(logger_, "DistributedServiceManager stopped with some service stop failures");
        }
        
        return all_stopped_successfully;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in DistributedServiceManager::stop: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to stop distributed services: " + std::string(e.what()));
    }
}

bool DistributedServiceManager::is_initialized() const {
    return initialized_;
}

bool DistributedServiceManager::is_running() const {
    return running_;
}

std::shared_ptr<ShardingService> DistributedServiceManager::get_sharding_service() const {
    std::lock_guard<std::mutex> lock(service_mutex_);
    return sharding_service_;
}

std::shared_ptr<ReplicationService> DistributedServiceManager::get_replication_service() const {
    std::lock_guard<std::mutex> lock(service_mutex_);
    return replication_service_;
}

std::shared_ptr<QueryRouter> DistributedServiceManager::get_query_router() const {
    std::lock_guard<std::mutex> lock(service_mutex_);
    return query_router_;
}

ClusterService* DistributedServiceManager::get_cluster_service() const {
    std::lock_guard<std::mutex> lock(service_mutex_);
    return cluster_service_.get();
}

Result<bool> DistributedServiceManager::join_cluster(const std::string& seed_node_host, int seed_node_port) {
    try {
        if (!config_.enable_clustering || !cluster_service_) {
            LOG_WARN(logger_, "Clustering is not enabled");
            return true;
        }
        
        LOG_INFO(logger_, "Joining cluster via seed node " + seed_node_host + ":" + 
                std::to_string(seed_node_port));
        
        auto result = cluster_service_->join_cluster(seed_node_host, seed_node_port);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to join cluster: " + 
                     ErrorHandler::format_error(result.error()));
            
            // Log security event for failed cluster join
            if (security_audit_logger_) {
                security_audit_logger_->log_security_event(
                    SecurityEvent(SecurityEventType::AUTHORIZATION_DENIED, "system", seed_node_host,
                                 "cluster_join", "join_cluster", false));
            }
            
            return result;
        }
        
        LOG_INFO(logger_, "Successfully joined cluster");
        
        // Log security event for successful cluster join
        if (security_audit_logger_) {
            security_audit_logger_->log_security_event(
                SecurityEvent(SecurityEventType::AUTHORIZATION_GRANTED, "system", seed_node_host,
                             "cluster_join", "join_cluster", true));
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in join_cluster: " + std::string(e.what()));
        
        // Log security event for exception during cluster join
        if (security_audit_logger_) {
            security_audit_logger_->log_security_event(
                SecurityEvent(SecurityEventType::SECURITY_POLICY_VIOLATION, "system", seed_node_host,
                             "cluster_join", "join_cluster", false));
        }
        
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to join cluster: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::leave_cluster() {
    try {
        if (!config_.enable_clustering || !cluster_service_) {
            LOG_WARN(logger_, "Clustering is not enabled");
            return true;
        }
        
        LOG_INFO(logger_, "Leaving cluster");
        
        // In a real implementation, this would notify other nodes
        // For now, we'll just stop the cluster service
        auto result = stop_cluster_service();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to stop cluster service: " + ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Successfully left cluster");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in leave_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to leave cluster: " + std::string(e.what()));
    }
}

Result<ClusterState> DistributedServiceManager::get_cluster_state() const {
    try {
        if (!config_.enable_clustering || !cluster_service_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Clustering is not enabled");
        }
        
        return cluster_service_->get_cluster_state();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_cluster_state: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get cluster state: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::is_cluster_healthy() const {
    try {
        if (!config_.enable_clustering || !cluster_service_) {
            return true; // No cluster means no cluster health issues
        }
        
        return cluster_service_->check_cluster_health();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_cluster_healthy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check cluster health: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::add_node_to_cluster(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Adding node to cluster: " + node_id);
        
        if (!config_.enable_clustering || !cluster_service_) {
            LOG_WARN(logger_, "Clustering is not enabled");
            return true;
        }
        
        // Notify cluster service about new node
        auto result = cluster_service_->add_node_to_cluster(node_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to add node to cluster service: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        // If sharding is enabled, update sharding service
        if (config_.enable_sharding && sharding_service_) {
            Database dummy_db;
            dummy_db.databaseId = "cluster_db";
            auto shard_result = sharding_service_->add_node_to_cluster(node_id);
            if (!shard_result.has_value()) {
                LOG_WARN(logger_, "Failed to update sharding service with new node: " + 
                        ErrorHandler::format_error(shard_result.error()));
            }
        }
        
        LOG_INFO(logger_, "Successfully added node to cluster: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_node_to_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add node to cluster: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::remove_node_from_cluster(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Removing node from cluster: " + node_id);
        
        if (!config_.enable_clustering || !cluster_service_) {
            LOG_WARN(logger_, "Clustering is not enabled");
            return true;
        }
        
        // Notify cluster service about node removal
        auto result = cluster_service_->remove_node_from_cluster(node_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to remove node from cluster service: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        // If sharding is enabled, update sharding service
        if (config_.enable_sharding && sharding_service_) {
            auto shard_result = sharding_service_->remove_node_from_cluster(node_id);
            if (!shard_result.has_value()) {
                LOG_WARN(logger_, "Failed to update sharding service with node removal: " + 
                        ErrorHandler::format_error(shard_result.error()));
            }
        }
        
        LOG_INFO(logger_, "Successfully removed node from cluster: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_node_from_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to remove node from cluster: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::handle_node_failure(const std::string& failed_node_id) {
    try {
        LOG_WARN(logger_, "Handling failure of node: " + failed_node_id);
        
        if (!config_.enable_clustering || !cluster_service_) {
            LOG_WARN(logger_, "Clustering is not enabled");
            return true;
        }
        
        // Notify cluster service about node failure
        auto result = cluster_service_->handle_node_failure(failed_node_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to handle node failure in cluster service: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        // If sharding is enabled, update sharding service
        if (config_.enable_sharding && sharding_service_) {
            auto shard_result = sharding_service_->handle_node_failure(failed_node_id);
            if (!shard_result.has_value()) {
                LOG_WARN(logger_, "Failed to update sharding service with node failure: " + 
                        ErrorHandler::format_error(shard_result.error()));
            }
        }
        
        // If replication is enabled, update replication service
        if (config_.enable_replication && replication_service_) {
            auto repl_result = replication_service_->handle_node_failure(failed_node_id);
            if (!repl_result.has_value()) {
                LOG_WARN(logger_, "Failed to update replication service with node failure: " + 
                        ErrorHandler::format_error(repl_result.error()));
            }
        }
        
        LOG_INFO(logger_, "Successfully handled failure of node: " + failed_node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_node_failure: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to handle node failure: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::update_distributed_config(const DistributedConfig& new_config) {
    try {
        LOG_INFO(logger_, "Updating distributed configuration");
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid distributed configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid distributed configuration");
        }
        
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            config_ = new_config;
        }
        
        // Update service configurations
        if (config_.enable_sharding && sharding_service_) {
            auto result = sharding_service_->update_sharding_config(config_.sharding_config);
            if (!result.has_value()) {
                LOG_WARN(logger_, "Failed to update sharding configuration: " + 
                        ErrorHandler::format_error(result.error()));
            }
        }
        
        if (config_.enable_replication && replication_service_) {
            auto result = replication_service_->update_replication_config(config_.replication_config);
            if (!result.has_value()) {
                LOG_WARN(logger_, "Failed to update replication configuration: " + 
                        ErrorHandler::format_error(result.error()));
            }
        }
        
        LOG_INFO(logger_, "Successfully updated distributed configuration");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_distributed_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update distributed configuration: " + std::string(e.what()));
    }
}

DistributedConfig DistributedServiceManager::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

Result<bool> DistributedServiceManager::create_shards_for_database(const Database& database) {
    try {
        if (!config_.enable_sharding || !sharding_service_) {
            LOG_WARN(logger_, "Sharding is not enabled");
            return true;
        }
        
        LOG_INFO(logger_, "Creating shards for database: " + database.databaseId);
        
        // Log configuration change for security auditing
        if (security_audit_logger_) {
            SecurityEvent event(
                SecurityEventType::CONFIGURATION_CHANGE,
                "unknown_user",  // In a real implementation, user ID would be passed
                "unknown_ip",    // In a real implementation, IP would be passed
                database.databaseId,
                "create_shards",
                true
            );
            event.details = "Creating shards for database configuration";
            security_audit_logger_->log_security_event(event);
        }
        
        // Perform performance benchmarking if enabled
        std::chrono::high_resolution_clock::time_point start_time;
        if (performance_benchmark_) {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        auto result = sharding_service_->create_shards_for_database(database);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to create shards for database " + database.databaseId + 
                     ": " + ErrorHandler::format_error(result.error()));
            
            // Log security event for failed configuration change
            if (security_audit_logger_) {
                security_audit_logger_->log_security_event(
                    SecurityEvent(SecurityEventType::CONFIGURATION_CHANGE, "system", "localhost",
                                 database.databaseId, "create_shards_for_database", false));
            }
            
            return result;
        }
        
        LOG_INFO(logger_, "Successfully created shards for database: " + database.databaseId);
        
        // Log performance metrics if benchmarking is available
        if (performance_benchmark_) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // In a real implementation, you might want to store this performance data
            LOG_DEBUG(logger_, "Shard creation completed in " + std::to_string(duration.count()) + " milliseconds");
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_shards_for_database: " + std::string(e.what()));
        
        // Log security event for exception during configuration change
        if (security_audit_logger_) {
            security_audit_logger_->log_security_event(
                SecurityEvent(SecurityEventType::SECURITY_POLICY_VIOLATION, "system", "localhost",
                             database.databaseId, "create_shards_for_database", false));
        }
        
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create shards for database: " + std::string(e.what()));
    }
}

Result<std::string> DistributedServiceManager::get_shard_for_vector(const std::string& vector_id, 
                                                                 const std::string& database_id) const {
    try {
        if (!config_.enable_sharding || !sharding_service_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Sharding is not enabled");
        }
        
        Database dummy_db;
        dummy_db.databaseId = database_id;
        
        Vector dummy_vec;
        dummy_vec.id = vector_id;
        
        auto result = sharding_service_->determine_shard(dummy_vec, dummy_db);
        if (!result.has_value()) {
            return result;
        }
        
        return result.value().shard_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_shard_for_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to determine shard for vector: " + std::string(e.what()));
    }
}

Result<std::string> DistributedServiceManager::get_node_for_shard(const std::string& shard_id) const {
    try {
        if (!config_.enable_sharding || !sharding_service_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Sharding is not enabled");
        }
        
        return sharding_service_->get_node_for_shard(shard_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_node_for_shard: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to determine node for shard: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::replicate_vector(const Vector& vector, const Database& database) {
    try {
        if (!config_.enable_replication || !replication_service_) {
            LOG_WARN(logger_, "Replication is not enabled");
            return true;
        }
        
        LOG_DEBUG(logger_, "Replicating vector " + vector.id + " for database " + database.databaseId);
        
        // Log data modification for security auditing
        if (security_audit_logger_) {
            SecurityEvent event(
                SecurityEventType::DATA_MODIFICATION,
                "unknown_user",  // In a real implementation, user ID would be passed
                "unknown_ip",    // In a real implementation, IP would be passed
                database.databaseId,
                "replicate",
                true
            );
            event.details = "Replicating vector: " + vector.id;
            security_audit_logger_->log_security_event(event);
        }
        
        // Perform performance benchmarking if enabled
        std::chrono::high_resolution_clock::time_point start_time;
        if (performance_benchmark_) {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        auto result = replication_service_->replicate_vector(vector, database);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to replicate vector " + vector.id + 
                     ": " + ErrorHandler::format_error(result.error()));
            
            // Log security event for failed replication
            if (security_audit_logger_) {
                security_audit_logger_->log_security_event(
                    SecurityEvent(SecurityEventType::DATA_MODIFICATION, "system", "localhost",
                                 database.databaseId, "replicate_vector", false));
            }
            
            return result;
        }
        
        LOG_DEBUG(logger_, "Successfully replicated vector " + vector.id);
        
        // Log performance metrics if benchmarking is available
        if (performance_benchmark_) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            // In a real implementation, you might want to store this performance data
            LOG_DEBUG(logger_, "Vector replication completed in " + std::to_string(duration.count()) + " microseconds");
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in replicate_vector: " + std::string(e.what()));
        
        // Log security event for exception during replication
        if (security_audit_logger_) {
            security_audit_logger_->log_security_event(
                SecurityEvent(SecurityEventType::SECURITY_POLICY_VIOLATION, "system", "localhost",
                             database.databaseId, "replicate_vector", false));
        }
        
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to replicate vector: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::is_vector_fully_replicated(const std::string& vector_id) const {
    try {
        if (!config_.enable_replication || !replication_service_) {
            return true; // No replication means no replication issues
        }
        
        return replication_service_->is_fully_replicated(vector_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_vector_fully_replicated: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check vector replication status: " + std::string(e.what()));
    }
}

Result<RouteInfo> DistributedServiceManager::route_operation(const std::string& database_id,
                                                         const std::string& operation_type,
                                                         const std::string& operation_key) const {
    try {
        if (!query_router_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Query router is not available");
        }
        
        LOG_DEBUG(logger_, "Routing operation " + operation_type + " for database " + database_id);
        
        // Log access to the database for security auditing
        if (security_audit_logger_) {
            SecurityEvent event(
                SecurityEventType::DATA_ACCESS,
                "unknown_user",  // In a real implementation, user ID would be passed
                "unknown_ip",    // In a real implementation, IP would be passed
                database_id,
                operation_type,
                true
            );
            event.details = "Routing operation for key: " + operation_key;
            security_audit_logger_->log_security_event(event);
        }
        
        // Perform performance benchmarking if enabled
        std::chrono::high_resolution_clock::time_point start_time;
        if (performance_benchmark_) {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        auto result = query_router_->route_operation(database_id, operation_type, operation_key);
        
        // Log performance metrics if benchmarking is available
        if (performance_benchmark_ && result.has_value()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            
            // In a real implementation, you might want to store this performance data
            LOG_DEBUG(logger_, "Route operation completed in " + std::to_string(duration.count()) + "ns");
        }
        
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_operation: " + std::string(e.what()));
        
        // Log security event for exception during routing
        if (security_audit_logger_) {
            security_audit_logger_->log_security_event(
                SecurityEvent(SecurityEventType::SECURITY_POLICY_VIOLATION, "system", "localhost",
                             database_id, "route_operation", false));
        }
        
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route operation: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, std::string>> DistributedServiceManager::get_distributed_stats() const {
    try {
        std::unordered_map<std::string, std::string> stats;
        
        if (config_.enable_clustering && cluster_service_) {
            auto cluster_stats_result = cluster_service_->get_cluster_stats();
            if (cluster_stats_result.has_value()) {
                auto cluster_stats = cluster_stats_result.value();
                for (const auto& entry : cluster_stats) {
                    stats["cluster_" + entry.first] = entry.second;
                }
            }
        }
        
        if (config_.enable_sharding && sharding_service_) {
            auto shard_dist_result = sharding_service_->get_shard_distribution();
            if (shard_dist_result.has_value()) {
                auto shard_dist = shard_dist_result.value();
                stats["shard_distribution"] = std::to_string(shard_dist.size()) + " shards";
            }
        }
        
        if (config_.enable_replication && replication_service_) {
            auto repl_stats_result = replication_service_->get_replication_stats();
            if (repl_stats_result.has_value()) {
                auto repl_stats = repl_stats_result.value();
                for (const auto& entry : repl_stats) {
                    stats["replication_" + entry.first] = std::to_string(entry.second);
                }
            }
        }
        
        if (query_router_) {
            auto routing_stats_result = query_router_->get_routing_stats();
            if (routing_stats_result.has_value()) {
                auto routing_stats = routing_stats_result.value();
                for (const auto& entry : routing_stats) {
                    stats["routing_" + entry.first] = std::to_string(entry.second);
                }
            }
        }
        
        LOG_DEBUG(logger_, "Generated distributed statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_distributed_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get distributed stats: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::check_distributed_health() const {
    try {
        // Check cluster health
        if (config_.enable_clustering && cluster_service_) {
            auto cluster_health_result = is_cluster_healthy();
            if (!cluster_health_result.has_value() || !cluster_health_result.value()) {
                LOG_WARN(logger_, "Cluster health check failed");
                if (cluster_health_result.has_value()) {
                    return cluster_health_result;
                } else {
                    RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Cluster health check failed: " + 
                                ErrorHandler::format_error(cluster_health_result.error()));
                }
            }
        }
        
        // Check sharding health
        if (config_.enable_sharding && sharding_service_) {
            auto shard_balance_result = sharding_service_->is_balanced();
            if (!shard_balance_result.has_value() || !shard_balance_result.value()) {
                LOG_WARN(logger_, "Sharding balance check failed");
                if (shard_balance_result.has_value()) {
                    return shard_balance_result;
                } else {
                    RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Sharding balance check failed: " + 
                                ErrorHandler::format_error(shard_balance_result.error()));
                }
            }
        }
        
        // Check replication health
        if (config_.enable_replication && replication_service_) {
            // In a real implementation, we would check replication health
            // For now, we'll assume it's healthy
        }
        
        LOG_DEBUG(logger_, "Distributed health check passed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_distributed_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check distributed health: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::rebalance_shards() {
    try {
        if (!config_.enable_sharding || !sharding_service_) {
            LOG_WARN(logger_, "Sharding is not enabled");
            return true;
        }
        
        LOG_INFO(logger_, "Rebalancing shards");
        
        auto result = sharding_service_->rebalance_shards();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to rebalance shards: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Shard rebalancing completed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in rebalance_shards: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to rebalance shards: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::force_replication_for_database(const std::string& database_id) {
    try {
        if (!config_.enable_replication || !replication_service_) {
            LOG_WARN(logger_, "Replication is not enabled");
            return true;
        }
        
        LOG_INFO(logger_, "Forcing replication for database: " + database_id);
        
        auto result = replication_service_->force_replication_for_database(database_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to force replication for database " + database_id + 
                     ": " + ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Forced replication completed for database: " + database_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in force_replication_for_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to force replication: " + std::string(e.what()));
    }
}

// Private methods

Result<bool> DistributedServiceManager::initialize_sharding_service() {
    try {
        LOG_DEBUG(logger_, "Initializing sharding service");
        
        std::lock_guard<std::mutex> lock(service_mutex_);
        sharding_service_ = std::make_shared<ShardingService>();
        
        auto result = sharding_service_->initialize(config_.sharding_config);
        if (!result) {
            LOG_ERROR(logger_, "Failed to initialize sharding service");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize sharding service");
        }
        
        LOG_DEBUG(logger_, "Sharding service initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize_sharding_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize sharding service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::initialize_replication_service() {
    try {
        LOG_DEBUG(logger_, "Initializing replication service");
        
        std::lock_guard<std::mutex> lock(service_mutex_);
        replication_service_ = std::make_shared<ReplicationService>();
        
        auto result = replication_service_->initialize(config_.replication_config);
        if (!result) {
            LOG_ERROR(logger_, "Failed to initialize replication service");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize replication service");
        }
        
        LOG_DEBUG(logger_, "Replication service initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize_replication_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize replication service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::initialize_query_router() {
    try {
        LOG_DEBUG(logger_, "Initializing query router");
        
        std::lock_guard<std::mutex> lock(service_mutex_);
        query_router_ = std::make_shared<QueryRouter>();
        
        RoutingConfig routing_config;
        routing_config.strategy = config_.routing_config.strategy;
        routing_config.max_route_cache_size = config_.routing_config.max_route_cache_size;
        routing_config.route_ttl_seconds = config_.routing_config.route_ttl_seconds;
        routing_config.enable_adaptive_routing = config_.routing_config.enable_adaptive_routing;
        routing_config.preferred_nodes = config_.routing_config.preferred_nodes;
        
        auto result = query_router_->initialize(routing_config);
        if (!result) {
            LOG_ERROR(logger_, "Failed to initialize query router");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize query router");
        }
        
        LOG_DEBUG(logger_, "Query router initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize_query_router: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize query router: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::initialize_cluster_service() {
    try {
        LOG_DEBUG(logger_, "Initializing cluster service");
        
        std::lock_guard<std::mutex> lock(service_mutex_);
        cluster_service_ = std::make_unique<ClusterService>(config_.cluster_host, config_.cluster_port);
        
        auto result = cluster_service_->initialize();
        if (!result) {
            LOG_ERROR(logger_, "Failed to initialize cluster service");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize cluster service");
        }
        
        LOG_DEBUG(logger_, "Cluster service initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize_cluster_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize cluster service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::start_sharding_service() {
    try {
        LOG_DEBUG(logger_, "Starting sharding service");
        
        // In a real implementation, this might involve connecting to cluster members
        // For now, we'll just log that it's started
        LOG_DEBUG(logger_, "Sharding service started");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_sharding_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start sharding service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::start_replication_service() {
    try {
        LOG_DEBUG(logger_, "Starting replication service");
        
        // In a real implementation, this might involve connecting to cluster members
        // For now, we'll just log that it's started
        LOG_DEBUG(logger_, "Replication service started");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_replication_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start replication service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::start_query_router() {
    try {
        LOG_DEBUG(logger_, "Starting query router");
        
        // In a real implementation, this might involve connecting to cluster members
        // For now, we'll just log that it's started
        LOG_DEBUG(logger_, "Query router started");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_query_router: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start query router: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::start_cluster_service() {
    try {
        LOG_DEBUG(logger_, "Starting cluster service");
        
        if (!cluster_service_) {
            LOG_WARN(logger_, "Cluster service not initialized");
            return true;
        }
        
        auto result = cluster_service_->start();
        if (!result) {
            LOG_ERROR(logger_, "Failed to start cluster service");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start cluster service");
        }
        
        // If seed nodes are provided, join the cluster
        if (!config_.seed_nodes.empty()) {
            bool joined_cluster = false;
            for (const auto& seed_node : config_.seed_nodes) {
                // Parse seed node (format: host:port)
                size_t colon_pos = seed_node.find(':');
                if (colon_pos != std::string::npos) {
                    std::string host = seed_node.substr(0, colon_pos);
                    int port = std::stoi(seed_node.substr(colon_pos + 1));
                    
                    auto join_result = cluster_service_->join_cluster(host, port);
                    if (!join_result.has_value()) {
                        LOG_WARN(logger_, "Failed to join cluster via seed node " + seed_node + 
                                ": " + ErrorHandler::format_error(join_result.error()));
                    } else {
                        LOG_INFO(logger_, "Successfully joined cluster via seed node: " + seed_node);
                        joined_cluster = true;
                        break; // Successfully joined, no need to try other seed nodes
                    }
                }
            }
            
            // If we couldn't join via any seed nodes, log a warning
            if (!joined_cluster) {
                LOG_WARN(logger_, "Failed to join cluster via any seed nodes");
            }
        }
        
        LOG_DEBUG(logger_, "Cluster service started");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_cluster_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start cluster service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::stop_sharding_service() {
    try {
        LOG_DEBUG(logger_, "Stopping sharding service");
        
        // In a real implementation, this might involve disconnecting from cluster members
        // For now, we'll just log that it's stopped
        LOG_DEBUG(logger_, "Sharding service stopped");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop_sharding_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to stop sharding service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::stop_replication_service() {
    try {
        LOG_DEBUG(logger_, "Stopping replication service");
        
        // In a real implementation, this might involve disconnecting from cluster members
        // For now, we'll just log that it's stopped
        LOG_DEBUG(logger_, "Replication service stopped");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop_replication_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to stop replication service: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::stop_query_router() {
    try {
        LOG_DEBUG(logger_, "Stopping query router");
        
        // In a real implementation, this might involve disconnecting from cluster members
        // For now, we'll just log that it's stopped
        LOG_DEBUG(logger_, "Query router stopped");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop_query_router: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to stop query router: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::stop_cluster_service() {
    try {
        LOG_DEBUG(logger_, "Stopping cluster service");
        
        if (cluster_service_) {
            cluster_service_->stop();
        }
        
        LOG_DEBUG(logger_, "Cluster service stopped");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop_cluster_service: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to stop cluster service: " + std::string(e.what()));
    }
}

bool DistributedServiceManager::validate_config(const DistributedConfig& config) const {
    // Basic validation
    if (config.enable_clustering && config.cluster_host.empty()) {
        LOG_ERROR(logger_, "Cluster host is required when clustering is enabled");
        return false;
    }
    
    if (config.enable_clustering && config.cluster_port <= 0) {
        LOG_ERROR(logger_, "Valid cluster port is required when clustering is enabled");
        return false;
    }
    
    if (config.enable_sharding && config.sharding_config.num_shards <= 0) {
        LOG_ERROR(logger_, "Valid number of shards is required when sharding is enabled");
        return false;
    }
    
    if (config.enable_replication && config.replication_config.default_replication_factor < 1) {
        LOG_ERROR(logger_, "Valid replication factor is required when replication is enabled");
        return false;
    }
    
    // Validate routing configuration
    if (!config.routing_config.strategy.empty() && 
        config.routing_config.strategy != "round_robin" && 
        config.routing_config.strategy != "least_loaded" && 
        config.routing_config.strategy != "consistent_hash" && 
        config.routing_config.strategy != "adaptive") {
        LOG_ERROR(logger_, "Invalid routing strategy: " + config.routing_config.strategy);
        return false;
    }
    
    // Additional validation: Check for consistency between service configurations
    if (config.enable_sharding && config.enable_replication) {
        // Ensure that the number of shards is compatible with replication requirements
        // The number of shards should not exceed the available nodes factoring in replication
        if (config.sharding_config.num_shards > 10000) { // Arbitrary large number to prevent issues
            LOG_WARN(logger_, "Number of shards is very large, which may impact performance");
        }
        
        // Check that replication factor is reasonable given the number of available nodes
        if (!config.seed_nodes.empty() && 
            config.replication_config.default_replication_factor > static_cast<int>(config.seed_nodes.size())) {
            LOG_WARN(logger_, "Replication factor exceeds number of available seed nodes, " 
                    "which may affect data availability");
        }
    }
    
    // Validate that cluster configuration makes sense with other services
    if (config.enable_clustering && config.seed_nodes.empty() && running_) {
        LOG_WARN(logger_, "No seed nodes specified for cluster but clustering is enabled");
    }
    
    // Validate sharding strategy is compatible with other services
    if (config.enable_sharding && 
        !config.sharding_config.strategy.empty() && 
        config.sharding_config.strategy != "hash" && 
        config.sharding_config.strategy != "range" && 
        config.sharding_config.strategy != "vector" && 
        config.sharding_config.strategy != "auto") {
        LOG_ERROR(logger_, "Invalid sharding strategy: " + config.sharding_config.strategy);
        return false;
    }
    
    return true;
}

Result<bool> DistributedServiceManager::coordinate_services() {
    try {
        LOG_DEBUG(logger_, "Coordinating distributed services");
        
        // Ensure services are aware of each other and properly configured
        if (config_.enable_sharding && sharding_service_ && 
            config_.enable_replication && replication_service_) {
            
            // Connect sharding and replication services to enable coordinated operations
            // For example, when a vector is assigned to a shard, information about replication should be shared
            LOG_DEBUG(logger_, "Connecting sharding and replication services");
            
            // Update service dependencies
            auto result = update_service_dependencies();
            if (!result.has_value()) {
                LOG_WARN(logger_, "Failed to update service dependencies: " + 
                        ErrorHandler::format_error(result.error()));
                // Continue anyway as this isn't critical for basic operation
            }
        }
        
        if (query_router_ && cluster_service_) {
            // Connect query router to cluster service to enable dynamic routing based on cluster state
            LOG_DEBUG(logger_, "Connecting query router to cluster service");
            
            // Register with cluster service to receive notifications about cluster changes
            cluster_service_->register_rpc_handlers();
        }
        
        // If sharding and clustering are enabled, ensure cluster service is aware of sharding
        if (config_.enable_sharding && sharding_service_ && 
            config_.enable_clustering && cluster_service_) {
            
            LOG_DEBUG(logger_, "Integrating sharding with clustering");
            
            // Register sharding-related RPC handlers with cluster service
            // This allows cluster nodes to coordinate sharding operations
            cluster_service_->register_rpc_handlers();
        }
        
        // If replication and clustering are enabled, ensure proper coordination
        if (config_.enable_replication && replication_service_ && 
            config_.enable_clustering && cluster_service_) {
            
            LOG_DEBUG(logger_, "Integrating replication with clustering");
            
            // Register replication-related RPC handlers with cluster service
            cluster_service_->register_rpc_handlers();
        }
        
        LOG_DEBUG(logger_, "Distributed services coordinated successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in coordinate_services: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to coordinate services: " + std::string(e.what()));
    }
}

void DistributedServiceManager::handle_service_failure(const std::string& service_name, 
                                                   const std::string& error_message) {
    LOG_ERROR(logger_, "Service " + service_name + " failed: " + error_message);
    
    // Trigger appropriate recovery procedures based on the failed service
    if (service_name == "sharding") {
        LOG_WARN(logger_, "Attempting to recover sharding service...");
        // Attempt to restart the sharding service
        if (sharding_service_) {
            Result<bool> restart_result = start_sharding_service();
            if (!restart_result.has_value()) {
                LOG_ERROR(logger_, "Failed to restart sharding service: " + 
                         ErrorHandler::format_error(restart_result.error()));
            } else {
                LOG_INFO(logger_, "Sharding service restarted successfully");
            }
        }
    } else if (service_name == "replication") {
        LOG_WARN(logger_, "Attempting to recover replication service...");
        // Attempt to restart the replication service
        if (replication_service_) {
            Result<bool> restart_result = start_replication_service();
            if (!restart_result.has_value()) {
                LOG_ERROR(logger_, "Failed to restart replication service: " + 
                         ErrorHandler::format_error(restart_result.error()));
            } else {
                LOG_INFO(logger_, "Replication service restarted successfully");
            }
        }
    } else if (service_name == "query_router") {
        LOG_WARN(logger_, "Attempting to recover query router service...");
        // Attempt to restart the query router service
        if (query_router_) {
            Result<bool> restart_result = start_query_router();
            if (!restart_result.has_value()) {
                LOG_ERROR(logger_, "Failed to restart query router: " + 
                         ErrorHandler::format_error(restart_result.error()));
            } else {
                LOG_INFO(logger_, "Query router restarted successfully");
            }
        }
    } else if (service_name == "cluster") {
        LOG_WARN(logger_, "Attempting to recover cluster service...");
        // Attempt to restart the cluster service
        if (cluster_service_) {
            Result<bool> restart_result = start_cluster_service();
            if (!restart_result.has_value()) {
                LOG_ERROR(logger_, "Failed to restart cluster service: " + 
                         ErrorHandler::format_error(restart_result.error()));
            } else {
                LOG_INFO(logger_, "Cluster service restarted successfully");
                
                // Rejoin the cluster if possible
                if (!config_.seed_nodes.empty()) {
                    for (const auto& seed_node : config_.seed_nodes) {
                        size_t colon_pos = seed_node.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string host = seed_node.substr(0, colon_pos);
                            int port = std::stoi(seed_node.substr(colon_pos + 1));
                            
                            auto join_result = cluster_service_->join_cluster(host, port);
                            if (join_result.has_value()) {
                                LOG_INFO(logger_, "Successfully rejoined cluster via seed node: " + seed_node);
                                break;
                            } else {
                                LOG_WARN(logger_, "Failed to rejoin cluster via seed node " + seed_node + 
                                        ": " + ErrorHandler::format_error(join_result.error()));
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Trigger overall system health check after service recovery attempt
    Result<bool> health_check = check_distributed_health();
    if (!health_check.has_value() || !health_check.value()) {
        LOG_WARN(logger_, "System health check failed after service recovery attempt");
    } else {
        LOG_INFO(logger_, "System health check passed after service recovery");
    }
}

Result<bool> DistributedServiceManager::update_service_dependencies() {
    try {
        LOG_DEBUG(logger_, "Updating service dependencies");
        
        // Update dependencies between sharding and replication services
        if (config_.enable_sharding && sharding_service_ && 
            config_.enable_replication && replication_service_) {
            
            // Ensure the sharding service is aware of replication requirements
            auto sharding_config = sharding_service_->get_config();
            auto replication_config = replication_service_->get_config();
            
            // Update sharding configuration based on replication factor to ensure
            // proper distribution of data across nodes for replication purposes
            sharding_config.replication_factor = replication_config.default_replication_factor;
            auto update_result = sharding_service_->update_sharding_config(sharding_config);
            if (!update_result.has_value()) {
                LOG_WARN(logger_, "Failed to update sharding config with replication factor: " + 
                        ErrorHandler::format_error(update_result.error()));
                // Don't return failure here as it's not critical
            }
        }
        
        // Update dependencies between query router and other services
        if (query_router_ && cluster_service_) {
            // Update routing configuration based on cluster state
            auto routing_config = query_router_->get_config();
            auto cluster_nodes_result = cluster_service_->get_all_nodes();
            
            if (cluster_nodes_result.has_value()) {
                auto cluster_nodes = cluster_nodes_result.value();
                std::vector<std::string> available_nodes;
                
                // Extract node IDs for routing
                for (const auto& node : cluster_nodes) {
                    if (node.is_alive) {
                        available_nodes.push_back(node.node_id);
                    }
                }
                
                // Update routing configuration with available nodes
                routing_config.preferred_nodes = available_nodes;
                auto update_result = query_router_->update_routing_config(routing_config);
                if (!update_result.has_value()) {
                    LOG_WARN(logger_, "Failed to update query router config with cluster nodes: " + 
                            ErrorHandler::format_error(update_result.error()));
                    // Don't return failure here as it's not critical
                }
            }
        }
        
        // Update dependencies between sharding and clustering
        if (config_.enable_sharding && sharding_service_ && 
            config_.enable_clustering && cluster_service_) {
            
            auto cluster_nodes_result = cluster_service_->get_all_nodes();
            if (cluster_nodes_result.has_value()) {
                auto cluster_nodes = cluster_nodes_result.value();
                std::vector<std::string> available_nodes;
                
                // Extract node IDs for sharding
                for (const auto& node : cluster_nodes) {
                    if (node.is_alive) {
                        available_nodes.push_back(node.node_id);
                    }
                }
                
                // Update sharding service with available nodes
                for (const auto& node_id : available_nodes) {
                    auto add_result = sharding_service_->add_node_to_cluster(node_id);
                    if (!add_result.has_value()) {
                        LOG_WARN(logger_, "Failed to add node " + node_id + 
                                " to sharding service: " + ErrorHandler::format_error(add_result.error()));
                    }
                }
            }
        }
        
        LOG_DEBUG(logger_, "Service dependencies updated successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_service_dependencies: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update service dependencies: " + std::string(e.what()));
    }
}

std::string DistributedServiceManager::generate_route_key(const std::string& database_id,
                                                      const std::string& operation_type,
                                                      const std::string& operation_key) const {
    return database_id + ":" + operation_type + ":" + operation_key;
}

Result<std::vector<std::string>> DistributedServiceManager::select_multiple_nodes(
    const std::string& database_id,
    const std::string& operation_type,
    const std::string& operation_key,
    int count) const {
    try {
        std::vector<std::string> nodes;
        
        // For now, we'll return dummy nodes
        // In a real implementation, this would select actual cluster nodes
        for (int i = 0; i < std::max(1, count); ++i) {
            nodes.push_back("node_" + database_id + "_" + std::to_string(i));
        }
        
        return nodes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in select_multiple_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to select nodes: " + std::string(e.what()));
    }
}

std::shared_ptr<SecurityAuditLogger> DistributedServiceManager::get_security_audit_logger() const {
    std::lock_guard<std::mutex> lock(service_mutex_);
    return security_audit_logger_;
}

std::shared_ptr<PerformanceBenchmark> DistributedServiceManager::get_performance_benchmark() const {
    std::lock_guard<std::mutex> lock(service_mutex_);
    return performance_benchmark_;
}

Result<BenchmarkResult> DistributedServiceManager::run_distributed_benchmark(const BenchmarkConfig& config,
                                                                           const std::function<Result<BenchmarkOperationResult>()>& operation_func) {
    try {
        if (!performance_benchmark_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Performance benchmark service is not available");
        }
        
        LOG_INFO(logger_, "Running distributed benchmark: " + config.benchmark_name);
        
        auto result = performance_benchmark_->run_custom_benchmark(config, operation_func);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to run distributed benchmark: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Completed distributed benchmark: " + config.benchmark_name);
        
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_distributed_benchmark: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run distributed benchmark: " + std::string(e.what()));
    }
}

Result<bool> DistributedServiceManager::audit_security_event(const SecurityEvent& event) {
    try {
        if (!security_audit_logger_) {
            LOG_WARN(logger_, "Security audit logger is not available");
            return true; // Don't fail operation if auditing fails
        }
        
        auto result = security_audit_logger_->log_security_event(event);
        if (!result.has_value()) {
            LOG_WARN(logger_, "Failed to log security event: " + ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_DEBUG(logger_, "Recorded security event: " + std::to_string(static_cast<int>(event.event_type)));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in audit_security_event: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to audit security event: " + std::string(e.what()));
    }
}

} // namespace jadevectordb