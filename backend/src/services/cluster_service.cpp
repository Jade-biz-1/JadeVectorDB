#include "cluster_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>

namespace jadevectordb {

ClusterService::ClusterService(const std::string& host, int port) 
    : host_(host), port_(port), current_role_(ClusterRole::FOLLOWER), 
      running_(false), heartbeat_running_(false), election_running_(false),
      current_term_(0) {
    logger_ = logging::LoggerManager::get_logger("ClusterService");
    initialize_node_id();
}

ClusterService::~ClusterService() {
    stop();
}

bool ClusterService::initialize() {
    try {
        LOG_INFO(logger_, "Initializing ClusterService for node: " + node_id_ + 
                " at " + host_ + ":" + std::to_string(port_));
        
        // Initialize cluster state
        cluster_state_.term = 0;
        cluster_state_.master_node_id = "";
        cluster_state_.last_election_time = std::chrono::steady_clock::now();
        
        LOG_INFO(logger_, "ClusterService initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in ClusterService::initialize: " + std::string(e.what()));
        return false;
    }
}

bool ClusterService::start() {
    try {
        if (running_) {
            LOG_WARN(logger_, "ClusterService is already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting ClusterService for node: " + node_id_);
        
        running_ = true;
        start_background_threads();
        
        LOG_INFO(logger_, "ClusterService started successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in ClusterService::start: " + std::string(e.what()));
        return false;
    }
}

void ClusterService::stop() {
    try {
        if (!running_) {
            LOG_DEBUG(logger_, "ClusterService is not running");
            return;
        }
        
        LOG_INFO(logger_, "Stopping ClusterService for node: " + node_id_);
        
        running_ = false;
        stop_background_threads();
        
        LOG_INFO(logger_, "ClusterService stopped successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in ClusterService::stop: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::join_cluster(const std::string& seed_node_host, int seed_node_port) {
    try {
        LOG_INFO(logger_, "Attempting to join cluster via seed node " + seed_node_host + ":" + 
                std::to_string(seed_node_port));
        
        // In a real implementation, this would:
        // 1. Connect to the seed node
        // 2. Exchange cluster information
        // 3. Register this node with the cluster
        // 4. Synchronize cluster state
        
        // For now, we'll just simulate joining
        ClusterNode new_node(node_id_, host_, port_, "follower");
        new_node.is_alive = true;
        new_node.last_heartbeat = static_cast<int>(std::time(nullptr));
        new_node.load_factor = 0.0;
        new_node.last_seen = std::chrono::steady_clock::now();
        
        // Add this node to the cluster state
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            cluster_state_.nodes.push_back(new_node);
        }
        
        LOG_INFO(logger_, "Successfully joined cluster via seed node " + seed_node_host + ":" + 
                std::to_string(seed_node_port));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in join_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to join cluster: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::leave_cluster() {
    try {
        LOG_INFO(logger_, "Leaving cluster");
        
        // In a real implementation, this would:
        // 1. Notify other nodes that this node is leaving
        // 2. Transfer responsibilities to other nodes
        // 3. Clean up cluster state
        
        // For now, we'll just remove this node from the cluster state
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            cluster_state_.nodes.erase(
                std::remove_if(cluster_state_.nodes.begin(), cluster_state_.nodes.end(),
                             [this](const ClusterNode& node) { return node.node_id == node_id_; }),
                cluster_state_.nodes.end()
            );
        }
        
        LOG_INFO(logger_, "Successfully left cluster");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in leave_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to leave cluster: " + std::string(e.what()));
    }
}

Result<ClusterState> ClusterService::get_cluster_state() const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return cluster_state_;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_cluster_state: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get cluster state: " + std::string(e.what()));
    }
}

ClusterService::ClusterRole ClusterService::get_current_role() const {
    return current_role_;
}

bool ClusterService::is_master() const {
    return current_role_ == ClusterRole::MASTER;
}

Result<ClusterNode> ClusterService::get_master_node() const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        
        if (cluster_state_.master_node_id.empty()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No master node elected");
        }
        
        // Find the master node in the cluster
        for (const auto& node : cluster_state_.nodes) {
            if (node.node_id == cluster_state_.master_node_id) {
                return node;
            }
        }
        
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Master node not found in cluster");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_master_node: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get master node: " + std::string(e.what()));
    }
}

Result<std::vector<ClusterNode>> ClusterService::get_all_nodes() const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        return cluster_state_.nodes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_all_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get all nodes: " + std::string(e.what()));
    }
}

bool ClusterService::is_node_in_cluster(const std::string& node_id) const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        
        for (const auto& node : cluster_state_.nodes) {
            if (node.node_id == node_id) {
                return true;
            }
        }
        
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_node_in_cluster: " + std::string(e.what()));
        return false;
    }
}

void ClusterService::send_heartbeat() {
    try {
        LOG_DEBUG(logger_, "Sending heartbeat from node: " + node_id_);
        
        // In a real implementation, this would:
        // 1. Send heartbeat messages to all other nodes
        // 2. Include current term and node state
        // 3. Handle responses from other nodes
        
        // For now, we'll just update our own last heartbeat time
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            for (auto& node : cluster_state_.nodes) {
                if (node.node_id == node_id_) {
                    node.last_heartbeat = static_cast<int>(std::time(nullptr));
                    node.last_seen = std::chrono::steady_clock::now();
                    break;
                }
            }
        }
        
        LOG_DEBUG(logger_, "Heartbeat sent successfully from node: " + node_id_);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_heartbeat: " + std::string(e.what()));
    }
}

void ClusterService::receive_heartbeat(const std::string& from_node_id, int term) {
    try {
        LOG_DEBUG(logger_, "Received heartbeat from node: " + from_node_id + 
                 " with term: " + std::to_string(term));
        
        // Update our term if necessary
        if (term > current_term_) {
            current_term_ = term;
            if (current_role_ != ClusterRole::FOLLOWER) {
                current_role_ = ClusterRole::FOLLOWER;
                LOG_INFO(logger_, "Stepped down to follower due to higher term from node: " + from_node_id);
            }
        }
        
        // Reset election timeout since we received a heartbeat
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            cluster_state_.last_election_time = std::chrono::steady_clock::now();
        }
        
        // Update the sender's last seen time
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            for (auto& node : cluster_state_.nodes) {
                if (node.node_id == from_node_id) {
                    node.last_heartbeat = static_cast<int>(std::time(nullptr));
                    node.last_seen = std::chrono::steady_clock::now();
                    node.is_alive = true;
                    break;
                }
            }
        }
        
        LOG_DEBUG(logger_, "Processed heartbeat from node: " + from_node_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in receive_heartbeat: " + std::string(e.what()));
    }
}

void ClusterService::trigger_election() {
    try {
        LOG_INFO(logger_, "Triggering election for node: " + node_id_);
        
        if (current_role_ == ClusterRole::MASTER) {
            LOG_DEBUG(logger_, "Node is already master, ignoring election trigger");
            return;
        }
        
        // Increment term and become candidate
        current_term_++;
        current_role_ = ClusterRole::CANDIDATE;
        voted_for_ = node_id_;
        votes_received_.clear();
        votes_received_[node_id_] = current_term_; // Vote for ourselves
        
        LOG_INFO(logger_, "Node " + node_id_ + " became candidate for term " + std::to_string(current_term_));
        
        // Request votes from other nodes
        perform_leader_election();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in trigger_election: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::request_vote(const std::string& candidate_id, int candidate_term, 
                                       int last_log_index, int last_log_term) {
    try {
        LOG_DEBUG(logger_, "Vote request received from candidate: " + candidate_id + 
                 " for term: " + std::to_string(candidate_term));
        
        // Check if we've already voted in this term
        if (votes_received_.find(candidate_id) != votes_received_.end() && 
            votes_received_.at(candidate_id) == candidate_term) {
            LOG_DEBUG(logger_, "Already voted for candidate " + candidate_id + " in term " + 
                     std::to_string(candidate_term));
            return true;
        }
        
        // Check if candidate's term is at least as large as ours
        if (candidate_term < current_term_) {
            LOG_DEBUG(logger_, "Rejecting vote request from candidate " + candidate_id + 
                     " - candidate term " + std::to_string(candidate_term) + 
                     " is less than current term " + std::to_string(current_term_));
            return false;
        }
        
        // Update our term if necessary
        if (candidate_term > current_term_) {
            current_term_ = candidate_term;
            voted_for_ = ""; // Reset our vote
            if (current_role_ != ClusterRole::FOLLOWER) {
                current_role_ = ClusterRole::FOLLOWER;
                LOG_INFO(logger_, "Stepped down to follower due to higher term from candidate: " + candidate_id);
            }
        }
        
        // Grant vote if we haven't voted yet or we're voting for this candidate
        if (voted_for_.empty() || voted_for_ == candidate_id) {
            voted_for_ = candidate_id;
            votes_received_[candidate_id] = candidate_term;
            LOG_DEBUG(logger_, "Granted vote to candidate: " + candidate_id + 
                     " for term: " + std::to_string(candidate_term));
            return true;
        }
        
        LOG_DEBUG(logger_, "Rejecting vote request from candidate: " + candidate_id + 
                 " - already voted for: " + voted_for_);
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in request_vote: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to process vote request: " + std::string(e.what()));
    }
}

void ClusterService::handle_node_failure(const std::string& node_id) {
    try {
        LOG_WARN(logger_, "Handling failure of node: " + node_id);
        
        // Mark node as failed
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            for (auto& node : cluster_state_.nodes) {
                if (node.node_id == node_id) {
                    node.is_alive = false;
                    LOG_INFO(logger_, "Marked node " + node_id + " as failed");
                    break;
                }
            }
        }
        
        // If this was the master node, trigger a new election
        {
            std::shared_lock<std::shared_mutex> lock(state_mutex_);
            if (cluster_state_.master_node_id == node_id && current_role_ != ClusterRole::MASTER) {
                LOG_INFO(logger_, "Master node failed, triggering new election");
                trigger_election();
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_node_failure: " + std::string(e.what()));
    }
}

void ClusterService::update_node_info(const ClusterNode& node) {
    try {
        LOG_DEBUG(logger_, "Updating node info for node: " + node.node_id);
        
        std::lock_guard<std::shared_mutex> lock(state_mutex_);
        bool found = false;
        
        // Update existing node or add new node
        for (auto& existing_node : cluster_state_.nodes) {
            if (existing_node.node_id == node.node_id) {
                existing_node = node;
                found = true;
                break;
            }
        }
        
        if (!found) {
            cluster_state_.nodes.push_back(node);
            LOG_DEBUG(logger_, "Added new node to cluster: " + node.node_id);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_node_info: " + std::string(e.what()));
    }
}

void ClusterService::perform_leader_election() {
    try {
        LOG_INFO(logger_, "Performing leader election for term: " + std::to_string(current_term_));
        
        // Count votes (including our own)
        int vote_count = static_cast<int>(votes_received_.size());
        int total_nodes = 0;
        
        {
            std::shared_lock<std::shared_mutex> lock(state_mutex_);
            total_nodes = static_cast<int>(cluster_state_.nodes.size());
        }
        
        // If we have majority of votes, become leader
        if (vote_count > total_nodes / 2) {
            current_role_ = ClusterRole::MASTER;
            cluster_state_.master_node_id = node_id_;
            
            {
                std::lock_guard<std::shared_mutex> lock(state_mutex_);
                cluster_state_.term = current_term_;
            }
            
            LOG_INFO(logger_, "Node " + node_id_ + " elected as master for term " + 
                    std::to_string(current_term_) + " with " + std::to_string(vote_count) + 
                    " votes out of " + std::to_string(total_nodes));
            
            // Notify other nodes of our leadership
            // In a real implementation, we would send messages to all nodes
        } else {
            LOG_DEBUG(logger_, "Insufficient votes for leadership: " + std::to_string(vote_count) + 
                     "/" + std::to_string(total_nodes / 2 + 1));
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in perform_leader_election: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::check_cluster_health() const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        
        // Check if we have any nodes
        if (cluster_state_.nodes.empty()) {
            LOG_WARN(logger_, "Cluster has no nodes");
            return false;
        }
        
        // Check how many nodes are alive
        int alive_nodes = 0;
        for (const auto& node : cluster_state_.nodes) {
            if (node.is_alive) {
                alive_nodes++;
            }
        }
        
        // Cluster is healthy if majority of nodes are alive
        bool is_healthy = alive_nodes > static_cast<int>(cluster_state_.nodes.size()) / 2;
        
        LOG_DEBUG(logger_, "Cluster health check: " + std::to_string(alive_nodes) + 
                 "/" + std::to_string(cluster_state_.nodes.size()) + " nodes alive, healthy: " + 
                 (is_healthy ? "true" : "false"));
        
        return is_healthy;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_cluster_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check cluster health: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, std::string>> ClusterService::get_cluster_stats() const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        
        std::unordered_map<std::string, std::string> stats;
        stats["node_id"] = node_id_;
        stats["role"] = [this]() -> std::string {
            switch (current_role_) {
                case ClusterRole::MASTER: return "master";
                case ClusterRole::WORKER: return "worker";
                case ClusterRole::CANDIDATE: return "candidate";
                case ClusterRole::FOLLOWER: return "follower";
                default: return "unknown";
            }
        }();
        stats["term"] = std::to_string(current_term_);
        stats["total_nodes"] = std::to_string(cluster_state_.nodes.size());
        
        int alive_nodes = 0;
        for (const auto& node : cluster_state_.nodes) {
            if (node.is_alive) {
                alive_nodes++;
            }
        }
        stats["alive_nodes"] = std::to_string(alive_nodes);
        stats["master_node"] = cluster_state_.master_node_id;
        
        LOG_DEBUG(logger_, "Generated cluster statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_cluster_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get cluster stats: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::add_node_to_cluster(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Adding node to cluster: " + node_id);
        
        // Create a new node entry
        ClusterNode new_node;
        new_node.node_id = node_id;
        new_node.host = "unknown"; // Would be populated in a real implementation
        new_node.port = 0; // Would be populated in a real implementation
        new_node.role = "follower";
        new_node.is_alive = true;
        new_node.last_heartbeat = static_cast<int>(std::time(nullptr));
        new_node.load_factor = 0.0;
        new_node.last_seen = std::chrono::steady_clock::now();
        
        // Add the node to the cluster state
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            cluster_state_.nodes.push_back(new_node);
        }
        
        LOG_INFO(logger_, "Successfully added node to cluster: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_node_to_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add node to cluster: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::remove_node_from_cluster(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Removing node from cluster: " + node_id);
        
        // Remove the node from the cluster state
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            cluster_state_.nodes.erase(
                std::remove_if(cluster_state_.nodes.begin(), cluster_state_.nodes.end(),
                             [&node_id](const ClusterNode& node) { return node.node_id == node_id; }),
                cluster_state_.nodes.end()
            );
        }
        
        LOG_INFO(logger_, "Successfully removed node from cluster: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_node_from_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to remove node from cluster: " + std::string(e.what()));
    }
}

// Private methods

void ClusterService::initialize_node_id() {
    // Generate a unique node ID based on host, port, and timestamp
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    node_id_ = host_ + ":" + std::to_string(port_) + "-" + std::to_string(nanoseconds);
    
    LOG_DEBUG(logger_, "Initialized node ID: " + node_id_);
}

void ClusterService::start_background_threads() {
    // Start heartbeat thread
    heartbeat_running_ = true;
    heartbeat_thread_ = std::thread(&ClusterService::run_heartbeat_task, this);
    
    // Start election thread
    election_running_ = true;
    election_thread_ = std::thread(&ClusterService::run_election_task, this);
    
    LOG_DEBUG(logger_, "Started background threads for node: " + node_id_);
}

void ClusterService::stop_background_threads() {
    // Stop heartbeat thread
    if (heartbeat_thread_.joinable()) {
        heartbeat_running_ = false;
        heartbeat_thread_.join();
    }
    
    // Stop election thread
    if (election_thread_.joinable()) {
        election_running_ = false;
        election_thread_.join();
    }
    
    LOG_DEBUG(logger_, "Stopped background threads for node: " + node_id_);
}

void ClusterService::run_heartbeat_task() {
    LOG_INFO(logger_, "Heartbeat task started for node: " + node_id_);
    
    while (heartbeat_running_ && running_) {
        try {
            // Send heartbeat
            send_heartbeat();
            
            // Sleep for heartbeat interval (e.g., 1 second)
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in heartbeat task: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Prevent tight loop on error
        }
    }
    
    LOG_INFO(logger_, "Heartbeat task stopped for node: " + node_id_);
}

void ClusterService::run_election_task() {
    LOG_INFO(logger_, "Election task started for node: " + node_id_);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(150, 300); // Random election timeout between 150-300ms
    
    while (election_running_ && running_) {
        try {
            // Check if we should trigger an election (follower/candidate role and timeout)
            if (current_role_ != ClusterRole::MASTER) {
                auto now = std::chrono::steady_clock::now();
                auto last_heartbeat = std::chrono::steady_clock::time_point();
                
                {
                    std::shared_lock<std::shared_mutex> lock(state_mutex_);
                    last_heartbeat = cluster_state_.last_election_time;
                }
                
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat).count();
                
                // If we haven't received a heartbeat in a while, trigger election
                if (elapsed > 500) { // 500ms timeout for demo
                    LOG_DEBUG(logger_, "Election timeout reached, triggering election");
                    trigger_election();
                }
            }
            
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in election task: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Prevent tight loop on error
        }
    }
    
    LOG_INFO(logger_, "Election task stopped for node: " + node_id_);
}

void ClusterService::register_rpc_handlers() {
    // Register RPC handlers for various distributed operations
    // In a real implementation, this would set up actual RPC endpoints
    
    LOG_DEBUG(logger_, "Registering RPC handlers for cluster service");
    
    // Example RPC handlers (conceptual, actual implementation would involve network layer)
    // - Sharding: handlers for shard assignment, migration, etc.
    // - Replication: handlers for replication requests, status checks, etc.
    // - Query routing: handlers for routing table updates
}

void ClusterService::update_cluster_role() {
    // Updates the cluster role based on the Raft algorithm
    LOG_DEBUG(logger_, "Updating cluster role for node: " + node_id_);
    
    // In a real Raft implementation, this would involve:
    // - Checking current term and votes received
    // - Comparing log lengths with other nodes
    // - Determining the appropriate role (follower, candidate, leader)
    
    // For now, we'll update based on heartbeat/timeout behavior
    auto now = std::chrono::steady_clock::now();
    auto last_heartbeat = std::chrono::steady_clock::time_point();
    
    {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        last_heartbeat = cluster_state_.last_election_time;
    }
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat).count();
    
    // If we haven't received a heartbeat in a while, we should become a candidate
    if (elapsed > 1000 && current_role_ != ClusterRole::MASTER) {  // 1 second timeout
        if (current_role_ != ClusterRole::CANDIDATE) {
            LOG_INFO(logger_, "Becoming candidate due to heartbeat timeout");
            current_role_ = ClusterRole::CANDIDATE;
        }
    }
}

bool ClusterService::is_election_timeout() const {
    auto now = std::chrono::steady_clock::now();
    auto last_heartbeat = std::chrono::steady_clock::time_point();
    
    {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        last_heartbeat = cluster_state_.last_election_time;
    }
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat).count();
    
    // Election timeout is typically longer than heartbeat interval
    // Using 1.5 seconds as an example timeout
    return elapsed > 1500;
}

std::chrono::milliseconds ClusterService::get_election_timeout() const {
    // Return a random timeout between the configured min and max
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(config_.election_timeout_min_ms, config_.election_timeout_max_ms);
    
    int timeout_ms = dis(gen);
    return std::chrono::milliseconds(timeout_ms);
}

Result<bool> ClusterService::connect_to_seed_node(const std::string& host, int port) {
    try {
        LOG_DEBUG(logger_, "Connecting to seed node: " + host + ":" + std::to_string(port));
        
        // In a real implementation, this would establish a network connection
        // For now, we'll just simulate the connection
        
        LOG_DEBUG(logger_, "Connected to seed node: " + host + ":" + std::to_string(port));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in connect_to_seed_node: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::NETWORK_ERROR, "Failed to connect to seed node: " + std::string(e.what()));
    }
}

Result<bool> ClusterService::register_node_with_cluster() {
    try {
        LOG_DEBUG(logger_, "Registering node " + node_id_ + " with the cluster");
        
        // In a real implementation, this would send a registration message to the cluster
        // For now, we'll just add this node to our local cluster state
        
        ClusterNode new_node(node_id_, host_, port_, "follower");
        new_node.is_alive = true;
        new_node.last_heartbeat = static_cast<int>(std::time(nullptr));
        new_node.load_factor = 0.0;
        new_node.last_seen = std::chrono::steady_clock::now();
        
        {
            std::lock_guard<std::shared_mutex> lock(state_mutex_);
            // Check if node already exists
            auto it = std::find_if(cluster_state_.nodes.begin(), cluster_state_.nodes.end(),
                                [this](const ClusterNode& node) { return node.node_id == node_id_; });
            
            if (it == cluster_state_.nodes.end()) {
                cluster_state_.nodes.push_back(new_node);
            } else {
                *it = new_node; // Update existing
            }
        }
        
        LOG_DEBUG(logger_, "Node " + node_id_ + " registered with the cluster");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in register_node_with_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to register node with cluster: " + std::string(e.what()));
    }
}

Result<std::vector<ClusterNode>> ClusterService::discover_cluster_nodes() {
    try {
        LOG_DEBUG(logger_, "Discovering cluster nodes for node: " + node_id_);
        
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        std::vector<ClusterNode> nodes = cluster_state_.nodes;
        
        LOG_DEBUG(logger_, "Discovered " + std::to_string(nodes.size()) + " cluster nodes");
        return nodes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in discover_cluster_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to discover cluster nodes: " + std::string(e.what()));
    }
}

} // namespace jadevectordb