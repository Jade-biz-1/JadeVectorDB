#ifndef JADEVECTORDB_CLUSTER_SERVICE_H
#define JADEVECTORDB_CLUSTER_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "lib/logging.h"

namespace jadevectordb {

// Represents a node in the cluster
struct ClusterNode {
    std::string node_id;
    std::string host;
    int port;
    std::string role;  // "master", "worker", "candidate", "follower"
    bool is_alive;
    int last_heartbeat;
    double load_factor;
    std::chrono::steady_clock::time_point last_seen;
    
    ClusterNode() : port(0), is_alive(false), last_heartbeat(0), load_factor(0.0) {}
    ClusterNode(const std::string& id, const std::string& h, int p, const std::string& r)
        : node_id(id), host(h), port(p), role(r), is_alive(true), 
          last_heartbeat(std::time(nullptr)), load_factor(0.0),
          last_seen(std::chrono::steady_clock::now()) {}
};

// Represents the cluster state
struct ClusterState {
    std::string master_node_id;
    std::vector<ClusterNode> nodes;
    int term;  // Current election term
    std::string voted_for;  // ID of candidate voted for in current term
    std::chrono::steady_clock::time_point last_election_time;
    
    ClusterState() : term(0) {}
};

/**
 * @brief Service for managing cluster membership and node coordination
 * 
 * This service handles cluster state, node discovery, master election,
 * and node failure detection in a distributed deployment.
 */
class ClusterService {
public:
    enum class ClusterRole {
        MASTER,
        WORKER,
        CANDIDATE,
        FOLLOWER
    };

private:
    ClusterState cluster_state_;
    mutable std::shared_mutex state_mutex_;
    std::shared_ptr<logging::Logger> logger_;
    
    std::string node_id_;
    std::string host_;
    int port_;
    ClusterRole current_role_;
    std::atomic<bool> running_;
    
    // Raft-related state
    int current_term_;
    std::string voted_for_;
    std::unordered_map<std::string, int> votes_received_;  // term -> votes
    
    // Background tasks
    std::thread heartbeat_thread_;
    std::thread election_thread_;
    std::atomic<bool> heartbeat_running_;
    std::atomic<bool> election_running_;
    
public:
    ClusterService(const std::string& host, int port);
    ~ClusterService();
    
    // Initialize the cluster service
    bool initialize();
    
    // Start the cluster service
    bool start();
    
    // Stop the cluster service
    void stop();
    
    // Join an existing cluster
    Result<bool> join_cluster(const std::string& seed_node_host, int seed_node_port);
    
    // Get current cluster state
    Result<ClusterState> get_cluster_state() const;
    
    // Get current node role
    ClusterRole get_current_role() const;
    
    // Check if current node is master
    bool is_master() const;
    
    // Get master node information
    Result<ClusterNode> get_master_node() const;
    
    // Get list of all nodes in the cluster
    Result<std::vector<ClusterNode>> get_all_nodes() const;
    
    // Check if a specific node is part of the cluster
    bool is_node_in_cluster(const std::string& node_id) const;
    
    // Send heartbeat to other nodes
    void send_heartbeat();
    
    // Receive heartbeat from other nodes
    void receive_heartbeat(const std::string& from_node_id, int term);
    
    // Trigger election process
    void trigger_election();
    
    // Request vote from other nodes
    Result<bool> request_vote(const std::string& candidate_id, int candidate_term, 
                            int last_log_index, int last_log_term);
    
    // Handle node failure detection
    void handle_node_failure(const std::string& node_id);
    
    // Update node information
    void update_node_info(const ClusterNode& node);
    
    // Perform leader election using Raft algorithm
    void perform_leader_election();
    
    // Check cluster health
    Result<bool> check_cluster_health() const;
    
    // Get cluster statistics
    Result<std::unordered_map<std::string, std::string>> get_cluster_stats() const;

private:
    // Initialize node ID
    void initialize_node_id();
    
    // Start background threads
    void start_background_threads();
    
    // Stop background threads
    void stop_background_threads();
    
    // Background heartbeat task
    void run_heartbeat_task();
    
    // Background election task
    void run_election_task();
    
    // Send RPC request to another node
    template<typename RequestType, typename ResponseType>
    Result<ResponseType> send_rpc_request(const std::string& node_id, 
                                        const RequestType& request);
    
    // Process incoming RPC requests
    void register_rpc_handlers();
    
    // Update cluster role based on Raft algorithm
    void update_cluster_role();
    
    // Check if election timeout has occurred
    bool is_election_timeout() const;
    
    // Get random election timeout
    std::chrono::milliseconds get_election_timeout() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_CLUSTER_SERVICE_H