#include "raft_consensus.h"
#include "cluster_service.h"
#include "api/grpc/distributed_master_client.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace jadevectordb {

RaftConsensus::RaftConsensus(const std::string& server_id)
    : server_id_(server_id), current_term_(0), commit_index_(0), last_applied_(0),
      election_timeout_base_ms_(3000), heartbeat_interval_ms_(150),
      snapshot_log_threshold_(10000),
      running_(false), state_(State::FOLLOWER), leader_id_("") {
    logger_ = logging::LoggerManager::get_logger("RaftConsensus");
    last_heartbeat_time_ = std::chrono::steady_clock::now();
    last_election_time_ = last_heartbeat_time_;
}

RaftConsensus::RaftConsensus(const std::string& server_id, std::shared_ptr<ClusterService> cluster_service)
    : server_id_(server_id), cluster_service_(cluster_service), current_term_(0), commit_index_(0), last_applied_(0),
      election_timeout_base_ms_(3000), heartbeat_interval_ms_(150),
      snapshot_log_threshold_(10000),
      running_(false), state_(State::FOLLOWER), leader_id_("") {
    logger_ = logging::LoggerManager::get_logger("RaftConsensus");
    last_heartbeat_time_ = std::chrono::steady_clock::now();
    last_election_time_ = last_heartbeat_time_;
}

RaftConsensus::RaftConsensus(const std::string& server_id,
                            std::shared_ptr<ClusterService> cluster_service,
                            std::shared_ptr<DistributedMasterClient> master_client)
    : server_id_(server_id), cluster_service_(cluster_service), master_client_(master_client),
      current_term_(0), commit_index_(0), last_applied_(0),
      election_timeout_base_ms_(3000), heartbeat_interval_ms_(150),
      snapshot_log_threshold_(10000),
      running_(false), state_(State::FOLLOWER), leader_id_("") {
    logger_ = logging::LoggerManager::get_logger("RaftConsensus");
    last_heartbeat_time_ = std::chrono::steady_clock::now();
    last_election_time_ = last_heartbeat_time_;
}

bool RaftConsensus::initialize() {
    try {
        LOG_INFO(logger_, "Initializing RaftConsensus for server: " + server_id_);

        // Load persisted state if available
        load_state();

        // Load snapshot if available
        load_snapshot();

        LOG_INFO(logger_, "RaftConsensus initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in RaftConsensus::initialize: " + std::string(e.what()));
        return false;
    }
}

void RaftConsensus::start() {
    try {
        LOG_INFO(logger_, "Starting RaftConsensus for server: " + server_id_);
        
        running_ = true;
        
        // Start background threads for heartbeat and election
        std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                if (state_ == State::LEADER) {
                    if (is_heartbeat_timeout()) {
                        send_heartbeats();
                        reset_heartbeat_timer();
                    }

                    // Check if we need to create a snapshot for log compaction
                    if (should_compact_log()) {
                        LOG_INFO(logger_, "Log size exceeds threshold, creating snapshot");
                        auto snapshot_result = create_snapshot();
                        if (snapshot_result.has_value()) {
                            // Compact the log after successful snapshot
                            compact_log_to_snapshot();
                        } else {
                            LOG_WARN(logger_, "Failed to create snapshot: " +
                                    snapshot_result.error().message);
                        }
                    }
                } else {
                    if (is_election_timeout()) {
                        handle_election_timeout();
                    }
                }
            }
        }).detach();
        
        LOG_INFO(logger_, "RaftConsensus started successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in RaftConsensus::start: " + std::string(e.what()));
    }
}

void RaftConsensus::stop() {
    try {
        LOG_INFO(logger_, "Stopping RaftConsensus for server: " + server_id_);
        running_ = false;
        LOG_INFO(logger_, "RaftConsensus stopped successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in RaftConsensus::stop: " + std::string(e.what()));
    }
}

RaftConsensus::State RaftConsensus::get_state() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return state_;
}

int RaftConsensus::get_current_term() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return current_term_;
}

bool RaftConsensus::is_leader() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return state_ == State::LEADER;
}

std::string RaftConsensus::get_leader_id() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return leader_id_;
}

std::string RaftConsensus::get_server_id() const {
    return server_id_;
}

RaftConsensus::RequestVoteReply RaftConsensus::handle_request_vote(const RequestVoteArgs& args) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    LOG_DEBUG(logger_, "Handling RequestVote from server " + args.candidate_id + 
             " for term " + std::to_string(args.term));
    
    RequestVoteReply reply;
    reply.term = current_term_;
    reply.vote_granted = false;
    
    // Update term if candidate's term is higher
    if (args.term > current_term_) {
        current_term_ = args.term;
        voted_for_ = "";
        become_follower(current_term_);
        persist_state();
    }
    
    // Grant vote if:
    // 1. We haven't voted for this term, OR we've already voted for this candidate
    // 2. The candidate's log is at least as up-to-date as ours
    bool candidate_log_ok = args.last_log_term > log_.empty() ? 0 : log_.back().term ||
                           (args.last_log_term == (log_.empty() ? 0 : log_.back().term) && 
                            args.last_log_index >= static_cast<int>(log_.size()));
    
    if ((voted_for_.empty() || voted_for_ == args.candidate_id) && candidate_log_ok) {
        voted_for_ = args.candidate_id;
        reply.vote_granted = true;
        reset_election_timer();  // Reset election timer since we received a valid request
        LOG_DEBUG(logger_, "Granted vote to candidate " + args.candidate_id + " for term " + std::to_string(args.term));
    } else {
        LOG_DEBUG(logger_, "Rejected vote for candidate " + args.candidate_id + " for term " + std::to_string(args.term));
    }
    
    reply.term = current_term_;
    return reply;
}

RaftConsensus::AppendEntriesReply RaftConsensus::handle_append_entries(const AppendEntriesArgs& args) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    LOG_DEBUG(logger_, "Handling AppendEntries from leader " + args.leader_id + 
             " for term " + std::to_string(args.term));
    
    AppendEntriesReply reply;
    reply.term = current_term_;
    reply.success = false;
    reply.conflict_index = 0;
    reply.conflict_term = 0;
    
    // Update term if leader's term is higher
    if (args.term > current_term_) {
        current_term_ = args.term;
        voted_for_ = "";
        become_follower(args.term);
        update_leader_id(args.leader_id);
        persist_state();
    } else if (args.term < current_term_) {
        // Leader is behind, return false
        reply.term = current_term_;
        reply.success = false;
        return reply;
    }
    
    // Reset election timer since we received a valid AppendEntries
    reset_election_timer();
    update_leader_id(args.leader_id);
    
    // Check if the previous log entry matches
    if (args.prev_log_index > static_cast<int>(log_.size())) {
        reply.conflict_index = static_cast<int>(log_.size()) + 1;
        reply.success = false;
        LOG_DEBUG(logger_, "AppendEntries failed: prev_log_index " + std::to_string(args.prev_log_index) + 
                 " is beyond log size " + std::to_string(log_.size()));
        return reply;
    }
    
    // If prev_log_index is 0, we're at the beginning of the log
    if (args.prev_log_index > 0) {
        if (static_cast<int>(log_.size()) < args.prev_log_index || 
            log_[args.prev_log_index - 1].term != args.prev_log_term) {
            // Conflict found
            reply.conflict_index = args.prev_log_index;
            if (args.prev_log_index <= static_cast<int>(log_.size())) {
                reply.conflict_term = log_[args.prev_log_index - 1].term;
            }
            reply.success = false;
            LOG_DEBUG(logger_, "AppendEntries failed: log mismatch at index " + std::to_string(args.prev_log_index));
            return reply;
        }
    }
    
    // Append new entries
    log_.erase(log_.begin() + args.prev_log_index, log_.end());
    log_.insert(log_.end(), args.entries.begin(), args.entries.end());
    
    // Update commit index if leader's commit index is greater
    if (args.leader_commit > commit_index_) {
        commit_index_ = std::min(args.leader_commit, static_cast<int>(log_.size()));
    }
    
    reply.term = current_term_;
    reply.success = true;
    
    LOG_DEBUG(logger_, "AppendEntries succeeded: added " + std::to_string(args.entries.size()) + 
             " entries, new log size: " + std::to_string(log_.size()));
    return reply;
}

Result<bool> RaftConsensus::add_command(const std::string& command_type, const std::string& command_data) {
    try {
        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        
        if (state_ != State::LEADER) {
            LOG_WARN(logger_, "Cannot add command: server is not the leader");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Server is not the leader");
        }
        
        // Create a new log entry
        LogEntry new_entry(current_term_, command_type, command_data);
        log_.push_back(new_entry);
        
        LOG_DEBUG(logger_, "Added command to log: " + command_type + 
                 ", log size is now " + std::to_string(log_.size()));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_command: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add command: " + std::string(e.what()));
    }
}

std::vector<LogEntry> RaftConsensus::get_log() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return log_;
}

Result<LogEntry> RaftConsensus::get_log_entry(int index) const {
    try {
        std::shared_lock<std::shared_mutex> lock(state_mutex_);
        
        if (index <= 0 || index > static_cast<int>(log_.size())) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Log entry not found at index " + std::to_string(index));
        }
        
        return log_[index - 1];  // Convert from 1-based to 0-based index
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_log_entry: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get log entry: " + std::string(e.what()));
    }
}

void RaftConsensus::apply_committed_entries() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    // Apply entries from last_applied up to commit_index_
    while (last_applied_ < commit_index_) {
        ++last_applied_;
        
        // Apply the log entry to the state machine
        apply_log_entry(log_[last_applied_ - 1]);
    }
}

int RaftConsensus::get_commit_index() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return commit_index_;
}

int RaftConsensus::get_last_applied() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return last_applied_;
}

void RaftConsensus::handle_election_timeout() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    LOG_DEBUG(logger_, "Election timeout for server: " + server_id_);
    
    // Convert to candidate and start election
    become_candidate();
    
    // Request votes from other nodes
    request_votes();
}

void RaftConsensus::handle_heartbeat_timeout() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    if (state_ == State::LEADER) {
        send_heartbeats();
        reset_heartbeat_timer();
    }
}

void RaftConsensus::become_follower(int new_term) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    LOG_INFO(logger_, "Server " + server_id_ + " becoming follower for term " + std::to_string(new_term));
    
    state_ = State::FOLLOWER;
    current_term_ = new_term;
    voted_for_ = "";
    leader_id_ = "";
    
    reset_election_timer();
    persist_state();
}

void RaftConsensus::become_candidate() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    LOG_INFO(logger_, "Server " + server_id_ + " becoming candidate for term " + std::to_string(current_term_ + 1));
    
    state_ = State::CANDIDATE;
    current_term_++;
    voted_for_ = server_id_;
    leader_id_ = "";
    
    reset_election_timer();
    persist_state();
}

void RaftConsensus::become_leader() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    LOG_INFO(logger_, "Server " + server_id_ + " becoming leader for term " + std::to_string(current_term_));
    
    state_ = State::LEADER;
    leader_id_ = server_id_;
    
    // Initialize next_index and match_index for all servers
    // In a real implementation, these would be initialized based on cluster configuration
    reset_heartbeat_timer();
    persist_state();
}

void RaftConsensus::request_votes() {
    LOG_INFO(logger_, "Requesting votes for term " + std::to_string(current_term_) +
             " from server " + server_id_);

    // If no cluster service, fall back to auto-promotion (for testing)
    if (!cluster_service_) {
        LOG_WARN(logger_, "No ClusterService available, auto-promoting to leader");
        become_leader();
        return;
    }

    // Get all nodes in the cluster
    auto nodes_result = cluster_service_->get_all_nodes();
    if (!nodes_result.has_value()) {
        LOG_ERROR(logger_, "Failed to get cluster nodes: " + nodes_result.error().message);
        return;
    }

    auto nodes = nodes_result.value();
    if (nodes.empty()) {
        LOG_WARN(logger_, "No other nodes in cluster, auto-promoting to leader");
        become_leader();
        return;
    }

    // Prepare RequestVote arguments
    RequestVoteArgs args;
    args.term = current_term_;
    args.candidate_id = server_id_;
    args.last_log_index = static_cast<int>(log_.size());
    args.last_log_term = log_.empty() ? 0 : log_.back().term;

    // Track votes received
    int votes_received = 1;  // Vote for self
    int votes_needed = (nodes.size() + 1) / 2 + 1;  // Majority (+1 for self)

    LOG_DEBUG(logger_, "Sending RequestVote to " + std::to_string(nodes.size()) +
             " nodes, need " + std::to_string(votes_needed) + " votes");

    // Send RequestVote RPCs to all other nodes in parallel
    std::vector<std::thread> vote_threads;
    std::mutex votes_mutex;

    for (const auto& node : nodes) {
        // Skip self
        if (node.node_id == server_id_) {
            continue;
        }

        // Skip dead nodes
        if (!node.is_alive) {
            LOG_DEBUG(logger_, "Skipping dead node: " + node.node_id);
            continue;
        }

        vote_threads.emplace_back([this, &node, &args, &votes_received, &votes_mutex]() {
            try {
                // Use ClusterService to send RequestVote RPC
                auto vote_result = cluster_service_->request_vote(
                    args.candidate_id,
                    args.term,
                    args.last_log_index,
                    args.last_log_term
                );

                if (vote_result.has_value() && vote_result.value()) {
                    std::lock_guard<std::mutex> lock(votes_mutex);
                    votes_received++;
                    LOG_DEBUG(logger_, "Received vote from node " + node.node_id +
                             ", total votes: " + std::to_string(votes_received));
                }
            } catch (const std::exception& e) {
                LOG_WARN(logger_, "Exception requesting vote from node " + node.node_id +
                        ": " + std::string(e.what()));
            }
        });
    }

    // Wait for all vote requests to complete
    for (auto& thread : vote_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Check if we won the election
    LOG_INFO(logger_, "Election results: received " + std::to_string(votes_received) +
             " votes out of " + std::to_string(nodes.size() + 1) + " total nodes");

    if (votes_received >= votes_needed) {
        LOG_INFO(logger_, "Won election for term " + std::to_string(current_term_));
        become_leader();
    } else {
        LOG_INFO(logger_, "Lost election for term " + std::to_string(current_term_) +
                ", remaining as candidate");
    }
}

void RaftConsensus::send_heartbeats() {
    if (state_ != State::LEADER) {
        return;
    }

    LOG_DEBUG(logger_, "Sending heartbeats from leader " + server_id_);

    // If no cluster service, nothing to do
    if (!cluster_service_) {
        return;
    }

    // Get all nodes in the cluster
    auto nodes_result = cluster_service_->get_all_nodes();
    if (!nodes_result.has_value()) {
        LOG_ERROR(logger_, "Failed to get cluster nodes for heartbeat: " + nodes_result.error().message);
        return;
    }

    auto nodes = nodes_result.value();

    // Send heartbeat to all followers
    for (const auto& node : nodes) {
        // Skip self
        if (node.node_id == server_id_) {
            continue;
        }

        // Skip dead nodes
        if (!node.is_alive) {
            continue;
        }

        // Prepare AppendEntries arguments (empty for heartbeat)
        AppendEntriesArgs args;
        args.term = current_term_;
        args.leader_id = server_id_;
        args.prev_log_index = static_cast<int>(log_.size());
        args.prev_log_term = log_.empty() ? 0 : log_.back().term;
        args.entries.clear();  // Empty for heartbeat
        args.leader_commit = commit_index_;

        // Send heartbeat asynchronously (don't block)
        std::thread([this, node, args]() {
            try {
                // Use ClusterService to send heartbeat
                cluster_service_->send_heartbeat();
                LOG_DEBUG(logger_, "Sent heartbeat to node " + node.node_id);
            } catch (const std::exception& e) {
                LOG_WARN(logger_, "Exception sending heartbeat to node " + node.node_id +
                        ": " + std::string(e.what()));
            }
        }).detach();
    }
}

RaftConsensus::AppendEntriesReply RaftConsensus::send_append_entries(const std::string& follower_id, 
                                                                   const AppendEntriesArgs& args) {
    AppendEntriesReply reply;
    reply.term = current_term_;
    reply.success = false;
    reply.conflict_index = 0;
    reply.conflict_term = 0;

    // Validate that we have a cluster service for RPC
    if (!cluster_service_) {
        LOG_WARN(logger_, "No ClusterService available for sending AppendEntries to " + follower_id);
        // In single-node mode, auto-succeed
        reply.success = true;
        return reply;
    }

    // Verify the target node is in the cluster and alive
    auto nodes_result = cluster_service_->get_all_nodes();
    if (!nodes_result.has_value()) {
        LOG_WARN(logger_, "Failed to get cluster nodes for AppendEntries to " + follower_id);
        return reply;
    }

    bool node_found = false;
    bool node_alive = false;
    for (const auto& node : nodes_result.value()) {
        if (node.node_id == follower_id) {
            node_found = true;
            node_alive = node.is_alive;
            break;
        }
    }

    if (!node_found) {
        LOG_WARN(logger_, "Target node not found in cluster for AppendEntries: " + follower_id);
        return reply;
    }

    if (!node_alive) {
        LOG_DEBUG(logger_, "Target node is not alive, skipping AppendEntries: " + follower_id);
        return reply;
    }

    LOG_DEBUG(logger_, "Sending AppendEntries to follower " + follower_id +
             " with " + std::to_string(args.entries.size()) + " entries for term " +
             std::to_string(args.term));

    // If we have a master client, use it for real RPC calls
    if (master_client_) {
        try {
            // Convert our args to DistributedMasterClient format
            DistributedMasterClient::AppendEntriesRequest request;
            request.term = args.term;
            request.leader_id = args.leader_id;
            request.prev_log_index = args.prev_log_index;
            request.prev_log_term = args.prev_log_term;
            request.leader_commit_index = args.leader_commit;

            // Convert log entries
            for (const auto& entry : args.entries) {
                DistributedMasterClient::LogEntry log_entry;
                log_entry.term = entry.term;
                log_entry.index = 0;  // Will be determined by receiver
                log_entry.type = 5;  // LOG_WRITE_OPERATION
                log_entry.data = entry.command_data;
                log_entry.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    entry.timestamp.time_since_epoch()
                ).count();
                request.entries.push_back(log_entry);
            }

            // Make the RPC call
            auto result = master_client_->append_entries(follower_id, request);

            if (result.has_value()) {
                reply.term = result.value().term;
                reply.success = result.value().success;
                reply.conflict_index = 0;  // Not used in current implementation
                reply.conflict_term = 0;

                LOG_DEBUG(logger_, "AppendEntries RPC to " + follower_id +
                         " returned: " + (reply.success ? "success" : "failure"));
            } else {
                LOG_WARN(logger_, "AppendEntries RPC to " + follower_id + " failed: " +
                        result.error().message);
                reply.success = false;
            }
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in send_append_entries to " + follower_id +
                     ": " + std::string(e.what()));
            reply.success = false;
        }
    } else {
        // No master client available - simulation mode for testing
        LOG_DEBUG(logger_, "No DistributedMasterClient available, simulating AppendEntries success");
        reply.success = true;
        reply.term = args.term;
    }

    return reply;
}

void RaftConsensus::update_node_match_index(const std::string& node_id, int match_index) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    
    if (state_ == State::LEADER) {
        match_index_[node_id] = match_index;
        
        // Update commit index based on match indices
        std::vector<int> match_indices;
        match_indices.reserve(match_index_.size() + 1); // +1 for the leader
        
        // Add match index for each node
        for (const auto& pair : match_index_) {
            match_indices.push_back(pair.second);
        }
        
        // Add leader's own match index (log size)
        match_indices.push_back(static_cast<int>(log_.size()));
        
        // Sort to find the median
        std::sort(match_indices.begin(), match_indices.end());
        
        // Calculate the majority index
        int majority_index = match_indices[match_indices.size() / 2];
        
        // Update commit index if the majority index is greater and corresponds to a log entry from the current term
        if (majority_index > commit_index_) {
            if (majority_index <= static_cast<int>(log_.size()) && 
                log_[majority_index - 1].term == current_term_) {
                commit_index_ = majority_index;
            }
        }
    }
}

bool RaftConsensus::has_majority_replicated(int log_index) const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    
    // Count how many nodes have replicated the log entry at log_index
    int replicated_count = 1; // Count this node
    
    for (const auto& match_pair : match_index_) {
        if (match_pair.second >= log_index) {
            replicated_count++;
        }
    }
    
    // For simplicity, assuming cluster size is match_index_.size() + 1 (this node)
    int total_nodes = match_index_.size() + 1;
    int majority = total_nodes / 2 + 1;
    
    return replicated_count >= majority;
}

bool RaftConsensus::has_majority_alive() const {
    // If no cluster service, assume majority is alive (for testing)
    if (!cluster_service_) {
        return true;
    }

    // Get all nodes in the cluster
    auto nodes_result = cluster_service_->get_all_nodes();
    if (!nodes_result.has_value()) {
        LOG_WARN(logger_, "Failed to get cluster nodes to check majority");
        return false;
    }

    auto nodes = nodes_result.value();
    if (nodes.empty()) {
        // Single node cluster, always has majority
        return true;
    }

    // Count alive nodes
    int alive_count = 0;
    for (const auto& node : nodes) {
        if (node.is_alive) {
            alive_count++;
        }
    }

    // Need majority of total nodes (including self if not in list)
    int total_nodes = nodes.size();
    bool self_found = false;
    for (const auto& node : nodes) {
        if (node.node_id == server_id_) {
            self_found = true;
            break;
        }
    }
    if (!self_found) {
        total_nodes++;  // Add self to count
        alive_count++;  // Self is alive
    }

    int majority = (total_nodes / 2) + 1;
    bool has_majority = alive_count >= majority;

    LOG_DEBUG(logger_, "Cluster has " + std::to_string(alive_count) +
             " alive nodes out of " + std::to_string(total_nodes) +
             ", majority requires " + std::to_string(majority) +
             ", has majority: " + (has_majority ? "yes" : "no"));

    return has_majority;
}

void RaftConsensus::reset_election_timer() {
    last_election_time_ = std::chrono::steady_clock::now();
}

void RaftConsensus::reset_heartbeat_timer() {
    last_heartbeat_time_ = std::chrono::steady_clock::now();
}

bool RaftConsensus::is_election_timeout() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_election_time_);
    auto random_timeout = generate_random_timeout(election_timeout_base_ms_, election_timeout_base_ms_ * 2);
    return elapsed.count() > random_timeout;
}

bool RaftConsensus::is_heartbeat_timeout() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat_time_);
    return elapsed.count() > heartbeat_interval_ms_;
}

// Private methods

int RaftConsensus::generate_random_timeout(int min_ms, int max_ms) const {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_ms, max_ms);
    return dis(gen);
}

void RaftConsensus::apply_log_entry(const LogEntry& entry) {
    // Apply the log entry to the state machine
    LOG_DEBUG(logger_, "Applying log entry to state machine: " + entry.command_type + 
             " (term " + std::to_string(entry.term) + ")");
    
    // In a real implementation, this would apply the command to the system's state
    // For example, if it's a "store_vector" command, it would store the vector in the database
    // If it's a "delete_vector" command, it would delete the vector from the database
}

void RaftConsensus::persist_state() {
    try {
        // Create state directory if it doesn't exist
        std::string state_dir = "/tmp/jadevectordb/raft/" + server_id_;
        try {
            std::filesystem::create_directories(state_dir);
        } catch (const std::exception& e) {
            LOG_WARN(logger_, "Failed to create state directory: " + state_dir + " - " + e.what());
            return;
        }

        // Persist to file
        std::string state_file = state_dir + "/raft_state.dat";
        std::ofstream ofs(state_file, std::ios::binary);
        if (!ofs.is_open()) {
            LOG_ERROR(logger_, "Failed to open state file for writing: " + state_file);
            return;
        }

        // Write current term
        ofs.write(reinterpret_cast<const char*>(&current_term_), sizeof(current_term_));

        // Write voted_for (length + string)
        size_t voted_for_len = voted_for_.size();
        ofs.write(reinterpret_cast<const char*>(&voted_for_len), sizeof(voted_for_len));
        ofs.write(voted_for_.data(), voted_for_len);

        // Write log size
        size_t log_size = log_.size();
        ofs.write(reinterpret_cast<const char*>(&log_size), sizeof(log_size));

        // Write each log entry
        for (const auto& entry : log_) {
            ofs.write(reinterpret_cast<const char*>(&entry.term), sizeof(entry.term));

            size_t cmd_type_len = entry.command_type.size();
            ofs.write(reinterpret_cast<const char*>(&cmd_type_len), sizeof(cmd_type_len));
            ofs.write(entry.command_type.data(), cmd_type_len);

            size_t cmd_data_len = entry.command_data.size();
            ofs.write(reinterpret_cast<const char*>(&cmd_data_len), sizeof(cmd_data_len));
            ofs.write(entry.command_data.data(), cmd_data_len);
        }

        ofs.close();
        LOG_DEBUG(logger_, "Persisted Raft state for server " + server_id_ +
                 ", term: " + std::to_string(current_term_) +
                 ", log size: " + std::to_string(log_.size()));
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in persist_state: " + std::string(e.what()));
    }
}

void RaftConsensus::load_state() {
    try {
        std::string state_file = "/tmp/jadevectordb/raft/" + server_id_ + "/raft_state.dat";

        // Check if file exists
        std::ifstream test_file(state_file);
        if (!test_file.good()) {
            LOG_DEBUG(logger_, "No persisted state found for server " + server_id_ + ", starting fresh");
            return;
        }
        test_file.close();

        // Load from file
        std::ifstream ifs(state_file, std::ios::binary);
        if (!ifs.is_open()) {
            LOG_ERROR(logger_, "Failed to open state file for reading: " + state_file);
            return;
        }

        // Read current term
        ifs.read(reinterpret_cast<char*>(&current_term_), sizeof(current_term_));

        // Read voted_for
        size_t voted_for_len = 0;
        ifs.read(reinterpret_cast<char*>(&voted_for_len), sizeof(voted_for_len));
        voted_for_.resize(voted_for_len);
        ifs.read(&voted_for_[0], voted_for_len);

        // Read log size
        size_t log_size = 0;
        ifs.read(reinterpret_cast<char*>(&log_size), sizeof(log_size));

        // Read each log entry
        log_.clear();
        log_.reserve(log_size);
        for (size_t i = 0; i < log_size; ++i) {
            LogEntry entry;
            ifs.read(reinterpret_cast<char*>(&entry.term), sizeof(entry.term));

            size_t cmd_type_len = 0;
            ifs.read(reinterpret_cast<char*>(&cmd_type_len), sizeof(cmd_type_len));
            entry.command_type.resize(cmd_type_len);
            ifs.read(&entry.command_type[0], cmd_type_len);

            size_t cmd_data_len = 0;
            ifs.read(reinterpret_cast<char*>(&cmd_data_len), sizeof(cmd_data_len));
            entry.command_data.resize(cmd_data_len);
            ifs.read(&entry.command_data[0], cmd_data_len);

            entry.timestamp = std::chrono::steady_clock::now();
            log_.push_back(entry);
        }

        ifs.close();
        LOG_INFO(logger_, "Loaded Raft state for server " + server_id_ +
                ", term: " + std::to_string(current_term_) +
                ", log size: " + std::to_string(log_.size()));
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in load_state: " + std::string(e.what()));
    }
}

void RaftConsensus::update_leader_id(const std::string& new_leader_id) {
    if (leader_id_ != new_leader_id) {
        LOG_INFO(logger_, "Leader changed from " + leader_id_ + " to " + new_leader_id);
        leader_id_ = new_leader_id;
    }
}

// ===== Snapshot Operations =====

Result<bool> RaftConsensus::create_snapshot() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);

    try {
        // Can't create snapshot if we haven't committed anything yet
        if (commit_index_ == 0) {
            return tl::make_unexpected(ErrorHandler::create_error(
                ErrorCode::INVALID_STATE,
                "Cannot create snapshot: no committed entries"
            ));
        }

        LOG_INFO(logger_, "Creating snapshot at commit index " + std::to_string(commit_index_));

        // Create snapshot metadata
        SnapshotMetadata metadata;
        metadata.last_included_index = commit_index_;
        metadata.last_included_term = (commit_index_ <= static_cast<int>(log_.size()))
                                      ? log_[commit_index_ - 1].term
                                      : last_snapshot_.metadata.last_included_term;

        // Get cluster members from cluster service
        if (cluster_service_) {
            auto nodes_result = cluster_service_->get_all_nodes();
            if (nodes_result.has_value()) {
                for (const auto& node : nodes_result.value()) {
                    metadata.cluster_members.push_back(node.node_id);
                }
            }
        }

        // Serialize committed log entries as state machine data
        // In a real implementation, this would serialize the actual state machine state
        // For now, we'll just serialize the committed log entries
        std::vector<uint8_t> snapshot_data;
        std::stringstream ss;

        // Write snapshot header
        ss << "RAFT_SNAPSHOT_V1\n";
        ss << "commit_index:" << commit_index_ << "\n";
        ss << "num_entries:" << commit_index_ << "\n";

        // Write log entries (up to commit_index)
        int entries_to_write = std::min(commit_index_, static_cast<int>(log_.size()));
        for (int i = 0; i < entries_to_write; i++) {
            const auto& entry = log_[i];
            ss << "ENTRY\n";
            ss << "term:" << entry.term << "\n";
            ss << "type:" << entry.command_type << "\n";
            ss << "data_len:" << entry.command_data.length() << "\n";
            ss.write(entry.command_data.c_str(), entry.command_data.length());
            ss << "\n";
        }

        std::string snapshot_str = ss.str();
        snapshot_data.assign(snapshot_str.begin(), snapshot_str.end());

        // Create snapshot
        Snapshot snapshot;
        snapshot.metadata = metadata;
        snapshot.data = snapshot_data;

        // Persist snapshot to disk
        last_snapshot_ = snapshot;
        persist_snapshot(snapshot);

        LOG_INFO(logger_, "Snapshot created successfully with " + std::to_string(entries_to_write) +
                 " entries, size: " + std::to_string(snapshot_data.size()) + " bytes");

        return true;
    } catch (const std::exception& e) {
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::INTERNAL_ERROR,
            "Exception in create_snapshot: " + std::string(e.what())
        ));
    }
}

void RaftConsensus::persist_snapshot(const Snapshot& snapshot) {
    try {
        // Create directory if it doesn't exist
        std::string state_dir = "/tmp/jadevectordb/raft/" + server_id_;
        std::filesystem::create_directories(state_dir);

        std::string snapshot_path = state_dir + "/snapshot.dat";
        std::ofstream ofs(snapshot_path, std::ios::binary);

        if (!ofs.is_open()) {
            LOG_ERROR(logger_, "Failed to open snapshot file for writing: " + snapshot_path);
            return;
        }

        // Write snapshot metadata
        ofs.write(reinterpret_cast<const char*>(&snapshot.metadata.last_included_index), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&snapshot.metadata.last_included_term), sizeof(int));

        // Write cluster members
        size_t num_members = snapshot.metadata.cluster_members.size();
        ofs.write(reinterpret_cast<const char*>(&num_members), sizeof(size_t));
        for (const auto& member : snapshot.metadata.cluster_members) {
            size_t member_len = member.length();
            ofs.write(reinterpret_cast<const char*>(&member_len), sizeof(size_t));
            ofs.write(member.c_str(), member_len);
        }

        // Write snapshot data
        size_t data_size = snapshot.data.size();
        ofs.write(reinterpret_cast<const char*>(&data_size), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(snapshot.data.data()), data_size);

        ofs.close();
        LOG_INFO(logger_, "Persisted snapshot to " + snapshot_path);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in persist_snapshot: " + std::string(e.what()));
    }
}

bool RaftConsensus::load_snapshot() {
    try {
        std::string state_dir = "/tmp/jadevectordb/raft/" + server_id_;
        std::string snapshot_path = state_dir + "/snapshot.dat";

        std::ifstream ifs(snapshot_path, std::ios::binary);
        if (!ifs.is_open()) {
            // No snapshot file exists yet
            return false;
        }

        Snapshot snapshot;

        // Read snapshot metadata
        ifs.read(reinterpret_cast<char*>(&snapshot.metadata.last_included_index), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&snapshot.metadata.last_included_term), sizeof(int));

        // Read cluster members
        size_t num_members = 0;
        ifs.read(reinterpret_cast<char*>(&num_members), sizeof(size_t));
        snapshot.metadata.cluster_members.resize(num_members);
        for (size_t i = 0; i < num_members; i++) {
            size_t member_len = 0;
            ifs.read(reinterpret_cast<char*>(&member_len), sizeof(size_t));
            std::string member(member_len, '\0');
            ifs.read(&member[0], member_len);
            snapshot.metadata.cluster_members[i] = member;
        }

        // Read snapshot data
        size_t data_size = 0;
        ifs.read(reinterpret_cast<char*>(&data_size), sizeof(size_t));
        snapshot.data.resize(data_size);
        ifs.read(reinterpret_cast<char*>(snapshot.data.data()), data_size);

        ifs.close();

        std::unique_lock<std::shared_mutex> lock(state_mutex_);
        last_snapshot_ = snapshot;
        last_applied_ = snapshot.metadata.last_included_index;
        commit_index_ = std::max(commit_index_, snapshot.metadata.last_included_index);

        LOG_INFO(logger_, "Loaded snapshot from " + snapshot_path +
                 ", last_included_index: " + std::to_string(snapshot.metadata.last_included_index) +
                 ", size: " + std::to_string(data_size) + " bytes");

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in load_snapshot: " + std::string(e.what()));
        return false;
    }
}

RaftConsensus::InstallSnapshotReply RaftConsensus::handle_install_snapshot(const InstallSnapshotArgs& args) {
    InstallSnapshotReply reply;
    reply.term = current_term_;
    reply.success = false;

    std::unique_lock<std::shared_mutex> lock(state_mutex_);

    // Reply false if term < currentTerm
    if (args.term < current_term_) {
        LOG_DEBUG(logger_, "Rejecting InstallSnapshot from " + args.leader_id +
                 " with stale term " + std::to_string(args.term));
        return reply;
    }

    // If term > currentTerm, become follower
    if (args.term > current_term_) {
        become_follower(args.term);
    }

    // Reset election timer since we heard from the leader
    reset_election_timer();
    update_leader_id(args.leader_id);

    try {
        // If this is the first chunk, create a new snapshot
        if (args.offset == 0) {
            last_snapshot_.metadata.last_included_index = args.last_included_index;
            last_snapshot_.metadata.last_included_term = args.last_included_term;
            last_snapshot_.data.clear();
        }

        // Append data to snapshot
        last_snapshot_.data.insert(last_snapshot_.data.end(), args.data.begin(), args.data.end());

        // If this is the last chunk, finalize the snapshot
        if (args.done) {
            // Discard any existing or partial log with last_included_index
            if (args.last_included_index >= static_cast<int>(log_.size())) {
                // Snapshot contains more than our entire log
                log_.clear();
            } else {
                // Snapshot contains part of our log, keep entries after snapshot
                log_.erase(log_.begin(), log_.begin() + args.last_included_index);
            }

            // Update commit index and last applied
            commit_index_ = std::max(commit_index_, args.last_included_index);
            last_applied_ = args.last_included_index;

            // Persist the snapshot
            persist_snapshot(last_snapshot_);

            LOG_INFO(logger_, "Installed snapshot from " + args.leader_id +
                     ", last_included_index: " + std::to_string(args.last_included_index) +
                     ", size: " + std::to_string(last_snapshot_.data.size()) + " bytes");

            reply.success = true;
        }

        reply.term = current_term_;
        return reply;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_install_snapshot: " + std::string(e.what()));
        reply.success = false;
        return reply;
    }
}

RaftConsensus::InstallSnapshotReply RaftConsensus::send_install_snapshot(const std::string& follower_id,
                                                                        const InstallSnapshotArgs& args) {
    InstallSnapshotReply reply;
    reply.term = current_term_;
    reply.success = false;

    // Validate that we have a cluster service for RPC
    if (!cluster_service_) {
        LOG_WARN(logger_, "No ClusterService available for sending InstallSnapshot to " + follower_id);
        return reply;
    }

    // Verify the target node is in the cluster and alive
    auto nodes_result = cluster_service_->get_all_nodes();
    if (!nodes_result.has_value()) {
        LOG_WARN(logger_, "Failed to get cluster nodes for InstallSnapshot to " + follower_id);
        return reply;
    }

    bool node_found = false;
    bool node_alive = false;
    for (const auto& node : nodes_result.value()) {
        if (node.node_id == follower_id) {
            node_found = true;
            node_alive = node.is_alive;
            break;
        }
    }

    if (!node_found) {
        LOG_WARN(logger_, "Target node not found in cluster for InstallSnapshot: " + follower_id);
        return reply;
    }

    if (!node_alive) {
        LOG_DEBUG(logger_, "Target node is not alive, skipping InstallSnapshot: " + follower_id);
        return reply;
    }

    LOG_DEBUG(logger_, "Sending InstallSnapshot to follower " + follower_id +
             " with last_included_index " + std::to_string(args.last_included_index) +
             " for term " + std::to_string(args.term));

    // Use DistributedMasterClient for actual RPC if available
    if (master_client_) {
        DistributedMasterClient::InstallSnapshotRequest rpc_request;
        rpc_request.term = args.term;
        rpc_request.leader_id = args.leader_id;
        rpc_request.last_included_index = args.last_included_index;
        rpc_request.last_included_term = args.last_included_term;
        rpc_request.offset = args.offset;
        rpc_request.data = args.data;
        rpc_request.done = args.done;

        auto rpc_result = master_client_->install_snapshot(follower_id, rpc_request);
        if (rpc_result.has_value()) {
            reply.term = rpc_result.value().term;
            reply.success = rpc_result.value().success;
            LOG_DEBUG(logger_, "InstallSnapshot RPC to " + follower_id + " returned: " +
                     (reply.success ? "success" : "failure"));
        } else {
            LOG_WARN(logger_, "InstallSnapshot RPC to " + follower_id + " failed: " +
                    rpc_result.error().message);
            reply.success = false;
        }
    } else {
        // No master client available, simulate success for testing
        LOG_DEBUG(logger_, "No master_client available, simulating InstallSnapshot success");
        reply.success = true;
        reply.term = args.term;
    }

    return reply;
}

const Snapshot& RaftConsensus::get_last_snapshot() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return last_snapshot_;
}

bool RaftConsensus::should_compact_log() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return static_cast<int>(log_.size()) > snapshot_log_threshold_;
}

void RaftConsensus::compact_log_to_snapshot() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);

    if (last_snapshot_.metadata.last_included_index == 0) {
        LOG_DEBUG(logger_, "No snapshot available for log compaction");
        return;
    }

    try {
        // Remove all log entries up to and including the snapshot's last_included_index
        int entries_to_remove = std::min(last_snapshot_.metadata.last_included_index,
                                        static_cast<int>(log_.size()));

        if (entries_to_remove > 0) {
            log_.erase(log_.begin(), log_.begin() + entries_to_remove);
            LOG_INFO(logger_, "Compacted log by removing " + std::to_string(entries_to_remove) +
                     " entries up to index " + std::to_string(last_snapshot_.metadata.last_included_index) +
                     ", remaining log size: " + std::to_string(log_.size()));
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in compact_log_to_snapshot: " + std::string(e.what()));
    }
}

} // namespace jadevectordb