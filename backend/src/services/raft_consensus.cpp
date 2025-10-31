#include "raft_consensus.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>

namespace jadevectordb {

RaftConsensus::RaftConsensus(const std::string& server_id) 
    : server_id_(server_id), current_term_(0), commit_index_(0), last_applied_(0), 
      election_timeout_base_ms_(3000), heartbeat_interval_ms_(150), 
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
    // In a real implementation, this would send RequestVote RPCs to other nodes
    // For now, we'll just log the request
    LOG_DEBUG(logger_, "Requesting votes for term " + std::to_string(current_term_) + 
             " from server " + server_id_);
    
    // This is where we would send RequestVote RPCs to other nodes
    // and wait for responses to determine if we received a majority
    // For simulation purposes, we'll assume we get votes from a majority
    // which promotes us to leader
    become_leader();
}

void RaftConsensus::send_heartbeats() {
    // In a real implementation, this would send AppendEntries RPCs to followers
    // For now, we'll just log that heartbeats are being sent
    if (state_ == State::LEADER) {
        LOG_DEBUG(logger_, "Sending heartbeats from leader " + server_id_);
        
        // This is where we would send AppendEntries RPCs to followers
    }
}

RaftConsensus::AppendEntriesReply RaftConsensus::send_append_entries(const std::string& follower_id, 
                                                                   const AppendEntriesArgs& args) {
    // In a real implementation, this would make an RPC call to the follower
    // For now, we'll return a success reply
    AppendEntriesReply reply;
    reply.term = current_term_;
    reply.success = true;
    reply.conflict_index = 0;
    reply.conflict_term = 0;
    
    LOG_DEBUG(logger_, "Simulated sending AppendEntries to follower " + follower_id);
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
    // In a real implementation, this would check cluster membership to determine
    // how many nodes are currently alive and responsive
    // For now, we'll return true to indicate that majority is alive
    return true;
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
    // In a real implementation, this would persist the Raft state to stable storage
    // For now, we'll just log that persistence was called
    LOG_DEBUG(logger_, "Persisting Raft state for server " + server_id_ + 
             ", term: " + std::to_string(current_term_) + 
             ", log size: " + std::to_string(log_.size()));
}

void RaftConsensus::load_state() {
    // In a real implementation, this would load the Raft state from stable storage
    // For now, we'll just log that loading was attempted
    LOG_DEBUG(logger_, "Loading Raft state for server " + server_id_);
}

void RaftConsensus::update_leader_id(const std::string& new_leader_id) {
    if (leader_id_ != new_leader_id) {
        LOG_INFO(logger_, "Leader changed from " + leader_id_ + " to " + new_leader_id);
        leader_id_ = new_leader_id;
    }
}

} // namespace jadevectordb