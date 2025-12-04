#ifndef JADEVECTORDB_RAFT_CONSENSUS_H
#define JADEVECTORDB_RAFT_CONSENSUS_H

#include "services/cluster_service.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

namespace jadevectordb {

// Forward declarations
class ClusterService;

// Log entry for the Raft log
struct LogEntry {
    int term;
    std::string command_type;  // "store_vector", "delete_vector", etc.
    std::string command_data;  // Serialized command data
    std::chrono::steady_clock::time_point timestamp;
    
    LogEntry() : term(0) {}
    LogEntry(int t, const std::string& cmd_type, const std::string& cmd_data)
        : term(t), command_type(cmd_type), command_data(cmd_data), 
          timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Implementation of Raft consensus algorithm for leader election and log replication
 * 
 * This class implements the Raft consensus algorithm to ensure consistency across
 * the distributed vector database cluster.
 */
class RaftConsensus {
public:
    // State of the Raft node
    enum class State {
        FOLLOWER,
        CANDIDATE,
        LEADER
    };

    // AppendEntries RPC arguments
    struct AppendEntriesArgs {
        int term;                    // Leader's term
        std::string leader_id;       // Leader's ID
        int prev_log_index;         // Index of log entry immediately preceding new ones
        int prev_log_term;          // Term of prevLogIndex entry
        std::vector<LogEntry> entries; // Log entries to store (empty for heartbeat)
        int leader_commit;          // Leader's commitIndex
    };

    // AppendEntries RPC reply
    struct AppendEntriesReply {
        int term;                   // Current term, for leader to update itself
        bool success;               // True if follower contained entry matching prevLogIndex and prevLogTerm
        int conflict_index;         // Index of conflicting entry, if any
        int conflict_term;          // Term of conflicting entry, if any
    };

    // RequestVote RPC arguments
    struct RequestVoteArgs {
        int term;                   // Candidate's term
        std::string candidate_id;   // Candidate's ID requesting vote
        int last_log_index;        // Index of candidate's last log entry
        int last_log_term;         // Term of candidate's last log entry
    };

    // RequestVote RPC reply
    struct RequestVoteReply {
        int term;                   // Current term, for candidate to update itself
        bool vote_granted;          // True means candidate received vote
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ClusterService> cluster_service_;  // For cluster membership and RPC

    // Raft state variables
    State state_;
    int current_term_;
    std::string voted_for_;
    std::vector<LogEntry> log_;
    
    // Volatile state
    int commit_index_;
    int last_applied_;
    
    // Leader state (reinitialized after election)
    std::unordered_map<std::string, int> next_index_;   // For each server, index of the next log entry to send to that server
    std::unordered_map<std::string, int> match_index_;  // For each server, index of highest log entry known to be replicated on server
    
    // Server state
    std::string server_id_;
    std::string leader_id_;
    int election_timeout_base_ms_;
    int heartbeat_interval_ms_;
    
    // Timing
    std::chrono::steady_clock::time_point last_heartbeat_time_;
    std::chrono::steady_clock::time_point last_election_time_;
    
    // Atomic flags
    std::atomic<bool> running_;
    
    // Mutex for thread safety
    mutable std::shared_mutex state_mutex_;

public:
    explicit RaftConsensus(const std::string& server_id);
    explicit RaftConsensus(const std::string& server_id, std::shared_ptr<ClusterService> cluster_service);
    ~RaftConsensus() = default;
    
    // Initialize Raft instance
    bool initialize();
    
    // Start the Raft consensus process
    void start();
    
    // Stop the Raft consensus process
    void stop();
    
    // Get current state of the Raft node
    State get_state() const;
    
    // Get current term
    int get_current_term() const;
    
    // Check if this node is the leader
    bool is_leader() const;
    
    // Get the leader ID
    std::string get_leader_id() const;
    
    // Get server ID
    std::string get_server_id() const;
    
    // Request vote RPC handler
    RequestVoteReply handle_request_vote(const RequestVoteArgs& args);
    
    // Append entries RPC handler
    AppendEntriesReply handle_append_entries(const AppendEntriesArgs& args);
    
    // Add a new command to the log (only for leader)
    Result<bool> add_command(const std::string& command_type, const std::string& command_data);
    
    // Get the current log
    std::vector<LogEntry> get_log() const;
    
    // Get log entry at specific index
    Result<LogEntry> get_log_entry(int index) const;
    
    // Apply committed entries to the state machine
    void apply_committed_entries();
    
    // Get commit index
    int get_commit_index() const;
    
    // Get last applied index
    int get_last_applied() const;
    
    // Handle election timeout
    void handle_election_timeout();
    
    // Handle heartbeat timeout (for leader)
    void handle_heartbeat_timeout();
    
    // Convert to follower state
    void become_follower(int new_term);
    
    // Convert to candidate state and start election
    void become_candidate();
    
    // Convert to leader state
    void become_leader();
    
    // Request votes from other nodes
    void request_votes();
    
    // Send heartbeat (AppendEntries) to followers
    void send_heartbeats();
    
    // Send AppendEntries RPC to a specific follower
    AppendEntriesReply send_append_entries(const std::string& follower_id, 
                                         const AppendEntriesArgs& args);
    
    // Update the node's next and match index
    void update_node_match_index(const std::string& node_id, int match_index);
    
    // Check if there's a majority of nodes that have replicated the log entry
    bool has_majority_replicated(int log_index) const;
    
    // Check if there's a majority of nodes that are alive
    bool has_majority_alive() const;
    
    // Reset election timer
    void reset_election_timer();
    
    // Reset heartbeat timer
    void reset_heartbeat_timer();
    
    // Check if election timeout has occurred
    bool is_election_timeout() const;
    
    // Check if heartbeat timeout has occurred
    bool is_heartbeat_timeout() const;

private:
    // Generate a random timeout within the specified range
    int generate_random_timeout(int min_ms, int max_ms) const;
    
    // Apply a single log entry to the state machine
    void apply_log_entry(const LogEntry& entry);
    
    // Persist Raft state to storage (for crash recovery)
    void persist_state();
    
    // Load Raft state from storage (for crash recovery)
    void load_state();
    
    // Update leader ID
    void update_leader_id(const std::string& new_leader_id);
    
    // Send RPC request to another node
    template<typename RequestType, typename ResponseType>
    Result<ResponseType> send_rpc_request(const std::string& node_id, 
                                        const RequestType& request);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_RAFT_CONSENSUS_H