#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace jadevectordb {
namespace middleware {

/**
 * @brief Tracks failed login attempts and blocks IPs
 * 
 * Automatically blocks IPs after exceeding max failed attempts.
 * Thread-safe and suitable for distributed environments.
 */
class IPBlocker {
public:
    /**
     * @brief Construct a new IPBlocker
     * 
     * @param max_failed_attempts Max failed attempts before blocking (default: 10)
     * @param block_duration_seconds Duration of block in seconds (default: 3600 = 1 hour)
     * @param failure_window_seconds Window to track failures (default: 600 = 10 minutes)
     */
    IPBlocker(size_t max_failed_attempts = 10, 
              int block_duration_seconds = 3600,
              int failure_window_seconds = 600);

    /**
     * @brief Check if IP is blocked
     * 
     * @param ip IP address to check
     * @return true if IP is currently blocked
     */
    bool is_blocked(const std::string& ip) const;

    /**
     * @brief Record a failed login attempt
     * 
     * @param ip IP address of failed attempt
     * @param reason Reason for failure (e.g., "invalid_password", "invalid_username")
     * @return true if IP is now blocked, false otherwise
     */
    bool record_failure(const std::string& ip, const std::string& reason = "");

    /**
     * @brief Record a successful login
     * 
     * Clears failure history for this IP
     * 
     * @param ip IP address of successful login
     */
    void record_success(const std::string& ip);

    /**
     * @brief Manually block an IP
     * 
     * @param ip IP address to block
     * @param duration_seconds Duration of block (0 = use default)
     * @param reason Reason for manual block
     */
    void block_ip(const std::string& ip, int duration_seconds = 0, const std::string& reason = "manual");

    /**
     * @brief Manually unblock an IP (admin action)
     * 
     * @param ip IP address to unblock
     * @return true if IP was blocked and is now unblocked
     */
    bool unblock_ip(const std::string& ip);

    /**
     * @brief Get remaining block time for IP
     * 
     * @param ip IP address to check
     * @return int Seconds remaining in block, 0 if not blocked
     */
    int remaining_block_seconds(const std::string& ip) const;

    /**
     * @brief Get failure count for IP in current window
     * 
     * @param ip IP address to check
     * @return size_t Number of failures in current window
     */
    size_t get_failure_count(const std::string& ip) const;

    /**
     * @brief Get list of blocked IPs
     * 
     * @return std::vector<std::string> List of currently blocked IPs
     */
    std::vector<std::string> get_blocked_ips() const;

    /**
     * @brief Get statistics
     * 
     * @return std::unordered_map<std::string, size_t> Stats including total_blocks, active_blocks, total_failures
     */
    std::unordered_map<std::string, size_t> get_stats() const;

    /**
     * @brief Clear all blocks and failure history
     */
    void clear_all();

private:
    struct FailureRecord {
        std::vector<std::chrono::steady_clock::time_point> failures;
        std::string last_reason;
    };

    struct BlockRecord {
        std::chrono::steady_clock::time_point block_time;
        int duration_seconds;
        std::string reason;
        size_t failure_count;  // Failures that led to block
    };

    void cleanup_old_failures();
    void cleanup_expired_blocks();

    size_t max_failed_attempts_;
    int block_duration_seconds_;
    int failure_window_seconds_;
    
    std::unordered_map<std::string, FailureRecord> failures_;
    std::unordered_map<std::string, BlockRecord> blocks_;
    
    mutable std::mutex mutex_;
    std::chrono::steady_clock::time_point last_cleanup_;
    
    // Statistics
    mutable size_t total_blocks_ = 0;
    mutable size_t total_failures_ = 0;
};

/**
 * @brief IP blocker configuration
 */
struct IPBlockerConfig {
    size_t max_failed_attempts = 10;      // Block after 10 failed attempts
    int block_duration_seconds = 3600;    // 1 hour block
    int failure_window_seconds = 600;     // Track failures in 10-minute window
    bool enable_auto_unblock = true;      // Auto-unblock after duration
    bool log_blocks = true;               // Log all blocks to audit log
};

} // namespace middleware
} // namespace jadevectordb
