#include "ip_blocker.h"
#include <algorithm>
#include <iostream>

namespace jadevectordb {
namespace middleware {

IPBlocker::IPBlocker(size_t max_failed_attempts, 
                     int block_duration_seconds,
                     int failure_window_seconds)
    : max_failed_attempts_(max_failed_attempts)
    , block_duration_seconds_(block_duration_seconds)
    , failure_window_seconds_(failure_window_seconds)
    , last_cleanup_(std::chrono::steady_clock::now()) {
}

bool IPBlocker::is_blocked(const std::string& ip) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = blocks_.find(ip);
    if (it == blocks_.end()) {
        return false;
    }
    
    // Check if block has expired
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - it->second.block_time).count();
    
    return elapsed < it->second.duration_seconds;
}

bool IPBlocker::record_failure(const std::string& ip, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_failures_++;
    
    // Periodic cleanup
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup_).count();
    if (elapsed >= 60) {  // Cleanup every minute
        cleanup_old_failures();
        cleanup_expired_blocks();
        last_cleanup_ = now;
    }
    
    // Check if already blocked
    if (blocks_.find(ip) != blocks_.end()) {
        return true;  // Already blocked
    }
    
    // Record failure
    auto& record = failures_[ip];
    record.failures.push_back(now);
    record.last_reason = reason;
    
    // Remove old failures outside window
    auto cutoff = now - std::chrono::seconds(failure_window_seconds_);
    record.failures.erase(
        std::remove_if(record.failures.begin(), record.failures.end(),
            [cutoff](const auto& time) { return time < cutoff; }),
        record.failures.end()
    );
    
    // Check if should be blocked
    if (record.failures.size() >= max_failed_attempts_) {
        BlockRecord block;
        block.block_time = now;
        block.duration_seconds = block_duration_seconds_;
        block.reason = reason.empty() ? "max_failures_exceeded" : reason;
        block.failure_count = record.failures.size();
        
        blocks_[ip] = block;
        failures_.erase(ip);  // Clear failure history
        total_blocks_++;
        
        std::cout << "[IPBlocker] Blocked IP: " << ip 
                  << " (failures: " << block.failure_count 
                  << ", reason: " << block.reason 
                  << ", duration: " << block.duration_seconds << "s)" << std::endl;
        
        return true;  // IP is now blocked
    }
    
    return false;  // Not blocked yet
}

void IPBlocker::record_success(const std::string& ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear failure history on successful login
    failures_.erase(ip);
}

void IPBlocker::block_ip(const std::string& ip, int duration_seconds, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    BlockRecord block;
    block.block_time = std::chrono::steady_clock::now();
    block.duration_seconds = duration_seconds > 0 ? duration_seconds : block_duration_seconds_;
    block.reason = reason;
    block.failure_count = 0;  // Manual block
    
    blocks_[ip] = block;
    failures_.erase(ip);  // Clear any pending failures
    total_blocks_++;
    
    std::cout << "[IPBlocker] Manually blocked IP: " << ip 
              << " (reason: " << reason 
              << ", duration: " << block.duration_seconds << "s)" << std::endl;
}

bool IPBlocker::unblock_ip(const std::string& ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = blocks_.find(ip);
    if (it == blocks_.end()) {
        return false;  // Not blocked
    }
    
    std::cout << "[IPBlocker] Manually unblocked IP: " << ip << std::endl;
    blocks_.erase(it);
    failures_.erase(ip);  // Also clear any pending failures
    
    return true;
}

int IPBlocker::remaining_block_seconds(const std::string& ip) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = blocks_.find(ip);
    if (it == blocks_.end()) {
        return 0;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - it->second.block_time).count();
    
    int remaining = it->second.duration_seconds - static_cast<int>(elapsed);
    return std::max(0, remaining);
}

size_t IPBlocker::get_failure_count(const std::string& ip) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = failures_.find(ip);
    if (it == failures_.end()) {
        return 0;
    }
    
    // Count failures within window
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::seconds(failure_window_seconds_);
    
    size_t count = 0;
    for (const auto& failure_time : it->second.failures) {
        if (failure_time >= cutoff) {
            count++;
        }
    }
    
    return count;
}

std::vector<std::string> IPBlocker::get_blocked_ips() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> result;
    auto now = std::chrono::steady_clock::now();
    
    for (const auto& [ip, block] : blocks_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - block.block_time).count();
        
        if (elapsed < block.duration_seconds) {
            result.push_back(ip);
        }
    }
    
    return result;
}

std::unordered_map<std::string, size_t> IPBlocker::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Count active blocks
    size_t active_blocks = 0;
    auto now = std::chrono::steady_clock::now();
    
    for (const auto& [ip, block] : blocks_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - block.block_time).count();
        
        if (elapsed < block.duration_seconds) {
            active_blocks++;
        }
    }
    
    return {
        {"total_blocks", total_blocks_},
        {"active_blocks", active_blocks},
        {"total_failures", total_failures_},
        {"tracking_ips", failures_.size()}
    };
}

void IPBlocker::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    blocks_.clear();
    failures_.clear();
    total_blocks_ = 0;
    total_failures_ = 0;
}

void IPBlocker::cleanup_old_failures() {
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::seconds(failure_window_seconds_);
    
    std::vector<std::string> ips_to_remove;
    
    for (auto& [ip, record] : failures_) {
        // Remove old failures
        record.failures.erase(
            std::remove_if(record.failures.begin(), record.failures.end(),
                [cutoff](const auto& time) { return time < cutoff; }),
            record.failures.end()
        );
        
        // If no recent failures, remove the record
        if (record.failures.empty()) {
            ips_to_remove.push_back(ip);
        }
    }
    
    for (const auto& ip : ips_to_remove) {
        failures_.erase(ip);
    }
}

void IPBlocker::cleanup_expired_blocks() {
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> ips_to_remove;
    
    for (const auto& [ip, block] : blocks_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - block.block_time).count();
        
        if (elapsed >= block.duration_seconds) {
            ips_to_remove.push_back(ip);
        }
    }
    
    for (const auto& ip : ips_to_remove) {
        std::cout << "[IPBlocker] Auto-unblocked IP: " << ip 
                  << " (block expired)" << std::endl;
        blocks_.erase(ip);
    }
}

} // namespace middleware
} // namespace jadevectordb
