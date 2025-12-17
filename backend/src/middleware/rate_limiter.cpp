#include "rate_limiter.h"
#include <algorithm>
#include <vector>

namespace jadevectordb {
namespace middleware {

// ============================================================================
// TokenBucket Implementation
// ============================================================================

TokenBucket::TokenBucket(size_t capacity, double refill_rate)
    : capacity_(capacity)
    , refill_rate_(refill_rate)
    , tokens_(static_cast<double>(capacity))
    , last_refill_(std::chrono::steady_clock::now()) {
}

void TokenBucket::refill() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_refill_).count();
    
    // Add tokens based on elapsed time
    double new_tokens = elapsed * refill_rate_;
    tokens_ = std::min(static_cast<double>(capacity_), tokens_ + new_tokens);
    
    last_refill_ = now;
}

bool TokenBucket::try_consume(size_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    refill();
    
    if (tokens_ >= static_cast<double>(tokens)) {
        tokens_ -= static_cast<double>(tokens);
        return true;
    }
    
    return false;
}

double TokenBucket::retry_after_seconds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (tokens_ >= 1.0) {
        return 0.0;
    }
    
    // Calculate time until at least 1 token is available
    double tokens_needed = 1.0 - tokens_;
    return tokens_needed / refill_rate_;
}

size_t TokenBucket::available_tokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(tokens_);
}

// ============================================================================
// RateLimiter Implementation
// ============================================================================

RateLimiter::RateLimiter(size_t capacity, double refill_rate, int cleanup_interval_seconds)
    : capacity_(capacity)
    , refill_rate_(refill_rate)
    , cleanup_interval_seconds_(cleanup_interval_seconds)
    , last_cleanup_(std::chrono::steady_clock::now()) {
}

bool RateLimiter::allow(const std::string& key, size_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_requests_++;
    
    // Periodic cleanup of old buckets
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup_).count();
    if (elapsed >= cleanup_interval_seconds_) {
        cleanup_old_buckets();
        last_cleanup_ = now;
    }
    
    // Get or create bucket for this key
    auto it = buckets_.find(key);
    if (it == buckets_.end()) {
        auto bucket = std::make_shared<TokenBucket>(capacity_, refill_rate_);
        buckets_[key] = bucket;
        it = buckets_.find(key);
    }
    
    // Try to consume tokens
    bool allowed = it->second->try_consume(tokens);
    
    if (!allowed) {
        rate_limited_requests_++;
    }
    
    return allowed;
}

double RateLimiter::retry_after_seconds(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = buckets_.find(key);
    if (it == buckets_.end()) {
        return 0.0;  // No bucket = no rate limit
    }
    
    return it->second->retry_after_seconds();
}

void RateLimiter::reset(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    buckets_.erase(key);
}

void RateLimiter::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    buckets_.clear();
    total_requests_ = 0;
    rate_limited_requests_ = 0;
}

std::unordered_map<std::string, size_t> RateLimiter::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"total_buckets", buckets_.size()},
        {"total_requests", total_requests_},
        {"rate_limited_requests", rate_limited_requests_}
    };
}

void RateLimiter::cleanup_old_buckets() {
    // Remove buckets that have been inactive (full capacity)
    // This prevents memory growth from one-time IPs
    std::vector<std::string> keys_to_remove;
    
    for (const auto& [key, bucket] : buckets_) {
        if (bucket->available_tokens() >= capacity_) {
            keys_to_remove.push_back(key);
        }
    }
    
    for (const auto& key : keys_to_remove) {
        buckets_.erase(key);
    }
}

} // namespace middleware
} // namespace jadevectordb
