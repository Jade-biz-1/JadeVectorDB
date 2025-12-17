#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace jadevectordb {
namespace middleware {

/**
 * @brief Token bucket for rate limiting
 * 
 * Thread-safe implementation of the token bucket algorithm.
 * Tokens are added at a constant rate and consumed per request.
 */
class TokenBucket {
public:
    /**
     * @brief Construct a new Token Bucket
     * 
     * @param capacity Maximum number of tokens
     * @param refill_rate Tokens added per second
     */
    TokenBucket(size_t capacity, double refill_rate);

    /**
     * @brief Try to consume a token
     * 
     * @param tokens Number of tokens to consume (default: 1)
     * @return true if tokens were consumed, false if bucket is empty
     */
    bool try_consume(size_t tokens = 1);

    /**
     * @brief Get seconds until next token is available
     * 
     * @return double Seconds to wait
     */
    double retry_after_seconds() const;

    /**
     * @brief Get current token count
     * 
     * @return size_t Number of tokens available
     */
    size_t available_tokens() const;

private:
    void refill();

    size_t capacity_;
    double refill_rate_;  // tokens per second
    double tokens_;
    std::chrono::steady_clock::time_point last_refill_;
    mutable std::mutex mutex_;
};

/**
 * @brief Rate limiter using token bucket algorithm
 * 
 * Supports per-key rate limiting (IP address, API key, user ID, etc.)
 * Thread-safe and suitable for high-concurrency environments.
 */
class RateLimiter {
public:
    /**
     * @brief Construct a new Rate Limiter
     * 
     * @param capacity Maximum requests per window
     * @param refill_rate Requests allowed per second
     * @param cleanup_interval_seconds Interval to clean up old buckets (default: 300s)
     */
    RateLimiter(size_t capacity, double refill_rate, int cleanup_interval_seconds = 300);

    /**
     * @brief Check if request is allowed for key
     * 
     * @param key Rate limit key (IP, user ID, API key, etc.)
     * @param tokens Number of tokens to consume (default: 1)
     * @return true if request is allowed, false if rate limit exceeded
     */
    bool allow(const std::string& key, size_t tokens = 1);

    /**
     * @brief Get retry-after seconds for key
     * 
     * @param key Rate limit key
     * @return double Seconds to wait before retry, 0 if allowed
     */
    double retry_after_seconds(const std::string& key) const;

    /**
     * @brief Reset rate limit for key
     * 
     * @param key Rate limit key to reset
     */
    void reset(const std::string& key);

    /**
     * @brief Clear all rate limit buckets
     */
    void clear_all();

    /**
     * @brief Get statistics
     * 
     * @return std::unordered_map<std::string, size_t> Stats including total_buckets, total_requests, rate_limited_requests
     */
    std::unordered_map<std::string, size_t> get_stats() const;

private:
    void cleanup_old_buckets();

    size_t capacity_;
    double refill_rate_;
    int cleanup_interval_seconds_;
    std::unordered_map<std::string, std::shared_ptr<TokenBucket>> buckets_;
    mutable std::mutex mutex_;
    std::chrono::steady_clock::time_point last_cleanup_;
    
    // Statistics
    mutable size_t total_requests_ = 0;
    mutable size_t rate_limited_requests_ = 0;
};

/**
 * @brief Rate limiter configuration for different endpoint types
 */
struct RateLimitConfig {
    size_t login_capacity = 5;              // 5 attempts per minute
    double login_refill_rate = 5.0 / 60.0;  // per second
    
    size_t registration_capacity = 3;       // 3 per hour
    double registration_refill_rate = 3.0 / 3600.0;
    
    size_t api_capacity = 1000;             // 1000 per minute
    double api_refill_rate = 1000.0 / 60.0;
    
    size_t password_reset_capacity = 3;     // 3 per hour
    double password_reset_refill_rate = 3.0 / 3600.0;
    
    size_t global_capacity = 10000;         // 10k requests per second
    double global_refill_rate = 10000.0;
};

} // namespace middleware
} // namespace jadevectordb
