#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>

namespace jadevectordb {
namespace utils {

/**
 * Circuit Breaker Pattern Implementation
 * 
 * Prevents cascading failures by failing fast when a service is unhealthy.
 * 
 * States:
 * - CLOSED: Normal operation, all requests pass through
 * - OPEN: Service unhealthy, all requests fail immediately
 * - HALF_OPEN: Testing if service recovered, limited requests allowed
 * 
 * Configuration:
 * - failure_threshold: Number of failures to open circuit
 * - success_threshold: Number of successes to close circuit from half-open
 * - timeout: Duration to wait before moving from open to half-open
 */
class CircuitBreaker {
public:
    enum class State {
        CLOSED,      // Normal operation
        OPEN,        // Failing fast
        HALF_OPEN    // Testing recovery
    };

    struct Config {
        size_t failure_threshold;
        size_t success_threshold;
        std::chrono::seconds timeout;
        std::chrono::seconds window;
        
        Config() 
            : failure_threshold(5)
            , success_threshold(2)
            , timeout(30)
            , window(60) {}
    };

    /**
     * Construct a circuit breaker with given configuration
     */
    explicit CircuitBreaker(const std::string& name);
    explicit CircuitBreaker(const std::string& name, const Config& config);

    /**
     * Execute a function with circuit breaker protection
     * 
     * @tparam Func Function type returning bool (true = success, false = failure)
     * @param func Function to execute
     * @return true if function executed successfully, false if circuit is open or function failed
     */
    template<typename Func>
    bool execute(Func&& func) {
        // Check if circuit is open
        if (is_open()) {
            // Try to transition to half-open if timeout expired
            if (should_attempt_reset()) {
                transition_to_half_open();
            } else {
                // Circuit still open, fail fast
                return false;
            }
        }

        // Execute the function
        try {
            bool success = func();
            if (success) {
                record_success();
            } else {
                record_failure();
            }
            return success;
        } catch (...) {
            record_failure();
            return false;
        }
    }

    /**
     * Manually reset the circuit breaker to closed state
     */
    void reset();

    /**
     * Get current state
     */
    State get_state() const;

    /**
     * Get state as string for logging
     */
    std::string get_state_string() const;

    /**
     * Get failure count in current window
     */
    size_t get_failure_count() const;

    /**
     * Get success count (in half-open state)
     */
    size_t get_success_count() const;

    /**
     * Check if circuit is open
     */
    bool is_open() const;

    /**
     * Check if circuit is closed
     */
    bool is_closed() const;

    /**
     * Check if circuit is half-open
     */
    bool is_half_open() const;

private:
    void record_success();
    void record_failure();
    void transition_to_open();
    void transition_to_half_open();
    void transition_to_closed();
    bool should_attempt_reset() const;
    void clear_old_failures();

    std::string name_;
    Config config_;
    
    std::atomic<State> state_{State::CLOSED};
    std::atomic<size_t> failure_count_{0};
    std::atomic<size_t> success_count_{0};
    std::atomic<std::chrono::steady_clock::time_point> last_failure_time_;
    std::atomic<std::chrono::steady_clock::time_point> circuit_opened_time_;
    
    mutable std::mutex mutex_;
    std::vector<std::chrono::steady_clock::time_point> failure_times_;
};

} // namespace utils
} // namespace jadevectordb
