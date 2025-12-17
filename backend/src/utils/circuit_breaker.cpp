#include "circuit_breaker.h"
#include "lib/logging.h"
#include <algorithm>

namespace jadevectordb {
namespace utils {

CircuitBreaker::CircuitBreaker(const std::string& name)
    : CircuitBreaker(name, Config{}) {}

CircuitBreaker::CircuitBreaker(const std::string& name, const Config& config)
    : name_(name), config_(config) {
    last_failure_time_.store(std::chrono::steady_clock::time_point{});
    circuit_opened_time_.store(std::chrono::steady_clock::time_point{});
    LOG_INFO_DEFAULT("CircuitBreaker '" + name_ + "' initialized: failure_threshold=" + 
                     std::to_string(config_.failure_threshold) + ", success_threshold=" + 
                     std::to_string(config_.success_threshold) + ", timeout=" + 
                     std::to_string(config_.timeout.count()) + "s, window=" + 
                     std::to_string(config_.window.count()) + "s");
}

void CircuitBreaker::record_success() {
    State current_state = state_.load();
    
    if (current_state == State::HALF_OPEN) {
        size_t successes = success_count_.fetch_add(1) + 1;
        LOG_DEBUG_DEFAULT("CircuitBreaker '" + name_ + "': Success recorded (" + 
                         std::to_string(successes) + "/" + 
                         std::to_string(config_.success_threshold) + ")");
        
        if (successes >= config_.success_threshold) {
            transition_to_closed();
        }
    } else if (current_state == State::CLOSED) {
        // In closed state, clear failure count on success
        std::lock_guard<std::mutex> lock(mutex_);
        failure_times_.clear();
        failure_count_.store(0);
    }
}

void CircuitBreaker::record_failure() {
    State current_state = state_.load();
    last_failure_time_.store(std::chrono::steady_clock::now());
    
    if (current_state == State::HALF_OPEN) {
        // Failure in half-open immediately reopens circuit
        LOG_WARN_DEFAULT("CircuitBreaker '" + name_ + "': Failure in HALF_OPEN state, reopening circuit");
        transition_to_open();
        return;
    }
    
    if (current_state == State::CLOSED) {
        // Record failure with timestamp
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        failure_times_.push_back(now);
        
        // Remove old failures outside the window
        clear_old_failures();
        
        size_t failures = failure_times_.size();
        failure_count_.store(failures);
        
        LOG_DEBUG_DEFAULT("CircuitBreaker '" + name_ + "': Failure recorded (" + 
                         std::to_string(failures) + "/" + 
                         std::to_string(config_.failure_threshold) + ")");
        
        if (failures >= config_.failure_threshold) {
            transition_to_open();
        }
    }
}

void CircuitBreaker::transition_to_open() {
    State expected = State::CLOSED;
    if (state_.compare_exchange_strong(expected, State::OPEN)) {
        circuit_opened_time_.store(std::chrono::steady_clock::now());
        LOG_ERROR_DEFAULT("CircuitBreaker '" + name_ + "': OPENED - failing fast for " + 
                         std::to_string(config_.timeout.count()) + "s");
    } else {
        expected = State::HALF_OPEN;
        if (state_.compare_exchange_strong(expected, State::OPEN)) {
            circuit_opened_time_.store(std::chrono::steady_clock::now());
            LOG_ERROR_DEFAULT("CircuitBreaker '" + name_ + "': REOPENED from HALF_OPEN");
        }
    }
}

void CircuitBreaker::transition_to_half_open() {
    State expected = State::OPEN;
    if (state_.compare_exchange_strong(expected, State::HALF_OPEN)) {
        success_count_.store(0);
        LOG_INFO_DEFAULT("CircuitBreaker '" + name_ + "': HALF_OPEN - testing recovery");
    }
}

void CircuitBreaker::transition_to_closed() {
    State expected = State::HALF_OPEN;
    if (state_.compare_exchange_strong(expected, State::CLOSED)) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_times_.clear();
        failure_count_.store(0);
        success_count_.store(0);
        LOG_INFO_DEFAULT("CircuitBreaker '" + name_ + "': CLOSED - service recovered");
    }
}

bool CircuitBreaker::should_attempt_reset() const {
    if (state_.load() != State::OPEN) {
        return false;
    }
    
    auto opened_time = circuit_opened_time_.load();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - opened_time);
    
    return elapsed >= config_.timeout;
}

void CircuitBreaker::clear_old_failures() {
    // Must be called with mutex locked
    auto now = std::chrono::steady_clock::now();
    auto cutoff = now - config_.window;
    
    failure_times_.erase(
        std::remove_if(failure_times_.begin(), failure_times_.end(),
                      [cutoff](const auto& time) { return time < cutoff; }),
        failure_times_.end()
    );
}

void CircuitBreaker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.store(State::CLOSED);
    failure_times_.clear();
    failure_count_.store(0);
    success_count_.store(0);
    LOG_INFO_DEFAULT("CircuitBreaker '" + name_ + "': Manually reset to CLOSED");
}

CircuitBreaker::State CircuitBreaker::get_state() const {
    return state_.load();
}

std::string CircuitBreaker::get_state_string() const {
    switch (state_.load()) {
        case State::CLOSED: return "CLOSED";
        case State::OPEN: return "OPEN";
        case State::HALF_OPEN: return "HALF_OPEN";
        default: return "UNKNOWN";
    }
}

size_t CircuitBreaker::get_failure_count() const {
    return failure_count_.load();
}

size_t CircuitBreaker::get_success_count() const {
    return success_count_.load();
}

bool CircuitBreaker::is_open() const {
    return state_.load() == State::OPEN;
}

bool CircuitBreaker::is_closed() const {
    return state_.load() == State::CLOSED;
}

bool CircuitBreaker::is_half_open() const {
    return state_.load() == State::HALF_OPEN;
}

} // namespace utils
} // namespace jadevectordb
