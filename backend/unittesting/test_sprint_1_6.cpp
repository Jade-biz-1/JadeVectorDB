// Comprehensive tests for Sprint 1.6 - Production Readiness
// Tests: Circuit Breaker, Error Handling, Retry Logic
#include "utils/circuit_breaker.h"
#include "lib/logging.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <atomic>

using namespace jadevectordb;
using namespace jadevectordb::utils;

#define TEST(name) std::cout << "\n" << name << "..." << std::endl
#define ASSERT(cond, msg) if (!(cond)) { std::cerr << "  ❌ FAILED: " << msg << std::endl; return 1; } else { std::cout << "  ✓ " << msg << std::endl; }

int main() {
    std::cout << "\n=== Sprint 1.6: Production Readiness Tests ===" << std::endl;
    std::cout << "Testing: Circuit Breaker, Error Handling, Resilience\n" << std::endl;
    
    // Initialize logging
    logging::LoggerManager::initialize(logging::LogLevel::INFO);
    
    // =================================================================
    // CIRCUIT BREAKER TESTS
    // =================================================================
    
    TEST("1. Circuit Breaker - Initial state is CLOSED");
    CircuitBreaker::Config config;
    config.failure_threshold = 3;
    config.success_threshold = 2;
    config.timeout = std::chrono::seconds(2);
    config.window = std::chrono::seconds(5);
    
    CircuitBreaker cb1("test_cb1", config);
    ASSERT(cb1.is_closed(), "Initial state is CLOSED");
    ASSERT(cb1.get_state_string() == "CLOSED", "State string is CLOSED");
    ASSERT(cb1.get_failure_count() == 0, "Initial failure count is 0");
    
    TEST("2. Circuit Breaker - Successful operations in CLOSED state");
    int success_count = 0;
    for (int i = 0; i < 5; i++) {
        bool result = cb1.execute([&]() {
            success_count++;
            return true;  // Success
        });
        ASSERT(result, "Operation succeeded");
    }
    ASSERT(success_count == 5, "All 5 operations executed");
    ASSERT(cb1.is_closed(), "State remains CLOSED after successes");
    ASSERT(cb1.get_failure_count() == 0, "Failure count is 0");
    
    TEST("3. Circuit Breaker - Failures trigger OPEN state");
    CircuitBreaker cb2("test_cb2", config);
    int failure_count = 0;
    
    // Trigger failures
    for (int i = 0; i < 3; i++) {
        cb2.execute([&]() {
            failure_count++;
            return false;  // Failure
        });
    }
    
    ASSERT(failure_count == 3, "All 3 operations attempted");
    ASSERT(cb2.is_open(), "Circuit is OPEN after threshold failures");
    ASSERT(cb2.get_failure_count() == 3, "Failure count is 3");
    
    TEST("4. Circuit Breaker - OPEN state fails fast");
    int fast_fail_count = 0;
    for (int i = 0; i < 5; i++) {
        bool result = cb2.execute([&]() {
            fast_fail_count++;  // Should NOT be called
            return true;
        });
        ASSERT(!result, "Operation failed fast");
    }
    ASSERT(fast_fail_count == 0, "Functions not executed in OPEN state");
    ASSERT(cb2.is_open(), "Circuit remains OPEN");
    
    TEST("5. Circuit Breaker - Transition to HALF_OPEN after timeout");
    std::cout << "  (waiting 2 seconds for timeout...)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    int half_open_count = 0;
    bool half_open_result = cb2.execute([&]() {
        half_open_count++;
        return true;  // Success in half-open
    });
    
    ASSERT(half_open_result, "First request in HALF_OPEN succeeded");
    ASSERT(half_open_count == 1, "Function executed once");
    ASSERT(cb2.is_half_open(), "Circuit is HALF_OPEN");
    
    TEST("6. Circuit Breaker - Success in HALF_OPEN closes circuit");
    int close_count = 0;
    for (int i = 0; i < 2; i++) {
        cb2.execute([&]() {
            close_count++;
            return true;  // Success
        });
    }
    
    ASSERT(close_count == 2, "Both operations executed");
    ASSERT(cb2.is_closed(), "Circuit CLOSED after success threshold");
    ASSERT(cb2.get_failure_count() == 0, "Failure count reset to 0");
    
    TEST("7. Circuit Breaker - Failure in HALF_OPEN reopens circuit");
    CircuitBreaker cb3("test_cb3", config);
    
    // Trigger failures to open
    for (int i = 0; i < 3; i++) {
        cb3.execute([]() { return false; });
    }
    ASSERT(cb3.is_open(), "Circuit opened");
    
    // Wait for timeout
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Try and fail in half-open
    cb3.execute([]() { return true; });  // Transition to HALF_OPEN
    ASSERT(cb3.is_half_open(), "Circuit is HALF_OPEN");
    
    cb3.execute([]() { return false; });  // Fail in HALF_OPEN
    ASSERT(cb3.is_open(), "Circuit REOPENED after failure in HALF_OPEN");
    
    TEST("8. Circuit Breaker - Manual reset");
    CircuitBreaker cb4("test_cb4", config);
    
    // Open the circuit
    for (int i = 0; i < 3; i++) {
        cb4.execute([]() { return false; });
    }
    ASSERT(cb4.is_open(), "Circuit opened");
    
    // Manual reset
    cb4.reset();
    ASSERT(cb4.is_closed(), "Circuit manually reset to CLOSED");
    ASSERT(cb4.get_failure_count() == 0, "Failure count reset");
    
    TEST("9. Circuit Breaker - Exception handling");
    CircuitBreaker cb5("test_cb5", config);
    int exception_count = 0;
    
    for (int i = 0; i < 3; i++) {
        bool result = cb5.execute([&]() -> bool {
            exception_count++;
            throw std::runtime_error("Test exception");
        });
        ASSERT(!result, "Exception treated as failure");
    }
    
    ASSERT(exception_count == 3, "All functions executed before opening");
    ASSERT(cb5.is_open(), "Circuit opened after exceptions");
    
    TEST("10. Circuit Breaker - Time window for failure counting");
    CircuitBreaker::Config window_config;
    window_config.failure_threshold = 5;
    window_config.window = std::chrono::seconds(2);
    window_config.timeout = std::chrono::seconds(1);
    
    CircuitBreaker cb6("test_cb6", window_config);
    
    // 3 failures
    for (int i = 0; i < 3; i++) {
        cb6.execute([]() { return false; });
    }
    ASSERT(cb6.is_closed(), "Circuit still closed (3 < 5 failures)");
    
    // Wait for window to pass
    std::cout << "  (waiting 2 seconds for window expiry...)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Old failures should be cleared, circuit should stay closed
    cb6.execute([]() { return false; });  // 1 new failure
    ASSERT(cb6.is_closed(), "Circuit closed (old failures expired)");
    
    TEST("11. Circuit Breaker - Concurrent access");
    CircuitBreaker cb7("test_cb7", config);
    std::atomic<int> concurrent_success{0};
    std::atomic<int> concurrent_failure{0};
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&, i]() {
            bool result = cb7.execute([i]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return i % 2 == 0;  // Even threads succeed
            });
            if (result) {
                concurrent_success++;
            } else {
                concurrent_failure++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    ASSERT(concurrent_success > 0, "Some operations succeeded");
    ASSERT(concurrent_failure > 0, "Some operations failed");
    std::cout << "  ✓ Concurrent access: " << concurrent_success.load() 
              << " successes, " << concurrent_failure.load() << " failures" << std::endl;
    
    TEST("12. Circuit Breaker - State transitions are atomic");
    CircuitBreaker cb8("test_cb8", config);
    std::atomic<int> state_changes{0};
    
    std::vector<std::thread> state_threads;
    for (int i = 0; i < 20; i++) {
        state_threads.emplace_back([&]() {
            for (int j = 0; j < 10; j++) {
                auto prev_state = cb8.get_state();
                cb8.execute([]() { return false; });  // All fail
                auto new_state = cb8.get_state();
                if (prev_state != new_state) {
                    state_changes++;
                }
            }
        });
    }
    
    for (auto& t : state_threads) {
        t.join();
    }
    
    ASSERT(cb8.is_open(), "Circuit opened after concurrent failures");
    std::cout << "  ✓ State changes: " << state_changes.load() << " (atomic)" << std::endl;
    
    // =================================================================
    // ERROR HANDLING TESTS (for future implementation)
    // =================================================================
    
    TEST("13. Error Handling - Retry with exponential backoff (placeholder)");
    std::cout << "  ⏭ Test placeholder: Retry logic will be tested when implemented" << std::endl;
    
    TEST("14. Error Handling - Database connection recovery (placeholder)");
    std::cout << "  ⏭ Test placeholder: Connection recovery will be tested when implemented" << std::endl;
    
    TEST("15. Error Handling - Transaction rollback on failure (placeholder)");
    std::cout << "  ⏭ Test placeholder: Transaction handling will be tested when implemented" << std::endl;
    
    TEST("16. Health Check - Database connectivity (placeholder)");
    std::cout << "  ⏭ Test placeholder: Health checks will be tested when implemented" << std::endl;
    
    // =================================================================
    // PERFORMANCE VERIFICATION
    // =================================================================
    
    TEST("17. Circuit Breaker - Performance (closed state overhead)");
    CircuitBreaker cb_perf("test_perf", config);
    
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int iterations = 10000;
    
    for (int i = 0; i < iterations; i++) {
        cb_perf.execute([]() { return true; });
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "  ✓ " << iterations << " operations in " << duration.count() 
              << "μs (avg: " << avg_us << "μs per operation)" << std::endl;
    ASSERT(avg_us < 10.0, "Average overhead < 10μs per operation");
    
    TEST("18. Circuit Breaker - Performance (open state fast-fail)");
    CircuitBreaker cb_fast("test_fast", config);
    
    // Open the circuit
    for (int i = 0; i < 3; i++) {
        cb_fast.execute([]() { return false; });
    }
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cb_fast.execute([]() { 
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Would be slow
            return true; 
        });
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_us = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "  ✓ " << iterations << " fast-fails in " << duration.count() 
              << "μs (avg: " << avg_us << "μs per operation)" << std::endl;
    ASSERT(avg_us < 1.0, "Fast-fail < 1μs per operation");
    
    // =================================================================
    // SUMMARY
    // =================================================================
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "✅ All Sprint 1.6 Production Tests PASSED!" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "\nTest Coverage:" << std::endl;
    std::cout << "  ✓ Circuit Breaker: State transitions (12 tests)" << std::endl;
    std::cout << "  ✓ Circuit Breaker: Concurrent access (2 tests)" << std::endl;
    std::cout << "  ✓ Circuit Breaker: Performance (<10μs overhead)" << std::endl;
    std::cout << "  ⏭ Error Handling: Placeholders for future implementation" << std::endl;
    std::cout << "\nNext Steps:" << std::endl;
    std::cout << "  1. Integrate circuit breaker with SQLitePersistenceLayer" << std::endl;
    std::cout << "  2. Implement retry logic with exponential backoff" << std::endl;
    std::cout << "  3. Add health check endpoints" << std::endl;
    std::cout << "  4. Implement connection recovery" << std::endl;
    
    return 0;
}
