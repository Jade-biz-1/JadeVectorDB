/**
 * @file test_permission_cache.cpp
 * @brief Unit tests for the Permission Cache system (T11.6.3)
 * 
 * Tests verify:
 * - Basic cache operations (get, put)
 * - LRU eviction at max capacity
 * - TTL-based expiration
 * - Cache invalidation (user, database, all)
 * - Thread safety
 * - Performance characteristics
 */

#include "../src/cache/permission_cache.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <vector>
#include <random>

using namespace jadevectordb::cache;

// Helper function for test assertions
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        if (test_func()) { \
            std::cout << "PASSED" << std::endl; \
            passed++; \
        } else { \
            std::cout << "FAILED" << std::endl; \
            failed++; \
        } \
        total++; \
    } while(0)

// Test 1: Basic cache operations (get, put)
bool test_basic_cache_operations() {
    PermissionCache cache(100, 300); // 100 entries, 5 min TTL
    
    // Test put and get
    cache.put("user1", "db1", "read", true);
    auto result = cache.get("user1", "db1", "read");
    TEST_ASSERT(result.has_value(), "Cache should contain the entry");
    TEST_ASSERT(result.value() == true, "Cache should return correct value");
    
    // Test cache miss
    auto miss = cache.get("user2", "db1", "read");
    TEST_ASSERT(!miss.has_value(), "Cache should return nullopt for missing entry");
    
    // Test overwrite
    cache.put("user1", "db1", "read", false);
    result = cache.get("user1", "db1", "read");
    TEST_ASSERT(result.has_value(), "Cache should contain the entry");
    TEST_ASSERT(result.value() == false, "Cache should return updated value");
    
    return true;
}

// Test 2: Cache statistics tracking
bool test_cache_statistics() {
    PermissionCache cache(100, 300);
    
    // Initial stats
    auto stats = cache.get_stats();
    TEST_ASSERT(stats.hits == 0, "Initial hits should be 0");
    TEST_ASSERT(stats.misses == 0, "Initial misses should be 0");
    
    // Add entries and track hits/misses
    cache.put("user1", "db1", "read", true);
    cache.get("user1", "db1", "read"); // Hit
    cache.get("user2", "db1", "read"); // Miss
    cache.get("user1", "db1", "read"); // Hit
    
    stats = cache.get_stats();
    TEST_ASSERT(stats.hits == 2, "Should have 2 hits");
    TEST_ASSERT(stats.misses == 1, "Should have 1 miss");
    TEST_ASSERT(stats.hit_rate() > 0.66 && stats.hit_rate() < 0.67, 
                "Hit rate should be ~66.67%");
    
    return true;
}

// Test 3: LRU eviction at max capacity
bool test_lru_eviction() {
    PermissionCache cache(3, 300); // Small cache for testing
    
    // Fill cache to capacity
    cache.put("user1", "db1", "read", true);
    cache.put("user2", "db1", "read", true);
    cache.put("user3", "db1", "read", true);
    
    // Access user1 to make it most recently used
    cache.get("user1", "db1", "read");
    
    // Add new entry - should evict user2 (LRU)
    cache.put("user4", "db1", "read", true);
    
    auto user1_result = cache.get("user1", "db1", "read");
    auto user2_result = cache.get("user2", "db1", "read");
    auto user3_result = cache.get("user3", "db1", "read");
    auto user4_result = cache.get("user4", "db1", "read");
    
    TEST_ASSERT(user1_result.has_value(), "User1 should still be in cache (accessed recently)");
    TEST_ASSERT(!user2_result.has_value(), "User2 should be evicted (LRU)");
    TEST_ASSERT(user3_result.has_value(), "User3 should still be in cache");
    TEST_ASSERT(user4_result.has_value(), "User4 should be in cache (just added)");
    
    auto stats = cache.get_stats();
    TEST_ASSERT(stats.evictions >= 1, "Should have at least 1 eviction");
    
    return true;
}

// Test 4: TTL-based expiration
bool test_ttl_expiration() {
    PermissionCache cache(100, 1); // 1 second TTL for fast testing
    
    cache.put("user1", "db1", "read", true);
    
    // Immediately should be valid
    auto result = cache.get("user1", "db1", "read");
    TEST_ASSERT(result.has_value(), "Entry should be valid immediately");
    
    // Wait for expiration
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    
    // Should be expired now
    result = cache.get("user1", "db1", "read");
    TEST_ASSERT(!result.has_value(), "Entry should be expired after TTL");
    
    return true;
}

// Test 5: User-specific invalidation
bool test_user_invalidation() {
    PermissionCache cache(100, 300);
    
    // Add multiple entries for different users
    cache.put("user1", "db1", "read", true);
    cache.put("user1", "db2", "write", true);
    cache.put("user2", "db1", "read", true);
    cache.put("user2", "db2", "write", true);
    
    // Invalidate user1's cache
    cache.invalidate_user("user1");
    
    auto user1_db1 = cache.get("user1", "db1", "read");
    auto user1_db2 = cache.get("user1", "db2", "write");
    auto user2_db1 = cache.get("user2", "db1", "read");
    auto user2_db2 = cache.get("user2", "db2", "write");
    
    TEST_ASSERT(!user1_db1.has_value(), "User1's db1 entry should be invalidated");
    TEST_ASSERT(!user1_db2.has_value(), "User1's db2 entry should be invalidated");
    TEST_ASSERT(user2_db1.has_value(), "User2's db1 entry should remain");
    TEST_ASSERT(user2_db2.has_value(), "User2's db2 entry should remain");
    
    return true;
}

// Test 6: Database-specific invalidation
bool test_database_invalidation() {
    PermissionCache cache(100, 300);
    
    // Add multiple entries for different databases
    cache.put("user1", "db1", "read", true);
    cache.put("user1", "db2", "write", true);
    cache.put("user2", "db1", "read", true);
    cache.put("user2", "db2", "write", true);
    
    // Invalidate db1 cache
    cache.invalidate_database("db1");
    
    auto user1_db1 = cache.get("user1", "db1", "read");
    auto user1_db2 = cache.get("user1", "db2", "write");
    auto user2_db1 = cache.get("user2", "db1", "read");
    auto user2_db2 = cache.get("user2", "db2", "write");
    
    TEST_ASSERT(!user1_db1.has_value(), "User1's db1 entry should be invalidated");
    TEST_ASSERT(user1_db2.has_value(), "User1's db2 entry should remain");
    TEST_ASSERT(!user2_db1.has_value(), "User2's db1 entry should be invalidated");
    TEST_ASSERT(user2_db2.has_value(), "User2's db2 entry should remain");
    
    return true;
}

// Test 7: Permission-specific invalidation
bool test_permission_invalidation() {
    PermissionCache cache(100, 300);
    
    // Add multiple entries with different permissions
    cache.put("user1", "db1", "read", true);
    cache.put("user1", "db1", "write", true);
    cache.put("user2", "db1", "read", true);
    
    // Invalidate specific permission
    cache.invalidate("user1", "db1", "read");
    
    auto user1_read = cache.get("user1", "db1", "read");
    auto user1_write = cache.get("user1", "db1", "write");
    auto user2_read = cache.get("user2", "db1", "read");
    
    TEST_ASSERT(!user1_read.has_value(), "User1's read permission should be invalidated");
    TEST_ASSERT(user1_write.has_value(), "User1's write permission should remain");
    TEST_ASSERT(user2_read.has_value(), "User2's read permission should remain");
    
    return true;
}

// Test 8: Clear all cache
bool test_clear_cache() {
    PermissionCache cache(100, 300);
    
    // Add multiple entries
    cache.put("user1", "db1", "read", true);
    cache.put("user2", "db2", "write", true);
    cache.put("user3", "db3", "delete", false);
    
    // Clear all
    cache.clear();
    
    auto user1 = cache.get("user1", "db1", "read");
    auto user2 = cache.get("user2", "db2", "write");
    auto user3 = cache.get("user3", "db3", "delete");
    
    TEST_ASSERT(!user1.has_value(), "All entries should be cleared");
    TEST_ASSERT(!user2.has_value(), "All entries should be cleared");
    TEST_ASSERT(!user3.has_value(), "All entries should be cleared");
    
    auto stats = cache.get_stats();
    TEST_ASSERT(stats.hits == 0, "Stats should be reset");
    TEST_ASSERT(stats.misses == 3, "Should have 3 misses from get operations");
    
    return true;
}

// Test 9: Thread safety with concurrent access
bool test_thread_safety() {
    PermissionCache cache(1000, 300);
    const int num_threads = 10;
    const int operations_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch multiple threads doing concurrent operations
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&cache, t, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; i++) {
                std::string user_id = "user" + std::to_string(t);
                std::string db_id = "db" + std::to_string(i % 10);
                std::string permission = (i % 2 == 0) ? "read" : "write";
                
                cache.put(user_id, db_id, permission, i % 2 == 0);
                cache.get(user_id, db_id, permission);
                
                if (i % 20 == 0) {
                    cache.invalidate_user(user_id);
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // No crashes = success for thread safety
    auto stats = cache.get_stats();
    TEST_ASSERT(stats.hits + stats.misses > 0, "Should have some cache activity");
    
    return true;
}

// Test 10: Performance benchmark
bool test_performance() {
    PermissionCache cache(100000, 300);
    const int num_operations = 10000;
    
    // Warm up cache
    for (int i = 0; i < 1000; i++) {
        cache.put("user" + std::to_string(i % 100), 
                 "db" + std::to_string(i % 10), 
                 "read", true);
    }
    
    // Benchmark cache hits
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; i++) {
        cache.get("user" + std::to_string(i % 100), 
                 "db" + std::to_string(i % 10), 
                 "read");
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = static_cast<double>(duration.count()) / num_operations;
    
    std::cout << "\n  Average cache lookup time: " << avg_time_us << " μs" << std::endl;
    
    // Target: < 0.001ms = 1 μs per lookup (for cache hits)
    TEST_ASSERT(avg_time_us < 1.0, "Average cache lookup should be < 1 μs");
    
    auto stats = cache.get_stats();
    std::cout << "  Cache hit rate: " << (stats.hit_rate() * 100) << "%" << std::endl;
    TEST_ASSERT(stats.hit_rate() > 0.5, "Should have reasonable hit rate");
    
    return true;
}

// Test 11: Memory usage with max capacity
bool test_memory_limits() {
    const size_t max_entries = 100000;
    PermissionCache cache(max_entries, 300);
    
    // Fill beyond capacity
    for (size_t i = 0; i < max_entries + 1000; i++) {
        cache.put("user" + std::to_string(i), "db1", "read", true);
    }
    
    auto stats = cache.get_stats();
    TEST_ASSERT(stats.evictions >= 1000, "Should have evicted at least 1000 entries");
    
    // Verify oldest entries were evicted (LRU)
    auto oldest = cache.get("user0", "db1", "read");
    TEST_ASSERT(!oldest.has_value(), "Oldest entry should be evicted");
    
    auto newest = cache.get("user" + std::to_string(max_entries + 999), "db1", "read");
    TEST_ASSERT(newest.has_value(), "Newest entry should still be in cache");
    
    return true;
}

// Test 12: Expired entry cleanup
bool test_expired_cleanup() {
    PermissionCache cache(100, 1); // 1 second TTL
    
    // Add entries
    for (int i = 0; i < 10; i++) {
        cache.put("user" + std::to_string(i), "db1", "read", true);
    }
    
    // Wait for expiration
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    
    // Trigger cleanup by trying to access
    for (int i = 0; i < 10; i++) {
        auto result = cache.get("user" + std::to_string(i), "db1", "read");
        TEST_ASSERT(!result.has_value(), "Expired entries should not be returned");
    }
    
    return true;
}

int main() {
    std::cout << "===== Permission Cache Unit Tests (T11.6.3) =====" << std::endl;
    std::cout << std::endl;
    
    int total = 0, passed = 0, failed = 0;
    
    // Run all tests
    RUN_TEST(test_basic_cache_operations);
    RUN_TEST(test_cache_statistics);
    RUN_TEST(test_lru_eviction);
    RUN_TEST(test_ttl_expiration);
    RUN_TEST(test_user_invalidation);
    RUN_TEST(test_database_invalidation);
    RUN_TEST(test_permission_invalidation);
    RUN_TEST(test_clear_cache);
    RUN_TEST(test_thread_safety);
    RUN_TEST(test_performance);
    RUN_TEST(test_memory_limits);
    RUN_TEST(test_expired_cleanup);
    
    // Summary
    std::cout << std::endl;
    std::cout << "===== Test Summary =====" << std::endl;
    std::cout << "Total:  " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success Rate: " << (100.0 * passed / total) << "%" << std::endl;
    
    return (failed == 0) ? 0 : 1;
}
