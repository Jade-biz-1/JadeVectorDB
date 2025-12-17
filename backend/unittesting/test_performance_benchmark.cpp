#include "services/sqlite_persistence_layer.h"
#include "models/auth.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <random>
#include <iomanip>

using namespace jadevectordb;

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "  ❌ FAILED: " << message << std::endl; \
            return false; \
        } \
        std::cout << "  ✓ " << message << std::endl; \
    } while (0)

#define BENCHMARK(name) \
    std::cout << "\n" << name << std::endl

// Timer utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
};

// Helper to generate random string
std::string random_string(size_t length) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);
    
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += charset[dis(gen)];
    }
    return result;
}

bool benchmark_user_operations(SQLitePersistenceLayer& persistence) {
    BENCHMARK("1. User Operations Benchmark");
    
    // Create 100 users and measure time
    const int num_users = 100;
    std::vector<std::string> user_ids;
    user_ids.reserve(num_users);
    
    Timer timer;
    for (int i = 0; i < num_users; ++i) {
        std::string username = "user_" + std::to_string(i);
        auto result = persistence.create_user(
            username,
            username + "@test.com",
            "hash_" + std::to_string(i),
            "salt_" + std::to_string(i)
        );
        if (result.has_value()) {
            user_ids.push_back(result.value());
        }
    }
    double create_time = timer.elapsed_ms();
    double avg_create_time = create_time / num_users;
    
    std::cout << "  Created " << num_users << " users in " << std::fixed << std::setprecision(2) 
              << create_time << "ms (avg: " << avg_create_time << "ms per user)" << std::endl;
    
    // Lookup benchmark
    timer.reset();
    int lookup_count = 0;
    for (const auto& user_id : user_ids) {
        auto result = persistence.get_user(user_id);
        if (result.has_value()) {
            lookup_count++;
        }
    }
    double lookup_time = timer.elapsed_ms();
    double avg_lookup_time = lookup_time / num_users;
    
    std::cout << "  Looked up " << lookup_count << " users in " << std::fixed << std::setprecision(2)
              << lookup_time << "ms (avg: " << avg_lookup_time << "ms per lookup)" << std::endl;
    
    ASSERT(avg_create_time < 10.0, "User creation avg < 10ms");
    ASSERT(avg_lookup_time < 10.0, "User lookup avg < 10ms");
    
    return true;
}

bool benchmark_permission_checks(SQLitePersistenceLayer& persistence) {
    BENCHMARK("2. Permission Check Benchmark");
    
    // Create a test user and database
    auto user_result = persistence.create_user("perm_user", "perm@test.com", "hash", "salt");
    if (!user_result.has_value()) {
        std::cerr << "  Failed to create test user" << std::endl;
        return false;
    }
    std::string user_id = user_result.value();
    
    // Assign role
    auto role_result = persistence.assign_role_to_user(user_id, "role_user");
    if (!role_result.has_value()) {
        std::cerr << "  Failed to assign role" << std::endl;
        return false;
    }
    
    // Create database
    auto db_result = persistence.store_database_metadata(
        "perm_test_db", "Test", user_id, 384, "HNSW", "{}"
    );
    if (!db_result.has_value()) {
        std::cerr << "  Failed to create database" << std::endl;
        return false;
    }
    std::string db_id = db_result.value();
    
    // Grant permission
    auto grant_result = persistence.grant_database_permission(
        db_id, "user", user_id, "perm_db_read", user_id
    );
    if (!grant_result.has_value()) {
        std::cerr << "  Failed to grant permission" << std::endl;
        return false;
    }
    
    // Benchmark permission checks
    const int num_checks = 1000;
    Timer timer;
    int success_count = 0;
    
    for (int i = 0; i < num_checks; ++i) {
        auto check_result = persistence.check_database_permission(db_id, user_id, "database:read");
        if (check_result.has_value() && check_result.value()) {
            success_count++;
        }
    }
    
    double total_time = timer.elapsed_ms();
    double avg_time = total_time / num_checks;
    
    std::cout << "  Performed " << success_count << "/" << num_checks << " permission checks in "
              << std::fixed << std::setprecision(2) << total_time << "ms (avg: " 
              << avg_time << "ms per check)" << std::endl;
    
    ASSERT(success_count == num_checks, "All permission checks succeeded");
    ASSERT(avg_time < 5.0, "Permission check avg < 5ms");
    
    return true;
}

bool benchmark_concurrent_access(SQLitePersistenceLayer& persistence) {
    BENCHMARK("3. Concurrent Access Benchmark");
    
    const int num_threads = 10;
    const int operations_per_thread = 100;
    const int total_operations = num_threads * operations_per_thread;
    
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    Timer timer;
    
    // Launch threads that create users concurrently
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&persistence, &success_count, &failure_count, t, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                std::string username = "concurrent_" + std::to_string(t) + "_" + std::to_string(i);
                auto result = persistence.create_user(
                    username,
                    username + "@test.com",
                    random_string(32),
                    random_string(16)
                );
                if (result.has_value()) {
                    success_count++;
                } else {
                    failure_count++;
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    double total_time = timer.elapsed_ms();
    double avg_time = total_time / total_operations;
    
    std::cout << "  " << num_threads << " threads performed " << total_operations 
              << " operations in " << std::fixed << std::setprecision(2) << total_time 
              << "ms (avg: " << avg_time << "ms per op)" << std::endl;
    std::cout << "  Success: " << success_count << ", Failures: " << failure_count << std::endl;
    
    ASSERT(success_count >= total_operations * 0.95, "At least 95% of concurrent operations succeeded");
    
    return true;
}

bool benchmark_batch_operations(SQLitePersistenceLayer& persistence) {
    BENCHMARK("4. Batch Operations Benchmark");
    
    // Create a system user first
    auto system_user_result = persistence.create_user("system", "system@test.com", "hash", "salt");
    if (!system_user_result.has_value()) {
        std::cerr << "  Failed to create system user" << std::endl;
        return false;
    }
    std::string system_user_id = system_user_result.value();
    
    // Create database for testing
    auto db_result = persistence.store_database_metadata(
        "batch_test_db", "Batch Test", system_user_id, 128, "HNSW", "{}"
    );
    if (!db_result.has_value()) {
        std::cerr << "  Failed to create test database" << std::endl;
        return false;
    }
    std::string db_id = db_result.value();
    
    // Benchmark: Create users and assign roles in batch
    const int batch_size = 50;
    Timer timer;
    
    for (int i = 0; i < batch_size; ++i) {
        std::string username = "batch_user_" + std::to_string(i);
        auto user_result = persistence.create_user(
            username, username + "@test.com", "hash_" + std::to_string(i), "salt_" + std::to_string(i)
        );
        if (user_result.has_value()) {
            auto role_result = persistence.assign_role_to_user(user_result.value(), "role_user");
        }
    }
    
    double batch_time = timer.elapsed_ms();
    double avg_batch_time = batch_time / batch_size;
    
    std::cout << "  Created and assigned roles to " << batch_size << " users in "
              << std::fixed << std::setprecision(2) << batch_time 
              << "ms (avg: " << avg_batch_time << "ms per user)" << std::endl;
    
    ASSERT(avg_batch_time < 15.0, "Batch operation avg < 15ms per user");
    
    return true;
}

bool benchmark_query_performance(SQLitePersistenceLayer& persistence) {
    BENCHMARK("5. Query Performance Benchmark");
    
    // List all users (should be many from previous tests)
    Timer timer;
    auto list_result = persistence.list_users(1000, 0);
    double list_time = timer.elapsed_ms();
    
    int user_count = 0;
    if (list_result.has_value()) {
        user_count = list_result.value().size();
    }
    
    std::cout << "  Listed " << user_count << " users in " 
              << std::fixed << std::setprecision(2) << list_time << "ms" << std::endl;
    
    // Get audit logs
    timer.reset();
    auto audit_result = persistence.get_audit_logs(100, 0);
    double audit_time = timer.elapsed_ms();
    
    int audit_count = 0;
    if (audit_result.has_value()) {
        audit_count = audit_result.value().size();
    }
    
    std::cout << "  Retrieved " << audit_count << " audit logs in "
              << std::fixed << std::setprecision(2) << audit_time << "ms" << std::endl;
    
    ASSERT(list_time < 100.0, "List users query < 100ms");
    ASSERT(audit_time < 50.0, "Audit log query < 50ms");
    
    return true;
}

int main() {
    std::cout << "=== SQLitePersistenceLayer Performance Benchmark ===" << std::endl;
    std::cout << "\nTargets:" << std::endl;
    std::cout << "  • User operations: < 10ms" << std::endl;
    std::cout << "  • Permission checks: < 5ms" << std::endl;
    std::cout << "  • Concurrent access: 1000+ operations" << std::endl;
    
    // Use temporary directory for testing
    const char* test_dir = "/tmp/jadevectordb_benchmark";
    system(("rm -rf " + std::string(test_dir)).c_str());
    system(("mkdir -p " + std::string(test_dir)).c_str());
    
    SQLitePersistenceLayer persistence(test_dir);
    auto init_result = persistence.initialize();
    if (!init_result.has_value()) {
        std::cerr << "Failed to initialize persistence layer: " 
                  << init_result.error().message << std::endl;
        return 1;
    }
    
    std::cout << "\nDatabase: " << test_dir << "/system.db" << std::endl;
    
    bool all_passed = true;
    
    all_passed = benchmark_user_operations(persistence) && all_passed;
    all_passed = benchmark_permission_checks(persistence) && all_passed;
    all_passed = benchmark_concurrent_access(persistence) && all_passed;
    all_passed = benchmark_batch_operations(persistence) && all_passed;
    all_passed = benchmark_query_performance(persistence) && all_passed;
    
    // Cleanup
    auto close_result = persistence.close();
    if (!close_result.has_value()) {
        std::cerr << "\nWarning: Failed to close persistence layer" << std::endl;
    }
    
    std::cout << "\n=== BENCHMARK " << (all_passed ? "PASSED ✓" : "FAILED ❌") << " ===" << std::endl;
    
    return all_passed ? 0 : 1;
}
