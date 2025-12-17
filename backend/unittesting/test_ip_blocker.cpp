#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "../src/middleware/ip_blocker.h"

using namespace jadevectordb::middleware;

// ============================================================================
// IPBlocker Tests
// ============================================================================

TEST(IPBlockerTest, NotBlockedInitially) {
    IPBlocker blocker(5, 3600, 600);
    
    EXPECT_FALSE(blocker.is_blocked("192.168.1.1"));
    EXPECT_EQ(blocker.get_failure_count("192.168.1.1"), 0);
}

TEST(IPBlockerTest, RecordFailures) {
    IPBlocker blocker(5, 3600, 600);
    
    std::string ip = "192.168.1.100";
    
    for (int i = 0; i < 4; i++) {
        bool blocked = blocker.record_failure(ip, "invalid_password");
        EXPECT_FALSE(blocked) << "Should not be blocked after " << (i + 1) << " failures";
    }
    
    EXPECT_EQ(blocker.get_failure_count(ip), 4);
    EXPECT_FALSE(blocker.is_blocked(ip));
}

TEST(IPBlockerTest, AutoBlockAfterMaxFailures) {
    IPBlocker blocker(3, 3600, 600);  // Block after 3 failures
    
    std::string ip = "192.168.1.200";
    
    EXPECT_FALSE(blocker.record_failure(ip));
    EXPECT_FALSE(blocker.record_failure(ip));
    
    // 3rd failure should trigger block
    EXPECT_TRUE(blocker.record_failure(ip));
    
    EXPECT_TRUE(blocker.is_blocked(ip));
}

TEST(IPBlockerTest, SuccessLoginClearsFailures) {
    IPBlocker blocker(5, 3600, 600);
    
    std::string ip = "192.168.1.50";
    
    blocker.record_failure(ip);
    blocker.record_failure(ip);
    EXPECT_EQ(blocker.get_failure_count(ip), 2);
    
    // Successful login should clear history
    blocker.record_success(ip);
    EXPECT_EQ(blocker.get_failure_count(ip), 0);
}

TEST(IPBlockerTest, ManualBlock) {
    IPBlocker blocker(10, 3600, 600);
    
    std::string ip = "192.168.1.99";
    
    EXPECT_FALSE(blocker.is_blocked(ip));
    
    blocker.block_ip(ip, 7200, "suspicious_activity");
    
    EXPECT_TRUE(blocker.is_blocked(ip));
    
    int remaining = blocker.remaining_block_seconds(ip);
    EXPECT_GT(remaining, 7190);  // Should be close to 7200
    EXPECT_LT(remaining, 7210);
}

TEST(IPBlockerTest, ManualUnblock) {
    IPBlocker blocker(3, 3600, 600);
    
    std::string ip = "192.168.1.150";
    
    // Trigger auto-block
    blocker.record_failure(ip);
    blocker.record_failure(ip);
    blocker.record_failure(ip);
    
    EXPECT_TRUE(blocker.is_blocked(ip));
    
    // Admin unblock
    EXPECT_TRUE(blocker.unblock_ip(ip));
    EXPECT_FALSE(blocker.is_blocked(ip));
    
    // Unblocking non-blocked IP should return false
    EXPECT_FALSE(blocker.unblock_ip("192.168.1.1"));
}

TEST(IPBlockerTest, RemainingBlockSeconds) {
    IPBlocker blocker(3, 3600, 600);
    
    std::string ip = "192.168.1.175";
    
    blocker.record_failure(ip);
    blocker.record_failure(ip);
    blocker.record_failure(ip);  // Blocked
    
    int remaining = blocker.remaining_block_seconds(ip);
    EXPECT_GT(remaining, 3590);  // Should be close to 3600
    EXPECT_LE(remaining, 3600);
    
    // Non-blocked IP should return 0
    EXPECT_EQ(blocker.remaining_block_seconds("192.168.1.1"), 0);
}

TEST(IPBlockerTest, GetBlockedIPs) {
    IPBlocker blocker(2, 3600, 600);
    
    blocker.record_failure("192.168.1.10");
    blocker.record_failure("192.168.1.10");  // Blocked
    
    blocker.record_failure("192.168.1.20");
    blocker.record_failure("192.168.1.20");  // Blocked
    
    auto blocked_ips = blocker.get_blocked_ips();
    
    EXPECT_EQ(blocked_ips.size(), 2);
    EXPECT_NE(std::find(blocked_ips.begin(), blocked_ips.end(), "192.168.1.10"), 
              blocked_ips.end());
    EXPECT_NE(std::find(blocked_ips.begin(), blocked_ips.end(), "192.168.1.20"), 
              blocked_ips.end());
}

TEST(IPBlockerTest, Statistics) {
    IPBlocker blocker(3, 3600, 600);
    
    blocker.record_failure("192.168.1.1");
    blocker.record_failure("192.168.1.2");
    blocker.record_failure("192.168.1.1");  // 3 total failures
    
    blocker.record_failure("192.168.1.3");
    blocker.record_failure("192.168.1.3");
    blocker.record_failure("192.168.1.3");  // IP 3 blocked, 6 total failures
    
    auto stats = blocker.get_stats();
    
    EXPECT_EQ(stats["total_blocks"], 1);
    EXPECT_EQ(stats["active_blocks"], 1);
    EXPECT_EQ(stats["total_failures"], 6);
}

TEST(IPBlockerTest, ClearAll) {
    IPBlocker blocker(2, 3600, 600);
    
    blocker.record_failure("192.168.1.1");
    blocker.record_failure("192.168.1.1");  // Blocked
    
    blocker.record_failure("192.168.1.2");
    
    EXPECT_TRUE(blocker.is_blocked("192.168.1.1"));
    EXPECT_EQ(blocker.get_failure_count("192.168.1.2"), 1);
    
    blocker.clear_all();
    
    EXPECT_FALSE(blocker.is_blocked("192.168.1.1"));
    EXPECT_EQ(blocker.get_failure_count("192.168.1.2"), 0);
}

TEST(IPBlockerTest, FailureWindow) {
    IPBlocker blocker(3, 3600, 1);  // 1 second failure window
    
    std::string ip = "192.168.1.88";
    
    blocker.record_failure(ip);
    blocker.record_failure(ip);
    
    EXPECT_EQ(blocker.get_failure_count(ip), 2);
    
    // Wait for window to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    
    // Old failures should be cleared
    EXPECT_EQ(blocker.get_failure_count(ip), 0);
}

TEST(IPBlockerTest, BlockExpiration) {
    IPBlocker blocker(2, 1, 600);  // 1 second block duration
    
    std::string ip = "192.168.1.77";
    
    blocker.record_failure(ip);
    blocker.record_failure(ip);  // Blocked
    
    EXPECT_TRUE(blocker.is_blocked(ip));
    
    // Wait for block to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    
    // Should no longer be blocked
    EXPECT_FALSE(blocker.is_blocked(ip));
}

TEST(IPBlockerTest, DifferentReasons) {
    IPBlocker blocker(3, 3600, 600);
    
    std::string ip = "192.168.1.66";
    
    blocker.record_failure(ip, "invalid_password");
    blocker.record_failure(ip, "invalid_username");
    blocker.record_failure(ip, "account_locked");
    
    EXPECT_TRUE(blocker.is_blocked(ip));
}

TEST(IPBlockerTest, MultipleIPsIndependent) {
    IPBlocker blocker(3, 3600, 600);
    
    blocker.record_failure("192.168.1.10");
    blocker.record_failure("192.168.1.10");
    blocker.record_failure("192.168.1.10");  // IP 10 blocked
    
    EXPECT_TRUE(blocker.is_blocked("192.168.1.10"));
    EXPECT_FALSE(blocker.is_blocked("192.168.1.20"));
    
    // IP 20 should be independent
    blocker.record_failure("192.168.1.20");
    EXPECT_FALSE(blocker.is_blocked("192.168.1.20"));
}

TEST(IPBlockerTest, ThreadSafety) {
    IPBlocker blocker(100, 3600, 600);
    
    std::atomic<int> block_triggers{0};
    
    std::vector<std::thread> threads;
    for (int t = 0; t < 10; t++) {
        threads.emplace_back([&, t]() {
            std::string ip = "192.168.1." + std::to_string(t);
            for (int i = 0; i < 150; i++) {
                // record_failure returns true if IP is/becomes blocked
                // Multiple threads on same IP can trigger this multiple times
                if (blocker.record_failure(ip)) {
                    block_triggers++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Each IP should be blocked exactly once, but record_failure 
    // returns true for every call after blocking, so count >= 10
    EXPECT_GE(block_triggers.load(), 10);
    
    // Verify all IPs are actually blocked
    for (int t = 0; t < 10; t++) {
        std::string ip = "192.168.1." + std::to_string(t);
        EXPECT_TRUE(blocker.is_blocked(ip));
    }
}

TEST(IPBlockerTest, HighConcurrencyDifferentIPs) {
    IPBlocker blocker(10, 3600, 600);
    
    std::atomic<int> total_failures{0};
    std::atomic<int> blocks{0};
    
    std::vector<std::thread> threads;
    for (int t = 0; t < 20; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; i++) {
                std::string ip = "10.0." + std::to_string(i / 10) + "." + std::to_string(i % 10);
                total_failures++;
                if (blocker.record_failure(ip)) {
                    blocks++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(total_failures.load(), 2000);
    
    auto stats = blocker.get_stats();
    std::cout << "High concurrency test: " 
              << stats["total_blocks"] << " blocks from " 
              << stats["total_failures"] << " failures" << std::endl;
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(IPBlockerTest, Performance) {
    IPBlocker blocker(1000, 3600, 600);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10000; i++) {
        std::string ip = "192.168." + std::to_string(i / 256) + "." + std::to_string(i % 256);
        blocker.record_failure(ip);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should process 10k failures in < 100ms (10 μs per operation)
    EXPECT_LT(duration.count(), 100000);
    
    std::cout << "IP blocker performance: " 
              << (duration.count() / 10000.0) 
              << " μs per failure record" << std::endl;
}
