#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "../src/middleware/rate_limiter.h"

using namespace jadevectordb::middleware;

// ============================================================================
// TokenBucket Tests
// ============================================================================

TEST(TokenBucketTest, InitialCapacity) {
    TokenBucket bucket(10, 1.0);  // 10 tokens, 1 per second
    
    // Should have full capacity initially
    EXPECT_EQ(bucket.available_tokens(), 10);
    EXPECT_TRUE(bucket.try_consume(1));
    EXPECT_EQ(bucket.available_tokens(), 9);
}

TEST(TokenBucketTest, ConsumeMultipleTokens) {
    TokenBucket bucket(10, 1.0);
    
    EXPECT_TRUE(bucket.try_consume(5));
    EXPECT_EQ(bucket.available_tokens(), 5);
    
    EXPECT_TRUE(bucket.try_consume(5));
    EXPECT_EQ(bucket.available_tokens(), 0);
    
    // Should fail when empty
    EXPECT_FALSE(bucket.try_consume(1));
}

TEST(TokenBucketTest, RefillRate) {
    TokenBucket bucket(10, 10.0);  // 10 tokens per second
    
    // Consume all tokens
    EXPECT_TRUE(bucket.try_consume(10));
    EXPECT_EQ(bucket.available_tokens(), 0);
    
    // Wait 100ms (should add ~1 token)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_TRUE(bucket.try_consume(1));
    EXPECT_EQ(bucket.available_tokens(), 0);
}

TEST(TokenBucketTest, MaxCapacity) {
    TokenBucket bucket(5, 10.0);  // 5 max, 10 per second
    
    // Start full
    EXPECT_EQ(bucket.available_tokens(), 5);
    
    // Wait (should not exceed capacity)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(bucket.available_tokens(), 5);  // Still at max
}

TEST(TokenBucketTest, RetryAfterSeconds) {
    TokenBucket bucket(10, 1.0);  // 1 token per second
    
    // Consume all tokens
    EXPECT_TRUE(bucket.try_consume(10));
    
    // Should indicate ~1 second wait
    double retry_after = bucket.retry_after_seconds();
    EXPECT_GT(retry_after, 0.9);
    EXPECT_LT(retry_after, 1.1);
}

// ============================================================================
// RateLimiter Tests
// ============================================================================

TEST(RateLimiterTest, BasicRateLimiting) {
    RateLimiter limiter(5, 1.0);  // 5 requests, 1 per second
    
    std::string key = "test_key";
    
    // Should allow first 5 requests
    for (int i = 0; i < 5; i++) {
        EXPECT_TRUE(limiter.allow(key)) << "Request " << i << " should be allowed";
    }
    
    // 6th request should be rate limited
    EXPECT_FALSE(limiter.allow(key));
}

TEST(RateLimiterTest, PerKeyRateLimiting) {
    RateLimiter limiter(3, 1.0);
    
    std::string key1 = "user1";
    std::string key2 = "user2";
    
    // Each key should have independent rate limits
    EXPECT_TRUE(limiter.allow(key1));
    EXPECT_TRUE(limiter.allow(key1));
    EXPECT_TRUE(limiter.allow(key1));
    EXPECT_FALSE(limiter.allow(key1));  // key1 rate limited
    
    // key2 should still be allowed
    EXPECT_TRUE(limiter.allow(key2));
    EXPECT_TRUE(limiter.allow(key2));
    EXPECT_TRUE(limiter.allow(key2));
    EXPECT_FALSE(limiter.allow(key2));  // key2 rate limited
}

TEST(RateLimiterTest, RetryAfter) {
    RateLimiter limiter(2, 1.0);
    
    std::string key = "test";
    
    EXPECT_TRUE(limiter.allow(key));
    EXPECT_TRUE(limiter.allow(key));
    EXPECT_FALSE(limiter.allow(key));
    
    // Should indicate wait time
    double retry_after = limiter.retry_after_seconds(key);
    EXPECT_GT(retry_after, 0.0);
    EXPECT_LT(retry_after, 2.0);
}

TEST(RateLimiterTest, Reset) {
    RateLimiter limiter(2, 1.0);
    
    std::string key = "test";
    
    EXPECT_TRUE(limiter.allow(key));
    EXPECT_TRUE(limiter.allow(key));
    EXPECT_FALSE(limiter.allow(key));
    
    // Reset should allow requests again
    limiter.reset(key);
    EXPECT_TRUE(limiter.allow(key));
}

TEST(RateLimiterTest, ClearAll) {
    RateLimiter limiter(1, 1.0);
    
    EXPECT_TRUE(limiter.allow("key1"));
    EXPECT_TRUE(limiter.allow("key2"));
    EXPECT_TRUE(limiter.allow("key3"));
    
    EXPECT_FALSE(limiter.allow("key1"));
    EXPECT_FALSE(limiter.allow("key2"));
    EXPECT_FALSE(limiter.allow("key3"));
    
    limiter.clear_all();
    
    EXPECT_TRUE(limiter.allow("key1"));
    EXPECT_TRUE(limiter.allow("key2"));
    EXPECT_TRUE(limiter.allow("key3"));
}

TEST(RateLimiterTest, Statistics) {
    RateLimiter limiter(5, 1.0);
    
    limiter.allow("key1");
    limiter.allow("key2");
    limiter.allow("key1");  // 3 total requests, 2 keys
    
    auto stats = limiter.get_stats();
    
    EXPECT_EQ(stats["total_buckets"], 2);
    EXPECT_EQ(stats["total_requests"], 3);
    EXPECT_EQ(stats["rate_limited_requests"], 0);
}

TEST(RateLimiterTest, ThreadSafety) {
    RateLimiter limiter(100, 100.0);
    std::string key = "concurrent_test";
    
    std::atomic<int> allowed_count{0};
    std::atomic<int> denied_count{0};
    
    // 10 threads, 20 requests each
    std::vector<std::thread> threads;
    for (int t = 0; t < 10; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 20; i++) {
                if (limiter.allow(key)) {
                    allowed_count++;
                } else {
                    denied_count++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Should have allowed exactly 100 requests (capacity)
    // and denied the rest (100 total - 100 allowed = 100 denied)
    EXPECT_EQ(allowed_count.load(), 100);
    EXPECT_EQ(denied_count.load(), 100);
}

TEST(RateLimiterTest, HighConcurrency) {
    RateLimiter limiter(1000, 1000.0);
    
    std::atomic<int> total_requests{0};
    std::atomic<int> allowed{0};
    
    std::vector<std::thread> threads;
    for (int t = 0; t < 20; t++) {
        threads.emplace_back([&]() {
            std::string key = "thread_" + std::to_string(t);
            for (int i = 0; i < 100; i++) {
                total_requests++;
                if (limiter.allow(key)) {
                    allowed++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(total_requests.load(), 2000);
    // Each thread should get 1000 capacity, all should be allowed
    EXPECT_EQ(allowed.load(), 2000);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(RateLimiterTest, Performance) {
    RateLimiter limiter(10000, 10000.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10000; i++) {
        limiter.allow("perf_test");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should process 10k requests in < 100ms (10 μs per request)
    EXPECT_LT(duration.count(), 100000);
    
    std::cout << "Rate limiter performance: " 
              << (duration.count() / 10000.0) 
              << " μs per request" << std::endl;
}
