#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <random>
#include <vector>
#include <algorithm>

class MemoryPressureTest : public ::testing::Test {
protected:
    std::string test_storage_path_;
    
    void SetUp() override {
        test_storage_path_ = "/tmp/jade_memory_pressure_test";
        std::filesystem::remove_all(test_storage_path_);
        std::filesystem::create_directories(test_storage_path_);
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_storage_path_);
    }
    
    std::vector<float> generate_random_vector(size_t dimension, unsigned int seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        std::vector<float> vec(dimension);
        for (size_t i = 0; i < dimension; i++) {
            vec[i] = dis(gen);
        }
        return vec;
    }
    
    size_t get_process_memory_mb() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.find("VmRSS:") != std::string::npos) {
                // Extract the number
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    size_t kb = std::stoul(line.substr(pos));
                    return kb / 1024; // Convert to MB
                }
            }
        }
        return 0;
    }
};

// Test 1: LRU eviction with many databases
TEST_F(MemoryPressureTest, LRUEvictionManyDatabases) {
    const int DIMENSION = 128;
    const int NUM_DATABASES = 150; // More than max_open_files (100)
    const int VECTORS_PER_DB = 100;
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(
        test_storage_path_, 50); // Max 50 open files
    
    std::cout << "Creating " << NUM_DATABASES << " databases (max 50 open at once)..." << std::endl;
    
    size_t memory_before = get_process_memory_mb();
    std::cout << "Memory before: " << memory_before << " MB" << std::endl;
    
    // Create and populate databases
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "db_" + std::to_string(db_idx);
        
        ASSERT_TRUE(store->create_vector_file(db_id, DIMENSION));
        
        for (int i = 0; i < VECTORS_PER_DB; i++) {
            auto vec = generate_random_vector(DIMENSION, db_idx * 1000 + i);
            ASSERT_TRUE(store->store_vector(db_id, "vec_" + std::to_string(i), vec));
        }
        
        if (db_idx % 25 == 0) {
            size_t memory_now = get_process_memory_mb();
            std::cout << "After " << db_idx << " databases: " << memory_now << " MB" << std::endl;
        }
    }
    
    size_t memory_after = get_process_memory_mb();
    std::cout << "Memory after: " << memory_after << " MB" << std::endl;
    std::cout << "Memory increase: " << (memory_after - memory_before) << " MB" << std::endl;
    
    // Verify LRU eviction worked - memory shouldn't grow unbounded
    EXPECT_LT(memory_after - memory_before, 500) << "Memory usage should be bounded by LRU eviction";
    
    // Flush all
    store->flush_all(false);
    
    // Access all databases in random order to test LRU
    std::cout << "\nAccessing all databases in random order..." << std::endl;
    std::vector<int> access_order;
    for (int i = 0; i < NUM_DATABASES; i++) {
        access_order.push_back(i);
    }
    std::shuffle(access_order.begin(), access_order.end(), std::mt19937(42));
    
    int success_count = 0;
    for (int db_idx : access_order) {
        std::string db_id = "db_" + std::to_string(db_idx);
        auto result = store->retrieve_vector(db_id, "vec_0");
        if (result.has_value()) {
            success_count++;
        }
    }
    
    EXPECT_EQ(success_count, NUM_DATABASES) << "All databases should be accessible despite LRU eviction";
    std::cout << "Successfully accessed all " << success_count << " databases" << std::endl;
}

// Test 2: Repeated access pattern with LRU
TEST_F(MemoryPressureTest, RepeatedAccessPattern) {
    const int DIMENSION = 256;
    const int NUM_DATABASES = 100;
    const int VECTORS_PER_DB = 50;
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(
        test_storage_path_, 20); // Max 20 open files
    
    // Create databases
    std::cout << "Creating " << NUM_DATABASES << " databases..." << std::endl;
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "db_" + std::to_string(db_idx);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < VECTORS_PER_DB; i++) {
            auto vec = generate_random_vector(DIMENSION, db_idx * 1000 + i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
    }
    
    store->flush_all(false);
    
    // Simulate workload: repeatedly access "hot" databases
    std::cout << "Simulating workload with hot databases..." << std::endl;
    std::vector<int> hot_dbs = {0, 1, 2, 3, 4}; // First 5 databases are "hot"
    
    size_t memory_before = get_process_memory_mb();
    
    const int NUM_ACCESSES = 10000;
    int hit_count = 0;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> hot_dist(0, 4);
    std::uniform_int_distribution<int> cold_dist(5, NUM_DATABASES - 1);
    std::uniform_int_distribution<int> hot_cold_dist(0, 9); // 80% hot, 20% cold
    
    for (int i = 0; i < NUM_ACCESSES; i++) {
        int db_idx;
        if (hot_cold_dist(gen) < 8) {
            db_idx = hot_dbs[hot_dist(gen)]; // Hot database
        } else {
            db_idx = cold_dist(gen); // Cold database
        }
        
        std::string db_id = "db_" + std::to_string(db_idx);
        auto result = store->retrieve_vector(db_id, "vec_0");
        if (result.has_value()) {
            hit_count++;
        }
    }
    
    size_t memory_after = get_process_memory_mb();
    
    std::cout << "Completed " << NUM_ACCESSES << " accesses with " << hit_count << " hits" << std::endl;
    std::cout << "Memory used: " << (memory_after - memory_before) << " MB" << std::endl;
    
    EXPECT_EQ(hit_count, NUM_ACCESSES);
    EXPECT_LT(memory_after - memory_before, 200) << "Memory should stay bounded with LRU";
}

// Test 3: Large vectors with limited open files
TEST_F(MemoryPressureTest, LargeVectorsLimitedFiles) {
    const int DIMENSION = 4096; // Large vectors
    const int NUM_DATABASES = 50;
    const int VECTORS_PER_DB = 100;
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(
        test_storage_path_, 10); // Very limited: only 10 open files
    
    std::cout << "Creating " << NUM_DATABASES << " databases with large vectors (dim=" << DIMENSION << ")..." << std::endl;
    
    size_t memory_start = get_process_memory_mb();
    
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "large_db_" + std::to_string(db_idx);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < VECTORS_PER_DB; i++) {
            auto vec = generate_random_vector(DIMENSION, db_idx * 10000 + i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        if (db_idx % 10 == 0) {
            std::cout << "Created " << db_idx << " databases" << std::endl;
        }
    }
    
    size_t memory_after_creation = get_process_memory_mb();
    std::cout << "Memory after creation: " << (memory_after_creation - memory_start) << " MB increase" << std::endl;
    
    // Access pattern that forces frequent eviction
    std::cout << "\nForcing frequent LRU eviction..." << std::endl;
    for (int round = 0; round < 5; round++) {
        for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
            std::string db_id = "large_db_" + std::to_string(db_idx);
            auto result = store->retrieve_vector(db_id, "vec_0");
            EXPECT_TRUE(result.has_value());
        }
        std::cout << "Round " << (round + 1) << " complete" << std::endl;
    }
    
    size_t memory_final = get_process_memory_mb();
    std::cout << "Final memory: " << (memory_final - memory_start) << " MB increase" << std::endl;
    
    // Memory shouldn't grow significantly despite accessing all databases
    EXPECT_LT(memory_final - memory_start, 500);
}

// Test 4: Concurrent access with limited open files
TEST_F(MemoryPressureTest, ConcurrentAccessLimitedFiles) {
    const int DIMENSION = 128;
    const int NUM_DATABASES = 80;
    const int VECTORS_PER_DB = 50;
    const int NUM_THREADS = 8;
    
    auto store = std::make_shared<jadevectordb::MemoryMappedVectorStore>(
        test_storage_path_, 15); // Limited to 15 open files
    
    // Create databases
    std::cout << "Creating " << NUM_DATABASES << " databases..." << std::endl;
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "db_" + std::to_string(db_idx);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < VECTORS_PER_DB; i++) {
            auto vec = generate_random_vector(DIMENSION, db_idx * 1000 + i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
    }
    
    store->flush_all(false);
    
    size_t memory_before = get_process_memory_mb();
    std::cout << "Memory before concurrent access: " << memory_before << " MB" << std::endl;
    
    // Concurrent access
    std::cout << "Starting " << NUM_THREADS << " concurrent threads..." << std::endl;
    std::vector<std::thread> threads;
    std::atomic<int> total_accesses{0};
    
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937 gen(t * 12345);
            std::uniform_int_distribution<int> db_dist(0, NUM_DATABASES - 1);
            std::uniform_int_distribution<int> vec_dist(0, VECTORS_PER_DB - 1);
            
            for (int i = 0; i < 1000; i++) {
                int db_idx = db_dist(gen);
                int vec_idx = vec_dist(gen);
                
                std::string db_id = "db_" + std::to_string(db_idx);
                std::string vec_id = "vec_" + std::to_string(vec_idx);
                
                auto result = store->retrieve_vector(db_id, vec_id);
                if (result.has_value()) {
                    total_accesses++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    size_t memory_after = get_process_memory_mb();
    std::cout << "Memory after concurrent access: " << memory_after << " MB" << std::endl;
    std::cout << "Total successful accesses: " << total_accesses.load() << std::endl;
    
    EXPECT_EQ(total_accesses.load(), NUM_THREADS * 1000);
    EXPECT_LT(memory_after - memory_before, 300);
}

// Test 5: Stress test with minimal open files
TEST_F(MemoryPressureTest, StressTestMinimalOpenFiles) {
    const int DIMENSION = 128;
    const int NUM_DATABASES = 100;
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(
        test_storage_path_, 5); // Extreme: only 5 open files
    
    std::cout << "Creating " << NUM_DATABASES << " databases with only 5 max open files..." << std::endl;
    
    // Create databases
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "db_" + std::to_string(db_idx);
        store->create_vector_file(db_id, DIMENSION);
        
        // Store a few vectors
        for (int i = 0; i < 10; i++) {
            auto vec = generate_random_vector(DIMENSION, db_idx * 100 + i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
    }
    
    // Rapid sequential access to force constant eviction
    std::cout << "Rapidly accessing all databases..." << std::endl;
    int successful_accesses = 0;
    
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "db_" + std::to_string(db_idx);
        auto result = store->retrieve_vector(db_id, "vec_0");
        if (result.has_value()) {
            successful_accesses++;
        }
    }
    
    std::cout << "Successfully accessed " << successful_accesses << "/" << NUM_DATABASES << " databases" << std::endl;
    EXPECT_EQ(successful_accesses, NUM_DATABASES);
    
    // Reverse order access
    std::cout << "Accessing in reverse order..." << std::endl;
    successful_accesses = 0;
    
    for (int db_idx = NUM_DATABASES - 1; db_idx >= 0; db_idx--) {
        std::string db_id = "db_" + std::to_string(db_idx);
        auto result = store->retrieve_vector(db_id, "vec_5");
        if (result.has_value()) {
            successful_accesses++;
        }
    }
    
    EXPECT_EQ(successful_accesses, NUM_DATABASES);
    std::cout << "Stress test completed successfully!" << std::endl;
}

// Test 6: Memory stability over time
TEST_F(MemoryPressureTest, MemoryStabilityOverTime) {
    const int DIMENSION = 256;
    const int NUM_DATABASES = 50;
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(
        test_storage_path_, 25);
    
    // Create databases
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "db_" + std::to_string(db_idx);
        store->create_vector_file(db_id, DIMENSION);
        
        for (int i = 0; i < 100; i++) {
            auto vec = generate_random_vector(DIMENSION, db_idx * 1000 + i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
    }
    
    store->flush_all(false);
    
    // Monitor memory over multiple access cycles
    std::cout << "\nMonitoring memory stability over 10 cycles..." << std::endl;
    
    std::vector<size_t> memory_samples;
    size_t baseline_memory = get_process_memory_mb();
    memory_samples.push_back(baseline_memory);
    
    for (int cycle = 0; cycle < 10; cycle++) {
        // Access all databases
        for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
            std::string db_id = "db_" + std::to_string(db_idx);
            store->retrieve_vector(db_id, "vec_" + std::to_string(cycle * 10));
        }
        
        size_t current_memory = get_process_memory_mb();
        memory_samples.push_back(current_memory);
        std::cout << "Cycle " << (cycle + 1) << ": " << current_memory << " MB" << std::endl;
    }
    
    // Check that memory is stable (not growing unbounded)
    size_t max_memory = *std::max_element(memory_samples.begin(), memory_samples.end());
    size_t min_memory = *std::min_element(memory_samples.begin(), memory_samples.end());
    
    std::cout << "\nMemory range: " << min_memory << " - " << max_memory << " MB" << std::endl;
    std::cout << "Variation: " << (max_memory - min_memory) << " MB" << std::endl;
    
    // Memory variation should be reasonable (not growing unbounded)
    EXPECT_LT(max_memory - baseline_memory, 200) << "Memory should remain stable";
}
