#include <gtest/gtest.h>
#include "storage/memory_mapped_vector_store.h"
#include <filesystem>
#include <random>
#include <chrono>
#include <iomanip>

class LargeDatasetTest : public ::testing::Test {
protected:
    std::string test_storage_path_;
    
    void SetUp() override {
        test_storage_path_ = "/tmp/jade_large_dataset_test";
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
    
    size_t get_directory_size(const std::string& path) {
        size_t total_size = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                total_size += entry.file_size();
            }
        }
        return total_size;
    }
    
    void print_memory_stats(const std::string& label) {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.find("VmRSS:") != std::string::npos || 
                line.find("VmSize:") != std::string::npos) {
                std::cout << label << " - " << line << std::endl;
            }
        }
    }
};

// Test 1: Store and retrieve 100K vectors
TEST_F(LargeDatasetTest, Store100KVectors) {
    const int DIMENSION = 128;
    const int NUM_VECTORS = 100000;
    const std::string db_id = "large_db_100k";
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create database
    ASSERT_TRUE(store->create_vector_file(db_id, DIMENSION));
    
    // Store vectors
    std::cout << "Storing " << NUM_VECTORS << " vectors..." << std::endl;
    for (int i = 0; i < NUM_VECTORS; i++) {
        std::string vec_id = "vec_" + std::to_string(i);
        auto vec = generate_random_vector(DIMENSION, i);
        ASSERT_TRUE(store->store_vector(db_id, vec_id, vec));
        
        if (i % 10000 == 0 && i > 0) {
            std::cout << "  Stored " << i << " vectors..." << std::endl;
        }
    }
    
    auto store_time = std::chrono::high_resolution_clock::now();
    auto store_duration = std::chrono::duration_cast<std::chrono::milliseconds>(store_time - start_time);
    std::cout << "Store time: " << store_duration.count() << " ms" << std::endl;
    std::cout << "Store rate: " << (NUM_VECTORS * 1000.0 / store_duration.count()) << " vectors/sec" << std::endl;
    
    // Flush
    store->flush_all(true);
    auto flush_time = std::chrono::high_resolution_clock::now();
    auto flush_duration = std::chrono::duration_cast<std::chrono::milliseconds>(flush_time - store_time);
    std::cout << "Flush time: " << flush_duration.count() << " ms" << std::endl;
    
    // Verify count
    EXPECT_EQ(store->get_vector_count(db_id), NUM_VECTORS);
    
    // File size
    size_t file_size = get_directory_size(test_storage_path_);
    std::cout << "Storage size: " << (file_size / 1024.0 / 1024.0) << " MB" << std::endl;
    double bytes_per_vector = file_size / (double)NUM_VECTORS;
    std::cout << "Bytes per vector: " << bytes_per_vector << std::endl;
    
    // Random retrieve test
    std::cout << "Random retrieve test..." << std::endl;
    auto retrieve_start = std::chrono::high_resolution_clock::now();
    
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> dis(0, NUM_VECTORS - 1);
    
    const int RETRIEVE_COUNT = 1000;
    int success_count = 0;
    for (int i = 0; i < RETRIEVE_COUNT; i++) {
        int idx = dis(gen);
        std::string vec_id = "vec_" + std::to_string(idx);
        auto result = store->retrieve_vector(db_id, vec_id);
        if (result.has_value()) {
            success_count++;
        }
    }
    
    auto retrieve_time = std::chrono::high_resolution_clock::now();
    auto retrieve_duration = std::chrono::duration_cast<std::chrono::microseconds>(retrieve_time - retrieve_start);
    std::cout << "Retrieved " << success_count << "/" << RETRIEVE_COUNT << " vectors" << std::endl;
    std::cout << "Average retrieve time: " << (retrieve_duration.count() / RETRIEVE_COUNT) << " µs" << std::endl;
    
    EXPECT_EQ(success_count, RETRIEVE_COUNT);
}

// Test 2: Multiple databases with 1M total vectors
TEST_F(LargeDatasetTest, MultipleDatabasesWith1MVectors) {
    const int NUM_DATABASES = 10;
    const int VECTORS_PER_DB = 100000; // Total: 1M vectors
    const int DIMENSION = 128;
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    print_memory_stats("Before storing");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Creating " << NUM_DATABASES << " databases with " 
              << VECTORS_PER_DB << " vectors each..." << std::endl;
    
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "large_db_" + std::to_string(db_idx);
        
        ASSERT_TRUE(store->create_vector_file(db_id, DIMENSION));
        
        std::cout << "Database " << (db_idx + 1) << "/" << NUM_DATABASES << "..." << std::endl;
        
        // Use batch store for better performance
        const int BATCH_SIZE = 1000;
        for (int batch_start = 0; batch_start < VECTORS_PER_DB; batch_start += BATCH_SIZE) {
            std::vector<std::pair<std::string, std::vector<float>>> batch;
            
            for (int i = 0; i < BATCH_SIZE && (batch_start + i) < VECTORS_PER_DB; i++) {
                int vec_idx = batch_start + i;
                std::string vec_id = "vec_" + std::to_string(vec_idx);
                auto vec = generate_random_vector(DIMENSION, db_idx * 1000000 + vec_idx);
                batch.emplace_back(vec_id, vec);
            }
            
            store->batch_store(db_id, batch);
            
            if ((batch_start + BATCH_SIZE) % 10000 == 0) {
                std::cout << "  Stored " << (batch_start + BATCH_SIZE) << " vectors..." << std::endl;
            }
        }
        
        // Verify count
        EXPECT_EQ(store->get_vector_count(db_id), VECTORS_PER_DB);
        
        // Flush this database
        store->flush(db_id, false);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nTotal time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Rate: " << (NUM_DATABASES * VECTORS_PER_DB / duration.count()) << " vectors/sec" << std::endl;
    
    // Final flush
    store->flush_all(true);
    
    print_memory_stats("After storing");
    
    // Storage statistics
    size_t total_size = get_directory_size(test_storage_path_);
    std::cout << "\nTotal storage: " << (total_size / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Storage per vector: " << (total_size / (NUM_DATABASES * VECTORS_PER_DB)) << " bytes" << std::endl;
    
    // Verify we can still access data from all databases
    std::cout << "\nVerifying data access across all databases..." << std::endl;
    for (int db_idx = 0; db_idx < NUM_DATABASES; db_idx++) {
        std::string db_id = "large_db_" + std::to_string(db_idx);
        
        // Check first and last vector
        auto first = store->retrieve_vector(db_id, "vec_0");
        auto last = store->retrieve_vector(db_id, "vec_" + std::to_string(VECTORS_PER_DB - 1));
        
        EXPECT_TRUE(first.has_value());
        EXPECT_TRUE(last.has_value());
    }
    
    std::cout << "All databases verified!" << std::endl;
}

// Test 3: Sequential scan performance
TEST_F(LargeDatasetTest, SequentialScanPerformance) {
    const int DIMENSION = 256;
    const int NUM_VECTORS = 50000;
    const std::string db_id = "scan_db";
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    // Setup
    store->create_vector_file(db_id, DIMENSION);
    for (int i = 0; i < NUM_VECTORS; i++) {
        auto vec = generate_random_vector(DIMENSION, i);
        store->store_vector(db_id, "vec_" + std::to_string(i), vec);
    }
    store->flush_all(true);
    
    // Sequential scan
    std::cout << "Sequential scan of " << NUM_VECTORS << " vectors..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    int success_count = 0;
    for (int i = 0; i < NUM_VECTORS; i++) {
        auto result = store->retrieve_vector(db_id, "vec_" + std::to_string(i));
        if (result.has_value()) {
            success_count++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Scanned " << success_count << " vectors in " << duration.count() << " ms" << std::endl;
    std::cout << "Scan rate: " << (NUM_VECTORS * 1000.0 / duration.count()) << " vectors/sec" << std::endl;
    
    EXPECT_EQ(success_count, NUM_VECTORS);
}

// Test 4: Update performance on large dataset
TEST_F(LargeDatasetTest, UpdatePerformanceLargeDataset) {
    const int DIMENSION = 128;
    const int NUM_VECTORS = 50000;
    const std::string db_id = "update_db";
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    // Initial population
    std::cout << "Initial population..." << std::endl;
    store->create_vector_file(db_id, DIMENSION);
    for (int i = 0; i < NUM_VECTORS; i++) {
        auto vec = generate_random_vector(DIMENSION, i);
        store->store_vector(db_id, "vec_" + std::to_string(i), vec);
    }
    store->flush_all(true);
    
    // Update 10% of vectors
    std::cout << "Updating 10% of vectors..." << std::endl;
    const int UPDATE_COUNT = NUM_VECTORS / 10;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < UPDATE_COUNT; i++) {
        auto vec = generate_random_vector(DIMENSION, i + 100000);
        store->update_vector(db_id, "vec_" + std::to_string(i), vec);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Updated " << UPDATE_COUNT << " vectors in " << duration.count() << " ms" << std::endl;
    std::cout << "Update rate: " << (UPDATE_COUNT * 1000.0 / duration.count()) << " vectors/sec" << std::endl;
    
    store->flush_all(true);
    
    // Verify updates
    auto result = store->retrieve_vector(db_id, "vec_0");
    ASSERT_TRUE(result.has_value());
    auto expected = generate_random_vector(DIMENSION, 100000);
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ((*result)[i], expected[i]);
    }
}

// Test 5: Delete performance on large dataset
TEST_F(LargeDatasetTest, DeletePerformanceLargeDataset) {
    const int DIMENSION = 128;
    const int NUM_VECTORS = 50000;
    const std::string db_id = "delete_db";
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    // Initial population
    std::cout << "Initial population..." << std::endl;
    store->create_vector_file(db_id, DIMENSION);
    for (int i = 0; i < NUM_VECTORS; i++) {
        auto vec = generate_random_vector(DIMENSION, i);
        store->store_vector(db_id, "vec_" + std::to_string(i), vec);
    }
    store->flush_all(true);
    
    size_t initial_size = get_directory_size(test_storage_path_);
    std::cout << "Initial storage: " << (initial_size / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Delete 50% of vectors
    std::cout << "Deleting 50% of vectors..." << std::endl;
    const int DELETE_COUNT = NUM_VECTORS / 2;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < DELETE_COUNT; i++) {
        store->delete_vector(db_id, "vec_" + std::to_string(i * 2));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Deleted " << DELETE_COUNT << " vectors in " << duration.count() << " ms" << std::endl;
    std::cout << "Delete rate: " << (DELETE_COUNT * 1000.0 / duration.count()) << " vectors/sec" << std::endl;
    
    store->flush_all(true);
    
    // Verify count
    size_t remaining = store->get_vector_count(db_id);
    EXPECT_EQ(remaining, NUM_VECTORS - DELETE_COUNT);
    
    std::cout << "Remaining vectors: " << remaining << std::endl;
}

// Test 6: Memory usage with different vector dimensions
TEST_F(LargeDatasetTest, MemoryUsageDifferentDimensions) {
    const int NUM_VECTORS = 10000;
    std::vector<int> dimensions = {64, 128, 256, 512, 1024, 2048};
    
    auto store = std::make_unique<jadevectordb::MemoryMappedVectorStore>(test_storage_path_);
    
    std::cout << "\nMemory usage by dimension:" << std::endl;
    std::cout << std::setw(10) << "Dimension" 
              << std::setw(15) << "File Size (MB)" 
              << std::setw(20) << "Bytes per Vector" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (int dim : dimensions) {
        std::string db_id = "dim_" + std::to_string(dim);
        store->create_vector_file(db_id, dim);
        
        for (int i = 0; i < NUM_VECTORS; i++) {
            auto vec = generate_random_vector(dim, i);
            store->store_vector(db_id, "vec_" + std::to_string(i), vec);
        }
        
        store->flush(db_id, true);
        
        // Measure storage for this database
        std::string db_path = test_storage_path_ + "/" + db_id;
        size_t db_size = get_directory_size(db_path);
        
        std::cout << std::setw(10) << dim
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << (db_size / 1024.0 / 1024.0)
                  << std::setw(20) << (db_size / NUM_VECTORS) << std::endl;
    }
}
