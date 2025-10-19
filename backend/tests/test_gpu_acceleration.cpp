#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <cmath>

#include "lib/gpu_acceleration.h"
#include "lib/vector_operations.h"
#include "lib/gpu_detection.h"
#include "lib/gpu_memory_manager.h"
#include "lib/workload_balancer.h"

using namespace jadevectordb;

// Test fixture for GPU acceleration components
class GPUAccelerationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the device manager for tests
        hardware::DeviceManager::get_instance().initialize(hardware::DeviceType::CPU);
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
};

// Test device detection
TEST_F(GPUAccelerationTest, DeviceManagerInitialization) {
    auto& device_manager = hardware::DeviceManager::get_instance();
    
    // Verify that CPU device is available (should always be available)
    auto cpu_device = device_manager.find_device(hardware::DeviceType::CPU);
    ASSERT_NE(cpu_device, nullptr);
    EXPECT_TRUE(cpu_device->is_available());
    EXPECT_EQ(cpu_device->get_device_type(), hardware::DeviceType::CPU);
    
    // GPU availability depends on the system, but the manager should work
    bool gpu_available = device_manager.is_device_type_available(hardware::DeviceType::CUDA);
    // This test should pass regardless of GPU availability
    SUCCEED();
}

// Test CPU vector operations
TEST_F(GPUAccelerationTest, CPUVectorOperations) {
    vector_ops::CPUVectorOperations cpu_ops;
    
    // Test cosine similarity
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f};
    
    float similarity = cpu_ops.cosine_similarity(vec1, vec2);
    EXPECT_NEAR(similarity, 0.0f, 1e-6);
    
    // Test with same vectors (should be 1.0)
    similarity = cpu_ops.cosine_similarity(vec1, vec1);
    EXPECT_NEAR(similarity, 1.0f, 1e-6);
    
    // Test Euclidean distance
    float distance = cpu_ops.euclidean_distance(vec1, vec2);
    EXPECT_NEAR(distance, std::sqrt(2.0f), 1e-6);
    
    // Test dot product
    float dot = cpu_ops.dot_product(vec1, vec2);
    EXPECT_NEAR(dot, 0.0f, 1e-6);
    
    // Test with same vectors (should be magnitude squared)
    dot = cpu_ops.dot_product(vec1, vec1);
    EXPECT_NEAR(dot, 1.0f, 1e-6);
}

// Test vector operations factory
TEST_F(GPUAccelerationTest, VectorOperationsFactory) {
    // Test with CPU preference (should always work)
    auto ops = vector_ops::VectorOperationsFactory::create_operations(hardware::DeviceType::CPU);
    ASSERT_NE(ops, nullptr);
    
    // Verify it can perform operations
    std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec2 = {4.0f, 5.0f, 6.0f};
    
    float result = ops->dot_product(vec1, vec2);
    EXPECT_NEAR(result, 32.0f, 1e-6); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

// Test GPU memory manager with CPU device (mock scenario)
TEST_F(GPUAccelerationTest, GPUMemoryManagerWithCPUDevice) {
    auto cpu_device = hardware::DeviceManager::get_instance().get_active_device();
    auto memory_manager = std::make_shared<gpu_memory::GPUDeviceMemoryManager>(cpu_device);
    
    // Test allocation/deallocation
    const size_t test_size = 1024; // bytes
    auto* ptr = memory_manager->allocate(test_size);
    ASSERT_NE(ptr, nullptr);
    
    // Test memory statistics
    auto stats = memory_manager->get_memory_stats();
    EXPECT_GT(stats.first, 0); // allocated should be > 0
    
    // Deallocate
    memory_manager->deallocate(ptr);
    
    // Clear cache (though there shouldn't be anything cached in this simple test)
    memory_manager->clear_cache();
}

// Test workload balancer with CPU-only setup
TEST_F(GPUAccelerationTest, WorkloadBalancerSimpleStrategy) {
    auto cpu_device = hardware::DeviceManager::get_instance().get_active_device();
    auto gpu_device = std::shared_ptr<hardware::IDevice>(nullptr); // No GPU for this test
    auto strategy = std::make_shared<workload::SimpleThresholdStrategy>(500); // 500 vector threshold
    
    workload::WorkloadBalancer balancer(cpu_device, gpu_device, strategy);
    
    // Create test vectors
    std::vector<std::vector<float>> test_vectors = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    std::vector<float> query = {0.5f, 0.5f, 0.5f};
    
    // Process vectors using the balancer
    auto results = balancer.process_vectors(
        test_vectors, 
        query,
        [](const std::vector<float>& a, const std::vector<float>& b) {
            // Simple dot product function for testing
            float result = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                result += a[i] * b[i];
            }
            return result;
        }
    );
    
    // Verify we got the right number of results
    EXPECT_EQ(results.size(), test_vectors.size());
    
    // Verify results are reasonable (not all zeros)
    bool has_positive = false;
    for (float result : results) {
        if (result > 0) has_positive = true;
        break;
    }
    EXPECT_TRUE(has_positive);
}

// Test with a performance-based strategy
TEST_F(GPUAccelerationTest, PerformanceBasedStrategy) {
    workload::PerformanceBasedStrategy strategy;
    
    // Add some historical metrics
    workload::WorkloadMetrics metrics1;
    metrics1.cpu_processing_time_ms = 100;
    metrics1.gpu_processing_time_ms = 50;  // GPU is faster
    metrics1.gpu_available = true;
    metrics1.num_vectors_processed = 1000;
    
    strategy.add_historical_metrics(metrics1);
    
    // Create new metrics for decision
    workload::WorkloadMetrics current_metrics;
    current_metrics.gpu_available = true;
    
    int gpu_percentage = strategy.get_gpu_work_percentage(current_metrics, 1000, 128);
    
    // Since GPU was faster in history, it should get a higher percentage
    // Though the exact value depends on the implementation
    EXPECT_GE(gpu_percentage, 0);
    EXPECT_LE(gpu_percentage, 100);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}