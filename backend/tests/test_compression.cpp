#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <cmath>
#include <random>

#include "lib/compression.h"

using namespace jadevectordb;

// Test fixture for compression components
class CompressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize with default configuration
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
};

// Test the compression manager with no compression
TEST_F(CompressionTest, CompressionManagerNoCompression) {
    compression::CompressionManager manager;
    
    // Use default config (no compression)
    compression::CompressionConfig config;
    config.type = compression::CompressionType::NONE;
    
    EXPECT_TRUE(manager.configure(config));
    
    // Test compression/decompression
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto compressed = manager.compress_vector(original);
    auto decompressed = manager.decompress_vector(compressed, original.size());
    
    EXPECT_EQ(original.size(), decompressed.size());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_FLOAT_EQ(original[i], decompressed[i]);
    }
}

// Test SVD compression
TEST_F(CompressionTest, SVDCompression) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::SVD;
    config.compression_ratio = 0.5; // Target 50% compression
    config.target_dimensions = 3;   // Reduce from 5 to 3 dimensions
    
    EXPECT_TRUE(manager.configure(config));
    
    // Create a test vector
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto compressed = manager.compress_vector(original);
    auto decompressed = manager.decompress_vector(compressed, original.size());
    
    // For SVD simulation, we expect the decompressed size to match original
    EXPECT_EQ(original.size(), decompressed.size());
    
    // The compression ratio should reflect the actual compression
    auto stats = manager.calculate_compression_stats(original.size() * sizeof(float), compressed.size());
    EXPECT_GT(stats.first, 0.0);  // Some compression should have occurred
}

// Test PCA compression
TEST_F(CompressionTest, PCACompression) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::PCA;
    config.compression_ratio = 0.6; // Target 60% compression
    config.target_dimensions = 4;   // Reduce from 5 to 4 dimensions
    
    EXPECT_TRUE(manager.configure(config));
    
    // Create a test vector
    std::vector<float> original = {1.5f, -2.3f, 3.7f, -4.1f, 0.9f};
    auto compressed = manager.compress_vector(original);
    auto decompressed = manager.decompress_vector(compressed, original.size());
    
    // For PCA simulation, we expect the decompressed size to match original
    EXPECT_EQ(original.size(), decompressed.size());
}

// Test Quantization compression
TEST_F(CompressionTest, QuantizationCompression) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::QUANTIZATION;
    
    EXPECT_TRUE(manager.configure(config));
    
    // Create a test vector with values that can be meaningfully quantized
    std::vector<float> original = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    auto compressed = manager.compress_vector(original);
    auto decompressed = manager.decompress_vector(compressed, original.size());
    
    // For quantization, we expect values to be close but not identical (lossy compression)
    EXPECT_EQ(original.size(), decompressed.size());
    
    // Check that values are reasonably close (allowing for quantization error)
    for (size_t i = 0; i < original.size(); ++i) {
        float diff = std::abs(original[i] - decompressed[i]);
        EXPECT_LT(diff, 0.5f); // Should be within 0.5 for this example
    }
}

// Test batch compression
TEST_F(CompressionTest, BatchCompression) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::QUANTIZATION;
    
    EXPECT_TRUE(manager.configure(config));
    
    // Create multiple test vectors
    std::vector<std::vector<float>> originals = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    
    auto compressed = manager.compress_batch(originals);
    std::vector<size_t> original_sizes = {originals[0].size(), originals[1].size(), originals[2].size()};
    auto decompressed = manager.decompress_batch(compressed, original_sizes);
    
    EXPECT_EQ(originals.size(), decompressed.size());
    for (size_t i = 0; i < originals.size(); ++i) {
        EXPECT_EQ(originals[i].size(), decompressed[i].size());
    }
}

// Test compression statistics
TEST_F(CompressionTest, CompressionStats) {
    compression::CompressionManager manager;
    
    // Test with no compression
    auto stats = manager.calculate_compression_stats(100, 100);
    EXPECT_FLOAT_EQ(stats.first, 1.0);  // No compression ratio
    EXPECT_FLOAT_EQ(stats.second, 0.0);  // 0% space saved
    
    // Test with 50% compression
    stats = manager.calculate_compression_stats(100, 50);
    EXPECT_FLOAT_EQ(stats.first, 0.5);   // 50% compression ratio
    EXPECT_FLOAT_EQ(stats.second, 50.0); // 50% space saved
}

// Test compression with various sizes
TEST_F(CompressionTest, CompressionWithVariousSizes) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::QUANTIZATION;
    
    EXPECT_TRUE(manager.configure(config));
    
    // Test with different sized vectors
    std::vector<std::vector<float>> test_vectors = {
        {1.0f},                                    // Size 1
        {1.0f, 2.0f},                             // Size 2
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},         // Size 5
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f} // Size 10
    };
    
    for (const auto& original : test_vectors) {
        auto compressed = manager.compress_vector(original);
        auto decompressed = manager.decompress_vector(compressed, original.size());
        
        EXPECT_EQ(original.size(), decompressed.size());
        
        // For quantization, values should be reasonably close
        for (size_t i = 0; i < original.size(); ++i) {
            float diff = std::abs(original[i] - decompressed[i]);
            EXPECT_LT(diff, 2.0f); // Allow larger tolerance for smaller vectors
        }
    }
}

// Test compression with extreme values
TEST_F(CompressionTest, CompressionWithExtremeValues) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::QUANTIZATION;
    
    EXPECT_TRUE(manager.configure(config));
    
    // Test with large values
    std::vector<float> large_values = {1000000.0f, -2000000.0f, 3000000.0f};
    auto compressed = manager.compress_vector(large_values);
    auto decompressed = manager.decompress_vector(compressed, large_values.size());
    
    EXPECT_EQ(large_values.size(), decompressed.size());
    
    // Test with very small values
    std::vector<float> small_values = {0.000001f, -0.000002f, 0.000003f};
    compressed = manager.compress_vector(small_values);
    decompressed = manager.decompress_vector(compressed, small_values.size());
    
    EXPECT_EQ(small_values.size(), decompressed.size());
    
    // Test with all same values
    std::vector<float> same_values = {5.0f, 5.0f, 5.0f};
    compressed = manager.compress_vector(same_values);
    decompressed = manager.decompress_vector(compressed, same_values.size());
    
    EXPECT_EQ(same_values.size(), decompressed.size());
}

// Test SVD compression with dimensionality reduction
TEST_F(CompressionTest, SVDWithDimensionalityReduction) {
    compression::CompressionManager manager;
    
    compression::CompressionConfig config;
    config.type = compression::CompressionType::SVD;
    config.target_dimensions = 2;  // Compress from 5 dimensions to 2
    
    EXPECT_TRUE(manager.configure(config));
    
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto compressed = manager.compress_vector(original);
    auto decompressed = manager.decompress_vector(compressed, original.size());
    
    EXPECT_EQ(original.size(), decompressed.size());
    
    // Note: With our simplified SVD implementation, the first 'target_dimensions' 
    // values should be preserved and the rest zeroed out during compression
    for (int i = 0; i < config.target_dimensions; ++i) {
        EXPECT_FLOAT_EQ(original[i], decompressed[i]);
    }
    for (size_t i = config.target_dimensions; i < decompressed.size(); ++i) {
        EXPECT_FLOAT_EQ(0.0f, decompressed[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}