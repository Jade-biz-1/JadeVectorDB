#include "compression.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>

// Note: For SVD/PCA, we'll need to use Eigen library which is already used in the project

namespace jadevectordb {
namespace compression {

// Forward declarations of compression algorithm implementations
class SVDCompression;
class PCACompression;
class QuantizationCompression;

// SVD Compression Implementation
class SVDCompression : public ICompressionAlgorithm {
private:
    CompressionConfig config_;
    double compression_ratio_;

public:
    explicit SVDCompression(const CompressionConfig& config)
        : config_(config), compression_ratio_(0.0) {}

    std::vector<uint8_t> compress(const std::vector<float>& input) override {
        if (input.empty()) {
            return std::vector<uint8_t>();
        }

        // For SVD compression, we need to work with matrices
        // For now, we'll implement a simplified version that just reduces dimensionality

        size_t original_size = input.size();
        int target_dim = config_.target_dimensions > 0 ? config_.target_dimensions :
                         static_cast<int>(original_size * config_.compression_ratio);

        // Limit target dimensions to be less than original
        target_dim = std::min(target_dim, static_cast<int>(original_size));

        // Simulate SVD by keeping only the first `target_dim` components
        // In a real implementation, we would perform actual SVD decomposition
        std::vector<float> compressed_data(input.begin(),
                                          input.begin() + std::min(target_dim, static_cast<int>(original_size)));

        // Convert to bytes
        size_t byte_size = compressed_data.size() * sizeof(float);
        std::vector<uint8_t> result(byte_size + sizeof(size_t));

        // Store original size first, then the compressed data
        std::memcpy(result.data(), &original_size, sizeof(size_t));
        std::memcpy(result.data() + sizeof(size_t), compressed_data.data(), byte_size);

        // Calculate compression ratio
        compression_ratio_ = static_cast<double>(byte_size + sizeof(size_t)) / (original_size * sizeof(float));

        return result;
    }

    std::vector<float> decompress(const std::vector<uint8_t>& compressed,
                                  size_t original_size) override {
        if (compressed.size() < sizeof(size_t)) {
            return std::vector<float>(original_size, 0.0f);
        }

        // Extract original size from compressed data
        size_t stored_original_size;
        std::memcpy(&stored_original_size, compressed.data(), sizeof(size_t));

        // Calculate how many floats we have after the size header
        size_t float_data_size = compressed.size() - sizeof(size_t);
        size_t num_compressed_values = float_data_size / sizeof(float);

        // Extract compressed values
        std::vector<float> compressed_values(num_compressed_values);
        std::memcpy(compressed_values.data(),
                   compressed.data() + sizeof(size_t),
                   float_data_size);

        // Reconstruct to original size (pad with zeros)
        std::vector<float> result(stored_original_size, 0.0f);
        std::copy(compressed_values.begin(),
                  compressed_values.begin() + std::min(compressed_values.size(), result.size()),
                  result.begin());

        return result;
    }

    double get_compression_ratio() const override {
        return compression_ratio_;
    }

    CompressionType get_compression_type() const override {
        return CompressionType::SVD;
    }

    std::string get_name() const override {
        return "SVD Compression";
    }
};

// PCA Compression Implementation
class PCACompression : public ICompressionAlgorithm {
private:
    CompressionConfig config_;
    double compression_ratio_;

public:
    explicit PCACompression(const CompressionConfig& config)
        : config_(config), compression_ratio_(0.0) {}

    std::vector<uint8_t> compress(const std::vector<float>& input) override {
        if (input.empty()) {
            return std::vector<uint8_t>();
        }

        size_t original_size = input.size();
        int target_dim = config_.target_dimensions > 0 ? config_.target_dimensions :
                         static_cast<int>(original_size * config_.compression_ratio);

        // Limit target dimensions to be less than original
        target_dim = std::min(target_dim, static_cast<int>(original_size));

        // Simulate PCA by keeping only the first `target_dim` components
        // In a real implementation, we would calculate principal components
        std::vector<float> compressed_data(input.begin(),
                                          input.begin() + std::min(target_dim, static_cast<int>(original_size)));

        // Convert to bytes
        size_t byte_size = compressed_data.size() * sizeof(float);
        std::vector<uint8_t> result(byte_size + sizeof(size_t));

        // Store original size first, then the compressed data
        std::memcpy(result.data(), &original_size, sizeof(size_t));
        std::memcpy(result.data() + sizeof(size_t), compressed_data.data(), byte_size);

        // Calculate compression ratio
        compression_ratio_ = static_cast<double>(byte_size + sizeof(size_t)) / (original_size * sizeof(float));

        return result;
    }

    std::vector<float> decompress(const std::vector<uint8_t>& compressed,
                                  size_t original_size) override {
        if (compressed.size() < sizeof(size_t)) {
            return std::vector<float>(original_size, 0.0f);
        }

        // Extract original size from compressed data
        size_t stored_original_size;
        std::memcpy(&stored_original_size, compressed.data(), sizeof(size_t));

        // Calculate how many floats we have after the size header
        size_t float_data_size = compressed.size() - sizeof(size_t);
        size_t num_compressed_values = float_data_size / sizeof(float);

        // Extract compressed values
        std::vector<float> compressed_values(num_compressed_values);
        std::memcpy(compressed_values.data(),
                   compressed.data() + sizeof(size_t),
                   float_data_size);

        // Reconstruct to original size (pad with zeros)
        std::vector<float> result(stored_original_size, 0.0f);
        std::copy(compressed_values.begin(),
                  compressed_values.begin() + std::min(compressed_values.size(), result.size()),
                  result.begin());

        return result;
    }

    double get_compression_ratio() const override {
        return compression_ratio_;
    }

    CompressionType get_compression_type() const override {
        return CompressionType::PCA;
    }

    std::string get_name() const override {
        return "PCA Compression";
    }
};

// Quantization Compression Implementation
class QuantizationCompression : public ICompressionAlgorithm {
private:
    CompressionConfig config_;
    double compression_ratio_;

public:
    explicit QuantizationCompression(const CompressionConfig& config)
        : config_(config), compression_ratio_(0.0) {}

    std::vector<uint8_t> compress(const std::vector<float>& input) override {
        if (input.empty()) {
            return std::vector<uint8_t>();
        }

        // Simple quantization: convert from float (4 bytes) to int8_t (1 byte)
        // This is a basic example - real quantization would be more sophisticated

        std::vector<int8_t> quantized(input.size());
        float min_val = *std::min_element(input.begin(), input.end());
        float max_val = *std::max_element(input.begin(), input.end());

        // Handle the case where all values are the same
        if (min_val == max_val) {
            std::fill(quantized.begin(), quantized.end(),
                     static_cast<int8_t>((min_val >= 0) ? 127 : -128));
        } else {
            // Normalize to [0, 255] then convert to [-128, 127]
            float range = max_val - min_val;
            for (size_t i = 0; i < input.size(); ++i) {
                float normalized = (input[i] - min_val) / range;  // [0, 1]
                int8_t value = static_cast<int8_t>(normalized * 255 - 128);  // [-128, 127]
                quantized[i] = value;
            }
        }

        // Store metadata (min, max) along with quantized data
        struct QuantizationHeader {
            float min_val;
            float max_val;
            size_t original_size;
        };

        QuantizationHeader header = {min_val, max_val, input.size()};

        size_t header_size = sizeof(QuantizationHeader);
        size_t quantized_size = quantized.size() * sizeof(int8_t);

        std::vector<uint8_t> result(header_size + quantized_size);

        // Copy header
        std::memcpy(result.data(), &header, header_size);

        // Copy quantized data
        std::memcpy(result.data() + header_size, quantized.data(), quantized_size);

        // Calculate compression ratio (4 bytes per float -> 1 byte per value)
        compression_ratio_ = static_cast<double>(header_size + quantized_size) / (input.size() * sizeof(float));

        return result;
    }

    std::vector<float> decompress(const std::vector<uint8_t>& compressed,
                                  size_t original_size) override {
        if (compressed.size() < sizeof(float) * 3) {  // Need at least min, max, and size
            return std::vector<float>(original_size, 0.0f);
        }

        // Extract header
        struct QuantizationHeader {
            float min_val;
            float max_val;
            size_t original_size;
        };

        QuantizationHeader header;
        std::memcpy(&header, compressed.data(), sizeof(QuantizationHeader));

        // Extract quantized values
        size_t header_size = sizeof(QuantizationHeader);
        size_t quantized_size = compressed.size() - header_size;
        size_t num_quantized_values = quantized_size / sizeof(int8_t);

        std::vector<int8_t> quantized_values(num_quantized_values);
        std::memcpy(quantized_values.data(),
                   compressed.data() + header_size,
                   quantized_size);

        // Dequantize
        std::vector<float> result(header.original_size);
        float range = header.max_val - header.min_val;

        if (range == 0.0f) {
            std::fill(result.begin(), result.end(), header.min_val);
        } else {
            for (size_t i = 0; i < quantized_values.size(); ++i) {
                float normalized = (static_cast<float>(quantized_values[i]) + 128.0f) / 255.0f;  // [0, 1]
                result[i] = header.min_val + normalized * range;
            }
            // Pad with zeros if needed
            for (size_t i = quantized_values.size(); i < result.size(); ++i) {
                result[i] = 0.0f;
            }
        }

        return result;
    }

    double get_compression_ratio() const override {
        return compression_ratio_;
    }

    CompressionType get_compression_type() const override {
        return CompressionType::QUANTIZATION;
    }

    std::string get_name() const override {
        return "Quantization Compression";
    }
};

// Now implement CompressionManager methods

CompressionManager::CompressionManager() {
    // Initialize with no compression by default
    CompressionConfig default_config;
    default_config.type = CompressionType::NONE;
    configure(default_config);
}

bool CompressionManager::configure(const CompressionConfig& config) {
    current_config_ = config;

    switch (config.type) {
        case CompressionType::SVD:
            active_algorithm_ = std::make_unique<SVDCompression>(config);
            break;

        case CompressionType::PCA:
            active_algorithm_ = std::make_unique<PCACompression>(config);
            break;

        case CompressionType::QUANTIZATION:
            active_algorithm_ = std::make_unique<QuantizationCompression>(config);
            break;

        case CompressionType::NONE:
            // Use uncompressed (no compression) algorithm
            active_algorithm_ = nullptr;
            break;

        default:
            // For now, unsupported compression types return false
            return false;
    }

    return true;
}

std::vector<uint8_t> CompressionManager::compress_vector(const std::vector<float>& input) {
    if (!active_algorithm_) {
        // If no algorithm is active, return the input as raw bytes
        std::vector<uint8_t> result(input.size() * sizeof(float));
        std::memcpy(result.data(), input.data(), input.size() * sizeof(float));
        return result;
    }

    return active_algorithm_->compress(input);
}

std::vector<float> CompressionManager::decompress_vector(const std::vector<uint8_t>& compressed,
                                                       size_t original_size) {
    if (!active_algorithm_) {
        // If no algorithm is active, treat compressed data as raw bytes
        std::vector<float> result(original_size);
        std::memcpy(result.data(), compressed.data(), original_size * sizeof(float));
        return result;
    }

    return active_algorithm_->decompress(compressed, original_size);
}

std::vector<std::vector<uint8_t>> CompressionManager::compress_batch(
    const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(input.size());

    for (const auto& vec : input) {
        results.push_back(compress_vector(vec));
    }

    return results;
}

std::vector<std::vector<float>> CompressionManager::decompress_batch(
    const std::vector<std::vector<uint8_t>>& compressed,
    const std::vector<size_t>& original_sizes) {

    if (compressed.size() != original_sizes.size()) {
        throw std::invalid_argument("Mismatched compressed and original_sizes vector sizes");
    }

    std::vector<std::vector<float>> results;
    results.reserve(compressed.size());

    for (size_t i = 0; i < compressed.size(); ++i) {
        results.push_back(decompress_vector(compressed[i], original_sizes[i]));
    }

    return results;
}

ICompressionAlgorithm* CompressionManager::get_active_algorithm() const {
    return active_algorithm_.get();
}

const CompressionConfig& CompressionManager::get_config() const {
    return current_config_;
}

std::pair<double, double> CompressionManager::calculate_compression_stats(
    size_t original, size_t compressed) const {
    if (original == 0) {
        return {0.0, 0.0};
    }

    double ratio = static_cast<double>(compressed) / original;
    double saved = (1.0 - ratio) * 100.0;

    return {ratio, saved};
}

} // namespace compression
} // namespace jadevectordb
