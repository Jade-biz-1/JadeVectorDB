#ifndef JADEVECTORDB_COMPRESSION_H
#define JADEVECTORDB_COMPRESSION_H

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

namespace jadevectordb {
namespace compression {

    // Enum for compression types
    enum class CompressionType {
        NONE,           // No compression
        SVD,            // Singular Value Decomposition
        PCA,            // Principal Component Analysis
        QUANTIZATION,   // Vector quantization
        AUTOENCODER,    // Neural network-based compression
        CUSTOM          // Custom compression algorithm
    };

    // Enum for compression quality/rate
    enum class CompressionQuality {
        LOSSLESS,       // Exact reconstruction
        HIGH,           // High quality, larger size
        MEDIUM,         // Balanced quality/size
        LOW,            // Low quality, smaller size
    };

    struct CompressionConfig {
        CompressionType type = CompressionType::NONE;
        CompressionQuality quality = CompressionQuality::MEDIUM;
        double compression_ratio = 0.5;  // Target compression ratio (0.0 to 1.0)
        int target_dimensions = 0;       // Target dimension for dimensionality reduction techniques
        std::string custom_params;       // Additional parameters for custom algorithms
    };

    /**
     * @brief Interface for compression algorithms
     * 
     * This interface defines the contract for all compression algorithms
     * used in the vector database system.
     */
    class ICompressionAlgorithm {
    public:
        virtual ~ICompressionAlgorithm() = default;
        
        /**
         * @brief Compress a vector
         * @param input Input vector to compress
         * @return Compressed representation
         */
        virtual std::vector<uint8_t> compress(const std::vector<float>& input) = 0;
        
        /**
         * @brief Decompress a vector
         * @param compressed Compressed data
         * @param original_size Size of the original vector
         * @return Decompressed vector
         */
        virtual std::vector<float> decompress(const std::vector<uint8_t>& compressed, 
                                            size_t original_size) = 0;
        
        /**
         * @brief Get the compression ratio achieved
         * @return Compression ratio (compressed_size / original_size)
         */
        virtual double get_compression_ratio() const = 0;
        
        /**
         * @brief Get the type of compression algorithm
         */
        virtual CompressionType get_compression_type() const = 0;
        
        /**
         * @brief Get a human-readable name for the algorithm
         */
        virtual std::string get_name() const = 0;
    };

    /**
     * @brief Compression manager to handle different compression algorithms
     * 
     * This class manages multiple compression algorithms and provides
     * a unified interface for compression/decompression operations.
     */
    class CompressionManager {
    private:
        std::unique_ptr<ICompressionAlgorithm> active_algorithm_;
        CompressionConfig current_config_;
        
    public:
        CompressionManager();
        ~CompressionManager() = default;
        
        /**
         * @brief Configure the compression algorithm
         * @param config Configuration for the compression algorithm
         * @return True if configuration was successful
         */
        bool configure(const CompressionConfig& config);
        
        /**
         * @brief Compress a vector using the active algorithm
         * @param input Input vector to compress
         * @return Compressed representation
         */
        std::vector<uint8_t> compress_vector(const std::vector<float>& input);
        
        /**
         * @brief Decompress a vector using the active algorithm
         * @param compressed Compressed data
         * @param original_size Size of the original vector
         * @return Decompressed vector
         */
        std::vector<float> decompress_vector(const std::vector<uint8_t>& compressed, 
                                           size_t original_size);
        
        /**
         * @brief Compress multiple vectors efficiently
         * @param input List of input vectors to compress
         * @return List of compressed representations
         */
        std::vector<std::vector<uint8_t>> compress_batch(const std::vector<std::vector<float>>& input);
        
        /**
         * @brief Decompress multiple vectors efficiently
         * @param compressed List of compressed data
         * @param original_sizes Sizes of the original vectors
         * @return List of decompressed vectors
         */
        std::vector<std::vector<float>> decompress_batch(const std::vector<std::vector<uint8_t>>& compressed,
                                                        const std::vector<size_t>& original_sizes);
        
        /**
         * @brief Get the active compression algorithm
         */
        ICompressionAlgorithm* get_active_algorithm() const;
        
        /**
         * @brief Get the current configuration
         */
        const CompressionConfig& get_config() const;
        
        /**
         * @brief Calculate compression statistics
         * @param original Original data size
         * @param compressed Compressed data size
         * @return Pair with compression ratio and space saved percentage
         */
        std::pair<double, double> calculate_compression_stats(size_t original, size_t compressed) const;
    };

    // Forward declaration of specific compression algorithms that will be implemented
    class SVDCompression;
    class PCACompression;
    class QuantizationCompression;

} // namespace compression
} // namespace jadevectordb

#endif // JADEVECTORDB_COMPRESSION_H