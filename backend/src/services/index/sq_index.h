#ifndef JADEVECTORDB_SQ_INDEX_H
#define JADEVECTORDB_SQ_INDEX_H

#include "models/index.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <shared_mutex>
#include <random>

namespace jadevectordb {

/**
 * @brief Implementation of Scalar Quantization (SQ) index algorithm
 * 
 * This implementation provides similarity search using scalar quantization,
 * where each dimension of the vector is quantized independently to a discrete value.
 * This results in a compressed representation with reduced memory requirements
 * while maintaining reasonable search accuracy.
 */
class SqIndex {
public:
    struct SqParams {
        int bits_per_dimension;           // Number of bits per dimension (defines number of levels)
        int num_levels;                   // Number of quantization levels (2^bits_per_dimension)
        std::vector<std::pair<float, float>> range_per_dimension;  // Min and max value for each dimension
        bool normalize_vectors;           // Whether to normalize vectors before quantization
        int random_seed;                  // Random seed (for initialization if needed)
        
        // Constructor
        SqParams() : bits_per_dimension(8), num_levels(256), normalize_vectors(false), 
                     random_seed(100) {}
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Index parameters
    SqParams params_;
    
    // Data structures
    std::vector<std::vector<float>> quantization_ranges_;  // [dimension_idx]{min_val, max_val}
    std::unordered_map<int, std::vector<uint8_t>> sq_codes_;  // Vector ID -> SQ codes
    std::unordered_map<int, std::vector<float>> original_vectors_;  // Vector ID -> original vector
    
    // Current state
    int dimension_ = 0;
    bool is_trained_ = false;      // Whether the quantization ranges are determined
    bool is_built_ = false;        // Whether the index is built with vectors
    
    // Thread safety
    mutable std::shared_mutex index_mutex_;
    
public:
    explicit SqIndex(const SqParams& params = SqParams{});
    ~SqIndex() = default;
    
    // Initialize the index with parameters
    bool initialize(const SqParams& params);
    
    // Add a vector to the index (will be encoded during build phase)
    Result<bool> add_vector(int vector_id, const std::vector<float>& vector);
    
    // Build the index (determine quantization ranges, encode vectors)
    Result<bool> build();
    
    // Search for similar vectors using SQ approximation
    Result<std::vector<std::pair<int, float>>> search(const std::vector<float>& query,
                                                     int k = 10,
                                                     float threshold = 0.0f) const;
    
    // Build the index from a set of vectors (batch operation)
    Result<bool> build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors);
    
    // Check if the index contains a specific vector
    bool contains(int vector_id) const;
    
    // Remove a vector from the index
    Result<bool> remove_vector(int vector_id);
    
    // Update a vector in the index
    Result<bool> update_vector(int vector_id, const std::vector<float>& new_vector);
    
    // Get the number of vectors in the index
    size_t size() const;
    
    // Get index statistics
    Result<std::unordered_map<std::string, std::string>> get_stats() const;
    
    // Clear the index
    void clear();
    
    // Check if the index is empty
    bool empty() const;
    
    // Get the dimension of vectors in the index
    int get_dimension() const;
    
    // Get the number of bits per dimension
    int get_bits_per_dimension() const;
    
    // Get the number of quantization levels
    int get_num_levels() const;
    
    // Encode a vector using scalar quantization
    std::vector<uint8_t> encode_vector(const std::vector<float>& vector) const;
    
    // Decode an SQ code to reconstruct an approximate vector
    std::vector<float> decode_code(const std::vector<uint8_t>& code) const;

private:
    // Internal implementation methods
    
    // Determine quantization ranges from the training vectors
    Result<bool> determine_quantization_ranges(const std::vector<std::vector<float>>& vectors);
    
    // Quantize a single float value to an integer based on the range
    uint8_t quantize_value(float value, int dim_idx) const;
    
    // Dequantize a single integer value back to float
    float dequantize_value(uint8_t code, int dim_idx) const;
    
    // Compute distance in the SQ space
    float compute_sq_distance(const std::vector<uint8_t>& code1, const std::vector<uint8_t>& code2) const;
    
    // Compute distance between a query vector and an SQ code
    float compute_asymmetric_distance(const std::vector<float>& query, const std::vector<uint8_t>& code) const;
    
    // Compute squared distance between two vectors
    float compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    
    // Validate index state
    bool validate() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SQ_INDEX_H