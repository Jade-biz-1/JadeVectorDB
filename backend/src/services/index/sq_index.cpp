#include "sq_index.h"
#include "lib/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>

namespace jadevectordb {

SqIndex::SqIndex(const SqParams& params) : params_(params) {
    logger_ = logging::LoggerManager::get_logger("index.sq");
    
    // Set num_levels based on bits_per_dimension if not already set
    if (params_.num_levels == 0 && params_.bits_per_dimension > 0) {
        params_.num_levels = 1 << params_.bits_per_dimension;  // 2^bits_per_dimension
    }
}

bool SqIndex::initialize(const SqParams& params) {
    params_ = params;
    if (params_.bits_per_dimension <= 0) {
        LOG_ERROR(logger_, "Bits per dimension must be positive");
        return false;
    }
    if (params_.num_levels <= 0) {
        params_.num_levels = 1 << params_.bits_per_dimension;  // 2^bits_per_dimension
    }
    if (params_.num_levels != (1 << params_.bits_per_dimension)) {
        LOG_WARN(logger_, "num_levels " << params_.num_levels << " does not match 2^bits_per_dimension " << 
                 (1 << params_.bits_per_dimension) << ", using calculated value " << (1 << params_.bits_per_dimension));
        params_.num_levels = 1 << params_.bits_per_dimension;
    }
    return true;
}

Result<bool> SqIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    if (vector.empty()) {
        return Result<bool>::failure("Vector cannot be empty");
    }
    
    if (dimension_ == 0) {
        dimension_ = vector.size();
        // Initialize quantization ranges vector
        quantization_ranges_.resize(dimension_);
        for (auto& range : quantization_ranges_) {
            range.first = std::numeric_limits<float>::max();   // min
            range.second = std::numeric_limits<float>::lowest(); // max
        }
    } else if (static_cast<int>(vector.size()) != dimension_) {
        return Result<bool>::failure("Vector dimension mismatch");
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Store the original vector for now
    original_vectors_[vector_id] = vector;
    
    return Result<bool>::success(true);
}

Result<bool> SqIndex::build() {
    if (original_vectors_.empty()) {
        return Result<bool>::success(true); // Nothing to build
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Extract vectors for determining quantization ranges
    std::vector<std::vector<float>> training_vectors;
    training_vectors.reserve(original_vectors_.size());
    
    for (const auto& pair : original_vectors_) {
        training_vectors.push_back(pair.second);
    }
    
    // Determine quantization ranges
    auto result = determine_quantization_ranges(training_vectors);
    if (!result.has_value()) {
        return Result<bool>::failure("Failed to determine quantization ranges: " + result.error().message);
    }
    
    is_trained_ = true;
    
    // Encode all vectors
    for (const auto& pair : original_vectors_) {
        int vector_id = pair.first;
        const auto& vector = pair.second;
        std::vector<uint8_t> code = encode_vector(vector);
        sq_codes_[vector_id] = std::move(code);
    }
    
    is_built_ = true;
    
    LOG_INFO(logger_, "SQ index built with " << original_vectors_.size() << " vectors");
    return Result<bool>::success(true);
}

Result<bool> SqIndex::build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors) {
    std::unique_lock<boost::shared_mutex> lock(index_mutex_);
    
    clear();
    
    // Add all vectors
    for (const auto& vec_pair : vectors) {
        auto result = add_vector(vec_pair.first, vec_pair.second);
        if (!result.has_value()) {
            return result;
        }
    }
    
    // Now build the index
    return build();
}

Result<std::vector<std::pair<int, float>>> SqIndex::search(const std::vector<float>& query, int k, float threshold) const {
    if (!is_built_) {
        return Result<std::vector<std::pair<int, float>>>::failure("Index is not built");
    }
    
    if (static_cast<int>(query.size()) != dimension_) {
        return Result<std::vector<std::pair<int, float>>>::failure("Query dimension mismatch");
    }
    
    boost::shared_lock<boost::shared_mutex> lock(index_mutex_);
    
    // Encode the query vector to SQ code if needed
    std::vector<uint8_t> query_code;
    if (is_trained_) {
        query_code = encode_vector(query);
    }
    
    // Perform search using asymmetric distance computation
    std::vector<std::pair<int, float>> results;
    
    // Compute distances to all stored codes
    for (const auto& code_pair : sq_codes_) {
        int vector_id = code_pair.first;
        const auto& code = code_pair.second;
        
        float dist;
        if (is_trained_) {
            dist = compute_asymmetric_distance(query, code);
        } else {
            // Fallback: use original vectors if index isn't trained
            auto it = original_vectors_.find(vector_id);
            if (it == original_vectors_.end()) continue;
            dist = compute_squared_distance(query, it->second);
        }
        
        // Apply threshold filter
        if (dist >= threshold) {
            results.emplace_back(vector_id, dist);
        }
    }
    
    // Sort by distance (ascending)
    std::sort(results.begin(), results.end(), 
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second < b.second;
              });
    
    // Return top k results
    if (k > 0 && static_cast<size_t>(k) < results.size()) {
        results.resize(k);
    }
    
    return Result<std::vector<std::pair<int, float>>>::success(std::move(results));
}

bool SqIndex::contains(int vector_id) const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return sq_codes_.find(vector_id) != sq_codes_.end();
}

Result<bool> SqIndex::remove_vector(int vector_id) {
    boost::unique_lock<boost::shared_mutex> lock(index_mutex_);
    
    auto it = sq_codes_.find(vector_id);
    if (it == sq_codes_.end()) {
        return Result<bool>::success(false); // Vector not found
    }
    
    sq_codes_.erase(it);
    original_vectors_.erase(vector_id);
    
    return Result<bool>::success(true);
}

Result<bool> SqIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    if (static_cast<int>(new_vector.size()) != dimension_) {
        return Result<bool>::failure("Vector dimension mismatch");
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Update the original vector
    auto it = original_vectors_.find(vector_id);
    if (it == original_vectors_.end()) {
        return Result<bool>::failure("Vector not found in index");
    }
    
    it->second = new_vector;
    
    // If the index is trained, update the SQ code
    if (is_trained_) {
        std::vector<uint8_t> new_code = encode_vector(new_vector);
        sq_codes_[vector_id] = std::move(new_code);
    }
    
    return Result<bool>::success(true);
}

size_t SqIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return sq_codes_.size();
}

Result<std::unordered_map<std::string, std::string>> SqIndex::get_stats() const {
    std::unordered_map<std::string, std::string> stats;
    stats["index_type"] = "SQ";
    stats["bits_per_dimension"] = std::to_string(params_.bits_per_dimension);
    stats["num_levels"] = std::to_string(params_.num_levels);
    stats["vector_count"] = std::to_string(size());
    stats["is_trained"] = is_trained_ ? "true" : "false";
    stats["is_built"] = is_built_ ? "true" : "false";
    stats["dimension"] = std::to_string(dimension_);
    
    return Result<std::unordered_map<std::string, std::string>>::success(std::move(stats));
}

void SqIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    quantization_ranges_.clear();
    sq_codes_.clear();
    original_vectors_.clear();
    dimension_ = 0;
    is_trained_ = false;
    is_built_ = false;
}

bool SqIndex::empty() const {
    return size() == 0;
}

int SqIndex::get_dimension() const {
    return dimension_;
}

int SqIndex::get_bits_per_dimension() const {
    return params_.bits_per_dimension;
}

int SqIndex::get_num_levels() const {
    return params_.num_levels;
}

std::vector<uint8_t> SqIndex::encode_vector(const std::vector<float>& vector) const {
    if (!is_trained_) {
        // If not trained, return empty code
        return std::vector<uint8_t>();
    }
    
    std::vector<uint8_t> code(dimension_);
    
    for (int i = 0; i < dimension_; ++i) {
        code[i] = quantize_value(vector[i], i);
    }
    
    return code;
}

std::vector<float> SqIndex::decode_code(const std::vector<uint8_t>& code) const {
    if (!is_trained_) {
        return std::vector<float>(); // Return empty if not trained
    }
    
    std::vector<float> reconstructed(dimension_);
    
    for (int i = 0; i < dimension_ && i < static_cast<int>(code.size()); ++i) {
        reconstructed[i] = dequantize_value(code[i], i);
    }
    
    return reconstructed;
}

Result<bool> SqIndex::determine_quantization_ranges(const std::vector<std::vector<float>>& vectors) {
    if (vectors.empty() || dimension_ == 0) {
        return Result<bool>::failure("Cannot determine ranges with empty vectors or unknown dimension");
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Initialize ranges to extreme values
    for (auto& range : quantization_ranges_) {
        range.first = std::numeric_limits<float>::max();   // min
        range.second = std::numeric_limits<float>::lowest(); // max
    }
    
    // Find min and max for each dimension
    for (const auto& vector : vectors) {
        for (int i = 0; i < dimension_ && i < static_cast<int>(vector.size()); ++i) {
            if (vector[i] < quantization_ranges_[i].first) {
                quantization_ranges_[i].first = vector[i];
            }
            if (vector[i] > quantization_ranges_[i].second) {
                quantization_ranges_[i].second = vector[i];
            }
        }
    }
    
    // Ensure ranges are valid (not equal)
    for (auto& range : quantization_ranges_) {
        if (range.first == range.second) {
            // If min equals max, add a small epsilon to avoid division by zero
            range.second += 1e-6f;
        }
    }
    
    LOG_INFO(logger_, "Determined quantization ranges for " << dimension_ << " dimensions");
    return Result<bool>::success(true);
}

uint8_t SqIndex::quantize_value(float value, int dim_idx) const {
    if (dim_idx >= static_cast<int>(quantization_ranges_.size())) {
        return 0; // Or handle error appropriately
    }
    
    const auto& range = quantization_ranges_[dim_idx];
    float min_val = range.first;
    float max_val = range.second;
    
    // Normalize the value to [0, 1]
    float normalized = (value - min_val) / (max_val - min_val);
    
    // Scale to the number of levels and clamp to valid range
    int quantized = static_cast<int>(normalized * (params_.num_levels - 1));
    quantized = std::max(0, std::min(params_.num_levels - 1, quantized));
    
    return static_cast<uint8_t>(quantized);
}

float SqIndex::dequantize_value(uint8_t code, int dim_idx) const {
    if (dim_idx >= static_cast<int>(quantization_ranges_.size())) {
        return 0.0f; // Or handle error appropriately
    }
    
    const auto& range = quantization_ranges_[dim_idx];
    float min_val = range.first;
    float max_val = range.second;
    
    // Convert the quantized value back to the original range
    float dequantized = min_val + (static_cast<float>(code) / (params_.num_levels - 1)) * (max_val - min_val);
    
    return dequantized;
}

float SqIndex::compute_sq_distance(const std::vector<uint8_t>& code1, const std::vector<uint8_t>& code2) const {
    if (!is_trained_) {
        return std::numeric_limits<float>::max();
    }
    
    float dist = 0.0f;
    
    int min_size = std::min(static_cast<int>(code1.size()), static_cast<int>(code2.size()));
    for (int i = 0; i < min_size; ++i) {
        // Get the dequantized values for comparison
        float val1 = dequantize_value(code1[i], i);
        float val2 = dequantize_value(code2[i], i);
        
        float diff = val1 - val2;
        dist += diff * diff;
    }
    
    return dist;
}

float SqIndex::compute_asymmetric_distance(const std::vector<float>& query, const std::vector<uint8_t>& code) const {
    if (!is_trained_) {
        return std::numeric_limits<float>::max();
    }
    
    float dist = 0.0f;
    
    int min_size = std::min(static_cast<int>(query.size()), static_cast<int>(code.size()));
    for (int i = 0; i < min_size; ++i) {
        // Dequantize the code value and compare with query value
        float dequantized_code_val = dequantize_value(code[i], i);
        float diff = query[i] - dequantized_code_val;
        dist += diff * diff;
    }
    
    return dist;
}

float SqIndex::compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::max();
    }
    
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    
    return dist;
}

bool SqIndex::validate() const {
    if (dimension_ <= 0) {
        LOG_ERROR(logger_, "Invalid dimension: " << dimension_);
        return false;
    }
    
    if (params_.bits_per_dimension <= 0) {
        LOG_ERROR(logger_, "Bits per dimension must be positive");
        return false;
    }
    
    if (params_.num_levels <= 0) {
        LOG_ERROR(logger_, "Number of levels must be positive");
        return false;
    }
    
    return true;
}

} // namespace jadevectordb