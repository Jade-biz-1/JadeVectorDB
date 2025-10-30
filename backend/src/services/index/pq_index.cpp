#include "pq_index.h"
#include "lib/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>
#include <shared_mutex>
#include <boost/thread/shared_mutex.hpp>
#include <sstream>

namespace jadevectordb {

PqIndex::PqIndex(const PqParams& params) : params_(params) {
    logger_ = logging::LoggerManager::get_logger("index.pq");
    initialize_random_generator();
    if (params_.subvector_dimension > 0) {
        // Compute number of subvectors based on the dimension
        params_.num_subvectors = 0; // Will be set when we know the actual vector dimension
    }
}

bool PqIndex::initialize(const PqParams& params) {
    params_ = params;
    if (params_.subvector_dimension <= 0) {
        logger_->error("Subvector dimension must be positive");
        return false;
    }
    if (params_.num_centroids <= 0) {
        logger_->error("Number of centroids must be positive");
        return false;
    }
    initialize_random_generator();
    return true;
}

void PqIndex::initialize_random_generator() {
    rng_.seed(params_.random_seed);
}

Result<bool> PqIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    if (vector.empty()) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector cannot be empty"));
    }
    
    if (dimension_ == 0) {
        dimension_ = vector.size();
        if (params_.num_subvectors == 0) {
            // Calculate number of subvectors based on vector dimension and subvector dimension
            params_.num_subvectors = dimension_ / params_.subvector_dimension;
            if (dimension_ % params_.subvector_dimension != 0) {
                LOG_WARN(logger_, "Vector dimension " << dimension_ << " is not divisible by subvector dimension " << 
                         params_.subvector_dimension << ". Last " << (dimension_ % params_.subvector_dimension) << 
                         " dimensions will be ignored.");
            }
        }
    } else if (static_cast<int>(vector.size()) != dimension_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector dimension mismatch"));
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Store the original vector for now
    original_vectors_[vector_id] = vector;
    
    return true;
}

Result<bool> PqIndex::build() {
    if (original_vectors_.empty()) {
        return true; // Nothing to build
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Extract vectors for training
    std::vector<std::vector<float>> training_vectors;
    for (const auto& pair : original_vectors_) {
        training_vectors.push_back(pair.second);
    }
    
    // Train the subvector centroids
    auto result = train_subvector_centroids(training_vectors);
    if (!result.has_value()) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to train subvector centroids: " + result.error().message));
    }
    
    is_trained_ = true;
    
    // Encode all vectors
    for (const auto& pair : original_vectors_) {
        int vector_id = pair.first;
        const auto& vector = pair.second;
        std::vector<uint8_t> code = encode_vector(vector);
        pq_codes_[vector_id] = std::move(code);
    }
    
    is_built_ = true;
    
    LOG_INFO(logger_, "PQ index built with " << original_vectors_.size() << " vectors");
    return true;
}

Result<bool> PqIndex::build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors) {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
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

Result<std::vector<std::pair<int, float>>> PqIndex::search(const std::vector<float>& query, int k, float threshold) const {
    if (!is_built_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_STATE, "Index is not built"));
    }
    
    if (static_cast<int>(query.size()) != dimension_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "Query dimension mismatch"));
    }
    
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    // Encode the query vector to PQ code if needed
    std::vector<uint8_t> query_code;
    if (is_trained_) {
        query_code = encode_vector(query);
    }
    
    // Perform search using asymmetric distance computation
    std::vector<std::pair<int, float>> results;
    
    // Compute distances to all stored codes
    for (const auto& code_pair : pq_codes_) {
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
    
    return std::move(results);
}

bool PqIndex::contains(int vector_id) const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return pq_codes_.find(vector_id) != pq_codes_.end();
}

Result<bool> PqIndex::remove_vector(int vector_id) {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    auto it = pq_codes_.find(vector_id);
    if (it == pq_codes_.end()) {
        return false; // Vector not found
    }
    
    pq_codes_.erase(it);
    original_vectors_.erase(vector_id);
    
    return true;
}

Result<bool> PqIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    if (static_cast<int>(new_vector.size()) != dimension_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector dimension mismatch"));
    }
    
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Update the original vector
    auto it = original_vectors_.find(vector_id);
    if (it == original_vectors_.end()) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::NOT_FOUND, "Vector not found in index"));
    }
    
    it->second = new_vector;
    
    // If the index is trained, update the PQ code
    if (is_trained_) {
        std::vector<uint8_t> new_code = encode_vector(new_vector);
        pq_codes_[vector_id] = std::move(new_code);
    }
    
    return true;
}

size_t PqIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return pq_codes_.size();
}

Result<std::unordered_map<std::string, std::string>> PqIndex::get_stats() const {
    std::unordered_map<std::string, std::string> stats;
    stats["index_type"] = "PQ";
    stats["subvector_dimension"] = std::to_string(params_.subvector_dimension);
    stats["num_centroids"] = std::to_string(params_.num_centroids);
    stats["num_subvectors"] = std::to_string(params_.num_subvectors);
    stats["vector_count"] = std::to_string(size());
    stats["is_trained"] = is_trained_ ? "true" : "false";
    stats["is_built"] = is_built_ ? "true" : "false";
    stats["dimension"] = std::to_string(dimension_);
    
    return std::move(stats);
}

void PqIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    subvector_centroids_.clear();
    pq_codes_.clear();
    original_vectors_.clear();
    dimension_ = 0;
    is_trained_ = false;
    is_built_ = false;
}

bool PqIndex::empty() const {
    return size() == 0;
}

int PqIndex::get_dimension() const {
    return dimension_;
}

int PqIndex::get_num_subvectors() const {
    return params_.num_subvectors;
}

int PqIndex::get_subvector_dimension() const {
    return params_.subvector_dimension;
}

std::vector<uint8_t> PqIndex::encode_vector(const std::vector<float>& vector) const {
    if (!is_trained_) {
        // If not trained, return empty code
        return std::vector<uint8_t>();
    }
    
    std::vector<uint8_t> code(params_.num_subvectors);
    std::vector<std::vector<float>> subvectors = split_into_subvectors(vector);
    
    for (int i = 0; i < params_.num_subvectors; ++i) {
        const auto& subvector = subvectors[i];
        const auto& centroids = subvector_centroids_[i];
        
        // Find the closest centroid
        float min_dist = std::numeric_limits<float>::max();
        int closest_centroid_idx = 0;
        
        for (int j = 0; j < params_.num_centroids; ++j) {
            float dist = compute_squared_distance(subvector, centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid_idx = j;
            }
        }
        
        // Ensure the index fits in uint8_t
        code[i] = static_cast<uint8_t>(std::min(closest_centroid_idx, 255));
    }
    
    return code;
}

std::vector<float> PqIndex::decode_code(const std::vector<uint8_t>& code) const {
    if (!is_trained_) {
        return std::vector<float>(); // Return empty if not trained
    }
    
    std::vector<float> reconstructed(dimension_);
    
    for (int i = 0; i < params_.num_subvectors; ++i) {
        if (i >= static_cast<int>(code.size())) break;
        
        uint8_t centroid_idx = code[i];
        const auto& centroid = subvector_centroids_[i][centroid_idx];
        
        // Copy centroid to the appropriate part of the reconstructed vector
        for (int j = 0; j < params_.subvector_dimension && (i * params_.subvector_dimension + j) < dimension_; ++j) {
            reconstructed[i * params_.subvector_dimension + j] = centroid[j];
        }
    }
    
    return reconstructed;
}

Result<bool> PqIndex::train_subvector_centroids(const std::vector<std::vector<float>>& vectors) {
    if (vectors.empty() || dimension_ == 0) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INVALID_STATE, "Cannot train with empty vectors or unknown dimension"));
    }
    
    if (params_.num_subvectors == 0) {
        params_.num_subvectors = dimension_ / params_.subvector_dimension;
        if (dimension_ % params_.subvector_dimension != 0) {
            LOG_WARN(logger_, "Vector dimension " << dimension_ << " is not divisible by subvector dimension " << 
                     params_.subvector_dimension << ". Last " << (dimension_ % params_.subvector_dimension) << 
                     " dimensions will be ignored.");
        }
    }
    
    subvector_centroids_.clear();
    subvector_centroids_.resize(params_.num_subvectors);
    
    // Split each vector into subvectors
    std::vector<std::vector<std::vector<float>>> subvectors(params_.num_subvectors);
    
    for (const auto& vector : vectors) {
        auto split_vecs = split_into_subvectors(vector);
        for (int i = 0; i < params_.num_subvectors && i < static_cast<int>(split_vecs.size()); ++i) {
            subvectors[i].push_back(std::move(split_vecs[i]));
        }
    }
    
    // Perform k-means clustering for each subvector set
    for (int i = 0; i < params_.num_subvectors; ++i) {
        if (subvectors[i].size() < static_cast<size_t>(params_.num_centroids)) {
            LOG_WARN(logger_, "Not enough subvectors (" << subvectors[i].size() << ") to train " << 
                     params_.num_centroids << " centroids for subvector " << i);
        }
        
        subvector_centroids_[i] = perform_kmeans_clustering_on_subvectors(subvectors[i], params_.num_centroids);
    }
    
    LOG_INFO(logger_, "Trained PQ centroids for " << params_.num_subvectors << " subvectors");
    return true;
}

std::vector<std::vector<float>> PqIndex::split_into_subvectors(const std::vector<float>& vector) const {
    std::vector<std::vector<float>> subvectors(params_.num_subvectors);
    
    for (int i = 0; i < params_.num_subvectors; ++i) {
        subvectors[i].resize(params_.subvector_dimension);
        for (int j = 0; j < params_.subvector_dimension; ++j) {
            int idx = i * params_.subvector_dimension + j;
            if (idx < dimension_) {
                subvectors[i][j] = vector[idx];
            } else {
                // If we run out of dimensions, pad with 0
                subvectors[i][j] = 0.0f;
            }
        }
    }
    
    return subvectors;
}

std::vector<float> PqIndex::reconstruct_from_code(const std::vector<uint8_t>& code) const {
    return decode_code(code);
}

float PqIndex::compute_pq_distance(const std::vector<uint8_t>& code1, const std::vector<uint8_t>& code2) const {
    if (!is_trained_) {
        return std::numeric_limits<float>::max();
    }
    
    float dist = 0.0f;
    
    // Sum the distances across all subvectors
    for (int i = 0; i < params_.num_subvectors; ++i) {
        if (i >= static_cast<int>(code1.size()) || i >= static_cast<int>(code2.size())) {
            break;
        }
        
        uint8_t centroid_idx1 = code1[i];
        uint8_t centroid_idx2 = code2[i];
        
        // Look up precomputed distance between the centroids if possible
        // For this implementation, we'll compute the distance directly
        const auto& centroid1 = subvector_centroids_[i][centroid_idx1];
        const auto& centroid2 = subvector_centroids_[i][centroid_idx2];
        
        dist += compute_squared_distance(centroid1, centroid2);
    }
    
    return dist;
}

float PqIndex::compute_asymmetric_distance(const std::vector<float>& query, const std::vector<uint8_t>& code) const {
    if (!is_trained_) {
        return std::numeric_limits<float>::max();
    }
    
    float dist = 0.0f;
    
    // Compute distance between query subvectors and code centroids
    auto query_subvectors = split_into_subvectors(query);
    
    for (int i = 0; i < params_.num_subvectors; ++i) {
        if (i >= static_cast<int>(code.size())) {
            break;
        }
        
        uint8_t centroid_idx = code[i];
        const auto& query_subvector = query_subvectors[i];
        const auto& centroid = subvector_centroids_[i][centroid_idx];
        
        dist += compute_squared_distance(query_subvector, centroid);
    }
    
    return dist;
}

std::vector<std::vector<float>> PqIndex::perform_kmeans_clustering_on_subvectors(
    const std::vector<std::vector<float>>& subvectors, int k) const {
    
    if (subvectors.empty() || k <= 0) {
        return std::vector<std::vector<float>>();
    }
    
    // Initialize centroids randomly
    std::vector<std::vector<float>> centroids(k, std::vector<float>(params_.subvector_dimension));
    
    // Use random samples as initial centroids
    std::vector<int> indices(subvectors.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    
    for (int i = 0; i < k && i < static_cast<int>(subvectors.size()); ++i) {
        centroids[i] = subvectors[indices[i]];
    }
    
    // Additional random initialization for remaining centroids if needed
    for (int i = subvectors.size(); i < k; ++i) {
        // Initialize with a random small vector
        for (int j = 0; j < params_.subvector_dimension; ++j) {
            centroids[i][j] = static_cast<float>(rng_()) / static_cast<float>(rng_.max());
        }
    }
    
    std::vector<int> assignments(subvectors.size());
    int iteration = 0;
    float prev_total_distance = std::numeric_limits<float>::max();
    
    while (iteration < params_.max_iterations) {
        // Assign each subvector to the closest centroid
        float total_distance = 0.0f;
        for (size_t i = 0; i < subvectors.size(); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int closest_centroid = 0;
            
            for (int j = 0; j < k; ++j) {
                float dist = compute_squared_distance(subvectors[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            
            assignments[i] = closest_centroid;
            total_distance += min_dist;
        }
        
        // Move centroids to the average of assigned subvectors
        std::vector<std::vector<float>> new_centroids = centroids;
        std::vector<int> count(k, 0);
        
        // Reset centroids to zero
        for (auto& centroid : new_centroids) {
            std::fill(centroid.begin(), centroid.end(), 0.0f);
        }
        
        // Sum up subvectors assigned to each centroid
        for (size_t i = 0; i < subvectors.size(); ++i) {
            int cluster_id = assignments[i];
            count[cluster_id]++;
            
            for (int j = 0; j < params_.subvector_dimension; ++j) {
                new_centroids[cluster_id][j] += subvectors[i][j];
            }
        }
        
        // Average to get the new centroids
        for (int i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (int j = 0; j < params_.subvector_dimension; ++j) {
                    new_centroids[i][j] /= static_cast<float>(count[i]);
                }
            }
        }
        
        centroids = std::move(new_centroids);
        
        // Check for convergence
        if (std::abs(total_distance - prev_total_distance) < params_.tolerance) {
            break;
        }
        
        prev_total_distance = total_distance;
        iteration++;
    }
    
    return centroids;
}

float PqIndex::compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const {
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

bool PqIndex::validate() const {
    if (dimension_ <= 0) {
        LOG_ERROR(logger_, "Invalid dimension: " << dimension_);
        return false;
    }
    
    if (params_.subvector_dimension <= 0) {
        LOG_ERROR(logger_, "Subvector dimension must be positive");
        return false;
    }
    
    if (params_.num_centroids <= 0) {
        LOG_ERROR(logger_, "Number of centroids must be positive");
        return false;
    }
    
    return true;
}

} // namespace jadevectordb