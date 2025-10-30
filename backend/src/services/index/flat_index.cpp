#include "services/index/flat_index.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <shared_mutex>
#include <cmath>

namespace jadevectordb {

FlatIndex::FlatIndex(const FlatParams& params) : params_(params) {
    logger_ = std::make_shared<logging::Logger>("FlatIndex");
}

bool FlatIndex::initialize(const FlatParams& params) {
    params_ = params;
    return true;
}

Result<bool> FlatIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    std::unique_lock lock(index_mutex_);

    if (dimension_ == 0) {
        dimension_ = vector.size();
    } else if (static_cast<int>(vector.size()) != dimension_) {
        return tl::make_unexpected(ErrorInfo(ErrorCode::INVALID_ARGUMENT, "Vector dimension mismatch"));
    }

    if (id_to_idx_.find(vector_id) != id_to_idx_.end()) {
        return tl::make_unexpected(ErrorInfo(ErrorCode::ALREADY_EXISTS, "Vector ID already exists"));
    }

    id_to_idx_[vector_id] = vectors_.size();
    vectors_.push_back({vector_id, vector});

    return true;
}

Result<bool> FlatIndex::build() {
    is_built_ = true;
    return true;
}

Result<std::vector<std::pair<int, float>>> FlatIndex::search(
    const std::vector<float>& query, int k, float threshold) const {

    std::shared_lock lock(index_mutex_);

    if (vectors_.empty()) {
        return std::vector<std::pair<int, float>>{};
    }

    std::vector<std::pair<int, float>> results;
    results.reserve(vectors_.size());

    // Simple brute force search - euclidean distance
    for (const auto& [id, vec] : vectors_) {
        float dist = 0.0f;
        for (size_t i = 0; i < query.size() && i < vec.size(); ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);

        if (threshold == 0.0f || dist <= threshold) {
            results.push_back({id, dist});
        }
    }

    // Sort by distance and keep top k
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (static_cast<int>(results.size()) > k) {
        results.resize(k);
    }

    return results;
}

Result<std::vector<std::pair<int, float>>> FlatIndex::search_with_distance(
    const std::vector<float>& query, int k, float threshold,
    std::function<float(const std::vector<float>&, const std::vector<float>&)> distance_func) const {
    // Stub implementation
    return search(query, k, threshold);
}

Result<bool> FlatIndex::build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors) {
    for (const auto& [id, vec] : vectors) {
        auto result = add_vector(id, vec);
        if (!result.has_value()) {
            return result;
        }
    }
    return build();
}

bool FlatIndex::contains(int vector_id) const {
    std::shared_lock lock(index_mutex_);
    return id_to_idx_.find(vector_id) != id_to_idx_.end();
}

Result<bool> FlatIndex::remove_vector(int vector_id) {
    std::unique_lock lock(index_mutex_);

    auto it = id_to_idx_.find(vector_id);
    if (it == id_to_idx_.end()) {
        return tl::make_unexpected(ErrorInfo(ErrorCode::NOT_FOUND, "Vector not found"));
    }

    size_t idx = it->second;
    id_to_idx_.erase(it);

    // Remove from vectors
    if (idx < vectors_.size() - 1) {
        vectors_[idx] = vectors_.back();
        id_to_idx_[vectors_[idx].first] = idx;
    }
    vectors_.pop_back();

    return true;
}

Result<bool> FlatIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    auto remove_result = remove_vector(vector_id);
    if (!remove_result.has_value()) {
        return remove_result;
    }
    return add_vector(vector_id, new_vector);
}

size_t FlatIndex::size() const {
    std::shared_lock lock(index_mutex_);
    return vectors_.size();
}

Result<std::unordered_map<std::string, std::string>> FlatIndex::get_stats() const {
    std::unordered_map<std::string, std::string> stats;
    stats["type"] = "FlatIndex";
    stats["size"] = std::to_string(size());
    stats["dimension"] = std::to_string(dimension_);
    return stats;
}

void FlatIndex::clear() {
    std::unique_lock lock(index_mutex_);
    vectors_.clear();
    id_to_idx_.clear();
    norms_.clear();
    dimension_ = 0;
    is_built_ = false;
}

bool FlatIndex::empty() const {
    std::shared_lock lock(index_mutex_);
    return vectors_.empty();
}

int FlatIndex::get_dimension() const {
    return dimension_;
}

} // namespace jadevectordb
