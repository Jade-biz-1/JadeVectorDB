#include "services/index/hnsw_index.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <cmath>

namespace jadevectordb {

HnswIndex::HnswIndex(const HnswParams& params)
    : params_(params),
      level_generator_(params.random_seed),
      uniform_distribution_(0.0, 1.0) {
    logger_ = std::make_shared<logging::Logger>("HnswIndex");
    nodes_.reserve(params.max_elements);
    // Note: link_locks_ cannot be pre-sized as std::mutex is not copyable
}

bool HnswIndex::initialize(const HnswParams& params) {
    params_ = params;
    return true;
}

Result<bool> HnswIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    // Stub: Just store the vector in a node
    std::unique_lock lock(index_mutex_);

    if (id_to_idx_.find(vector_id) != id_to_idx_.end()) {
        return tl::make_unexpected(ErrorInfo(ErrorCode::ALREADY_EXISTS, "Vector already exists"));
    }

    int level = 0; // Stub: always use level 0
    auto node = std::make_unique<HnswNode>(vector_id, vector, level);

    id_to_idx_[vector_id] = nodes_.size();
    nodes_.push_back(std::move(node));
    cur_element_count_++;

    if (entry_point_ == -1) {
        entry_point_ = 0;
    }

    return true;
}

Result<std::vector<std::pair<int, float>>> HnswIndex::search(
    const std::vector<float>& query, int k, float threshold) const {
    std::shared_lock lock(index_mutex_);

    // Stub: Simple linear search
    std::vector<std::pair<int, float>> results;
    for (const auto& node : nodes_) {
        float dist = 0.0f;
        for (size_t i = 0; i < query.size() && i < node->vector.size(); ++i) {
            float diff = query[i] - node->vector[i];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);
        if (threshold == 0.0f || dist <= threshold) {
            results.emplace_back(node->id, dist);
        }
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    if (static_cast<int>(results.size()) > k) {
        results.resize(k);
    }
    return results;
}

Result<bool> HnswIndex::build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors) {
    for (const auto& [id, vec] : vectors) {
        auto result = add_vector(id, vec);
        if (!result.has_value()) {
            return result;
        }
    }
    return true;
}

bool HnswIndex::contains(int vector_id) const {
    std::shared_lock lock(index_mutex_);
    return id_to_idx_.find(vector_id) != id_to_idx_.end();
}

Result<bool> HnswIndex::remove_vector(int vector_id) {
    std::unique_lock lock(index_mutex_);

    auto it = id_to_idx_.find(vector_id);
    if (it == id_to_idx_.end()) {
        return tl::make_unexpected(ErrorInfo(ErrorCode::NOT_FOUND, "Vector not found"));
    }

    id_to_idx_.erase(it);
    cur_element_count_--;
    return true;
}

Result<bool> HnswIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    auto remove_result = remove_vector(vector_id);
    if (!remove_result.has_value()) {
        return remove_result;
    }
    return add_vector(vector_id, new_vector);
}

size_t HnswIndex::size() const {
    std::shared_lock lock(index_mutex_);
    return cur_element_count_;
}

Result<std::unordered_map<std::string, std::string>> HnswIndex::get_stats() const {
    std::unordered_map<std::string, std::string> stats;
    stats["type"] = "HnswIndex";
    stats["size"] = std::to_string(size());
    stats["max_level"] = std::to_string(max_level_);
    return stats;
}

void HnswIndex::clear() {
    std::unique_lock lock(index_mutex_);
    nodes_.clear();
    id_to_idx_.clear();
    element_levels_.clear();
    cur_element_count_ = 0;
    max_level_ = 0;
    entry_point_ = -1;
}

bool HnswIndex::empty() const {
    std::shared_lock lock(index_mutex_);
    return cur_element_count_ == 0;
}

} // namespace jadevectordb
