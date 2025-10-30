#include "services/index/ivf_index.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <cmath>

namespace jadevectordb {

IvfIndex::IvfIndex(const IvfParams& params) : params_(params) {
    logger_ = std::make_shared<logging::Logger>("IvfIndex");
}

bool IvfIndex::initialize(const IvfParams& params) {
    params_ = params;
    return true;
}

Result<bool> IvfIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    vectors_[vector_id] = vector;
    if (dimension_ == 0) {
        dimension_ = vector.size();
    }
    return true;
}

Result<bool> IvfIndex::build() {
    is_trained_ = true;
    return true;
}

Result<std::vector<std::pair<int, float>>> IvfIndex::search(
    const std::vector<float>& query, int k, float threshold) const {
    std::vector<std::pair<int, float>> results;
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
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    if (static_cast<int>(results.size()) > k) {
        results.resize(k);
    }
    return results;
}

bool IvfIndex::contains(int vector_id) const {
    return vectors_.find(vector_id) != vectors_.end();
}

Result<bool> IvfIndex::remove_vector(int vector_id) {
    vectors_.erase(vector_id);
    return true;
}

Result<bool> IvfIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    vectors_[vector_id] = new_vector;
    return true;
}

size_t IvfIndex::size() const {
    return vectors_.size();
}

void IvfIndex::clear() {
    vectors_.clear();
    centroids_.clear();
    vector_cluster_map_.clear();
    cluster_map_.clear();
    dimension_ = 0;
    is_trained_ = false;
}

} // namespace jadevectordb
