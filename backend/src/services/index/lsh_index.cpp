#include "services/index/lsh_index.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <cmath>

namespace jadevectordb {

LshIndex::LshIndex(const LshParams& params) : params_(params) {
    logger_ = std::make_shared<logging::Logger>("LshIndex");
}

bool LshIndex::initialize(const LshParams& params) {
    params_ = params;
    return true;
}

Result<bool> LshIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    vectors_[vector_id] = vector;
    if (dimension_ == 0) {
        dimension_ = vector.size();
    }
    return true;
}

Result<bool> LshIndex::build() {
    is_built_ = true;
    return true;
}

Result<std::vector<std::pair<int, float>>> LshIndex::search(
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

bool LshIndex::contains(int vector_id) const {
    return vectors_.find(vector_id) != vectors_.end();
}

Result<bool> LshIndex::remove_vector(int vector_id) {
    vectors_.erase(vector_id);
    return true;
}

Result<bool> LshIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    vectors_[vector_id] = new_vector;
    return true;
}

size_t LshIndex::size() const {
    return vectors_.size();
}

void LshIndex::clear() {
    vectors_.clear();
    hash_tables_.clear();
    dimension_ = 0;
    is_built_ = false;
}

} // namespace jadevectordb
