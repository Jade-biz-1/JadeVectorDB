#include "services/index/hnsw_index.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <cmath>
#include <algorithm>
#include <unordered_set>

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
    std::unique_lock lock(index_mutex_);

    if (id_to_idx_.find(vector_id) != id_to_idx_.end()) {
        return tl::make_unexpected(ErrorInfo(ErrorCode::ALREADY_EXISTS, "Vector already exists"));
    }

    // Generate random level for this element
    int level = getRandomLevel();
    auto node = std::make_unique<HnswNode>(vector_id, vector, level);

    int new_idx = nodes_.size();
    id_to_idx_[vector_id] = new_idx;
    nodes_.push_back(std::move(node));

    // If this is the first element, make it the entry point
    if (entry_point_ == -1) {
        entry_point_ = new_idx;
        max_level_ = level;
        cur_element_count_++;
        return true;
    }

    // Update max level if needed
    if (level > max_level_) {
        max_level_ = level;
    }

    // Find nearest neighbors and connect at all levels
    int cur_c = entry_point_;

    // Search for neighbors at each level
    for (int lc = max_level_; lc >= 0; lc--) {
        if (lc <= level) {
            // Greedy search at this level to find nearest neighbors
            cur_c = greedySearch(vector, cur_c, lc);

            // Get candidates by searching around the closest point
            std::vector<std::pair<float, int>> candidates;
            candidates.push_back({calculateDistance(vector, nodes_[cur_c]->vector), cur_c});

            // Add neighbors of the current closest point as candidates
            if (lc < static_cast<int>(nodes_[cur_c]->neighbors.size())) {
                for (int neighbor_idx : nodes_[cur_c]->neighbors[lc]) {
                    float dist = calculateDistance(vector, nodes_[neighbor_idx]->vector);
                    candidates.push_back({dist, neighbor_idx});
                }
            }

            // Select M best neighbors
            auto selected = getNeighborsByHeuristic(candidates, params_.M);

            // Link the new element to selected neighbors
            for (const auto& [dist, neighbor_idx] : selected) {
                link(new_idx, neighbor_idx, lc);
                link(neighbor_idx, new_idx, lc);
            }
        } else {
            // Just search without connecting
            cur_c = greedySearch(vector, cur_c, lc);
        }
    }

    cur_element_count_++;
    return true;
}

Result<std::vector<std::pair<int, float>>> HnswIndex::search(
    const std::vector<float>& query, int k, float threshold) const {
    std::shared_lock lock(index_mutex_);

    if (entry_point_ == -1 || nodes_.empty()) {
        return std::vector<std::pair<int, float>>();
    }

    // Start from entry point at the highest level
    int cur_c = entry_point_;

    // Search through levels from top to bottom
    for (int lc = max_level_; lc > 0; lc--) {
        cur_c = greedySearch(query, cur_c, lc);
    }

    // At level 0, perform ef_search beam search to find k nearest neighbors
    std::vector<std::pair<float, int>> candidates;
    std::vector<std::pair<float, int>> visited;
    std::unordered_set<int> visited_set;

    candidates.push_back({calculateDistance(query, nodes_[cur_c]->vector), cur_c});
    visited_set.insert(cur_c);

    int ef = std::max(params_.ef_search, k);

    // Beam search at level 0
    while (!candidates.empty()) {
        // Get closest unvisited candidate
        std::sort(candidates.begin(), candidates.end());
        auto [dist, idx] = candidates.front();
        candidates.erase(candidates.begin());

        visited.push_back({dist, idx});

        // If we have enough good results, we can stop
        if (visited.size() >= static_cast<size_t>(ef)) {
            break;
        }

        // Explore neighbors
        if (idx < static_cast<int>(nodes_.size()) &&
            0 < static_cast<int>(nodes_[idx]->neighbors.size())) {
            for (int neighbor_idx : nodes_[idx]->neighbors[0]) {
                if (visited_set.find(neighbor_idx) == visited_set.end()) {
                    visited_set.insert(neighbor_idx);
                    float neighbor_dist = calculateDistance(query, nodes_[neighbor_idx]->vector);

                    // Add to candidates if better than worst current candidate
                    if (candidates.size() < static_cast<size_t>(ef) ||
                        neighbor_dist < candidates.back().first) {
                        candidates.push_back({neighbor_dist, neighbor_idx});
                    }
                }
            }
        }
    }

    // Convert to result format and filter by threshold
    std::vector<std::pair<int, float>> results;
    std::sort(visited.begin(), visited.end());

    for (const auto& [dist, idx] : visited) {
        if (threshold == 0.0f || dist <= threshold) {
            results.emplace_back(nodes_[idx]->id, dist);
            if (static_cast<int>(results.size()) >= k) {
                break;
            }
        }
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

int HnswIndex::getRandomLevel() {
    double r = uniform_distribution_(level_generator_);
    return static_cast<int>(-log(r) * params_.level_mult);
}

int HnswIndex::greedySearch(const std::vector<float>& query, int enterpoint, int level) const {
    int cur_best = enterpoint;
    float cur_dist = calculateDistance(query, nodes_[cur_best]->vector);
    bool changed = true;

    while (changed) {
        changed = false;

        // Check neighbors at this level
        if (level < static_cast<int>(nodes_[cur_best]->neighbors.size())) {
            for (int neighbor_idx : nodes_[cur_best]->neighbors[level]) {
                float dist = calculateDistance(query, nodes_[neighbor_idx]->vector);
                if (dist < cur_dist) {
                    cur_dist = dist;
                    cur_best = neighbor_idx;
                    changed = true;
                }
            }
        }
    }

    return cur_best;
}

float HnswIndex::calculateDistance(const std::vector<float>& a, const std::vector<float>& b) const {
    float dist = 0.0f;
    size_t dim = std::min(a.size(), b.size());
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

std::vector<std::pair<float, int>> HnswIndex::getNeighborsByHeuristic(
    const std::vector<std::pair<float, int>>& candidates, int max_size) const {

    // Simple heuristic: select closest neighbors
    std::vector<std::pair<float, int>> sorted_candidates = candidates;
    std::sort(sorted_candidates.begin(), sorted_candidates.end());

    if (static_cast<int>(sorted_candidates.size()) > max_size) {
        sorted_candidates.resize(max_size);
    }

    return sorted_candidates;
}

void HnswIndex::link(int src_idx, int dest_idx, int level) {
    // Ensure the neighbors vector has enough levels
    while (static_cast<int>(nodes_[src_idx]->neighbors.size()) <= level) {
        nodes_[src_idx]->neighbors.push_back(std::vector<int>());
    }

    // Check if already linked
    auto& neighbors = nodes_[src_idx]->neighbors[level];
    if (std::find(neighbors.begin(), neighbors.end(), dest_idx) == neighbors.end()) {
        neighbors.push_back(dest_idx);

        // Prune if exceeds M connections
        if (static_cast<int>(neighbors.size()) > params_.M) {
            // Remove furthest neighbor
            int furthest_idx = 0;
            float furthest_dist = calculateDistance(
                nodes_[src_idx]->vector,
                nodes_[neighbors[0]]->vector);

            for (size_t i = 1; i < neighbors.size(); ++i) {
                float dist = calculateDistance(
                    nodes_[src_idx]->vector,
                    nodes_[neighbors[i]]->vector);
                if (dist > furthest_dist) {
                    furthest_dist = dist;
                    furthest_idx = i;
                }
            }

            neighbors.erase(neighbors.begin() + furthest_idx);
        }
    }
}

int HnswIndex::get_dimension() const {
    std::shared_lock lock(index_mutex_);
    if (nodes_.empty()) {
        return 0;
    }
    return nodes_[0]->vector.size();
}

int HnswIndex::get_current_levels() const {
    std::shared_lock lock(index_mutex_);
    return max_level_ + 1;
}

} // namespace jadevectordb
