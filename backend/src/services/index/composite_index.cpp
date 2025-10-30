#include "composite_index.h"
#include "lib/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace jadevectordb {

CompositeIndex::CompositeIndex(const CompositeIndexParams& params) : params_(params) {
    logger_ = logging::LoggerManager::get_logger("index.composite");
}

bool CompositeIndex::initialize(const CompositeIndexParams& params) {
    params_ = params;
    return true;
}

Result<bool> CompositeIndex::add_component_index(const std::string& component_id,
                                                 CompositeIndexType type,
                                                 const std::unordered_map<std::string, std::string>& params,
                                                 float weight) {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    // Check if component ID already exists
    if (id_to_idx_map_.find(component_id) != id_to_idx_map_.end()) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Component ID already exists: " + component_id));
    }
    
    // Create an index of the appropriate type based on the type parameter
    void* index_ptr = nullptr;
    
    switch (type) {
        case CompositeIndexType::HNSW: {
            auto* hnsw_index = new HnswIndex();
            // Apply parameters if provided
            HnswIndex::HnswParams hnsw_params;
            // Apply params from the map if possible
            // This would require mapping string params to the specific HnswParams fields
            hnsw_index->initialize(hnsw_params);
            index_ptr = hnsw_index;
            break;
        }
        case CompositeIndexType::IVF: {
            auto* ivf_index = new IvfIndex();
            IvfIndex::IvfParams ivf_params;
            // Apply params from the map if possible
            ivf_index->initialize(ivf_params);
            index_ptr = ivf_index;
            break;
        }
        case CompositeIndexType::LSH: {
            auto* lsh_index = new LshIndex();
            LshIndex::LshParams lsh_params;
            // Apply params from the map if possible
            lsh_index->initialize(lsh_params);
            index_ptr = lsh_index;
            break;
        }
        case CompositeIndexType::FLAT: {
            auto* flat_index = new FlatIndex();
            FlatIndex::FlatParams flat_params;
            // Apply params from the map if possible
            flat_index->initialize(flat_params);
            index_ptr = flat_index;
            break;
        }
        case CompositeIndexType::PQ: {
            auto* pq_index = new PqIndex();
            PqIndex::PqParams pq_params;
            // Apply params from the map if possible
            pq_index->initialize(pq_params);
            index_ptr = pq_index;
            break;
        }
        case CompositeIndexType::OPQ: {
            auto* opq_index = new OpqIndex();
            OpqIndex::OpqParams opq_params;
            // Apply params from the map if possible
            opq_index->initialize(opq_params);
            index_ptr = opq_index;
            break;
        }
        case CompositeIndexType::SQ: {
            auto* sq_index = new SqIndex();
            SqIndex::SqParams sq_params;
            // Apply params from the map if possible
            sq_index->initialize(sq_params);
            index_ptr = sq_index;
            break;
        }
        default:
            return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Unsupported index type"));
    }
    
    // Create the component and add it to our list
    CompositeIndexComponent component(component_id, type, index_ptr, params, weight);
    size_t idx = components_.size();
    components_.push_back(std::move(component));
    id_to_idx_map_[component_id] = idx;
    
    LOG_INFO(logger_, "Added component index " + component_id + " of type " + 
             std::to_string(static_cast<int>(type)));
    
    return true;
}

Result<bool> CompositeIndex::add_vector(int vector_id, const std::vector<float>& vector) {
    if (vector.empty()) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Vector cannot be empty"));
    }
    
    if (dimension_ == 0) {
        dimension_ = vector.size();
    } else if (static_cast<int>(vector.size()) != dimension_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Vector dimension mismatch"));
    }
    
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    // Add the vector to all component indices
    for (auto& component : components_) {
        Result<bool> result;
        
        switch (component.type) {
            case CompositeIndexType::HNSW: {
                auto* hnsw_index = static_cast<HnswIndex*>(component.index_ptr.get());
                result = hnsw_index->add_vector(vector_id, vector);
                break;
            }
            case CompositeIndexType::IVF: {
                auto* ivf_index = static_cast<IvfIndex*>(component.index_ptr.get());
                result = ivf_index->add_vector(vector_id, vector);
                break;
            }
            case CompositeIndexType::LSH: {
                auto* lsh_index = static_cast<LshIndex*>(component.index_ptr.get());
                result = lsh_index->add_vector(vector_id, vector);
                break;
            }
            case CompositeIndexType::FLAT: {
                auto* flat_index = static_cast<FlatIndex*>(component.index_ptr.get());
                result = flat_index->add_vector(vector_id, vector);
                break;
            }
            case CompositeIndexType::PQ: {
                auto* pq_index = static_cast<PqIndex*>(component.index_ptr.get());
                result = pq_index->add_vector(vector_id, vector);
                break;
            }
            case CompositeIndexType::OPQ: {
                auto* opq_index = static_cast<OpqIndex*>(component.index_ptr.get());
                result = opq_index->add_vector(vector_id, vector);
                break;
            }
            case CompositeIndexType::SQ: {
                auto* sq_index = static_cast<SqIndex*>(component.index_ptr.get());
                result = sq_index->add_vector(vector_id, vector);
                break;
            }
            default:
                return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Unsupported index type"));
        }
        
        if (!result.has_value()) {
            logger_->error("Failed to add vector to component " + component.id + ": " + result.error().message);
            // For now, we continue adding to other components even if one fails
        }
    }
    
    return true;
}

Result<bool> CompositeIndex::build() {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    // Build all component indices
    for (auto& component : components_) {
        Result<bool> result;
        
        switch (component.type) {
            case CompositeIndexType::HNSW: {
                auto* hnsw_index = static_cast<HnswIndex*>(component.index_ptr.get());
                result = hnsw_index->build();
                break;
            }
            case CompositeIndexType::IVF: {
                auto* ivf_index = static_cast<IvfIndex*>(component.index_ptr.get());
                result = ivf_index->build();
                break;
            }
            case CompositeIndexType::LSH: {
                auto* lsh_index = static_cast<LshIndex*>(component.index_ptr.get());
                result = lsh_index->build();
                break;
            }
            case CompositeIndexType::FLAT: {
                auto* flat_index = static_cast<FlatIndex*>(component.index_ptr.get());
                result = flat_index->build();
                break;
            }
            case CompositeIndexType::PQ: {
                auto* pq_index = static_cast<PqIndex*>(component.index_ptr.get());
                result = pq_index->build();
                break;
            }
            case CompositeIndexType::OPQ: {
                auto* opq_index = static_cast<OpqIndex*>(component.index_ptr.get());
                result = opq_index->build();
                break;
            }
            case CompositeIndexType::SQ: {
                auto* sq_index = static_cast<SqIndex*>(component.index_ptr.get());
                result = sq_index->build();
                break;
            }
            default:
                return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Unsupported index type"));
        }
        
        if (!result.has_value()) {
            logger_->error("Failed to build component " + component.id + ": " + result.error().message);
            return result;
        }
    }
    
    is_built_ = true;
    logger_->info("Composite index built with " + std::to_string(components_.size()) + " components");
    
    return true;
}

Result<std::vector<std::pair<int, float>>> CompositeIndex::search(const std::vector<float>& query, 
                                                                 int k, 
                                                                 float threshold) const {
    if (!is_built_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Composite index is not built"));
    }
    
    if (static_cast<int>(query.size()) != dimension_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Query dimension mismatch"));
    }
    
    if (components_.empty()) {
        return std::vector<std::pair<int, float>>();
    }
    
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    // Search in all components
    std::vector<std::vector<std::pair<int, float>>> component_results;
    component_results.reserve(components_.size());
    
    for (size_t i = 0; i < components_.size(); ++i) {
        auto result = search_component(i, query, k, threshold);
        if (result.has_value()) {
            component_results.push_back(std::move(result.value()));
        } else {
            LOG_WARN(logger_, "Search failed for component " + components_[i].id + ": " + result.error().message);
            // Add empty result for this component
            component_results.push_back(std::vector<std::pair<int, float>>());
        }
    }
    
    // Fuse the results based on the fusion method
    std::vector<std::pair<int, float>> final_results;
    
    switch (params_.fusion_method) {
        case CompositeIndexParams::RRF:
            final_results = fuse_results_rrf(component_results);
            break;
        case CompositeIndexParams::WEIGHTED:
            final_results = fuse_results_weighted(component_results);
            break;
        case CompositeIndexParams::SIMPLE:
            final_results = fuse_results_simple(component_results);
            break;
        default:
            final_results = fuse_results_simple(component_results);  // Default to simple
            break;
    }
    
    // Return top k results
    if (k > 0 && static_cast<size_t>(k) < final_results.size()) {
        final_results.resize(k);
    }
    
    return std::move(final_results);
}

Result<bool> CompositeIndex::build_from_vectors(const std::vector<std::pair<int, std::vector<float>>>& vectors) {
    if (vectors.empty()) {
        logger_->warn("Building composite index with empty vector set");
        return true;
    }
    
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    // Add all vectors to all components
    for (const auto& vec_pair : vectors) {
        auto result = add_vector(vec_pair.first, vec_pair.second);
        if (!result.has_value()) {
            return result;
        }
    }
    
    // Now build all components
    return build();
}

bool CompositeIndex::contains(int vector_id) const {
    // A vector is considered contained if it's in at least one component
    // In a real implementation, we might want to be more strict
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    for (const auto& component : components_) {
        bool contains_result = false;
        
        switch (component.type) {
            case CompositeIndexType::HNSW: {
                auto* hnsw_index = static_cast<const HnswIndex*>(component.index_ptr.get());
                contains_result = hnsw_index->contains(vector_id);
                break;
            }
            case CompositeIndexType::IVF: {
                auto* ivf_index = static_cast<const IvfIndex*>(component.index_ptr.get());
                contains_result = ivf_index->contains(vector_id);
                break;
            }
            case CompositeIndexType::LSH: {
                auto* lsh_index = static_cast<const LshIndex*>(component.index_ptr.get());
                contains_result = lsh_index->contains(vector_id);
                break;
            }
            case CompositeIndexType::FLAT: {
                auto* flat_index = static_cast<const FlatIndex*>(component.index_ptr.get());
                contains_result = flat_index->contains(vector_id);
                break;
            }
            case CompositeIndexType::PQ: {
                auto* pq_index = static_cast<const PqIndex*>(component.index_ptr.get());
                contains_result = pq_index->contains(vector_id);
                break;
            }
            case CompositeIndexType::OPQ: {
                auto* opq_index = static_cast<const OpqIndex*>(component.index_ptr.get());
                contains_result = opq_index->contains(vector_id);
                break;
            }
            case CompositeIndexType::SQ: {
                auto* sq_index = static_cast<const SqIndex*>(component.index_ptr.get());
                contains_result = sq_index->contains(vector_id);
                break;
            }
            default:
                // Skip unsupported types
                continue;
        }
        
        if (contains_result) {
            return true;
        }
    }
    
    return false;
}

Result<bool> CompositeIndex::remove_vector(int vector_id) {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    bool any_success = false;
    std::string last_error;
    
    for (auto& component : components_) {
        Result<bool> result;
        
        switch (component.type) {
            case CompositeIndexType::HNSW: {
                auto* hnsw_index = static_cast<HnswIndex*>(component.index_ptr.get());
                result = hnsw_index->remove_vector(vector_id);
                break;
            }
            case CompositeIndexType::IVF: {
                auto* ivf_index = static_cast<IvfIndex*>(component.index_ptr.get());
                result = ivf_index->remove_vector(vector_id);
                break;
            }
            case CompositeIndexType::LSH: {
                auto* lsh_index = static_cast<LshIndex*>(component.index_ptr.get());
                result = lsh_index->remove_vector(vector_id);
                break;
            }
            case CompositeIndexType::FLAT: {
                auto* flat_index = static_cast<FlatIndex*>(component.index_ptr.get());
                result = flat_index->remove_vector(vector_id);
                break;
            }
            case CompositeIndexType::PQ: {
                auto* pq_index = static_cast<PqIndex*>(component.index_ptr.get());
                result = pq_index->remove_vector(vector_id);
                break;
            }
            case CompositeIndexType::OPQ: {
                auto* opq_index = static_cast<OpqIndex*>(component.index_ptr.get());
                result = opq_index->remove_vector(vector_id);
                break;
            }
            case CompositeIndexType::SQ: {
                auto* sq_index = static_cast<SqIndex*>(component.index_ptr.get());
                result = sq_index->remove_vector(vector_id);
                break;
            }
            default:
                continue; // Skip unsupported types
        }
        
        if (result.has_value() && result.value()) {
            any_success = true;
        } else if (!result.has_value()) {
            last_error = result.error().message;
        }
    }
    
    if (any_success) {
        return true;
    } else {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "No component index contained the vector or all removals failed: " + last_error));
    }
}

Result<bool> CompositeIndex::update_vector(int vector_id, const std::vector<float>& new_vector) {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    if (static_cast<int>(new_vector.size()) != dimension_) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Vector dimension mismatch"));
    }
    
    bool any_success = false;
    std::string last_error;
    
    for (auto& component : components_) {
        Result<bool> result;
        
        switch (component.type) {
            case CompositeIndexType::HNSW: {
                auto* hnsw_index = static_cast<HnswIndex*>(component.index_ptr.get());
                result = hnsw_index->update_vector(vector_id, new_vector);
                break;
            }
            case CompositeIndexType::IVF: {
                auto* ivf_index = static_cast<IvfIndex*>(component.index_ptr.get());
                result = ivf_index->update_vector(vector_id, new_vector);
                break;
            }
            case CompositeIndexType::LSH: {
                auto* lsh_index = static_cast<LshIndex*>(component.index_ptr.get());
                result = lsh_index->update_vector(vector_id, new_vector);
                break;
            }
            case CompositeIndexType::FLAT: {
                auto* flat_index = static_cast<FlatIndex*>(component.index_ptr.get());
                result = flat_index->update_vector(vector_id, new_vector);
                break;
            }
            case CompositeIndexType::PQ: {
                auto* pq_index = static_cast<PqIndex*>(component.index_ptr.get());
                result = pq_index->update_vector(vector_id, new_vector);
                break;
            }
            case CompositeIndexType::OPQ: {
                auto* opq_index = static_cast<OpqIndex*>(component.index_ptr.get());
                result = opq_index->update_vector(vector_id, new_vector);
                break;
            }
            case CompositeIndexType::SQ: {
                auto* sq_index = static_cast<SqIndex*>(component.index_ptr.get());
                result = sq_index->update_vector(vector_id, new_vector);
                break;
            }
            default:
                continue; // Skip unsupported types
        }
        
        if (result.has_value() && result.value()) {
            any_success = true;
        } else if (!result.has_value()) {
            last_error = result.error().message;
        }
    }
    
    if (any_success) {
        return true;
    } else {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "No component index contained the vector or all updates failed: " + last_error));
    }
}

size_t CompositeIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    
    // For a composite index, we might want to return the size of the largest component
    // or some other aggregation of sizes across components
    // For now, let's take the size of the first component as a representative value
    if (components_.empty()) {
        return 0;
    }
    
    return get_component_size(0);
}

Result<std::unordered_map<std::string, std::string>> CompositeIndex::get_stats() const {
    std::unordered_map<std::string, std::string> stats;
    stats["index_type"] = "Composite";
    stats["num_components"] = std::to_string(components_.size());
    stats["fusion_method"] = std::to_string(static_cast<int>(params_.fusion_method));
    stats["vector_count"] = std::to_string(size());
    stats["is_built"] = is_built_ ? "true" : "false";
    stats["dimension"] = std::to_string(dimension_);
    
    return std::move(stats);
}

void CompositeIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(index_mutex_);
    
    for (auto& component : components_) {
        switch (component.type) {
            case CompositeIndexType::HNSW: {
                auto* hnsw_index = static_cast<HnswIndex*>(component.index_ptr.get());
                hnsw_index->clear();
                break;
            }
            case CompositeIndexType::IVF: {
                auto* ivf_index = static_cast<IvfIndex*>(component.index_ptr.get());
                ivf_index->clear();
                break;
            }
            case CompositeIndexType::LSH: {
                auto* lsh_index = static_cast<LshIndex*>(component.index_ptr.get());
                lsh_index->clear();
                break;
            }
            case CompositeIndexType::FLAT: {
                auto* flat_index = static_cast<FlatIndex*>(component.index_ptr.get());
                flat_index->clear();
                break;
            }
            case CompositeIndexType::PQ: {
                auto* pq_index = static_cast<PqIndex*>(component.index_ptr.get());
                pq_index->clear();
                break;
            }
            case CompositeIndexType::OPQ: {
                auto* opq_index = static_cast<OpqIndex*>(component.index_ptr.get());
                opq_index->clear();
                break;
            }
            case CompositeIndexType::SQ: {
                auto* sq_index = static_cast<SqIndex*>(component.index_ptr.get());
                sq_index->clear();
                break;
            }
            default:
                // Skip unsupported types
                break;
        }
    }
    
    is_built_ = false;
}

bool CompositeIndex::empty() const {
    return size() == 0;
}

int CompositeIndex::get_dimension() const {
    return dimension_;
}

size_t CompositeIndex::get_num_components() const {
    std::shared_lock<std::shared_mutex> lock(index_mutex_);
    return components_.size();
}

CompositeIndex::CompositeIndexParams::FusionMethod CompositeIndex::get_fusion_method() const {
    return params_.fusion_method;
}

Result<std::vector<std::pair<int, float>>> CompositeIndex::search_component(
    size_t component_idx, 
    const std::vector<float>& query, 
    int k, 
    float threshold) const {
    
    if (component_idx >= components_.size()) {
        return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Invalid component index"));
    }
    
    const auto& component = components_[component_idx];
    
    switch (component.type) {
        case CompositeIndexType::HNSW: {
            auto* hnsw_index = static_cast<const HnswIndex*>(component.index_ptr.get());
            return hnsw_index->search(query, k, threshold);
        }
        case CompositeIndexType::IVF: {
            auto* ivf_index = static_cast<const IvfIndex*>(component.index_ptr.get());
            return ivf_index->search(query, k, threshold);
        }
        case CompositeIndexType::LSH: {
            auto* lsh_index = static_cast<const LshIndex*>(component.index_ptr.get());
            return lsh_index->search(query, k, threshold);
        }
        case CompositeIndexType::FLAT: {
            auto* flat_index = static_cast<const FlatIndex*>(component.index_ptr.get());
            return flat_index->search(query, k, threshold);
        }
        case CompositeIndexType::PQ: {
            auto* pq_index = static_cast<const PqIndex*>(component.index_ptr.get());
            return pq_index->search(query, k, threshold);
        }
        case CompositeIndexType::OPQ: {
            auto* opq_index = static_cast<const OpqIndex*>(component.index_ptr.get());
            return opq_index->search(query, k, threshold);
        }
        case CompositeIndexType::SQ: {
            auto* sq_index = static_cast<const SqIndex*>(component.index_ptr.get());
            return sq_index->search(query, k, threshold);
        }
        default:
            return tl::make_unexpected(MAKE_ERROR(ErrorCode::INTERNAL_ERROR, "Unsupported index type"));
    }
}

std::vector<std::pair<int, float>> CompositeIndex::fuse_results_rrf(
    const std::vector<std::vector<std::pair<int, float>>>& component_results) const {
    
    // Map from vector ID to aggregated score using RRF
    std::unordered_map<int, float> score_map;
    
    for (size_t comp_idx = 0; comp_idx < component_results.size(); ++comp_idx) {
        const auto& results = component_results[comp_idx];
        auto weight = components_[comp_idx].weight;
        
        for (size_t rank = 0; rank < results.size(); ++rank) {
            int vector_id = results[rank].first;
            // In RRF, the score is based on rank: 1.0 / (params_.rrf_k + rank)
            float score = weight * (1.0f / (params_.rrf_k + rank));
            score_map[vector_id] += score;
        }
    }
    
    // Convert map to vector and sort by score (descending)
    std::vector<std::pair<int, float>> final_results;
    final_results.reserve(score_map.size());
    
    for (const auto& pair : score_map) {
        final_results.emplace_back(pair.first, pair.second);
    }
    
    std::sort(final_results.begin(), final_results.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;  // Descending order
              });
    
    return final_results;
}

std::vector<std::pair<int, float>> CompositeIndex::fuse_results_weighted(
    const std::vector<std::vector<std::pair<int, float>>>& component_results) const {
    
    // Map from vector ID to aggregated weighted score
    std::unordered_map<int, float> score_map;
    
    for (size_t comp_idx = 0; comp_idx < component_results.size(); ++comp_idx) {
        const auto& results = component_results[comp_idx];
        auto weight = components_[comp_idx].weight;
        
        for (const auto& result : results) {
            int vector_id = result.first;
            float similarity = result.second;  // Assuming smaller distance means higher similarity
            // For similarity scores, we might need to convert distances to similarities
            // For now, we'll use the inverse of the distance (with a small constant to avoid division by zero)
            float score = weight / (similarity + 1e-6f);
            score_map[vector_id] += score;
        }
    }
    
    // Convert map to vector and sort by score (descending)
    std::vector<std::pair<int, float>> final_results;
    final_results.reserve(score_map.size());
    
    for (const auto& pair : score_map) {
        // Convert back to distance representation
        final_results.emplace_back(pair.first, 1.0f / pair.second);
    }
    
    std::sort(final_results.begin(), final_results.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second < b.second;  // Ascending order (smaller distance is better)
              });
    
    return final_results;
}

std::vector<std::pair<int, float>> CompositeIndex::fuse_results_simple(
    const std::vector<std::vector<std::pair<int, float>>>& component_results) const {
    
    // Simple approach: collect all results and deduplicate
    std::unordered_map<int, std::vector<float>> all_results;  // vector_id -> list of distances
    
    for (const auto& results : component_results) {
        for (const auto& result : results) {
            all_results[result.first].push_back(result.second);
        }
    }
    
    // Average distances for duplicate results
    std::vector<std::pair<int, float>> final_results;
    for (const auto& pair : all_results) {
        int vector_id = pair.first;
        const auto& distances = pair.second;
        
        float avg_distance = 0.0f;
        for (float dist : distances) {
            avg_distance += dist;
        }
        avg_distance /= distances.size();
        
        final_results.emplace_back(vector_id, avg_distance);
    }
    
    // Sort by distance (ascending)
    std::sort(final_results.begin(), final_results.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second < b.second;
              });
    
    return final_results;
}

float CompositeIndex::compute_squared_distance(const std::vector<float>& a, const std::vector<float>& b) const {
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

bool CompositeIndex::validate() const {
    if (dimension_ <= 0) {
        logger_->error("Invalid dimension: {}", std::to_string(dimension_));
        return false;
    }
    
    if (components_.empty()) {
        logger_->error("Composite index has no components");
        return false;
    }
    
    return true;
}

size_t CompositeIndex::get_component_size(size_t component_idx) const {
    if (component_idx >= components_.size()) {
        return 0;
    }
    
    const auto& component = components_[component_idx];
    
    switch (component.type) {
        case CompositeIndexType::HNSW: {
            auto* hnsw_index = static_cast<const HnswIndex*>(component.index_ptr.get());
            return hnsw_index->size();
        }
        case CompositeIndexType::IVF: {
            auto* ivf_index = static_cast<const IvfIndex*>(component.index_ptr.get());
            return ivf_index->size();
        }
        case CompositeIndexType::LSH: {
            auto* lsh_index = static_cast<const LshIndex*>(component.index_ptr.get());
            return lsh_index->size();
        }
        case CompositeIndexType::FLAT: {
            auto* flat_index = static_cast<const FlatIndex*>(component.index_ptr.get());
            return flat_index->size();
        }
        case CompositeIndexType::PQ: {
            auto* pq_index = static_cast<const PqIndex*>(component.index_ptr.get());
            return pq_index->size();
        }
        case CompositeIndexType::OPQ: {
            auto* opq_index = static_cast<const OpqIndex*>(component.index_ptr.get());
            return opq_index->size();
        }
        case CompositeIndexType::SQ: {
            auto* sq_index = static_cast<const SqIndex*>(component.index_ptr.get());
            return sq_index->size();
        }
        default:
            return 0;
    }
}

} // namespace jadevectordb