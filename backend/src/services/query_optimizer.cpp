#include "query_optimizer.h"
#include "models/vector.h"
#include "similarity_search.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <iomanip>

namespace jadevectordb {

QueryOptimizer::QueryOptimizer() {
    // Initialize with default index types
}

QueryOptimizationPlan QueryOptimizer::generate_query_plan(const std::string& database_id,
                                             const Vector& query_vector,
                                             const SearchParams& params) {
    // Check cache first
    std::string params_hash = hash_params(params);
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const QueryOptimizationPlan* cached = get_cached_plan(database_id, params_hash);
        if (cached) {
            return *cached;
        }
    }
    
    QueryOptimizationPlan plan;
    plan.plan_id = database_id + "_" + params_hash;
    
    // Select best index based on cost model
    plan.selected_index = select_best_index(database_id, params);
    
    // Calculate filter selectivity
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto stats_it = query_stats_.find(database_id);
    size_t total_vectors = stats_it != query_stats_.end() ? 
                          stats_it->second.vectors_scanned : 10000;
    
    double selectivity = calculate_filter_selectivity(params, total_vectors);
    
    // Determine if filter pushdown is beneficial
    plan.use_filter_pushdown = should_use_filter_pushdown(params, selectivity);
    
    // Optimize filter order (most selective first)
    if (plan.use_filter_pushdown) {
        plan.filter_order = optimize_filter_order(params);
    }
    
    // Early termination for top-k queries with high selectivity
    plan.use_early_termination = (params.top_k > 0 && selectivity < 0.3);
    
    // Calculate estimated cost and time
    auto index_stats_it = index_stats_.find(database_id);
    if (index_stats_it != index_stats_.end()) {
        auto idx_it = index_stats_it->second.find(plan.selected_index);
        if (idx_it != index_stats_it->second.end()) {
            plan.estimated_cost = calculate_index_cost(plan.selected_index,
                                                      idx_it->second,
                                                      params);
            plan.estimated_time_ms = idx_it->second.avg_query_time_ms;
            
            // Adjust for filters
            if (plan.use_filter_pushdown) {
                plan.estimated_time_ms *= selectivity;
            }
        }
    }
    
    // Generate reasoning
    std::stringstream reasoning;
    reasoning << "Selected " << static_cast<int>(plan.selected_index) << " index";
    if (plan.use_filter_pushdown) {
        reasoning << " with filter pushdown (selectivity: " 
                 << std::fixed << std::setprecision(2) << selectivity << ")";
    }
    if (plan.use_early_termination) {
        reasoning << ", using early termination";
    }
    plan.reasoning = reasoning.str();
    
    // Cache the plan
    {
        std::lock_guard<std::mutex> cache_lock(cache_mutex_);
        plan_cache_[database_id][params_hash] = plan;
        evict_old_plans(database_id);
    }
    
    return plan;
}

void QueryOptimizer::update_index_stats(const std::string& database_id,
                                       IndexType index_type,
                                       const IndexStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    index_stats_[database_id][index_type] = stats;
}

void QueryOptimizer::record_query_execution(const std::string& database_id,
                                           const QueryOptimizationPlan& plan,
                                           double actual_time_ms,
                                           size_t vectors_scanned,
                                           size_t results_returned) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto& stats = query_stats_[database_id];
    
    // Update running averages
    double total_latency = stats.avg_latency_ms * stats.total_queries + actual_time_ms;
    stats.total_queries++;
    stats.avg_latency_ms = total_latency / stats.total_queries;
    stats.vectors_scanned += vectors_scanned;
    stats.results_returned += results_returned;
    stats.last_query_time = std::chrono::system_clock::now();
    
    // Update index statistics with actual performance
    if (index_stats_.find(database_id) != index_stats_.end()) {
        auto& idx_stats = index_stats_[database_id][plan.selected_index];
        double total_time = idx_stats.avg_query_time_ms * (stats.total_queries - 1) + actual_time_ms;
        idx_stats.avg_query_time_ms = total_time / stats.total_queries;
        idx_stats.last_updated = std::chrono::system_clock::now();
    }
}

const QueryOptimizationPlan* QueryOptimizer::get_cached_plan(const std::string& database_id,
                                                const std::string& params_hash) const {
    auto db_it = plan_cache_.find(database_id);
    if (db_it != plan_cache_.end()) {
        auto plan_it = db_it->second.find(params_hash);
        if (plan_it != db_it->second.end()) {
            return &plan_it->second;
        }
    }
    return nullptr;
}

double QueryOptimizer::calculate_filter_selectivity(const SearchParams& params,
                                                   size_t total_vectors) const {
    if (total_vectors == 0) return 1.0;
    
    double selectivity = 1.0;
    
    // Estimate selectivity based on filters
    if (!params.filter_tags.empty()) {
        // Tags typically reduce results by 10-30%
        selectivity *= 0.2;
    }
    
    if (!params.filter_owner.empty()) {
        // Owner filter is typically selective (5-20% of vectors)
        selectivity *= 0.15;
    }
    
    if (!params.filter_category.empty()) {
        // Category filter is moderately selective (10-40% of vectors)
        selectivity *= 0.25;
    }
    
    if (params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
        // Score filters can be very selective
        double score_range = params.filter_max_score - params.filter_min_score;
        selectivity *= score_range;
    }
    
    if (params.threshold > 0.0f) {
        // Similarity threshold is typically very selective
        selectivity *= 0.1;
    }
    
    return std::max(0.001, std::min(1.0, selectivity));
}

std::vector<std::string> QueryOptimizer::optimize_filter_order(const SearchParams& params) const {
    std::vector<std::pair<std::string, double>> filters;
    
    // Assign selectivity estimates to each filter
    if (!params.filter_owner.empty()) {
        filters.emplace_back("owner", 0.15);  // Most selective
    }
    
    if (!params.filter_tags.empty()) {
        filters.emplace_back("tags", 0.2);
    }
    
    if (!params.filter_category.empty()) {
        filters.emplace_back("category", 0.25);
    }
    
    if (params.filter_min_score > 0.0f || params.filter_max_score < 1.0f) {
        double score_range = params.filter_max_score - params.filter_min_score;
        filters.emplace_back("score", score_range);
    }
    
    if (params.threshold > 0.0f) {
        filters.emplace_back("threshold", 0.1);  // Very selective
    }
    
    // Sort by selectivity (most selective first)
    std::sort(filters.begin(), filters.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; });
    
    std::vector<std::string> order;
    for (const auto& [name, _] : filters) {
        order.push_back(name);
    }
    
    return order;
}

void QueryOptimizer::clear_plan_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    plan_cache_.clear();
}

QueryStats QueryOptimizer::get_query_stats(const std::string& database_id) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto it = query_stats_.find(database_id);
    if (it != query_stats_.end()) {
        return it->second;
    }
    return QueryStats{};
}

double QueryOptimizer::calculate_index_cost(IndexType index_type,
                                           const IndexStats& stats,
                                           const SearchParams& params) const {
    double cost = 0.0;
    
    switch (index_type) {
        case IndexType::FLAT:
            // Brute force: O(n * d) where n is vector count, d is dimension
            cost = static_cast<double>(stats.vector_count) * stats.dimension * 0.001;
            break;
            
        case IndexType::HNSW:
            // HNSW: O(log(n) * d) average case
            cost = std::log2(static_cast<double>(stats.vector_count)) * 
                   stats.dimension * 0.01;
            // Account for recall rate (lower recall = lower cost but less accuracy)
            cost /= (stats.recall_rate > 0.0 ? stats.recall_rate : 1.0);
            break;
            
        case IndexType::IVF:
            // IVF: O(sqrt(n) * d) with proper clustering
            cost = std::sqrt(static_cast<double>(stats.vector_count)) * 
                   stats.dimension * 0.005;
            break;
            
        case IndexType::LSH:
            // LSH: O(d) for hash computation + O(k) for bucket scan
            cost = stats.dimension * 0.002 + params.top_k * 0.001;
            break;
            
        case IndexType::COMPOSITE:
            // Composite: depends on underlying indices
            cost = std::min(
                std::log2(static_cast<double>(stats.vector_count)) * stats.dimension * 0.01,
                static_cast<double>(stats.vector_count) * stats.dimension * 0.001
            );
            break;
    }
    
    // Add cost for result sorting and limiting
    if (params.top_k > 0) {
        cost += params.top_k * std::log2(params.top_k) * 0.0001;
    }
    
    return cost;
}

IndexType QueryOptimizer::select_best_index(const std::string& database_id,
                                           const SearchParams& params) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto it = index_stats_.find(database_id);
    if (it == index_stats_.end() || it->second.empty()) {
        // Default to FLAT if no statistics available
        return IndexType::FLAT;
    }
    
    IndexType best_index = IndexType::FLAT;
    double min_cost = std::numeric_limits<double>::max();
    
    for (const auto& [index_type, stats] : it->second) {
        double cost = calculate_index_cost(index_type, stats, params);
        
        if (cost < min_cost) {
            min_cost = cost;
            best_index = index_type;
        }
    }
    
    return best_index;
}

std::string QueryOptimizer::hash_params(const SearchParams& params) const {
    std::stringstream ss;
    ss << params.top_k << "_"
       << params.threshold << "_"
       << params.include_vector_data << "_"
       << params.include_metadata << "_"
       << params.filter_owner << "_"
       << params.filter_category << "_"
       << params.filter_min_score << "_"
       << params.filter_max_score;
    
    for (const auto& tag : params.filter_tags) {
        ss << "_" << tag;
    }
    
    return ss.str();
}

bool QueryOptimizer::should_use_filter_pushdown(const SearchParams& params,
                                               double selectivity) const {
    // Use filter pushdown if:
    // 1. Filters are present
    // 2. Selectivity is low (filters will reduce the dataset significantly)
    // 3. Cost of filtering is less than cost of computing similarities
    
    bool has_filters = !params.filter_tags.empty() || 
                      !params.filter_owner.empty() || 
                      !params.filter_category.empty() ||
                      params.filter_min_score > 0.0f ||
                      params.filter_max_score < 1.0f ||
                      params.threshold > 0.0f;
    
    // Use pushdown if filters eliminate >70% of vectors
    return has_filters && selectivity < 0.3;
}

void QueryOptimizer::evict_old_plans(const std::string& database_id) {
    auto& db_cache = plan_cache_[database_id];
    
    if (db_cache.size() > MAX_CACHE_SIZE) {
        // Simple LRU: remove 10% of oldest entries
        size_t to_remove = MAX_CACHE_SIZE / 10;
        
        // In a real implementation, we'd track access times
        // For now, just remove first entries
        auto it = db_cache.begin();
        for (size_t i = 0; i < to_remove && it != db_cache.end(); ++i) {
            it = db_cache.erase(it);
        }
    }
}

} // namespace jadevectordb
