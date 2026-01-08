#include "score_fusion.h"
#include <limits>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace jadedb {
namespace search {

ScoreFusion::ScoreFusion(const FusionConfig& config)
    : config_(config) {
}

std::unordered_map<std::string, size_t> ScoreFusion::build_rank_map(
    const std::vector<SearchResult>& results) const {

    std::unordered_map<std::string, size_t> rank_map;

    for (size_t i = 0; i < results.size(); i++) {
        rank_map[results[i].doc_id] = i + 1;  // Ranks start at 1
    }

    return rank_map;
}

std::vector<SearchResult> ScoreFusion::reciprocal_rank_fusion(
    const std::vector<std::vector<SearchResult>>& results_list,
    int k) const {

    // Build rank maps for each result list
    std::vector<std::unordered_map<std::string, size_t>> rank_maps;
    for (const auto& results : results_list) {
        rank_maps.push_back(build_rank_map(results));
    }

    // Collect all unique document IDs
    std::unordered_map<std::string, double> rrf_scores;

    for (const auto& rank_map : rank_maps) {
        for (const auto& [doc_id, rank] : rank_map) {
            // RRF formula: 1 / (k + rank)
            double rrf_contribution = 1.0 / (k + static_cast<double>(rank));
            rrf_scores[doc_id] += rrf_contribution;
        }
    }

    // Convert to SearchResult vector
    std::vector<SearchResult> fused_results;
    fused_results.reserve(rrf_scores.size());

    for (const auto& [doc_id, score] : rrf_scores) {
        fused_results.emplace_back(doc_id, score);
    }

    // Sort by RRF score descending
    std::sort(fused_results.begin(), fused_results.end());

    return fused_results;
}

double ScoreFusion::calculate_mean(const std::vector<SearchResult>& results) const {
    if (results.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (const auto& result : results) {
        sum += result.score;
    }

    return sum / results.size();
}

double ScoreFusion::calculate_stddev(
    const std::vector<SearchResult>& results,
    double mean) const {

    if (results.size() <= 1) {
        return 1.0;  // Avoid division by zero
    }

    double sum_squared_diff = 0.0;
    for (const auto& result : results) {
        double diff = result.score - mean;
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(sum_squared_diff / results.size());
}

void ScoreFusion::normalize_min_max(std::vector<SearchResult>& results) const {
    if (results.empty()) {
        return;
    }

    // Find min and max scores
    double min_score = std::numeric_limits<double>::max();
    double max_score = std::numeric_limits<double>::lowest();

    for (const auto& result : results) {
        min_score = std::min(min_score, result.score);
        max_score = std::max(max_score, result.score);
    }

    // Normalize to [0.01, 1] to avoid zero scores
    // This ensures all documents have some positive contribution
    double range = max_score - min_score;

    if (range < 1e-9) {
        // All scores are the same, set to 0.5 (middle value)
        for (auto& result : results) {
            result.score = 0.5;
        }
    } else {
        for (auto& result : results) {
            // Normalize to [0.01, 1] instead of [0, 1]
            result.score = 0.01 + 0.99 * (result.score - min_score) / range;
        }
    }
}

void ScoreFusion::normalize_z_score(std::vector<SearchResult>& results) const {
    if (results.empty()) {
        return;
    }

    // Calculate mean and standard deviation
    double mean = calculate_mean(results);
    double stddev = calculate_stddev(results, mean);

    if (stddev < 1e-9) {
        // All scores are the same, set to 0.0
        for (auto& result : results) {
            result.score = 0.0;
        }
    } else {
        // Apply z-score normalization
        for (auto& result : results) {
            result.score = (result.score - mean) / stddev;
        }
    }
}

std::vector<SearchResult> ScoreFusion::weighted_linear_fusion(
    const std::vector<SearchResult>& results1,
    const std::vector<SearchResult>& results2,
    double alpha,
    NormalizationMethod norm_method) const {

    // Validate alpha
    if (alpha < 0.0 || alpha > 1.0) {
        alpha = 0.7;  // Default
    }

    // Create copies for normalization
    std::vector<SearchResult> normalized1 = results1;
    std::vector<SearchResult> normalized2 = results2;

    // Apply normalization
    if (norm_method == NormalizationMethod::MIN_MAX) {
        normalize_min_max(normalized1);
        normalize_min_max(normalized2);
    } else if (norm_method == NormalizationMethod::Z_SCORE) {
        normalize_z_score(normalized1);
        normalize_z_score(normalized2);
    }

    // Build score maps
    std::unordered_map<std::string, double> score_map1;
    std::unordered_map<std::string, double> score_map2;

    for (const auto& result : normalized1) {
        score_map1[result.doc_id] = result.score;
    }

    for (const auto& result : normalized2) {
        score_map2[result.doc_id] = result.score;
    }

    // Combine scores
    std::unordered_map<std::string, double> fused_scores;

    // Add documents from first source
    for (const auto& [doc_id, score1] : score_map1) {
        double score2 = 0.0;
        auto it = score_map2.find(doc_id);
        if (it != score_map2.end()) {
            score2 = it->second;
        }

        // Weighted linear combination
        fused_scores[doc_id] = alpha * score1 + (1.0 - alpha) * score2;
    }

    // Add documents that only appear in second source
    for (const auto& [doc_id, score2] : score_map2) {
        if (score_map1.find(doc_id) == score_map1.end()) {
            fused_scores[doc_id] = (1.0 - alpha) * score2;
        }
    }

    // Convert to SearchResult vector
    std::vector<SearchResult> fused_results;
    fused_results.reserve(fused_scores.size());

    for (const auto& [doc_id, score] : fused_scores) {
        fused_results.emplace_back(doc_id, score);
    }

    // Sort by fused score descending
    std::sort(fused_results.begin(), fused_results.end());

    return fused_results;
}

std::vector<SearchResult> ScoreFusion::merge_results(
    const std::vector<std::vector<SearchResult>>& results_list) const {

    // Map to keep highest score for each document
    std::unordered_map<std::string, double> max_scores;

    for (const auto& results : results_list) {
        for (const auto& result : results) {
            auto it = max_scores.find(result.doc_id);
            if (it == max_scores.end()) {
                max_scores[result.doc_id] = result.score;
            } else {
                max_scores[result.doc_id] = std::max(it->second, result.score);
            }
        }
    }

    // Convert to SearchResult vector
    std::vector<SearchResult> merged;
    merged.reserve(max_scores.size());

    for (const auto& [doc_id, score] : max_scores) {
        merged.emplace_back(doc_id, score);
    }

    return merged;
}

std::vector<SearchResult> ScoreFusion::get_top_k(
    const std::vector<SearchResult>& results,
    size_t k) const {

    if (results.size() <= k) {
        return results;
    }

    // Return first k elements (assuming results are already sorted)
    return std::vector<SearchResult>(results.begin(), results.begin() + k);
}

} // namespace search
} // namespace jadedb
