#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace jadedb {
namespace search {

/**
 * @brief Search result with score
 */
struct SearchResult {
    std::string doc_id;
    double score;

    SearchResult() : score(0.0) {}
    SearchResult(const std::string& id, double s) : doc_id(id), score(s) {}

    bool operator<(const SearchResult& other) const {
        return score > other.score;  // Descending order
    }
};

/**
 * @brief Configuration for score fusion
 */
struct FusionConfig {
    // For RRF
    int rrf_k = 60;  // Standard RRF constant

    // For weighted linear fusion
    double alpha = 0.7;  // Weight for vector scores (0.0 to 1.0)

    FusionConfig() = default;
    FusionConfig(int k, double a) : rrf_k(k), alpha(a) {}
};

/**
 * @brief Score normalization methods
 */
enum class NormalizationMethod {
    NONE,           // No normalization
    MIN_MAX,        // Min-max normalization to [0, 1]
    Z_SCORE         // Z-score standardization
};

/**
 * @brief Score fusion algorithms for hybrid search
 *
 * Combines results from multiple retrieval systems (e.g., vector search + BM25)
 * using different fusion strategies:
 *
 * 1. Reciprocal Rank Fusion (RRF): Rank-based fusion
 * 2. Weighted Linear Fusion: Score-based fusion with normalization
 */
class ScoreFusion {
public:
    ScoreFusion() = default;
    explicit ScoreFusion(const FusionConfig& config);

    /**
     * @brief Reciprocal Rank Fusion (RRF)
     *
     * Combines rankings from multiple sources using reciprocal ranks.
     * Formula: RRF(d) = Σ 1 / (k + rank_i(d))
     *
     * Advantages:
     * - No score normalization needed
     * - Robust to different score scales
     * - Standard in information retrieval
     *
     * @param results_list Vector of result lists from different sources
     * @param k RRF constant (default: 60)
     * @return Fused and sorted results
     */
    std::vector<SearchResult> reciprocal_rank_fusion(
        const std::vector<std::vector<SearchResult>>& results_list,
        int k = 60
    ) const;

    /**
     * @brief Weighted Linear Fusion
     *
     * Combines normalized scores from two sources.
     * Formula: hybrid_score = α × norm_score1 + (1 - α) × norm_score2
     *
     * @param results1 Results from first source (e.g., vector search)
     * @param results2 Results from second source (e.g., BM25)
     * @param alpha Weight for first source (0.0 to 1.0)
     * @param norm_method Normalization method
     * @return Fused and sorted results
     */
    std::vector<SearchResult> weighted_linear_fusion(
        const std::vector<SearchResult>& results1,
        const std::vector<SearchResult>& results2,
        double alpha = 0.7,
        NormalizationMethod norm_method = NormalizationMethod::MIN_MAX
    ) const;

    /**
     * @brief Normalize scores using min-max normalization
     *
     * Transforms scores to [0, 1] range:
     * normalized = (score - min) / (max - min)
     *
     * @param results Results to normalize (modified in-place)
     */
    void normalize_min_max(std::vector<SearchResult>& results) const;

    /**
     * @brief Normalize scores using z-score standardization
     *
     * Transforms scores to have mean=0, stddev=1:
     * z_score = (score - mean) / stddev
     *
     * @param results Results to normalize (modified in-place)
     */
    void normalize_z_score(std::vector<SearchResult>& results) const;

    /**
     * @brief Merge and deduplicate results from multiple sources
     *
     * When the same document appears in multiple result lists,
     * keep the highest score.
     *
     * @param results_list Vector of result lists
     * @return Merged results (unsorted)
     */
    std::vector<SearchResult> merge_results(
        const std::vector<std::vector<SearchResult>>& results_list
    ) const;

    /**
     * @brief Get top-K results from a list
     *
     * @param results Input results
     * @param k Number of results to return
     * @return Top-K results
     */
    std::vector<SearchResult> get_top_k(
        const std::vector<SearchResult>& results,
        size_t k
    ) const;

    /**
     * @brief Set fusion configuration
     */
    void set_config(const FusionConfig& config) {
        config_ = config;
    }

    /**
     * @brief Get fusion configuration
     */
    const FusionConfig& get_config() const {
        return config_;
    }

private:
    FusionConfig config_;

    /**
     * @brief Calculate mean of scores
     */
    double calculate_mean(const std::vector<SearchResult>& results) const;

    /**
     * @brief Calculate standard deviation of scores
     */
    double calculate_stddev(const std::vector<SearchResult>& results, double mean) const;

    /**
     * @brief Build rank map (doc_id -> rank) from results
     */
    std::unordered_map<std::string, size_t> build_rank_map(
        const std::vector<SearchResult>& results
    ) const;
};

} // namespace search
} // namespace jadedb
