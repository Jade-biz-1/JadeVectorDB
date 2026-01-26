#pragma once

#include "subprocess_manager.h"
#include "score_fusion.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <type_traits>

namespace jadedb {
namespace search {

/**
 * @brief Adaptive reranking metric type
 */
enum class AdaptiveMetric {
    VARIANCE,       // Score variance (low variance = high confidence)
    SPREAD,         // Max - Min score spread
    TOP_SCORE_GAP,  // Gap between top-1 and top-2 scores
    ENTROPY         // Score distribution entropy
};

/**
 * @brief Configuration for reranking
 */
struct RerankingConfig {
    std::string model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    int batch_size = 32;
    double score_threshold = 0.0;  // Minimum rerank score to keep
    bool combine_scores = true;    // Combine with original scores
    double rerank_weight = 0.7;    // Weight for rerank score (0-1)

    // Adaptive reranking configuration
    bool enable_adaptive = false;                           // Enable adaptive mode
    AdaptiveMetric adaptive_metric = AdaptiveMetric::VARIANCE;  // Metric to use
    double adaptive_threshold = 0.1;                        // Confidence threshold
    // If metric value < threshold, scores are confident (don't rerank)
    // If metric value >= threshold, scores are uncertain (do rerank)

    RerankingConfig() = default;
};

/**
 * @brief Result from reranking
 */
struct RerankingResult {
    std::string doc_id;
    double rerank_score;       // Cross-encoder score
    double original_score;     // Original similarity/hybrid score
    double combined_score;     // Weighted combination
    std::unordered_map<std::string, std::string> metadata;

    RerankingResult()
        : rerank_score(0.0), original_score(0.0), combined_score(0.0) {}

    RerankingResult(const std::string& id, double rerank, double original, double combined)
        : doc_id(id), rerank_score(rerank), original_score(original), combined_score(combined) {}

    bool operator<(const RerankingResult& other) const {
        return combined_score > other.combined_score;  // Descending
    }
};

/**
 * @brief Statistics for reranking service
 */
struct RerankingStatistics {
    size_t total_requests = 0;
    size_t failed_requests = 0;
    double avg_latency_ms = 0.0;
    size_t total_documents_reranked = 0;

    RerankingStatistics() = default;
};

/**
 * @brief Service for reranking search results
 *
 * Provides high-level API for reranking using cross-encoder models.
 * Manages subprocess lifecycle and handles score fusion.
 */
class RerankingService {
public:
    /**
     * @brief Construct reranking service
     * @param database_id Database identifier
     * @param config Reranking configuration
     */
    RerankingService(
        const std::string& database_id,
        const RerankingConfig& config = RerankingConfig()
    );

    ~RerankingService();

    // Non-copyable (subprocess resource)
    RerankingService(const RerankingService&) = delete;
    RerankingService& operator=(const RerankingService&) = delete;

    /**
     * @brief Initialize service (starts subprocess)
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> initialize();

    /**
     * @brief Shutdown service (stops subprocess)
     */
    void shutdown();

    /**
     * @brief Rerank search results
     *
     * @param query Query text
     * @param results Vector of search results with doc_id and score
     * @param document_texts Map of doc_id to document text
     * @return Result<std::vector<RerankingResult>> Reranked results
     */
    jadevectordb::Result<std::vector<RerankingResult>> rerank(
        const std::string& query,
        const std::vector<SearchResult>& results,
        const std::unordered_map<std::string, std::string>& document_texts
    );

    /**
     * @brief Rerank with document texts directly
     *
     * @param query Query text
     * @param doc_ids Document IDs
     * @param documents Document texts
     * @param original_scores Original scores (optional, empty means no scores)
     * @return Result<std::vector<RerankingResult>> Reranked results
     */
    jadevectordb::Result<std::vector<RerankingResult>> rerank_batch(
        const std::string& query,
        const std::vector<std::string>& doc_ids,
        const std::vector<std::string>& documents,
        const std::vector<double>& original_scores = {}
    );

    /**
     * @brief Check if service is ready
     */
    bool is_ready() const;

    /**
     * @brief Check if subprocess is alive
     */
    bool is_subprocess_alive() const;

    /**
     * @brief Get current configuration
     */
    const RerankingConfig& get_config() const;

    /**
     * @brief Update configuration
     * Note: Requires reinitialization to take effect
     */
    void set_config(const RerankingConfig& config);

    /**
     * @brief Get service statistics
     */
    RerankingStatistics get_statistics() const;

    /**
     * @brief Reset statistics
     */
    void reset_statistics();

    /**
     * @brief Check if reranking should be applied based on adaptive logic
     *
     * Analyzes the confidence of search results to determine if reranking
     * would provide meaningful improvement.
     *
     * @param results Search results to analyze
     * @return true if reranking should be applied, false otherwise
     */
    template<typename T>
    bool should_apply_reranking(const std::vector<T>& results) const;

private:
    std::string database_id_;
    RerankingConfig config_;

    // Subprocess manager
    std::unique_ptr<SubprocessManager> subprocess_;

    // Thread safety
    mutable std::mutex mutex_;
    mutable std::mutex stats_mutex_;

    // Statistics
    RerankingStatistics stats_;
    double total_latency_ms_;

    // Logger
    std::shared_ptr<jadevectordb::logging::Logger> logger_;

    /**
     * @brief Combine rerank scores with original scores
     */
    double combine_score(double rerank_score, double original_score) const;

    /**
     * @brief Update statistics after request
     */
    void update_statistics(bool success, double latency_ms, size_t num_documents);

    /**
     * @brief Calculate score variance
     */
    template<typename T>
    double calculate_score_variance(const std::vector<T>& results) const;

    /**
     * @brief Calculate score spread (max - min)
     */
    template<typename T>
    double calculate_score_spread(const std::vector<T>& results) const;

    /**
     * @brief Calculate top score gap (top-1 minus top-2)
     */
    template<typename T>
    double calculate_top_score_gap(const std::vector<T>& results) const;

    /**
     * @brief Calculate score entropy
     */
    template<typename T>
    double calculate_score_entropy(const std::vector<T>& results) const;
};

} // namespace search
} // namespace jadedb
