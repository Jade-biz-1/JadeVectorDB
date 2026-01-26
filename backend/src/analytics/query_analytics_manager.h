#pragma once

#include "query_logger.h"
#include "services/similarity_search.h"
#include "services/search/hybrid_search_engine.h"
#include "services/search/reranking_service.h"
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace jadedb {
namespace analytics {

/**
 * @brief Manager for query analytics integration
 *
 * Provides high-level API for logging queries from various search services.
 * Handles timing, metadata collection, and error tracking automatically.
 */
class QueryAnalyticsManager {
public:
    /**
     * @brief Construct analytics manager
     * @param database_id Database identifier
     * @param analytics_db_path Path to analytics SQLite database
     */
    QueryAnalyticsManager(
        const std::string& database_id,
        const std::string& analytics_db_path
    );

    ~QueryAnalyticsManager();

    /**
     * @brief Initialize analytics manager
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> initialize();

    /**
     * @brief Shutdown analytics manager
     */
    void shutdown();

    /**
     * @brief Check if manager is ready
     */
    bool is_ready() const;

    /**
     * @brief Log a vector similarity search query
     *
     * @param query_vector Query vector
     * @param results Search results
     * @param retrieval_time_ms Time spent in retrieval
     * @param total_time_ms Total time including processing
     * @param top_k Number of results requested
     * @param metric Distance metric used
     * @param user_id Optional user identifier
     * @param session_id Optional session identifier
     * @param client_ip Optional client IP
     * @return Result<std::string> Query ID or error
     */
    jadevectordb::Result<std::string> log_vector_search(
        const std::vector<float>& query_vector,
        const std::vector<jadevectordb::SearchResult>& results,
        int64_t retrieval_time_ms,
        int64_t total_time_ms,
        int top_k,
        const std::string& metric = "cosine",
        const std::string& user_id = "",
        const std::string& session_id = "",
        const std::string& client_ip = ""
    );

    /**
     * @brief Log a hybrid search query
     *
     * @param query_text Query text for BM25
     * @param query_vector Query vector for similarity
     * @param results Search results
     * @param retrieval_time_ms Time spent in retrieval
     * @param total_time_ms Total time including processing
     * @param top_k Number of results requested
     * @param alpha Weight for vector vs BM25 (0-1)
     * @param fusion_method Fusion method used
     * @param user_id Optional user identifier
     * @param session_id Optional session identifier
     * @param client_ip Optional client IP
     * @return Result<std::string> Query ID or error
     */
    jadevectordb::Result<std::string> log_hybrid_search(
        const std::string& query_text,
        const std::vector<float>& query_vector,
        const std::vector<jadedb::search::SearchResult>& results,
        int64_t retrieval_time_ms,
        int64_t total_time_ms,
        int top_k,
        double alpha,
        const std::string& fusion_method,
        const std::string& user_id = "",
        const std::string& session_id = "",
        const std::string& client_ip = ""
    );

    /**
     * @brief Log a re-ranking operation
     *
     * @param query_text Original query text
     * @param initial_results Initial search results
     * @param reranked_results Re-ranked results
     * @param retrieval_time_ms Time for initial retrieval
     * @param reranking_time_ms Time for re-ranking
     * @param total_time_ms Total time
     * @param model_name Re-ranking model used
     * @param user_id Optional user identifier
     * @param session_id Optional session identifier
     * @param client_ip Optional client IP
     * @return Result<std::string> Query ID or error
     */
    jadevectordb::Result<std::string> log_reranking(
        const std::string& query_text,
        const std::vector<jadedb::search::SearchResult>& initial_results,
        const std::vector<jadedb::search::RerankingResult>& reranked_results,
        int64_t retrieval_time_ms,
        int64_t reranking_time_ms,
        int64_t total_time_ms,
        const std::string& model_name,
        const std::string& user_id = "",
        const std::string& session_id = "",
        const std::string& client_ip = ""
    );

    /**
     * @brief Log a search error
     *
     * @param query_text Query that caused the error
     * @param error_message Error description
     * @param query_type Type of query that failed
     * @param user_id Optional user identifier
     * @param session_id Optional session identifier
     * @param client_ip Optional client IP
     * @return Result<std::string> Query ID or error
     */
    jadevectordb::Result<std::string> log_error(
        const std::string& query_text,
        const std::string& error_message,
        const std::string& query_type,
        const std::string& user_id = "",
        const std::string& session_id = "",
        const std::string& client_ip = ""
    );

    /**
     * @brief Get query logger statistics
     */
    struct Statistics {
        size_t total_logged;
        size_t total_dropped;
        size_t queue_size;
    };

    Statistics get_statistics() const;

    /**
     * @brief Flush pending logs
     */
    jadevectordb::Result<void> flush();

private:
    std::string database_id_;
    std::unique_ptr<QueryLogger> logger_;
    std::shared_ptr<jadevectordb::logging::Logger> log_;

    /**
     * @brief Calculate score statistics from results
     */
    template<typename T>
    void calculate_score_stats(
        const std::vector<T>& results,
        double& avg_score,
        double& min_score,
        double& max_score
    ) const;

    /**
     * @brief Create a query log entry
     */
    QueryLogEntry create_entry(
        const std::string& query_text,
        const std::string& query_type
    ) const;
};

} // namespace analytics
} // namespace jadedb
