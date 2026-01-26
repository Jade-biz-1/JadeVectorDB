#include "query_analytics_manager.h"
#include <algorithm>
#include <numeric>

namespace jadedb {
namespace analytics {

QueryAnalyticsManager::QueryAnalyticsManager(
    const std::string& database_id,
    const std::string& analytics_db_path
)
    : database_id_(database_id),
      log_(jadevectordb::logging::LoggerManager::get_logger("QueryAnalyticsManager"))
{
    QueryLoggerConfig config;
    config.database_path = analytics_db_path;
    config.enable_async = true;
    config.batch_size = 100;
    config.flush_interval_ms = 5000;
    config.max_queue_size = 10000;

    logger_ = std::make_unique<QueryLogger>(database_id, config);
}

QueryAnalyticsManager::~QueryAnalyticsManager() {
    shutdown();
}

jadevectordb::Result<void> QueryAnalyticsManager::initialize() {
    if (!logger_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Logger not created"
        ));
    }

    return logger_->initialize();
}

void QueryAnalyticsManager::shutdown() {
    if (logger_) {
        logger_->shutdown();
    }
}

bool QueryAnalyticsManager::is_ready() const {
    return logger_ && logger_->is_ready();
}

jadevectordb::Result<std::string> QueryAnalyticsManager::log_vector_search(
    const std::vector<float>& query_vector,
    const std::vector<jadevectordb::SearchResult>& results,
    int64_t retrieval_time_ms,
    int64_t total_time_ms,
    int top_k,
    const std::string& metric,
    const std::string& user_id,
    const std::string& session_id,
    const std::string& client_ip
) {
    if (!is_ready()) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics manager not initialized"
        ));
    }

    auto entry = create_entry(
        "vector[" + std::to_string(query_vector.size()) + "d]",
        "vector"
    );

    entry.retrieval_time_ms = retrieval_time_ms;
    entry.total_time_ms = total_time_ms;
    entry.num_results = static_cast<int>(results.size());
    entry.top_k = top_k;
    entry.vector_metric = metric;
    entry.user_id = user_id;
    entry.session_id = session_id;
    entry.client_ip = client_ip;

    // Calculate score statistics
    if (!results.empty()) {
        std::vector<double> scores;
        scores.reserve(results.size());
        for (const auto& result : results) {
            scores.push_back(static_cast<double>(result.similarity_score));
        }

        entry.avg_similarity_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        entry.min_similarity_score = *std::min_element(scores.begin(), scores.end());
        entry.max_similarity_score = *std::max_element(scores.begin(), scores.end());
    }

    std::string query_id = entry.query_id;
    auto result = logger_->log_query(entry);

    if (!result.has_value()) {
        return tl::make_unexpected(result.error());
    }

    return query_id;
}

jadevectordb::Result<std::string> QueryAnalyticsManager::log_hybrid_search(
    const std::string& query_text,
    const std::vector<float>& query_vector,
    const std::vector<jadedb::search::SearchResult>& results,
    int64_t retrieval_time_ms,
    int64_t total_time_ms,
    int top_k,
    double alpha,
    const std::string& fusion_method,
    const std::string& user_id,
    const std::string& session_id,
    const std::string& client_ip
) {
    if (!is_ready()) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics manager not initialized"
        ));
    }

    auto entry = create_entry(query_text, "hybrid");

    entry.retrieval_time_ms = retrieval_time_ms;
    entry.total_time_ms = total_time_ms;
    entry.num_results = static_cast<int>(results.size());
    entry.top_k = top_k;
    entry.hybrid_alpha = alpha;
    entry.fusion_method = fusion_method;
    entry.user_id = user_id;
    entry.session_id = session_id;
    entry.client_ip = client_ip;

    // Calculate score statistics
    if (!results.empty()) {
        std::vector<double> scores;
        scores.reserve(results.size());
        for (const auto& result : results) {
            scores.push_back(result.score);
        }

        entry.avg_similarity_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        entry.min_similarity_score = *std::min_element(scores.begin(), scores.end());
        entry.max_similarity_score = *std::max_element(scores.begin(), scores.end());
    }

    std::string query_id = entry.query_id;
    auto result = logger_->log_query(entry);

    if (!result.has_value()) {
        return tl::make_unexpected(result.error());
    }

    return query_id;
}

jadevectordb::Result<std::string> QueryAnalyticsManager::log_reranking(
    const std::string& query_text,
    const std::vector<jadedb::search::SearchResult>& initial_results,
    const std::vector<jadedb::search::RerankingResult>& reranked_results,
    int64_t retrieval_time_ms,
    int64_t reranking_time_ms,
    int64_t total_time_ms,
    const std::string& model_name,
    const std::string& user_id,
    const std::string& session_id,
    const std::string& client_ip
) {
    if (!is_ready()) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics manager not initialized"
        ));
    }

    auto entry = create_entry(query_text, "rerank");

    entry.retrieval_time_ms = retrieval_time_ms;
    entry.reranking_time_ms = reranking_time_ms;
    entry.total_time_ms = total_time_ms;
    entry.num_results = static_cast<int>(reranked_results.size());
    entry.used_reranking = true;
    entry.reranking_model = model_name;
    entry.user_id = user_id;
    entry.session_id = session_id;
    entry.client_ip = client_ip;

    // Calculate score statistics from reranked results
    if (!reranked_results.empty()) {
        std::vector<double> scores;
        scores.reserve(reranked_results.size());
        for (const auto& result : reranked_results) {
            scores.push_back(result.combined_score);
        }

        entry.avg_similarity_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        entry.min_similarity_score = *std::min_element(scores.begin(), scores.end());
        entry.max_similarity_score = *std::max_element(scores.begin(), scores.end());
    }

    std::string query_id = entry.query_id;
    auto result = logger_->log_query(entry);

    if (!result.has_value()) {
        return tl::make_unexpected(result.error());
    }

    return query_id;
}

jadevectordb::Result<std::string> QueryAnalyticsManager::log_error(
    const std::string& query_text,
    const std::string& error_message,
    const std::string& query_type,
    const std::string& user_id,
    const std::string& session_id,
    const std::string& client_ip
) {
    if (!is_ready()) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics manager not initialized"
        ));
    }

    auto entry = create_entry(query_text, query_type);

    entry.has_error = true;
    entry.error_message = error_message;
    entry.user_id = user_id;
    entry.session_id = session_id;
    entry.client_ip = client_ip;

    std::string query_id = entry.query_id;
    auto result = logger_->log_query(entry);

    if (!result.has_value()) {
        return tl::make_unexpected(result.error());
    }

    return query_id;
}

QueryAnalyticsManager::Statistics QueryAnalyticsManager::get_statistics() const {
    Statistics stats;
    if (logger_) {
        stats.total_logged = logger_->get_total_logged();
        stats.total_dropped = logger_->get_total_dropped();
        stats.queue_size = logger_->get_queue_size();
    }
    return stats;
}

jadevectordb::Result<void> QueryAnalyticsManager::flush() {
    if (!logger_) {
        return jadevectordb::Result<void>{};
    }
    return logger_->flush();
}

QueryLogEntry QueryAnalyticsManager::create_entry(
    const std::string& query_text,
    const std::string& query_type
) const {
    QueryLogEntry entry;
    entry.query_id = QueryLogger::generate_query_id();
    entry.database_id = database_id_;
    entry.query_text = query_text;
    entry.query_type = query_type;
    entry.timestamp = QueryLogger::get_current_timestamp_ms();
    return entry;
}

} // namespace analytics
} // namespace jadedb
