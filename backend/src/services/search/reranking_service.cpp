#include "reranking_service.h"
#include "hybrid_search_engine.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstdlib>

namespace jadedb {
namespace search {

RerankingService::RerankingService(
    const std::string& database_id,
    const RerankingConfig& config
)
    : database_id_(database_id),
      config_(config),
      subprocess_(nullptr),
      total_latency_ms_(0.0),
      logger_(jadevectordb::logging::LoggerManager::get_logger("RerankingService")) {

    LOG_INFO(logger_, "Creating reranking service for database: " << database_id_);
}

RerankingService::~RerankingService() {
    shutdown();
}

jadevectordb::Result<void> RerankingService::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (subprocess_ && subprocess_->is_ready()) {
        LOG_DEBUG(logger_, "Subprocess already initialized");
        return jadevectordb::Result<void>{};
    }

    LOG_INFO(logger_, "Initializing reranking service");

    // Create subprocess configuration
    SubprocessConfig subprocess_config;
    subprocess_config.model_name = config_.model_name;
    subprocess_config.batch_size = config_.batch_size;

    // Allow overriding script path via environment variable (used by tests)
    const char* script_path_env = std::getenv("RERANKING_SCRIPT_PATH");
    if (script_path_env) {
        subprocess_config.script_path = script_path_env;
    }

    // Create and start subprocess
    subprocess_ = std::make_unique<SubprocessManager>(subprocess_config);

    auto start_result = subprocess_->start();
    if (!start_result.has_value()) {
        LOG_ERROR(logger_, "Failed to start subprocess: " << start_result.error().message);
        subprocess_.reset();
        return start_result;
    }

    LOG_INFO(logger_, "Reranking service initialized successfully");
    return jadevectordb::Result<void>{};
}

void RerankingService::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (subprocess_) {
        LOG_INFO(logger_, "Shutting down reranking service");
        subprocess_->stop();
        subprocess_.reset();
    }
}

jadevectordb::Result<std::vector<RerankingResult>> RerankingService::rerank(
    const std::string& query,
    const std::vector<SearchResult>& results,
    const std::unordered_map<std::string, std::string>& document_texts
) {
    auto start_time = std::chrono::steady_clock::now();

    // Extract doc_ids, documents, and original scores
    std::vector<std::string> doc_ids;
    std::vector<std::string> documents;
    std::vector<double> original_scores;

    doc_ids.reserve(results.size());
    documents.reserve(results.size());
    original_scores.reserve(results.size());

    for (const auto& result : results) {
        doc_ids.push_back(result.doc_id);

        // Find document text
        auto it = document_texts.find(result.doc_id);
        if (it != document_texts.end()) {
            documents.push_back(it->second);
        } else {
            LOG_WARN(logger_, "No text found for doc_id: " << result.doc_id
                     << ", using doc_id as fallback");
            documents.push_back(result.doc_id);
        }

        original_scores.push_back(result.score);
    }

    // Call rerank_batch
    auto rerank_result = rerank_batch(query, doc_ids, documents, original_scores);

    auto end_time = std::chrono::steady_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time
    ).count();
    double latency_ms = static_cast<double>(latency_us) / 1000.0;

    update_statistics(rerank_result.has_value(), latency_ms, documents.size());

    return rerank_result;
}

jadevectordb::Result<std::vector<RerankingResult>> RerankingService::rerank_batch(
    const std::string& query,
    const std::vector<std::string>& doc_ids,
    const std::vector<std::string>& documents,
    const std::vector<double>& original_scores
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!subprocess_ || !subprocess_->is_ready()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_UNAVAILABLE,
            "Reranking service not initialized"
        ));
    }

    if (doc_ids.size() != documents.size()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::INVALID_ARGUMENT,
            "doc_ids and documents size mismatch"
        ));
    }

    if (!original_scores.empty() && original_scores.size() != doc_ids.size()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::INVALID_ARGUMENT,
            "original_scores size mismatch"
        ));
    }

    // Handle empty input
    if (documents.empty()) {
        return jadevectordb::Result<std::vector<RerankingResult>>(
            std::vector<RerankingResult>{}
        );
    }

    // Handle single document
    if (documents.size() == 1) {
        std::vector<RerankingResult> results;
        RerankingResult result;
        result.doc_id = doc_ids[0];
        result.rerank_score = 1.0;  // Single doc gets perfect score
        result.original_score = original_scores.empty() ? 1.0 : original_scores[0];
        result.combined_score = config_.combine_scores ?
            combine_score(result.rerank_score, result.original_score) :
            result.rerank_score;
        results.push_back(result);
        return jadevectordb::Result<std::vector<RerankingResult>>(results);
    }

    LOG_DEBUG(logger_, "Reranking " << documents.size() << " documents for query: "
              << query.substr(0, 50));

    auto start_time = std::chrono::steady_clock::now();

    // Prepare request
    nlohmann::json request;
    request["query"] = query;
    request["documents"] = documents;

    // Send request
    auto response = subprocess_->send_request(request);
    if (!response.has_value()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::SERVICE_ERROR,
            "Reranking request failed: " + response.error().message
        ));
    }

    auto json_response = response.value();

    // Parse scores
    if (!json_response.contains("scores")) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::DESERIALIZATION_ERROR,
            "Response missing 'scores' field"
        ));
    }

    auto scores_json = json_response["scores"];
    if (!scores_json.is_array()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::DESERIALIZATION_ERROR,
            "'scores' field is not an array"
        ));
    }

    if (scores_json.size() != documents.size()) {
        return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(
            jadevectordb::ErrorCode::DESERIALIZATION_ERROR,
            "Score count mismatch: expected " + std::to_string(documents.size()) +
            ", got " + std::to_string(scores_json.size())
        ));
    }

    // Build results
    std::vector<RerankingResult> results;
    results.reserve(documents.size());

    for (size_t i = 0; i < documents.size(); i++) {
        RerankingResult result;
        result.doc_id = doc_ids[i];
        result.rerank_score = scores_json[i].get<double>();
        result.original_score = original_scores.empty() ? 0.0 : original_scores[i];

        // Combine scores if enabled
        if (config_.combine_scores && !original_scores.empty()) {
            result.combined_score = combine_score(result.rerank_score, result.original_score);
        } else {
            result.combined_score = result.rerank_score;
        }

        // Apply score threshold
        if (result.combined_score >= config_.score_threshold) {
            results.push_back(result);
        }
    }

    // Sort by combined score (descending)
    std::sort(results.begin(), results.end());

    auto end_time = std::chrono::steady_clock::now();
    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    LOG_DEBUG(logger_, "Reranking completed in " << latency_ms << "ms, "
              << "returned " << results.size() << " results (threshold filtered: "
              << (documents.size() - results.size()) << ")");

    return jadevectordb::Result<std::vector<RerankingResult>>(results);
}

bool RerankingService::is_ready() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return subprocess_ && subprocess_->is_ready();
}

bool RerankingService::is_subprocess_alive() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return subprocess_ && subprocess_->is_ready();
}

const RerankingConfig& RerankingService::get_config() const {
    return config_;
}

void RerankingService::set_config(const RerankingConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    LOG_INFO(logger_, "Configuration updated (requires reinitialization)");
}

RerankingStatistics RerankingService::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void RerankingService::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = RerankingStatistics();
    total_latency_ms_ = 0.0;
    LOG_DEBUG(logger_, "Statistics reset");
}

double RerankingService::combine_score(double rerank_score, double original_score) const {
    // Weighted combination: combined = rerank_weight * rerank + (1 - rerank_weight) * original
    return config_.rerank_weight * rerank_score +
           (1.0 - config_.rerank_weight) * original_score;
}

void RerankingService::update_statistics(
    bool success,
    double latency_ms,
    size_t num_documents
) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_requests++;
    if (!success) {
        stats_.failed_requests++;
    }

    total_latency_ms_ += latency_ms;
    stats_.avg_latency_ms = total_latency_ms_ / stats_.total_requests;
    stats_.total_documents_reranked += num_documents;
}

// Template implementations for adaptive reranking

template<typename T>
bool RerankingService::should_apply_reranking(const std::vector<T>& results) const {
    // If adaptive mode is disabled, always apply reranking
    if (!config_.enable_adaptive) {
        return true;
    }

    // Not enough results to analyze
    if (results.size() < 2) {
        return false;  // No point reranking single result
    }

    // Calculate the selected metric
    double metric_value = 0.0;
    switch (config_.adaptive_metric) {
        case AdaptiveMetric::VARIANCE:
            metric_value = calculate_score_variance(results);
            break;
        case AdaptiveMetric::SPREAD:
            metric_value = calculate_score_spread(results);
            break;
        case AdaptiveMetric::TOP_SCORE_GAP:
            metric_value = calculate_top_score_gap(results);
            break;
        case AdaptiveMetric::ENTROPY:
            metric_value = calculate_score_entropy(results);
            break;
    }

    // Decision logic:
    // - Low metric value = high confidence (scores are clear) -> skip reranking
    // - High metric value = low confidence (scores are uncertain) -> apply reranking
    bool should_rerank = metric_value >= config_.adaptive_threshold;

    LOG_DEBUG(logger_, "Adaptive reranking decision: metric=" << metric_value
              << ", threshold=" << config_.adaptive_threshold
              << ", rerank=" << (should_rerank ? "YES" : "NO"));

    return should_rerank;
}

template<typename T>
double RerankingService::calculate_score_variance(const std::vector<T>& results) const {
    if (results.empty()) return 0.0;

    // Extract scores
    std::vector<double> scores;
    scores.reserve(results.size());
    for (const auto& result : results) {
        if constexpr (std::is_same_v<T, SearchResult>) {
            scores.push_back(result.score);
        } else if constexpr (std::is_same_v<T, HybridSearchResult>) {
            scores.push_back(result.hybrid_score);
        } else {
            scores.push_back(result.similarity_score);
        }
    }

    // Calculate mean
    double mean = 0.0;
    for (double score : scores) {
        mean += score;
    }
    mean /= scores.size();

    // Calculate variance
    double variance = 0.0;
    for (double score : scores) {
        double diff = score - mean;
        variance += diff * diff;
    }
    variance /= scores.size();

    return variance;
}

template<typename T>
double RerankingService::calculate_score_spread(const std::vector<T>& results) const {
    if (results.empty()) return 0.0;

    // Find min and max scores
    double min_score = std::numeric_limits<double>::max();
    double max_score = std::numeric_limits<double>::lowest();

    for (const auto& result : results) {
        double score;
        if constexpr (std::is_same_v<T, SearchResult>) {
            score = result.score;
        } else if constexpr (std::is_same_v<T, HybridSearchResult>) {
            score = result.hybrid_score;
        } else {
            score = result.similarity_score;
        }

        min_score = std::min(min_score, score);
        max_score = std::max(max_score, score);
    }

    return max_score - min_score;
}

template<typename T>
double RerankingService::calculate_top_score_gap(const std::vector<T>& results) const {
    if (results.size() < 2) return 0.0;

    // Get top 2 scores
    auto get_score = [](const T& result) -> double {
        if constexpr (std::is_same_v<T, SearchResult>) {
            return result.score;
        } else if constexpr (std::is_same_v<T, HybridSearchResult>) {
            return result.hybrid_score;
        } else {
            return result.similarity_score;
        }
    };

    double top1 = get_score(results[0]);
    double top2 = get_score(results[1]);

    // Results should already be sorted, but check anyway
    for (size_t i = 2; i < results.size(); i++) {
        double score = get_score(results[i]);
        if (score > top1) {
            top2 = top1;
            top1 = score;
        } else if (score > top2) {
            top2 = score;
        }
    }

    return top1 - top2;
}

template<typename T>
double RerankingService::calculate_score_entropy(const std::vector<T>& results) const {
    if (results.empty()) return 0.0;

    // Normalize scores to probabilities
    std::vector<double> scores;
    scores.reserve(results.size());
    double total = 0.0;

    for (const auto& result : results) {
        double score;
        if constexpr (std::is_same_v<T, SearchResult>) {
            score = result.score;
        } else if constexpr (std::is_same_v<T, HybridSearchResult>) {
            score = result.hybrid_score;
        } else {
            score = result.similarity_score;
        }
        scores.push_back(score);
        total += score;
    }

    // Calculate entropy: H = -Σ p(x) * log(p(x))
    double entropy = 0.0;
    for (double score : scores) {
        if (total > 0.0 && score > 0.0) {
            double p = score / total;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

// Explicit template instantiations
template bool RerankingService::should_apply_reranking(const std::vector<SearchResult>&) const;
template bool RerankingService::should_apply_reranking(const std::vector<HybridSearchResult>&) const;

template double RerankingService::calculate_score_variance(const std::vector<SearchResult>&) const;
template double RerankingService::calculate_score_variance(const std::vector<HybridSearchResult>&) const;

template double RerankingService::calculate_score_spread(const std::vector<SearchResult>&) const;
template double RerankingService::calculate_score_spread(const std::vector<HybridSearchResult>&) const;

template double RerankingService::calculate_top_score_gap(const std::vector<SearchResult>&) const;
template double RerankingService::calculate_top_score_gap(const std::vector<HybridSearchResult>&) const;

template double RerankingService::calculate_score_entropy(const std::vector<SearchResult>&) const;
template double RerankingService::calculate_score_entropy(const std::vector<HybridSearchResult>&) const;

} // namespace search
} // namespace jadedb
