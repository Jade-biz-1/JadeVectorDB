#include "analytics_engine.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cctype>
#include <set>
#include <cmath>

namespace jadedb {
namespace analytics {

namespace {
    // Common stop words for query normalization
    const std::set<std::string> STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
    };
}

AnalyticsEngine::AnalyticsEngine(const std::string& database_path)
    : database_path_(database_path),
      db_(nullptr),
      logger_(jadevectordb::logging::LoggerManager::get_logger("AnalyticsEngine"))
{
}

AnalyticsEngine::~AnalyticsEngine() {
    shutdown();
}

jadevectordb::Result<void> AnalyticsEngine::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (db_) {
        return jadevectordb::Result<void>{};
    }

    int rc = sqlite3_open(database_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to open analytics database: " + std::string(sqlite3_errmsg(db_))
        ));
    }

    logger_->info("AnalyticsEngine initialized");
    return jadevectordb::Result<void>{};
}

void AnalyticsEngine::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        logger_->info("AnalyticsEngine shutdown");
    }
}

jadevectordb::Result<std::vector<QueryStatistics>> AnalyticsEngine::compute_statistics(
    const std::string& database_id,
    int64_t start_time,
    int64_t end_time,
    TimeBucket bucket_type
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics engine not initialized"
        ));
    }

    std::vector<QueryStatistics> results;
    std::unordered_map<int64_t, QueryStatistics> bucket_map;

    // Query all logs in the time range
    const char* sql = R"(
        SELECT
            timestamp, total_time_ms, num_results, has_error,
            avg_similarity_score, query_type, user_id, session_id
        FROM query_log
        WHERE database_id = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_))
        ));
    }

    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, start_time);
    sqlite3_bind_int64(stmt, 3, end_time);

    std::set<std::string> unique_users;
    std::set<std::string> unique_sessions;
    std::unordered_map<int64_t, std::vector<double>> latencies_per_bucket;

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int64_t timestamp = sqlite3_column_int64(stmt, 0);
        int64_t total_time_ms = sqlite3_column_int64(stmt, 1);
        int num_results = sqlite3_column_int(stmt, 2);
        int has_error = sqlite3_column_int(stmt, 3);
        double avg_score = sqlite3_column_double(stmt, 4);
        const char* query_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        const char* user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        const char* session_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));

        int64_t bucket = get_time_bucket(timestamp, bucket_type);

        if (bucket_map.find(bucket) == bucket_map.end()) {
            QueryStatistics stats;
            stats.database_id = database_id;
            stats.time_bucket = bucket;
            stats.bucket_type = bucket_type;
            bucket_map[bucket] = stats;
        }

        auto& stats = bucket_map[bucket];
        stats.total_queries++;

        if (has_error) {
            stats.failed_queries++;
        } else {
            stats.successful_queries++;
        }

        if (num_results == 0) {
            stats.zero_result_queries++;
        }

        stats.avg_results += num_results;
        stats.avg_similarity_score += avg_score;
        latencies_per_bucket[bucket].push_back(static_cast<double>(total_time_ms));

        if (total_time_ms > stats.max_latency_ms) {
            stats.max_latency_ms = total_time_ms;
        }

        std::string qtype = query_type ? query_type : "";
        if (qtype == "vector") stats.vector_queries++;
        else if (qtype == "hybrid") stats.hybrid_queries++;
        else if (qtype == "bm25") stats.bm25_queries++;
        else if (qtype == "rerank") stats.reranked_queries++;

        if (user_id) unique_users.insert(user_id);
        if (session_id) unique_sessions.insert(session_id);
    }

    sqlite3_finalize(stmt);

    // Calculate final statistics
    for (auto& [bucket, stats] : bucket_map) {
        if (stats.total_queries > 0) {
            stats.avg_results /= stats.total_queries;
            stats.avg_similarity_score /= stats.total_queries;
            stats.unique_users = unique_users.size();
            stats.unique_sessions = unique_sessions.size();

            // Calculate latency percentiles
            auto& latencies = latencies_per_bucket[bucket];
            if (!latencies.empty()) {
                std::sort(latencies.begin(), latencies.end());
                stats.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
                stats.p50_latency_ms = calculate_percentile(latencies, 0.50);
                stats.p95_latency_ms = calculate_percentile(latencies, 0.95);
                stats.p99_latency_ms = calculate_percentile(latencies, 0.99);
            }
        }

        results.push_back(stats);
    }

    // Sort by time bucket
    std::sort(results.begin(), results.end(),
        [](const QueryStatistics& a, const QueryStatistics& b) {
            return a.time_bucket < b.time_bucket;
        });

    return results;
}

jadevectordb::Result<std::vector<SearchPattern>> AnalyticsEngine::identify_patterns(
    const std::string& database_id,
    int64_t start_time,
    int64_t end_time,
    size_t min_count,
    size_t limit
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics engine not initialized"
        ));
    }

    std::unordered_map<std::string, SearchPattern> pattern_map;

    const char* sql = R"(
        SELECT query_text, total_time_ms, num_results, avg_similarity_score, timestamp
        FROM query_log
        WHERE database_id = ? AND timestamp >= ? AND timestamp <= ? AND has_error = 0
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare statement"
        ));
    }

    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, start_time);
    sqlite3_bind_int64(stmt, 3, end_time);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* query_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        int64_t total_time_ms = sqlite3_column_int64(stmt, 1);
        int num_results = sqlite3_column_int(stmt, 2);
        double avg_score = sqlite3_column_double(stmt, 3);
        int64_t timestamp = sqlite3_column_int64(stmt, 4);

        if (!query_text) continue;

        std::string normalized = normalize_query_text(query_text);
        if (normalized.empty()) continue;

        if (pattern_map.find(normalized) == pattern_map.end()) {
            SearchPattern pattern;
            pattern.pattern_text = query_text;
            pattern.normalized_text = normalized;
            pattern.first_seen = timestamp;
            pattern.last_seen = timestamp;
            pattern_map[normalized] = pattern;
        }

        auto& pattern = pattern_map[normalized];
        pattern.query_count++;
        pattern.avg_latency_ms += total_time_ms;
        pattern.avg_results += num_results;
        pattern.avg_similarity_score += avg_score;
        pattern.last_seen = std::max(pattern.last_seen, timestamp);
    }

    sqlite3_finalize(stmt);

    // Calculate averages and filter
    std::vector<SearchPattern> results;
    for (auto& [key, pattern] : pattern_map) {
        if (pattern.query_count >= min_count) {
            pattern.avg_latency_ms /= pattern.query_count;
            pattern.avg_results /= pattern.query_count;
            pattern.avg_similarity_score /= pattern.query_count;
            results.push_back(pattern);
        }
    }

    // Sort by query count descending
    std::sort(results.begin(), results.end(),
        [](const SearchPattern& a, const SearchPattern& b) {
            return a.query_count > b.query_count;
        });

    // Limit results
    if (results.size() > limit) {
        results.resize(limit);
    }

    return results;
}

jadevectordb::Result<std::vector<SlowQuery>> AnalyticsEngine::detect_slow_queries(
    const std::string& database_id,
    int64_t start_time,
    int64_t end_time,
    int64_t latency_threshold_ms,
    size_t limit
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics engine not initialized"
        ));
    }

    std::vector<SlowQuery> results;

    const char* sql = R"(
        SELECT query_id, query_text, query_type, total_time_ms, timestamp, num_results
        FROM query_log
        WHERE database_id = ? AND timestamp >= ? AND timestamp <= ?
            AND total_time_ms >= ?
        ORDER BY total_time_ms DESC
        LIMIT ?
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare statement"
        ));
    }

    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, start_time);
    sqlite3_bind_int64(stmt, 3, end_time);
    sqlite3_bind_int64(stmt, 4, latency_threshold_ms);
    sqlite3_bind_int(stmt, 5, static_cast<int>(limit));

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        SlowQuery query;
        query.query_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        query.query_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        query.query_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        query.total_time_ms = sqlite3_column_int64(stmt, 3);
        query.timestamp = sqlite3_column_int64(stmt, 4);
        query.num_results = sqlite3_column_int(stmt, 5);
        results.push_back(query);
    }

    sqlite3_finalize(stmt);
    return results;
}

jadevectordb::Result<std::vector<ZeroResultQuery>> AnalyticsEngine::analyze_zero_results(
    const std::string& database_id,
    int64_t start_time,
    int64_t end_time,
    size_t min_count,
    size_t limit
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics engine not initialized"
        ));
    }

    std::unordered_map<std::string, ZeroResultQuery> zero_map;

    const char* sql = R"(
        SELECT query_text, query_type, timestamp
        FROM query_log
        WHERE database_id = ? AND timestamp >= ? AND timestamp <= ?
            AND num_results = 0 AND has_error = 0
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare statement"
        ));
    }

    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, start_time);
    sqlite3_bind_int64(stmt, 3, end_time);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* query_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const char* query_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        int64_t timestamp = sqlite3_column_int64(stmt, 2);

        if (!query_text) continue;

        std::string normalized = normalize_query_text(query_text);
        if (normalized.empty()) continue;

        if (zero_map.find(normalized) == zero_map.end()) {
            ZeroResultQuery query;
            query.query_text = query_text;
            query.query_type = query_type ? query_type : "";
            query.occurrence_count = 0;
            query.last_seen = timestamp;
            zero_map[normalized] = query;
        }

        auto& query = zero_map[normalized];
        query.occurrence_count++;
        query.last_seen = std::max(query.last_seen, timestamp);
    }

    sqlite3_finalize(stmt);

    // Filter and convert to vector
    std::vector<ZeroResultQuery> results;
    for (const auto& [key, query] : zero_map) {
        if (query.occurrence_count >= min_count) {
            results.push_back(query);
        }
    }

    // Sort by occurrence count descending
    std::sort(results.begin(), results.end(),
        [](const ZeroResultQuery& a, const ZeroResultQuery& b) {
            return a.occurrence_count > b.occurrence_count;
        });

    // Limit results
    if (results.size() > limit) {
        results.resize(limit);
    }

    return results;
}

jadevectordb::Result<std::vector<TrendingQuery>> AnalyticsEngine::detect_trending(
    const std::string& database_id,
    int64_t current_start,
    int64_t current_end,
    TimeBucket bucket_type,
    double min_growth_rate,
    size_t limit
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Analytics engine not initialized"
        ));
    }

    // Calculate previous period
    int64_t period_duration = current_end - current_start;
    int64_t previous_start = current_start - period_duration;
    int64_t previous_end = current_start;

    // Count queries in current period
    std::unordered_map<std::string, size_t> current_counts;
    std::unordered_map<std::string, std::string> query_texts;

    const char* sql_current = R"(
        SELECT query_text FROM query_log
        WHERE database_id = ? AND timestamp >= ? AND timestamp <= ? AND has_error = 0
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql_current, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare statement"
        ));
    }

    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, current_start);
    sqlite3_bind_int64(stmt, 3, current_end);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* query_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (!query_text) continue;

        std::string normalized = normalize_query_text(query_text);
        if (normalized.empty()) continue;

        current_counts[normalized]++;
        query_texts[normalized] = query_text;
    }

    sqlite3_finalize(stmt);

    // Count queries in previous period
    std::unordered_map<std::string, size_t> previous_counts;

    const char* sql_previous = R"(
        SELECT query_text FROM query_log
        WHERE database_id = ? AND timestamp >= ? AND timestamp <= ? AND has_error = 0
    )";

    rc = sqlite3_prepare_v2(db_, sql_previous, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare statement"
        ));
    }

    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, previous_start);
    sqlite3_bind_int64(stmt, 3, previous_end);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* query_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (!query_text) continue;

        std::string normalized = normalize_query_text(query_text);
        if (normalized.empty()) continue;

        previous_counts[normalized]++;
    }

    sqlite3_finalize(stmt);

    // Calculate growth rates
    std::vector<TrendingQuery> results;
    for (const auto& [normalized, current_count] : current_counts) {
        size_t previous_count = previous_counts[normalized];

        // Calculate growth rate
        double growth_rate = 0.0;
        if (previous_count == 0) {
            growth_rate = 100.0;  // New query
        } else {
            growth_rate = ((static_cast<double>(current_count) - previous_count) / previous_count) * 100.0;
        }

        if (growth_rate >= min_growth_rate) {
            TrendingQuery trending;
            trending.query_text = query_texts[normalized];
            trending.current_count = current_count;
            trending.previous_count = previous_count;
            trending.growth_rate = growth_rate;
            trending.time_bucket = get_time_bucket(current_start, bucket_type);
            results.push_back(trending);
        }
    }

    // Sort by growth rate descending
    std::sort(results.begin(), results.end(),
        [](const TrendingQuery& a, const TrendingQuery& b) {
            return a.growth_rate > b.growth_rate;
        });

    // Limit results
    if (results.size() > limit) {
        results.resize(limit);
    }

    return results;
}

jadevectordb::Result<AnalyticsInsights> AnalyticsEngine::generate_insights(
    const std::string& database_id,
    int64_t start_time,
    int64_t end_time
) {
    AnalyticsInsights insights;

    // Get top patterns
    auto patterns_result = identify_patterns(database_id, start_time, end_time, 2, 10);
    if (patterns_result.has_value()) {
        insights.top_patterns = patterns_result.value();
    }

    // Get slow queries
    auto slow_result = detect_slow_queries(database_id, start_time, end_time, 500, 10);
    if (slow_result.has_value()) {
        insights.slow_queries = slow_result.value();
    }

    // Get zero-result queries
    auto zero_result = analyze_zero_results(database_id, start_time, end_time, 1, 10);
    if (zero_result.has_value()) {
        insights.zero_result_queries = zero_result.value();
    }

    // Get trending queries
    auto trending_result = detect_trending(database_id, start_time, end_time, TimeBucket::DAILY, 50.0, 10);
    if (trending_result.has_value()) {
        insights.trending_queries = trending_result.value();
    }

    // Calculate summary metrics
    auto stats_result = compute_statistics(database_id, start_time, end_time, TimeBucket::HOURLY);
    if (stats_result.has_value()) {
        const auto& stats = stats_result.value();
        if (!stats.empty()) {
            size_t total_queries = 0;
            size_t successful_queries = 0;
            double max_qps = 0.0;
            int64_t peak_hour = 0;

            for (const auto& stat : stats) {
                total_queries += stat.total_queries;
                successful_queries += stat.successful_queries;

                double qps = stat.total_queries / 3600.0;  // Queries per second
                if (qps > max_qps) {
                    max_qps = qps;
                    peak_hour = stat.time_bucket;
                }
            }

            if (total_queries > 0) {
                insights.overall_success_rate = (static_cast<double>(successful_queries) / total_queries) * 100.0;
            }

            insights.qps_peak = max_qps;
            insights.peak_hour = peak_hour;

            int64_t period_duration_seconds = (end_time - start_time) / 1000;
            if (period_duration_seconds > 0) {
                insights.qps_avg = static_cast<double>(total_queries) / period_duration_seconds;
            }
        }
    }

    return insights;
}

std::string AnalyticsEngine::normalize_query_text(const std::string& query_text) {
    std::string result;
    std::istringstream iss(query_text);
    std::string word;

    while (iss >> word) {
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        // Skip stop words and empty words
        if (word.empty() || STOP_WORDS.find(word) != STOP_WORDS.end()) {
            continue;
        }

        if (!result.empty()) {
            result += " ";
        }
        result += word;
    }

    return result;
}

double AnalyticsEngine::calculate_percentile(const std::vector<double>& sorted_values, double percentile) {
    if (sorted_values.empty()) {
        return 0.0;
    }

    double index = percentile * (sorted_values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) {
        return sorted_values[lower];
    }

    double weight = index - lower;
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

int64_t AnalyticsEngine::get_time_bucket(int64_t timestamp, TimeBucket bucket_type) {
    int64_t bucket_duration = get_bucket_duration_ms(bucket_type);
    return (timestamp / bucket_duration) * bucket_duration;
}

int64_t AnalyticsEngine::get_bucket_duration_ms(TimeBucket bucket_type) {
    switch (bucket_type) {
        case TimeBucket::HOURLY:
            return 3600000;  // 1 hour in ms
        case TimeBucket::DAILY:
            return 86400000;  // 1 day in ms
        case TimeBucket::WEEKLY:
            return 604800000;  // 1 week in ms
        case TimeBucket::MONTHLY:
            return 2592000000;  // 30 days in ms
        default:
            return 3600000;
    }
}

} // namespace analytics
} // namespace jadedb
