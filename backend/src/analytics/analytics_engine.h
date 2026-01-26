#pragma once

#include "lib/error_handling.h"
#include "lib/logging.h"
#include <sqlite3.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace jadedb {
namespace analytics {

/**
 * @brief Time bucket granularity
 */
enum class TimeBucket {
    HOURLY,
    DAILY,
    WEEKLY,
    MONTHLY
};

/**
 * @brief Query statistics for a time period
 */
struct QueryStatistics {
    std::string database_id;
    int64_t time_bucket;
    TimeBucket bucket_type;

    // Query counts
    size_t total_queries = 0;
    size_t successful_queries = 0;
    size_t failed_queries = 0;
    size_t zero_result_queries = 0;

    // User metrics
    size_t unique_users = 0;
    size_t unique_sessions = 0;

    // Latency metrics (milliseconds)
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double max_latency_ms = 0.0;

    // Result metrics
    double avg_results = 0.0;
    double avg_similarity_score = 0.0;

    // Query type breakdown
    size_t vector_queries = 0;
    size_t hybrid_queries = 0;
    size_t bm25_queries = 0;
    size_t reranked_queries = 0;
};

/**
 * @brief Search pattern with metrics
 */
struct SearchPattern {
    std::string pattern_text;
    std::string normalized_text;
    size_t query_count = 0;
    double avg_latency_ms = 0.0;
    double avg_results = 0.0;
    double avg_similarity_score = 0.0;
    int64_t first_seen = 0;
    int64_t last_seen = 0;
};

/**
 * @brief Slow query information
 */
struct SlowQuery {
    std::string query_id;
    std::string query_text;
    std::string query_type;
    int64_t total_time_ms;
    int64_t timestamp;
    int num_results;
};

/**
 * @brief Zero-result query information
 */
struct ZeroResultQuery {
    std::string query_text;
    std::string query_type;
    size_t occurrence_count;
    int64_t last_seen;
};

/**
 * @brief Trending query information
 */
struct TrendingQuery {
    std::string query_text;
    size_t current_count;
    size_t previous_count;
    double growth_rate;  // Percentage
    int64_t time_bucket;
};

/**
 * @brief Analytics insights
 */
struct AnalyticsInsights {
    std::vector<SearchPattern> top_patterns;
    std::vector<SlowQuery> slow_queries;
    std::vector<ZeroResultQuery> zero_result_queries;
    std::vector<TrendingQuery> trending_queries;

    // Summary metrics
    double overall_success_rate = 0.0;
    double qps_avg = 0.0;
    double qps_peak = 0.0;
    int64_t peak_hour = 0;
};

/**
 * @brief Engine for computing analytics and generating insights
 *
 * Provides methods to analyze query logs and generate actionable insights
 * about search performance, patterns, and trends.
 */
class AnalyticsEngine {
public:
    /**
     * @brief Construct analytics engine
     * @param database_path Path to analytics SQLite database
     */
    explicit AnalyticsEngine(const std::string& database_path);

    ~AnalyticsEngine();

    /**
     * @brief Initialize analytics engine
     */
    jadevectordb::Result<void> initialize();

    /**
     * @brief Shutdown analytics engine
     */
    void shutdown();

    /**
     * @brief Compute statistics for a time range
     *
     * @param database_id Database identifier
     * @param start_time Start timestamp (ms)
     * @param end_time End timestamp (ms)
     * @param bucket_type Time bucket granularity
     * @return Result<std::vector<QueryStatistics>> Statistics per time bucket
     */
    jadevectordb::Result<std::vector<QueryStatistics>> compute_statistics(
        const std::string& database_id,
        int64_t start_time,
        int64_t end_time,
        TimeBucket bucket_type = TimeBucket::HOURLY
    );

    /**
     * @brief Identify common search patterns
     *
     * @param database_id Database identifier
     * @param start_time Start timestamp (ms)
     * @param end_time End timestamp (ms)
     * @param min_count Minimum query count to include
     * @param limit Maximum patterns to return
     * @return Result<std::vector<SearchPattern>> Top patterns
     */
    jadevectordb::Result<std::vector<SearchPattern>> identify_patterns(
        const std::string& database_id,
        int64_t start_time,
        int64_t end_time,
        size_t min_count = 2,
        size_t limit = 100
    );

    /**
     * @brief Detect slow queries
     *
     * @param database_id Database identifier
     * @param start_time Start timestamp (ms)
     * @param end_time End timestamp (ms)
     * @param latency_threshold_ms Minimum latency to consider slow
     * @param limit Maximum queries to return
     * @return Result<std::vector<SlowQuery>> Slow queries
     */
    jadevectordb::Result<std::vector<SlowQuery>> detect_slow_queries(
        const std::string& database_id,
        int64_t start_time,
        int64_t end_time,
        int64_t latency_threshold_ms = 1000,
        size_t limit = 100
    );

    /**
     * @brief Analyze zero-result queries
     *
     * @param database_id Database identifier
     * @param start_time Start timestamp (ms)
     * @param end_time End timestamp (ms)
     * @param min_count Minimum occurrences to include
     * @param limit Maximum queries to return
     * @return Result<std::vector<ZeroResultQuery>> Zero-result queries
     */
    jadevectordb::Result<std::vector<ZeroResultQuery>> analyze_zero_results(
        const std::string& database_id,
        int64_t start_time,
        int64_t end_time,
        size_t min_count = 1,
        size_t limit = 100
    );

    /**
     * @brief Detect trending queries
     *
     * @param database_id Database identifier
     * @param current_start Start of current period (ms)
     * @param current_end End of current period (ms)
     * @param bucket_type Time bucket for comparison
     * @param min_growth_rate Minimum growth rate percentage
     * @param limit Maximum queries to return
     * @return Result<std::vector<TrendingQuery>> Trending queries
     */
    jadevectordb::Result<std::vector<TrendingQuery>> detect_trending(
        const std::string& database_id,
        int64_t current_start,
        int64_t current_end,
        TimeBucket bucket_type = TimeBucket::DAILY,
        double min_growth_rate = 50.0,
        size_t limit = 50
    );

    /**
     * @brief Generate comprehensive insights
     *
     * @param database_id Database identifier
     * @param start_time Start timestamp (ms)
     * @param end_time End timestamp (ms)
     * @return Result<AnalyticsInsights> Complete insights
     */
    jadevectordb::Result<AnalyticsInsights> generate_insights(
        const std::string& database_id,
        int64_t start_time,
        int64_t end_time
    );

    /**
     * @brief Normalize query text for pattern matching
     *
     * @param query_text Raw query text
     * @return Normalized text (lowercase, trimmed, stop words removed)
     */
    static std::string normalize_query_text(const std::string& query_text);

private:
    std::string database_path_;
    sqlite3* db_;
    mutable std::mutex mutex_;
    std::shared_ptr<jadevectordb::logging::Logger> logger_;

    /**
     * @brief Execute SQL query
     */
    jadevectordb::Result<void> execute_sql(const std::string& sql);

    /**
     * @brief Calculate percentile from sorted values
     */
    static double calculate_percentile(const std::vector<double>& sorted_values, double percentile);

    /**
     * @brief Get time bucket value
     */
    static int64_t get_time_bucket(int64_t timestamp, TimeBucket bucket_type);

    /**
     * @brief Get bucket duration in milliseconds
     */
    static int64_t get_bucket_duration_ms(TimeBucket bucket_type);
};

} // namespace analytics
} // namespace jadedb
