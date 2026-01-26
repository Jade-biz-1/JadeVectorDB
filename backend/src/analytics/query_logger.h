#pragma once

#include "lib/error_handling.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

namespace jadedb {
namespace analytics {

/**
 * @brief Entry for a single query log record
 */
struct QueryLogEntry {
    std::string query_id;
    std::string database_id;
    std::string query_text;
    std::string query_type;  // "vector", "hybrid", "bm25", "rerank"
    int64_t retrieval_time_ms;
    int64_t total_time_ms;
    int num_results;
    double avg_similarity_score;
    double min_similarity_score;
    double max_similarity_score;
    std::string user_id;
    std::string session_id;
    std::string client_ip;
    int64_t timestamp;  // Unix timestamp in milliseconds

    // Search-specific metadata
    int top_k;
    std::string vector_metric;  // "cosine", "euclidean", "dot_product"

    // Hybrid search specific
    double hybrid_alpha;  // Weight for vector vs BM25
    std::string fusion_method;  // "rrf", "linear"

    // Re-ranking specific
    bool used_reranking;
    std::string reranking_model;
    int64_t reranking_time_ms;

    // Error tracking
    bool has_error;
    std::string error_message;

    QueryLogEntry()
        : retrieval_time_ms(0), total_time_ms(0), num_results(0),
          avg_similarity_score(0.0), min_similarity_score(0.0), max_similarity_score(0.0),
          timestamp(0), top_k(0), hybrid_alpha(0.0),
          used_reranking(false), reranking_time_ms(0),
          has_error(false) {}
};

/**
 * @brief Configuration for query logger
 */
struct QueryLoggerConfig {
    std::string database_path;  // SQLite database for logs
    size_t batch_size = 100;    // Number of entries to batch before writing
    int64_t flush_interval_ms = 5000;  // Max time to hold entries before flush
    size_t max_queue_size = 10000;  // Max entries in memory queue
    bool enable_async = true;   // Use async writes (recommended)

    QueryLoggerConfig() = default;
};

/**
 * @brief Asynchronous query logger with minimal overhead
 *
 * Design goals:
 * - <1ms logging overhead (async queue)
 * - Batch writes to SQLite for performance
 * - Thread-safe operation
 * - Graceful shutdown with flush
 */
class QueryLogger {
public:
    /**
     * @brief Construct query logger
     * @param database_id Database identifier
     * @param config Logger configuration
     */
    QueryLogger(
        const std::string& database_id,
        const QueryLoggerConfig& config = QueryLoggerConfig()
    );

    ~QueryLogger();

    // Non-copyable (thread resource)
    QueryLogger(const QueryLogger&) = delete;
    QueryLogger& operator=(const QueryLogger&) = delete;

    /**
     * @brief Initialize logger (create tables, start background thread)
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> initialize();

    /**
     * @brief Shutdown logger (flush pending entries, stop thread)
     */
    void shutdown();

    /**
     * @brief Log a query entry (async, <1ms overhead)
     * @param entry Query log entry
     * @return Result<void> Success or error (queue full)
     */
    jadevectordb::Result<void> log_query(const QueryLogEntry& entry);

    /**
     * @brief Log a query entry (synchronous, for testing)
     * @param entry Query log entry
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> log_query_sync(const QueryLogEntry& entry);

    /**
     * @brief Force flush all pending entries
     * @return Result<void> Success or error
     */
    jadevectordb::Result<void> flush();

    /**
     * @brief Get current queue size
     */
    size_t get_queue_size() const;

    /**
     * @brief Get total entries logged
     */
    size_t get_total_logged() const;

    /**
     * @brief Get total entries dropped (queue full)
     */
    size_t get_total_dropped() const;

    /**
     * @brief Check if logger is ready
     */
    bool is_ready() const;

    /**
     * @brief Generate unique query ID
     */
    static std::string generate_query_id();

    /**
     * @brief Get current timestamp in milliseconds
     */
    static int64_t get_current_timestamp_ms();

private:
    std::string database_id_;
    QueryLoggerConfig config_;

    // SQLite database handle (raw pointer for C API)
    void* db_;  // sqlite3*

    // Async write queue
    std::queue<QueryLogEntry> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Background writer thread
    std::thread writer_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> ready_;

    // Statistics
    std::atomic<size_t> total_logged_;
    std::atomic<size_t> total_dropped_;

    // Logger
    std::shared_ptr<jadevectordb::logging::Logger> logger_;

    /**
     * @brief Background writer thread function
     */
    void writer_thread_func();

    /**
     * @brief Write batch of entries to database
     */
    jadevectordb::Result<void> write_batch(const std::vector<QueryLogEntry>& entries);

    /**
     * @brief Create database tables
     */
    jadevectordb::Result<void> create_tables();

    /**
     * @brief Insert entry into database
     */
    jadevectordb::Result<void> insert_entry(const QueryLogEntry& entry);

    /**
     * @brief Execute SQL statement
     */
    jadevectordb::Result<void> execute_sql(const std::string& sql);
};

} // namespace analytics
} // namespace jadedb
