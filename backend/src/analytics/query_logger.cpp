#include "query_logger.h"
#include <sqlite3.h>
#include <chrono>
#include <sstream>
#include <random>
#include <iomanip>

namespace jadedb {
namespace analytics {

namespace {
    // Generate UUID-like query ID
    std::string generate_uuid() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        static std::uniform_int_distribution<uint64_t> dis;

        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        oss << std::setw(16) << dis(gen);
        oss << std::setw(16) << dis(gen);
        return oss.str();
    }
}

QueryLogger::QueryLogger(
    const std::string& database_id,
    const QueryLoggerConfig& config
)
    : database_id_(database_id),
      config_(config),
      db_(nullptr),
      running_(false),
      ready_(false),
      total_logged_(0),
      total_dropped_(0),
      logger_(jadevectordb::logging::LoggerManager::get_logger("QueryLogger"))
{
}

QueryLogger::~QueryLogger() {
    shutdown();
}

jadevectordb::Result<void> QueryLogger::initialize() {
    if (ready_.load()) {
        return jadevectordb::Result<void>{};
    }

    // Open SQLite database
    int rc = sqlite3_open(config_.database_path.c_str(),
                          reinterpret_cast<sqlite3**>(&db_));
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to open analytics database: " + std::string(sqlite3_errmsg(reinterpret_cast<sqlite3*>(db_)))
        ));
    }

    // Enable WAL mode for better concurrency
    auto result = execute_sql("PRAGMA journal_mode=WAL");
    if (!result.has_value()) {
        sqlite3_close(reinterpret_cast<sqlite3*>(db_));
        db_ = nullptr;
        return result;
    }

    // Create tables
    result = create_tables();
    if (!result.has_value()) {
        sqlite3_close(reinterpret_cast<sqlite3*>(db_));
        db_ = nullptr;
        return result;
    }

    // Start background writer thread
    if (config_.enable_async) {
        running_.store(true);
        writer_thread_ = std::thread(&QueryLogger::writer_thread_func, this);
    }

    ready_.store(true);
    logger_->info("QueryLogger initialized for database: " + database_id_);

    return jadevectordb::Result<void>{};
}

void QueryLogger::shutdown() {
    if (!ready_.load()) {
        return;
    }

    logger_->info("Shutting down QueryLogger...");

    // Stop background thread
    if (config_.enable_async && running_.load()) {
        running_.store(false);
        queue_cv_.notify_all();
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }

    // Flush any remaining entries
    auto flush_result = flush();
    (void)flush_result;  // Ignore result during shutdown

    // Close database
    if (db_) {
        sqlite3_close(reinterpret_cast<sqlite3*>(db_));
        db_ = nullptr;
    }

    ready_.store(false);
    logger_->info("QueryLogger shutdown complete");
}

jadevectordb::Result<void> QueryLogger::log_query(const QueryLogEntry& entry) {
    if (!ready_.load()) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Logger not initialized"
        ));
    }

    if (config_.enable_async) {
        // Async path - push to queue
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (queue_.size() >= config_.max_queue_size) {
            total_dropped_.fetch_add(1);
            return tl::make_unexpected(jadevectordb::ErrorInfo(
                jadevectordb::ErrorCode::RESOURCE_EXHAUSTED,
                "Queue full, entry dropped"
            ));
        }

        queue_.push(entry);
        queue_cv_.notify_one();

        return jadevectordb::Result<void>{};
    } else {
        // Sync path - write immediately
        auto result = insert_entry(entry);
        if (result.has_value()) {
            total_logged_.fetch_add(1);
        }
        return result;
    }
}

jadevectordb::Result<void> QueryLogger::log_query_sync(const QueryLogEntry& entry) {
    if (!ready_.load()) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::INVALID_STATE,
            "Logger not initialized"
        ));
    }

    auto result = insert_entry(entry);
    if (result.has_value()) {
        total_logged_.fetch_add(1);
    }
    return result;
}

jadevectordb::Result<void> QueryLogger::flush() {
    if (!ready_.load()) {
        return jadevectordb::Result<void>{};
    }

    if (!config_.enable_async) {
        return jadevectordb::Result<void>{};
    }

    // Extract all pending entries
    std::vector<QueryLogEntry> entries;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!queue_.empty()) {
            entries.push_back(queue_.front());
            queue_.pop();
        }
    }

    if (entries.empty()) {
        return jadevectordb::Result<void>{};
    }

    return write_batch(entries);
}

size_t QueryLogger::get_queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

size_t QueryLogger::get_total_logged() const {
    return total_logged_.load();
}

size_t QueryLogger::get_total_dropped() const {
    return total_dropped_.load();
}

bool QueryLogger::is_ready() const {
    return ready_.load();
}

std::string QueryLogger::generate_query_id() {
    return generate_uuid();
}

int64_t QueryLogger::get_current_timestamp_ms() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void QueryLogger::writer_thread_func() {
    logger_->info("QueryLogger writer thread started");

    std::vector<QueryLogEntry> batch;
    batch.reserve(config_.batch_size);

    while (running_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Wait for entries or timeout
        queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.flush_interval_ms),
            [this]() { return !queue_.empty() || !running_.load(); });

        // Extract batch
        while (!queue_.empty() && batch.size() < config_.batch_size) {
            batch.push_back(queue_.front());
            queue_.pop();
        }

        lock.unlock();

        // Write batch if we have entries
        if (!batch.empty()) {
            auto result = write_batch(batch);
            if (!result.has_value()) {
                logger_->error("Failed to write batch: " + result.error().message);
            }
            batch.clear();
        }
    }

    logger_->info("QueryLogger writer thread stopped");
}

jadevectordb::Result<void> QueryLogger::write_batch(const std::vector<QueryLogEntry>& entries) {
    if (entries.empty()) {
        return jadevectordb::Result<void>{};
    }

    // Begin transaction
    auto result = execute_sql("BEGIN TRANSACTION");
    if (!result.has_value()) {
        return result;
    }

    // Insert all entries
    size_t success_count = 0;
    for (const auto& entry : entries) {
        auto insert_result = insert_entry(entry);
        if (insert_result.has_value()) {
            success_count++;
        } else {
            logger_->warn("Failed to insert entry: " + insert_result.error().message);
        }
    }

    // Commit transaction
    result = execute_sql("COMMIT");
    if (!result.has_value()) {
        auto rollback_result = execute_sql("ROLLBACK");
        (void)rollback_result;  // Ignore rollback result
        return result;
    }

    total_logged_.fetch_add(success_count);

    return jadevectordb::Result<void>{};
}

jadevectordb::Result<void> QueryLogger::create_tables() {
    const std::string create_sql = R"(
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            database_id TEXT NOT NULL,
            query_text TEXT NOT NULL,
            query_type TEXT NOT NULL,
            retrieval_time_ms INTEGER NOT NULL,
            total_time_ms INTEGER NOT NULL,
            num_results INTEGER NOT NULL,
            avg_similarity_score REAL NOT NULL,
            min_similarity_score REAL NOT NULL,
            max_similarity_score REAL NOT NULL,
            user_id TEXT,
            session_id TEXT,
            client_ip TEXT,
            timestamp INTEGER NOT NULL,
            top_k INTEGER,
            vector_metric TEXT,
            hybrid_alpha REAL,
            fusion_method TEXT,
            used_reranking INTEGER,
            reranking_model TEXT,
            reranking_time_ms INTEGER,
            has_error INTEGER,
            error_message TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_query_log_database_id ON query_log(database_id);
        CREATE INDEX IF NOT EXISTS idx_query_log_timestamp ON query_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_query_log_query_type ON query_log(query_type);
        CREATE INDEX IF NOT EXISTS idx_query_log_session_id ON query_log(session_id);
        CREATE INDEX IF NOT EXISTS idx_query_log_has_error ON query_log(has_error);
    )";

    return execute_sql(create_sql);
}

jadevectordb::Result<void> QueryLogger::insert_entry(const QueryLogEntry& entry) {
    sqlite3* sqlite_db = reinterpret_cast<sqlite3*>(db_);

    const std::string insert_sql = R"(
        INSERT INTO query_log (
            query_id, database_id, query_text, query_type,
            retrieval_time_ms, total_time_ms, num_results,
            avg_similarity_score, min_similarity_score, max_similarity_score,
            user_id, session_id, client_ip, timestamp,
            top_k, vector_metric, hybrid_alpha, fusion_method,
            used_reranking, reranking_model, reranking_time_ms,
            has_error, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(sqlite_db, insert_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to prepare insert statement: " + std::string(sqlite3_errmsg(sqlite_db))
        ));
    }

    // Bind parameters
    sqlite3_bind_text(stmt, 1, entry.query_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, entry.database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, entry.query_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, entry.query_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, entry.retrieval_time_ms);
    sqlite3_bind_int64(stmt, 6, entry.total_time_ms);
    sqlite3_bind_int(stmt, 7, entry.num_results);
    sqlite3_bind_double(stmt, 8, entry.avg_similarity_score);
    sqlite3_bind_double(stmt, 9, entry.min_similarity_score);
    sqlite3_bind_double(stmt, 10, entry.max_similarity_score);
    sqlite3_bind_text(stmt, 11, entry.user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 12, entry.session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 13, entry.client_ip.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 14, entry.timestamp);
    sqlite3_bind_int(stmt, 15, entry.top_k);
    sqlite3_bind_text(stmt, 16, entry.vector_metric.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 17, entry.hybrid_alpha);
    sqlite3_bind_text(stmt, 18, entry.fusion_method.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 19, entry.used_reranking ? 1 : 0);
    sqlite3_bind_text(stmt, 20, entry.reranking_model.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 21, entry.reranking_time_ms);
    sqlite3_bind_int(stmt, 22, entry.has_error ? 1 : 0);
    sqlite3_bind_text(stmt, 23, entry.error_message.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "Failed to insert entry: " + std::string(sqlite3_errmsg(sqlite_db))
        ));
    }

    return jadevectordb::Result<void>{};
}

jadevectordb::Result<void> QueryLogger::execute_sql(const std::string& sql) {
    sqlite3* sqlite_db = reinterpret_cast<sqlite3*>(db_);

    char* err_msg = nullptr;
    int rc = sqlite3_exec(sqlite_db, sql.c_str(), nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        std::string error = err_msg ? err_msg : "Unknown error";
        if (err_msg) {
            sqlite3_free(err_msg);
        }
        return tl::make_unexpected(jadevectordb::ErrorInfo(
            jadevectordb::ErrorCode::STORAGE_IO_ERROR,
            "SQL execution failed: " + error
        ));
    }

    return jadevectordb::Result<void>{};
}

} // namespace analytics
} // namespace jadedb
