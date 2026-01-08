#pragma once

#include "bm25_scorer.h"
#include "inverted_index.h"
#include "bm25_index_persistence.h"
#include "hybrid_search_engine.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>

namespace jadedb {
namespace search {

/**
 * @brief Status of BM25 index building operation
 */
enum class BuildStatus {
    IDLE,           // No build in progress
    IN_PROGRESS,    // Currently building
    COMPLETED,      // Build completed successfully
    FAILED          // Build failed
};

/**
 * @brief Progress information for index building
 */
struct BuildProgress {
    BuildStatus status = BuildStatus::IDLE;
    size_t total_documents = 0;
    size_t processed_documents = 0;
    size_t indexed_terms = 0;
    double progress_percentage = 0.0;
    std::string error_message;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point completed_at;
    int64_t duration_ms = 0;

    bool is_complete() const {
        return status == BuildStatus::COMPLETED || status == BuildStatus::FAILED;
    }

    double get_progress() const {
        if (total_documents == 0) return 0.0;
        return (static_cast<double>(processed_documents) / total_documents) * 100.0;
    }
};

/**
 * @brief Configuration for index building
 */
struct BuildConfig {
    // Batch size for processing documents
    size_t batch_size = 1000;

    // Whether to persist the index after building
    bool persist_on_completion = true;

    // Whether to rebuild from scratch (vs incremental)
    bool force_rebuild = false;

    // Path for index persistence
    std::string persistence_path;

    // BM25 parameters
    BM25Config bm25_config;

    BuildConfig() = default;
};

/**
 * @brief Document source provider function
 *
 * Returns a vector of BM25Documents for indexing.
 * This allows flexible integration with different data sources.
 */
using DocumentSourceProvider = std::function<std::vector<BM25Document>()>;

/**
 * @brief Callback for build progress updates
 *
 * Called periodically during the build process to report progress.
 */
using ProgressCallback = std::function<void(const BuildProgress&)>;

/**
 * @brief Service for building and managing BM25 indexes
 *
 * Provides a comprehensive pipeline for:
 * - Building BM25 indexes from document sources
 * - Incremental updates when documents are added/removed
 * - Rebuilding existing indexes
 * - Progress tracking and status reporting
 * - Persistence and recovery
 *
 * Thread-safe for concurrent builds across different databases.
 */
class BM25IndexBuilder {
public:
    /**
     * @brief Construct BM25 index builder
     * @param database_id Database identifier
     * @param config Build configuration
     */
    BM25IndexBuilder(
        const std::string& database_id,
        const BuildConfig& config = BuildConfig()
    );

    ~BM25IndexBuilder() = default;

    /**
     * @brief Build BM25 index from documents
     *
     * Performs full index build from the provided documents.
     * This is an asynchronous operation that returns immediately.
     * Use get_progress() to monitor build status.
     *
     * @param documents Vector of documents to index
     * @param callback Optional progress callback
     * @return true if build started successfully, false otherwise
     */
    bool build_from_documents(
        const std::vector<BM25Document>& documents,
        ProgressCallback callback = nullptr
    );

    /**
     * @brief Build BM25 index from a document source provider
     *
     * Allows lazy loading of documents from external sources.
     *
     * @param provider Function that returns documents
     * @param callback Optional progress callback
     * @return true if build started successfully, false otherwise
     */
    bool build_from_provider(
        DocumentSourceProvider provider,
        ProgressCallback callback = nullptr
    );

    /**
     * @brief Rebuild existing index
     *
     * Clears the current index and rebuilds from scratch.
     *
     * @param documents New set of documents
     * @param callback Optional progress callback
     * @return true if rebuild started successfully, false otherwise
     */
    bool rebuild_index(
        const std::vector<BM25Document>& documents,
        ProgressCallback callback = nullptr
    );

    /**
     * @brief Add documents to existing index (incremental update)
     *
     * Adds new documents without rebuilding the entire index.
     * Updates IDF scores and statistics.
     *
     * @param documents Documents to add
     * @return true on success
     */
    bool add_documents(const std::vector<BM25Document>& documents);

    /**
     * @brief Remove documents from index
     *
     * Removes documents by their IDs and updates statistics.
     *
     * @param doc_ids Document IDs to remove
     * @return true on success
     */
    bool remove_documents(const std::vector<std::string>& doc_ids);

    /**
     * @brief Update existing documents
     *
     * Efficiently updates documents by removing old versions
     * and adding new ones.
     *
     * @param documents Updated documents
     * @return true on success
     */
    bool update_documents(const std::vector<BM25Document>& documents);

    /**
     * @brief Get current build progress
     */
    BuildProgress get_progress() const;

    /**
     * @brief Check if a build is currently in progress
     */
    bool is_building() const;

    /**
     * @brief Wait for current build to complete
     *
     * Blocks until the build finishes or timeout is reached.
     *
     * @param timeout_ms Timeout in milliseconds (0 = infinite)
     * @return true if build completed, false if timeout
     */
    bool wait_for_completion(int timeout_ms = 0);

    /**
     * @brief Cancel ongoing build
     *
     * Attempts to cancel the current build operation.
     * The index may be in an inconsistent state after cancellation.
     */
    void cancel_build();

    /**
     * @brief Persist index to storage
     *
     * Saves the current index state to persistent storage.
     *
     * @param path Optional custom path (uses config path if empty)
     * @return true on success
     */
    bool persist_index(const std::string& path = "");

    /**
     * @brief Load index from storage
     *
     * Loads a previously persisted index.
     *
     * @param path Path to index storage
     * @return true on success
     */
    bool load_index(const std::string& path);

    /**
     * @brief Get index statistics
     *
     * @param total_docs Output: total documents
     * @param total_terms Output: total unique terms
     * @param avg_doc_length Output: average document length
     */
    void get_index_stats(
        size_t& total_docs,
        size_t& total_terms,
        double& avg_doc_length
    ) const;

    /**
     * @brief Check if index is ready for searching
     */
    bool is_index_ready() const;

    /**
     * @brief Get the hybrid search engine with this BM25 index
     *
     * Returns a configured HybridSearchEngine that uses this
     * BM25 index for keyword search.
     */
    std::shared_ptr<HybridSearchEngine> get_search_engine();

    /**
     * @brief Update build configuration
     */
    void set_config(const BuildConfig& config);

    /**
     * @brief Get current configuration
     */
    const BuildConfig& get_config() const;

private:
    std::string database_id_;
    BuildConfig config_;

    // Index components
    std::unique_ptr<BM25Scorer> bm25_scorer_;
    std::unique_ptr<InvertedIndex> inverted_index_;
    std::unique_ptr<BM25IndexPersistence> persistence_;

    // Search engine integration
    std::shared_ptr<HybridSearchEngine> search_engine_;

    // Build state
    mutable std::mutex build_mutex_;
    std::atomic<bool> building_;
    std::atomic<bool> cancel_requested_;
    BuildProgress progress_;

    // Logger
    std::shared_ptr<jadevectordb::logging::Logger> logger_;

    /**
     * @brief Internal build implementation
     */
    void build_internal(
        const std::vector<BM25Document>& documents,
        ProgressCallback callback
    );

    /**
     * @brief Update progress information
     */
    void update_progress(size_t processed, size_t total);

    /**
     * @brief Process a batch of documents
     */
    bool process_batch(
        const std::vector<BM25Document>& batch,
        size_t& indexed_terms
    );

    /**
     * @brief Finalize index after build
     */
    void finalize_build();

    /**
     * @brief Clear all index data
     */
    void clear_index();
};

} // namespace search
} // namespace jadedb
