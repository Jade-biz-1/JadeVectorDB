#pragma once

#include "bm25_scorer.h"
#include "inverted_index.h"
#include <sqlite3.h>
#include <string>
#include <memory>
#include <mutex>

namespace jadedb {
namespace search {

/**
 * @brief Persistence layer for BM25 index and inverted index
 *
 * Manages SQLite storage for:
 * - Inverted index (term -> postings)
 * - BM25 configuration (k1, b, avg_doc_length, total_docs)
 * - Document metadata (doc_id, doc_length, indexed_at)
 *
 * Schema:
 * - bm25_index: term, doc_frequency, postings_blob
 * - bm25_metadata: doc_id, doc_length, indexed_at
 * - bm25_config: database_id, k1, b, avg_doc_length, total_docs
 */
class BM25IndexPersistence {
public:
    /**
     * @brief Construct persistence layer
     * @param database_id Database identifier
     * @param db_path Path to SQLite database file
     */
    BM25IndexPersistence(const std::string& database_id, const std::string& db_path);
    ~BM25IndexPersistence();

    // Prevent copying
    BM25IndexPersistence(const BM25IndexPersistence&) = delete;
    BM25IndexPersistence& operator=(const BM25IndexPersistence&) = delete;

    /**
     * @brief Initialize database schema
     * Creates tables if they don't exist
     * @return true on success
     */
    bool initialize();

    /**
     * @brief Save complete index to database
     *
     * @param scorer BM25 scorer instance
     * @param index Inverted index instance
     * @return true on success
     */
    bool save_index(const BM25Scorer& scorer, const InvertedIndex& index);

    /**
     * @brief Load index from database
     *
     * @param scorer BM25 scorer to populate
     * @param index Inverted index to populate
     * @return true on success
     */
    bool load_index(BM25Scorer& scorer, InvertedIndex& index);

    /**
     * @brief Add single document to persisted index
     *
     * @param doc BM25 document
     * @param term_frequencies Term frequencies for this document
     * @return true on success
     */
    bool add_document(
        const BM25Document& doc,
        const std::unordered_map<std::string, int>& term_frequencies
    );

    /**
     * @brief Remove document from persisted index
     *
     * @param doc_id Document ID to remove
     * @return true on success
     */
    bool remove_document(const std::string& doc_id);

    /**
     * @brief Clear entire index
     * @return true on success
     */
    bool clear_index();

    /**
     * @brief Check if index exists for this database
     * @return true if index data exists
     */
    bool index_exists() const;

    /**
     * @brief Get last index update timestamp
     * @return Timestamp string or empty if no index
     */
    std::string get_last_update_time() const;

    /**
     * @brief Get index statistics
     *
     * @param total_docs Output: total documents
     * @param total_terms Output: total terms
     * @param avg_doc_length Output: average document length
     * @return true on success
     */
    bool get_index_stats(size_t& total_docs, size_t& total_terms, double& avg_doc_length) const;

private:
    std::string database_id_;
    std::string db_path_;
    sqlite3* db_;
    mutable std::mutex mutex_;

    /**
     * @brief Create database schema
     */
    bool create_schema();

    /**
     * @brief Execute SQL statement
     */
    bool execute_sql(const std::string& sql);

    /**
     * @brief Serialize postings list to blob
     */
    std::vector<uint8_t> serialize_postings(const PostingsList& postings) const;

    /**
     * @brief Deserialize postings list from blob
     */
    PostingsList deserialize_postings(const uint8_t* data, size_t size) const;

    /**
     * @brief Begin transaction
     */
    bool begin_transaction();

    /**
     * @brief Commit transaction
     */
    bool commit_transaction();

    /**
     * @brief Rollback transaction
     */
    bool rollback_transaction();
};

} // namespace search
} // namespace jadedb
