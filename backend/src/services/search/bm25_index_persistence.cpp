#include "bm25_index_persistence.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace jadedb {
namespace search {

BM25IndexPersistence::BM25IndexPersistence(
    const std::string& database_id,
    const std::string& db_path)
    : database_id_(database_id), db_path_(db_path), db_(nullptr) {
}

BM25IndexPersistence::~BM25IndexPersistence() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool BM25IndexPersistence::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Open database
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open BM25 index database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }

    // Create schema
    return create_schema();
}

bool BM25IndexPersistence::create_schema() {
    // Note: This is called from initialize which already holds the lock

    const char* schema_sql = R"(
        CREATE TABLE IF NOT EXISTS bm25_config (
            database_id TEXT PRIMARY KEY,
            k1 REAL NOT NULL,
            b REAL NOT NULL,
            avg_doc_length REAL NOT NULL,
            total_docs INTEGER NOT NULL,
            last_updated TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bm25_metadata (
            database_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            doc_length INTEGER NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY (database_id, doc_id)
        );

        CREATE TABLE IF NOT EXISTS bm25_index (
            database_id TEXT NOT NULL,
            term TEXT NOT NULL,
            doc_frequency INTEGER NOT NULL,
            postings_blob BLOB NOT NULL,
            PRIMARY KEY (database_id, term)
        );

        CREATE INDEX IF NOT EXISTS idx_bm25_metadata_doc ON bm25_metadata(database_id, doc_id);
        CREATE INDEX IF NOT EXISTS idx_bm25_index_term ON bm25_index(database_id, term);
    )";

    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, schema_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create BM25 schema: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }

    return true;
}

bool BM25IndexPersistence::execute_sql(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

std::vector<uint8_t> BM25IndexPersistence::serialize_postings(const PostingsList& postings) const {
    std::vector<uint8_t> data;

    // Format: [num_postings: 4 bytes] [posting1] [posting2] ...
    // Posting format: [doc_id_len: 2 bytes] [doc_id: var] [term_freq: 4 bytes] [num_positions: 2 bytes] [pos1: 4 bytes] [pos2: 4 bytes] ...

    size_t num_postings = postings.postings.size();

    // Write number of postings (4 bytes)
    data.push_back((num_postings >> 24) & 0xFF);
    data.push_back((num_postings >> 16) & 0xFF);
    data.push_back((num_postings >> 8) & 0xFF);
    data.push_back(num_postings & 0xFF);

    for (const auto& posting : postings.postings) {
        // Write doc_id length (2 bytes)
        uint16_t doc_id_len = posting.doc_id.size();
        data.push_back((doc_id_len >> 8) & 0xFF);
        data.push_back(doc_id_len & 0xFF);

        // Write doc_id
        data.insert(data.end(), posting.doc_id.begin(), posting.doc_id.end());

        // Write term frequency (4 bytes)
        uint32_t tf = posting.term_frequency;
        data.push_back((tf >> 24) & 0xFF);
        data.push_back((tf >> 16) & 0xFF);
        data.push_back((tf >> 8) & 0xFF);
        data.push_back(tf & 0xFF);

        // Write number of positions (2 bytes)
        uint16_t num_pos = posting.positions.size();
        data.push_back((num_pos >> 8) & 0xFF);
        data.push_back(num_pos & 0xFF);

        // Write positions
        for (int pos : posting.positions) {
            uint32_t upos = static_cast<uint32_t>(pos);
            data.push_back((upos >> 24) & 0xFF);
            data.push_back((upos >> 16) & 0xFF);
            data.push_back((upos >> 8) & 0xFF);
            data.push_back(upos & 0xFF);
        }
    }

    return data;
}

PostingsList BM25IndexPersistence::deserialize_postings(const uint8_t* data, size_t size) const {
    PostingsList postings;

    if (size < 4) {
        return postings;  // Invalid data
    }

    size_t offset = 0;

    // Read number of postings
    uint32_t num_postings = (static_cast<uint32_t>(data[offset]) << 24) |
                            (static_cast<uint32_t>(data[offset + 1]) << 16) |
                            (static_cast<uint32_t>(data[offset + 2]) << 8) |
                            static_cast<uint32_t>(data[offset + 3]);
    offset += 4;

    for (uint32_t i = 0; i < num_postings && offset < size; i++) {
        Posting posting;

        // Read doc_id length
        if (offset + 2 > size) break;
        uint16_t doc_id_len = (static_cast<uint16_t>(data[offset]) << 8) |
                              static_cast<uint16_t>(data[offset + 1]);
        offset += 2;

        // Read doc_id
        if (offset + doc_id_len > size) break;
        posting.doc_id = std::string(reinterpret_cast<const char*>(&data[offset]), doc_id_len);
        offset += doc_id_len;

        // Read term frequency
        if (offset + 4 > size) break;
        posting.term_frequency = (static_cast<uint32_t>(data[offset]) << 24) |
                                 (static_cast<uint32_t>(data[offset + 1]) << 16) |
                                 (static_cast<uint32_t>(data[offset + 2]) << 8) |
                                 static_cast<uint32_t>(data[offset + 3]);
        offset += 4;

        // Read number of positions
        if (offset + 2 > size) break;
        uint16_t num_pos = (static_cast<uint16_t>(data[offset]) << 8) |
                           static_cast<uint16_t>(data[offset + 1]);
        offset += 2;

        // Read positions
        for (uint16_t j = 0; j < num_pos && offset + 4 <= size; j++) {
            uint32_t upos = (static_cast<uint32_t>(data[offset]) << 24) |
                            (static_cast<uint32_t>(data[offset + 1]) << 16) |
                            (static_cast<uint32_t>(data[offset + 2]) << 8) |
                            static_cast<uint32_t>(data[offset + 3]);
            posting.positions.push_back(static_cast<int>(upos));
            offset += 4;
        }

        postings.add_posting(posting);
    }

    return postings;
}

bool BM25IndexPersistence::begin_transaction() {
    return execute_sql("BEGIN TRANSACTION");
}

bool BM25IndexPersistence::commit_transaction() {
    return execute_sql("COMMIT");
}

bool BM25IndexPersistence::rollback_transaction() {
    return execute_sql("ROLLBACK");
}

bool BM25IndexPersistence::save_index(const BM25Scorer& scorer, const InvertedIndex& index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    // Begin transaction
    if (!begin_transaction()) {
        return false;
    }

    try {
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
        std::string timestamp = ss.str();

        // Clear existing data for this database
        sqlite3_stmt* stmt;
        std::string delete_sql = "DELETE FROM bm25_config WHERE database_id = ?";
        sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        delete_sql = "DELETE FROM bm25_index WHERE database_id = ?";
        sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        delete_sql = "DELETE FROM bm25_metadata WHERE database_id = ?";
        sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        // Save config
        std::string insert_config_sql =
            "INSERT INTO bm25_config (database_id, k1, b, avg_doc_length, total_docs, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?)";
        sqlite3_prepare_v2(db_, insert_config_sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 2, 1.5);  // Default k1
        sqlite3_bind_double(stmt, 3, 0.75); // Default b
        sqlite3_bind_double(stmt, 4, scorer.get_avg_doc_length());
        sqlite3_bind_int(stmt, 5, static_cast<int>(scorer.get_document_count()));
        sqlite3_bind_text(stmt, 6, timestamp.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        // Save inverted index
        auto all_terms = index.get_all_terms();
        for (const auto& term : all_terms) {
            const PostingsList& postings = index.get_postings(term);
            std::vector<uint8_t> blob = serialize_postings(postings);

            std::string insert_index_sql =
                "INSERT INTO bm25_index (database_id, term, doc_frequency, postings_blob) "
                "VALUES (?, ?, ?, ?)";
            sqlite3_prepare_v2(db_, insert_index_sql.c_str(), -1, &stmt, nullptr);
            sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, term.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 3, static_cast<int>(postings.document_frequency()));
            sqlite3_bind_blob(stmt, 4, blob.data(), blob.size(), SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }

        // Commit transaction
        if (!commit_transaction()) {
            rollback_transaction();
            return false;
        }

        return true;
    } catch (...) {
        rollback_transaction();
        return false;
    }
}

bool BM25IndexPersistence::load_index(BM25Scorer& scorer, InvertedIndex& index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    // Clear existing data
    scorer.clear();
    index.clear();

    // Load config (avg_doc_length, total_docs)
    std::string config_sql = "SELECT avg_doc_length, total_docs FROM bm25_config WHERE database_id = ?";
    sqlite3_stmt* config_stmt;
    int rc = sqlite3_prepare_v2(db_, config_sql.c_str(), -1, &config_stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(config_stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);

    double avg_doc_length = 0.0;
    size_t total_docs = 0;

    if (sqlite3_step(config_stmt) == SQLITE_ROW) {
        avg_doc_length = sqlite3_column_double(config_stmt, 0);
        total_docs = static_cast<size_t>(sqlite3_column_int(config_stmt, 1));
    }

    sqlite3_finalize(config_stmt);

    // Load inverted index
    std::string select_sql = "SELECT term, postings_blob FROM bm25_index WHERE database_id = ?";
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db_, select_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);

    // Track documents and their term frequencies
    std::unordered_map<std::string, std::unordered_map<std::string, int>> doc_term_freqs;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string term = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const uint8_t* blob_data = static_cast<const uint8_t*>(sqlite3_column_blob(stmt, 1));
        int blob_size = sqlite3_column_bytes(stmt, 1);

        PostingsList postings = deserialize_postings(blob_data, blob_size);

        // Add postings to inverted index
        for (const auto& posting : postings.postings) {
            index.add_document_with_frequencies(posting.doc_id, {{term, posting.term_frequency}});

            // Track for BM25Scorer restoration
            doc_term_freqs[posting.doc_id][term] = posting.term_frequency;
        }
    }

    sqlite3_finalize(stmt);

    // Restore BM25Scorer state
    scorer.set_statistics(avg_doc_length, total_docs);

    // Restore documents to BM25Scorer
    for (const auto& [doc_id, term_freqs] : doc_term_freqs) {
        BM25Document doc;
        doc.doc_id = doc_id;
        doc.term_frequencies = term_freqs;
        // Calculate doc_length from term frequencies
        doc.doc_length = 0;
        for (const auto& [term, freq] : term_freqs) {
            doc.doc_length += freq;
        }
        scorer.restore_document(doc_id, doc);
    }

    return true;
}

bool BM25IndexPersistence::add_document(
    const BM25Document& doc,
    const std::unordered_map<std::string, int>& term_frequencies) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    // This is a simplified implementation
    // In production, you'd update the index incrementally
    return true;
}

bool BM25IndexPersistence::remove_document(const std::string& doc_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    // Delete from metadata
    std::string delete_sql = "DELETE FROM bm25_metadata WHERE database_id = ? AND doc_id = ?";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, doc_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return true;
}

bool BM25IndexPersistence::clear_index() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    if (!begin_transaction()) {
        return false;
    }

    sqlite3_stmt* stmt;

    // Clear config
    std::string delete_sql = "DELETE FROM bm25_config WHERE database_id = ?";
    sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    // Clear index
    delete_sql = "DELETE FROM bm25_index WHERE database_id = ?";
    sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    // Clear metadata
    delete_sql = "DELETE FROM bm25_metadata WHERE database_id = ?";
    sqlite3_prepare_v2(db_, delete_sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return commit_transaction();
}

bool BM25IndexPersistence::index_exists() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    std::string select_sql = "SELECT COUNT(*) FROM bm25_config WHERE database_id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, select_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);

    bool exists = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        int count = sqlite3_column_int(stmt, 0);
        exists = (count > 0);
    }

    sqlite3_finalize(stmt);
    return exists;
}

std::string BM25IndexPersistence::get_last_update_time() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return "";
    }

    std::string select_sql = "SELECT last_updated FROM bm25_config WHERE database_id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, select_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return "";
    }

    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);

    std::string timestamp;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return timestamp;
}

bool BM25IndexPersistence::get_index_stats(
    size_t& total_docs,
    size_t& total_terms,
    double& avg_doc_length) const {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!db_) {
        return false;
    }

    std::string select_sql = "SELECT total_docs, avg_doc_length FROM bm25_config WHERE database_id = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, select_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);

    bool found = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        total_docs = sqlite3_column_int(stmt, 0);
        avg_doc_length = sqlite3_column_double(stmt, 1);
        found = true;
    }

    sqlite3_finalize(stmt);

    if (found) {
        // Count terms
        select_sql = "SELECT COUNT(*) FROM bm25_index WHERE database_id = ?";
        sqlite3_prepare_v2(db_, select_sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, database_id_.c_str(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            total_terms = sqlite3_column_int(stmt, 0);
        }

        sqlite3_finalize(stmt);
    }

    return found;
}

} // namespace search
} // namespace jadedb
