#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>

namespace jadedb {
namespace search {

/**
 * @brief Configuration parameters for BM25 scoring
 */
struct BM25Config {
    double k1 = 1.5;        // Term frequency saturation parameter
    double b = 0.75;        // Length normalization parameter

    BM25Config() = default;
    BM25Config(double k1_val, double b_val) : k1(k1_val), b(b_val) {}
};

/**
 * @brief Document representation for BM25 indexing
 */
struct BM25Document {
    std::string doc_id;
    std::string text;
    std::unordered_map<std::string, int> term_frequencies;
    size_t doc_length;

    BM25Document() : doc_length(0) {}
    BM25Document(const std::string& id, const std::string& content)
        : doc_id(id), text(content), doc_length(0) {}
};

/**
 * @brief BM25 scoring engine for keyword-based relevance ranking
 *
 * Implements the BM25 (Best Matching 25) algorithm for text retrieval.
 * BM25 is a probabilistic relevance framework that ranks documents based
 * on query term frequencies while accounting for document length and
 * term importance (IDF).
 *
 * Formula:
 * BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) /
 *                          (f(qi, d) + k1 × (1 - b + b × |d| / avgdl))
 *
 * where:
 * - q = query terms
 * - d = document
 * - f(qi, d) = frequency of term qi in document d
 * - |d| = length of document d (in tokens)
 * - avgdl = average document length in collection
 * - k1 = term frequency saturation parameter (default 1.5)
 * - b = length normalization parameter (default 0.75)
 * - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1.0)
 * - N = total number of documents
 * - df(qi) = number of documents containing qi
 */
class BM25Scorer {
public:
    /**
     * @brief Construct a new BM25Scorer object
     * @param config BM25 configuration parameters
     */
    explicit BM25Scorer(const BM25Config& config = BM25Config());

    /**
     * @brief Tokenize text into terms
     *
     * Tokenization process:
     * 1. Convert to lowercase
     * 2. Split on whitespace and punctuation
     * 3. Remove stop words
     * 4. Filter out empty tokens
     *
     * @param text Input text to tokenize
     * @return std::vector<std::string> Vector of tokens
     */
    std::vector<std::string> tokenize(const std::string& text) const;

    /**
     * @brief Index a collection of documents for BM25 scoring
     *
     * Builds internal data structures:
     * - Term frequencies for each document
     * - Document frequencies for each term
     * - Average document length
     *
     * @param documents Vector of documents to index
     */
    void index_documents(const std::vector<BM25Document>& documents);

    /**
     * @brief Score a document against a query using BM25
     *
     * @param query_text Query string
     * @param doc_id Document ID to score
     * @return double BM25 relevance score (higher is better)
     */
    double score(const std::string& query_text, const std::string& doc_id) const;

    /**
     * @brief Score all indexed documents against a query
     *
     * @param query_text Query string
     * @return std::vector<std::pair<std::string, double>> Vector of (doc_id, score) pairs
     */
    std::vector<std::pair<std::string, double>> score_all(const std::string& query_text) const;

    /**
     * @brief Get IDF (Inverse Document Frequency) for a term
     *
     * IDF formula: log((N - df(t) + 0.5) / (df(t) + 0.5) + 1.0)
     *
     * @param term The term to get IDF for
     * @return double IDF value
     */
    double get_idf(const std::string& term) const;

    /**
     * @brief Get the number of indexed documents
     * @return size_t Number of documents
     */
    size_t get_document_count() const { return indexed_docs_.size(); }

    /**
     * @brief Get average document length
     * @return double Average document length in tokens
     */
    double get_avg_doc_length() const { return avg_doc_length_; }

    /**
     * @brief Check if a document is indexed
     * @param doc_id Document ID
     * @return bool True if document is indexed
     */
    bool is_indexed(const std::string& doc_id) const {
        return indexed_docs_.find(doc_id) != indexed_docs_.end();
    }

    /**
     * @brief Clear all indexed data
     */
    void clear();

    /**
     * @brief Add custom stop words to the default list
     * @param words Vector of stop words to add
     */
    void add_stop_words(const std::vector<std::string>& words);

    /**
     * @brief Set custom stop words (replaces default list)
     * @param words Vector of stop words
     */
    void set_stop_words(const std::vector<std::string>& words);

    /**
     * @brief Restore indexed document state (for persistence loading)
     * @param doc_id Document ID
     * @param doc Document to restore
     */
    void restore_document(const std::string& doc_id, const BM25Document& doc);

    /**
     * @brief Set statistics for persistence restoration
     * @param avg_length Average document length
     * @param total Total number of documents
     */
    void set_statistics(double avg_length, size_t total);

private:
    BM25Config config_;

    // Indexed documents (doc_id -> document)
    std::unordered_map<std::string, BM25Document> indexed_docs_;

    // Document frequency: term -> number of documents containing term
    std::unordered_map<std::string, int> doc_frequencies_;

    // Average document length
    double avg_doc_length_;

    // Total number of documents
    size_t total_docs_;

    // Stop words set (English common words)
    std::unordered_set<std::string> stop_words_;

    /**
     * @brief Initialize default English stop words
     */
    void init_default_stop_words();

    /**
     * @brief Calculate term frequencies for a document
     * @param tokens Vector of tokens
     * @return std::unordered_map<std::string, int> Term frequency map
     */
    std::unordered_map<std::string, int> calculate_term_frequencies(
        const std::vector<std::string>& tokens) const;

    /**
     * @brief Compute BM25 score for a single term
     * @param term Query term
     * @param doc Document to score
     * @return double Term-specific BM25 contribution
     */
    double compute_term_score(const std::string& term, const BM25Document& doc) const;
};

} // namespace search
} // namespace jadedb
