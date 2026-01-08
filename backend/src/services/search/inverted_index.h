#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <cstdint>

namespace jadedb {
namespace search {

/**
 * @brief Represents a single posting in the inverted index
 *
 * A posting contains information about a term's occurrence in a document.
 */
struct Posting {
    std::string doc_id;
    int term_frequency;
    std::vector<int> positions;  // Optional: positions for phrase search

    Posting() : term_frequency(0) {}
    Posting(const std::string& id, int freq)
        : doc_id(id), term_frequency(freq) {}
    Posting(const std::string& id, int freq, const std::vector<int>& pos)
        : doc_id(id), term_frequency(freq), positions(pos) {}
};

/**
 * @brief List of postings for a term
 */
struct PostingsList {
    std::vector<Posting> postings;

    PostingsList() = default;

    /**
     * @brief Add a posting to the list
     */
    void add_posting(const Posting& posting) {
        postings.push_back(posting);
    }

    /**
     * @brief Get posting for a specific document
     * @param doc_id Document ID
     * @return Pointer to posting if found, nullptr otherwise
     */
    const Posting* get_posting(const std::string& doc_id) const {
        for (const auto& posting : postings) {
            if (posting.doc_id == doc_id) {
                return &posting;
            }
        }
        return nullptr;
    }

    /**
     * @brief Get number of documents containing this term
     */
    size_t document_frequency() const {
        return postings.size();
    }
};

/**
 * @brief Statistics about the inverted index
 */
struct InvertedIndexStats {
    size_t total_documents;
    size_t total_terms;
    size_t total_postings;
    size_t memory_bytes;
    double avg_postings_per_term;

    InvertedIndexStats()
        : total_documents(0), total_terms(0), total_postings(0),
          memory_bytes(0), avg_postings_per_term(0.0) {}
};

/**
 * @brief Inverted index data structure for fast keyword lookup
 *
 * Maps terms to their postings lists, enabling efficient retrieval
 * of all documents containing a specific term.
 *
 * Structure:
 *   term1 -> [(doc1, freq, positions), (doc3, freq, positions), ...]
 *   term2 -> [(doc2, freq, positions), (doc4, freq, positions), ...]
 *   ...
 *
 * Performance targets:
 * - Index build: <5 minutes for 50K documents
 * - Lookup latency: <1ms per term
 * - Memory usage: <100MB for 50K documents
 */
class InvertedIndex {
public:
    InvertedIndex();
    ~InvertedIndex() = default;

    /**
     * @brief Add a document to the index
     *
     * @param doc_id Unique document identifier
     * @param terms List of terms (tokens) in the document
     * @param store_positions If true, store term positions for phrase search
     */
    void add_document(
        const std::string& doc_id,
        const std::vector<std::string>& terms,
        bool store_positions = false
    );

    /**
     * @brief Add a document with pre-computed term frequencies
     *
     * @param doc_id Unique document identifier
     * @param term_frequencies Map of term to frequency
     */
    void add_document_with_frequencies(
        const std::string& doc_id,
        const std::unordered_map<std::string, int>& term_frequencies
    );

    /**
     * @brief Get posting list for a term
     *
     * @param term The term to lookup
     * @return Const reference to PostingsList (empty if term not found)
     */
    const PostingsList& get_postings(const std::string& term) const;

    /**
     * @brief Check if a term exists in the index
     *
     * @param term The term to check
     * @return true if term exists
     */
    bool contains_term(const std::string& term) const;

    /**
     * @brief Get document frequency for a term
     *
     * Number of documents containing the term.
     *
     * @param term The term
     * @return Document frequency (0 if term not found)
     */
    size_t get_document_frequency(const std::string& term) const;

    /**
     * @brief Get all terms in the index
     *
     * @return Vector of all indexed terms
     */
    std::vector<std::string> get_all_terms() const;

    /**
     * @brief Get all document IDs in the index
     *
     * @return Vector of all indexed document IDs
     */
    std::vector<std::string> get_all_documents() const;

    /**
     * @brief Remove a document from the index
     *
     * @param doc_id Document ID to remove
     * @return true if document was found and removed
     */
    bool remove_document(const std::string& doc_id);

    /**
     * @brief Clear the entire index
     */
    void clear();

    /**
     * @brief Get index statistics
     *
     * @return InvertedIndexStats structure
     */
    InvertedIndexStats get_stats() const;

    /**
     * @brief Get number of unique terms in index
     */
    size_t term_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.size();
    }

    /**
     * @brief Get number of indexed documents
     */
    size_t document_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return doc_ids_.size();
    }

    /**
     * @brief Check if index is empty
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.empty();
    }

    /**
     * @brief Estimate memory usage in bytes
     *
     * @return Approximate memory usage
     */
    size_t estimate_memory_usage() const;

private:
    // Main inverted index: term -> PostingsList
    std::unordered_map<std::string, PostingsList> index_;

    // Set of all document IDs (for fast document_count and removal)
    std::unordered_set<std::string> doc_ids_;

    // Thread safety
    mutable std::mutex mutex_;

    // Empty postings list for returning when term not found
    static const PostingsList empty_postings_;

    /**
     * @brief Calculate term frequencies from term list
     *
     * @param terms List of terms
     * @param positions Optional output parameter for term positions
     * @return Map of term to frequency
     */
    std::unordered_map<std::string, int> calculate_term_frequencies(
        const std::vector<std::string>& terms,
        std::unordered_map<std::string, std::vector<int>>* positions = nullptr
    ) const;
};

} // namespace search
} // namespace jadedb
