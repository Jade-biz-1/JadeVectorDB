#include "inverted_index.h"
#include <algorithm>
#include <numeric>

namespace jadedb {
namespace search {

// Static member initialization
const PostingsList InvertedIndex::empty_postings_;

InvertedIndex::InvertedIndex() {
}

std::unordered_map<std::string, int> InvertedIndex::calculate_term_frequencies(
    const std::vector<std::string>& terms,
    std::unordered_map<std::string, std::vector<int>>* positions) const {

    std::unordered_map<std::string, int> term_freqs;

    for (size_t i = 0; i < terms.size(); i++) {
        const std::string& term = terms[i];
        term_freqs[term]++;

        // Store positions if requested
        if (positions) {
            (*positions)[term].push_back(static_cast<int>(i));
        }
    }

    return term_freqs;
}

void InvertedIndex::add_document(
    const std::string& doc_id,
    const std::vector<std::string>& terms,
    bool store_positions) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Add document ID to set
    doc_ids_.insert(doc_id);

    if (terms.empty()) {
        return;
    }

    // Calculate term frequencies and optionally positions
    std::unordered_map<std::string, std::vector<int>> positions;
    std::unordered_map<std::string, int> term_freqs;

    if (store_positions) {
        term_freqs = calculate_term_frequencies(terms, &positions);
    } else {
        term_freqs = calculate_term_frequencies(terms, nullptr);
    }

    // Add postings to inverted index
    for (const auto& [term, freq] : term_freqs) {
        Posting posting(doc_id, freq);

        if (store_positions) {
            posting.positions = positions[term];
        }

        index_[term].add_posting(posting);
    }
}

void InvertedIndex::add_document_with_frequencies(
    const std::string& doc_id,
    const std::unordered_map<std::string, int>& term_frequencies) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Add document ID to set
    doc_ids_.insert(doc_id);

    // Add postings to inverted index
    for (const auto& [term, freq] : term_frequencies) {
        Posting posting(doc_id, freq);
        index_[term].add_posting(posting);
    }
}

const PostingsList& InvertedIndex::get_postings(const std::string& term) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = index_.find(term);
    if (it != index_.end()) {
        return it->second;
    }

    return empty_postings_;
}

bool InvertedIndex::contains_term(const std::string& term) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.find(term) != index_.end();
}

size_t InvertedIndex::get_document_frequency(const std::string& term) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = index_.find(term);
    if (it != index_.end()) {
        return it->second.document_frequency();
    }

    return 0;
}

std::vector<std::string> InvertedIndex::get_all_terms() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> terms;
    terms.reserve(index_.size());

    for (const auto& [term, postings] : index_) {
        terms.push_back(term);
    }

    return terms;
}

std::vector<std::string> InvertedIndex::get_all_documents() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> docs;
    docs.reserve(doc_ids_.size());

    for (const auto& doc_id : doc_ids_) {
        docs.push_back(doc_id);
    }

    return docs;
}

bool InvertedIndex::remove_document(const std::string& doc_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if document exists
    if (doc_ids_.find(doc_id) == doc_ids_.end()) {
        return false;
    }

    // Remove document from set
    doc_ids_.erase(doc_id);

    // Remove postings for this document from all terms
    for (auto& [term, postings_list] : index_) {
        auto& postings = postings_list.postings;

        // Remove posting with matching doc_id
        postings.erase(
            std::remove_if(postings.begin(), postings.end(),
                [&doc_id](const Posting& p) {
                    return p.doc_id == doc_id;
                }),
            postings.end()
        );
    }

    // Remove terms with empty postings lists
    for (auto it = index_.begin(); it != index_.end(); ) {
        if (it->second.postings.empty()) {
            it = index_.erase(it);
        } else {
            ++it;
        }
    }

    return true;
}

void InvertedIndex::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.clear();
    doc_ids_.clear();
}

InvertedIndexStats InvertedIndex::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    InvertedIndexStats stats;
    stats.total_documents = doc_ids_.size();
    stats.total_terms = index_.size();

    size_t total_postings = 0;
    for (const auto& [term, postings_list] : index_) {
        total_postings += postings_list.postings.size();
    }
    stats.total_postings = total_postings;

    if (stats.total_terms > 0) {
        stats.avg_postings_per_term = static_cast<double>(total_postings) / stats.total_terms;
    }

    stats.memory_bytes = estimate_memory_usage();

    return stats;
}

size_t InvertedIndex::estimate_memory_usage() const {
    // Note: This is called from get_stats which already holds the lock
    // Don't lock again to avoid deadlock

    size_t total_bytes = 0;

    // Size of index map structure
    total_bytes += index_.size() * sizeof(std::pair<std::string, PostingsList>);

    // Size of terms (keys)
    for (const auto& [term, postings_list] : index_) {
        total_bytes += term.capacity();

        // Size of postings
        for (const auto& posting : postings_list.postings) {
            total_bytes += sizeof(Posting);
            total_bytes += posting.doc_id.capacity();
            total_bytes += posting.positions.capacity() * sizeof(int);
        }
    }

    // Size of doc_ids set
    total_bytes += doc_ids_.size() * sizeof(std::string);
    for (const auto& doc_id : doc_ids_) {
        total_bytes += doc_id.capacity();
    }

    return total_bytes;
}

} // namespace search
} // namespace jadedb
