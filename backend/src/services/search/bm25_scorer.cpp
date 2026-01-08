#include "bm25_scorer.h"
#include <algorithm>
#include <numeric>

namespace jadedb {
namespace search {

BM25Scorer::BM25Scorer(const BM25Config& config)
    : config_(config),
      avg_doc_length_(0.0),
      total_docs_(0) {
    init_default_stop_words();
}

void BM25Scorer::init_default_stop_words() {
    // Common English stop words
    std::vector<std::string> default_stops = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "the", "this", "but", "they", "have",
        "had", "what", "when", "where", "who", "which", "why", "how", "or"
    };

    stop_words_.insert(default_stops.begin(), default_stops.end());
}

std::vector<std::string> BM25Scorer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;

    if (text.empty()) {
        return tokens;
    }

    std::string current_token;
    for (char ch : text) {
        // Convert to lowercase
        char lower_ch = std::tolower(static_cast<unsigned char>(ch));

        // Check if character is alphanumeric
        if (std::isalnum(static_cast<unsigned char>(lower_ch))) {
            current_token += lower_ch;
        } else {
            // Delimiter found, save current token if not empty
            if (!current_token.empty()) {
                // Filter out stop words
                if (stop_words_.find(current_token) == stop_words_.end()) {
                    tokens.push_back(current_token);
                }
                current_token.clear();
            }
        }
    }

    // Don't forget the last token
    if (!current_token.empty()) {
        if (stop_words_.find(current_token) == stop_words_.end()) {
            tokens.push_back(current_token);
        }
    }

    return tokens;
}

std::unordered_map<std::string, int> BM25Scorer::calculate_term_frequencies(
    const std::vector<std::string>& tokens) const {
    std::unordered_map<std::string, int> term_freqs;

    for (const auto& token : tokens) {
        term_freqs[token]++;
    }

    return term_freqs;
}

void BM25Scorer::index_documents(const std::vector<BM25Document>& documents) {
    // Clear existing index
    clear();

    total_docs_ = documents.size();

    if (total_docs_ == 0) {
        return;
    }

    size_t total_length = 0;

    // First pass: calculate term frequencies and document lengths
    for (const auto& doc : documents) {
        BM25Document indexed_doc = doc;

        // Tokenize document text
        std::vector<std::string> tokens = tokenize(doc.text);

        // Calculate term frequencies
        indexed_doc.term_frequencies = calculate_term_frequencies(tokens);
        indexed_doc.doc_length = tokens.size();

        total_length += indexed_doc.doc_length;

        // Store indexed document
        indexed_docs_[doc.doc_id] = indexed_doc;
    }

    // Calculate average document length
    avg_doc_length_ = static_cast<double>(total_length) / total_docs_;

    // Second pass: calculate document frequencies
    for (const auto& [doc_id, doc] : indexed_docs_) {
        for (const auto& [term, freq] : doc.term_frequencies) {
            doc_frequencies_[term]++;
        }
    }
}

double BM25Scorer::get_idf(const std::string& term) const {
    if (total_docs_ == 0) {
        return 0.0;
    }

    // Get document frequency for term
    int df = 0;
    auto it = doc_frequencies_.find(term);
    if (it != doc_frequencies_.end()) {
        df = it->second;
    }

    // IDF formula: log((N - df + 0.5) / (df + 0.5) + 1.0)
    double n = static_cast<double>(total_docs_);
    double idf = std::log((n - df + 0.5) / (df + 0.5) + 1.0);

    return idf;
}

double BM25Scorer::compute_term_score(const std::string& term, const BM25Document& doc) const {
    // Get term frequency in document
    int tf = 0;
    auto tf_it = doc.term_frequencies.find(term);
    if (tf_it != doc.term_frequencies.end()) {
        tf = tf_it->second;
    }

    if (tf == 0) {
        return 0.0;
    }

    // Get IDF for term
    double idf = get_idf(term);

    // BM25 formula for this term
    // score = IDF(term) × (f(term, doc) × (k1 + 1)) /
    //                     (f(term, doc) + k1 × (1 - b + b × |doc| / avgdl))

    double k1 = config_.k1;
    double b = config_.b;

    double doc_len = static_cast<double>(doc.doc_length);
    double tf_double = static_cast<double>(tf);

    double numerator = tf_double * (k1 + 1.0);
    double denominator = tf_double + k1 * (1.0 - b + b * doc_len / avg_doc_length_);

    double term_score = idf * (numerator / denominator);

    return term_score;
}

double BM25Scorer::score(const std::string& query_text, const std::string& doc_id) const {
    // Check if document exists
    auto doc_it = indexed_docs_.find(doc_id);
    if (doc_it == indexed_docs_.end()) {
        return 0.0;
    }

    const BM25Document& doc = doc_it->second;

    // Tokenize query
    std::vector<std::string> query_tokens = tokenize(query_text);

    if (query_tokens.empty()) {
        return 0.0;
    }

    // Calculate BM25 score as sum of term scores
    double total_score = 0.0;
    for (const auto& term : query_tokens) {
        total_score += compute_term_score(term, doc);
    }

    return total_score;
}

std::vector<std::pair<std::string, double>> BM25Scorer::score_all(
    const std::string& query_text) const {
    std::vector<std::pair<std::string, double>> results;

    // Tokenize query
    std::vector<std::string> query_tokens = tokenize(query_text);

    if (query_tokens.empty()) {
        return results;
    }

    // Score all documents
    for (const auto& [doc_id, doc] : indexed_docs_) {
        double score = 0.0;

        for (const auto& term : query_tokens) {
            score += compute_term_score(term, doc);
        }

        if (score > 0.0) {
            results.emplace_back(doc_id, score);
        }
    }

    // Sort by score descending
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return results;
}

void BM25Scorer::clear() {
    indexed_docs_.clear();
    doc_frequencies_.clear();
    avg_doc_length_ = 0.0;
    total_docs_ = 0;
}

void BM25Scorer::add_stop_words(const std::vector<std::string>& words) {
    stop_words_.insert(words.begin(), words.end());
}

void BM25Scorer::set_stop_words(const std::vector<std::string>& words) {
    stop_words_.clear();
    stop_words_.insert(words.begin(), words.end());
}

void BM25Scorer::restore_document(const std::string& doc_id, const BM25Document& doc) {
    indexed_docs_[doc_id] = doc;

    // Update document frequencies
    for (const auto& [term, freq] : doc.term_frequencies) {
        doc_frequencies_[term]++;
    }
}

void BM25Scorer::set_statistics(double avg_length, size_t total) {
    avg_doc_length_ = avg_length;
    total_docs_ = total;
}

} // namespace search
} // namespace jadedb
