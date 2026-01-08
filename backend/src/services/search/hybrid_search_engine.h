#pragma once

#include "bm25_scorer.h"
#include "inverted_index.h"
#include "bm25_index_persistence.h"
#include "score_fusion.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace jadedb {
namespace search {

/**
 * @brief Fusion method for hybrid search
 */
enum class FusionMethod {
    RRF,        // Reciprocal Rank Fusion
    LINEAR      // Weighted Linear Fusion
};

/**
 * @brief Configuration for hybrid search
 */
struct HybridSearchConfig {
    // Fusion method
    FusionMethod fusion_method = FusionMethod::RRF;

    // For RRF
    int rrf_k = 60;

    // For LINEAR fusion
    double alpha = 0.7;  // Weight for vector scores

    // Candidate pool sizes
    size_t vector_candidates = 100;
    size_t bm25_candidates = 100;

    // BM25 configuration
    BM25Config bm25_config;

    HybridSearchConfig() = default;
};

/**
 * @brief Result from hybrid search
 */
struct HybridSearchResult {
    std::string doc_id;
    double vector_score;
    double bm25_score;
    double hybrid_score;
    std::unordered_map<std::string, std::string> metadata;

    HybridSearchResult()
        : vector_score(0.0), bm25_score(0.0), hybrid_score(0.0) {}

    bool operator<(const HybridSearchResult& other) const {
        return hybrid_score > other.hybrid_score;  // Descending
    }
};

/**
 * @brief Hybrid search engine combining vector and keyword search
 *
 * Orchestrates:
 * 1. Vector similarity search (HNSW/IVF indexes)
 * 2. BM25 keyword search (inverted index)
 * 3. Score fusion (RRF or weighted linear)
 * 4. Result re-ranking
 *
 * Pipeline:
 *   Query → [Vector Search] → top-100 candidates
 *         → [BM25 Search]   → top-100 candidates
 *         → [Merge + Fusion] → top-K results
 */
class HybridSearchEngine {
public:
    /**
     * @brief Construct hybrid search engine
     * @param database_id Database identifier
     * @param config Hybrid search configuration
     */
    HybridSearchEngine(
        const std::string& database_id,
        const HybridSearchConfig& config = HybridSearchConfig()
    );

    ~HybridSearchEngine() = default;

    /**
     * @brief Build BM25 index from documents
     *
     * @param documents Vector of BM25 documents with text
     * @return true on success
     */
    bool build_bm25_index(const std::vector<BM25Document>& documents);

    /**
     * @brief Load BM25 index from persistence
     *
     * @param persistence_path Path to SQLite database
     * @return true on success
     */
    bool load_bm25_index(const std::string& persistence_path);

    /**
     * @brief Save BM25 index to persistence
     *
     * @param persistence_path Path to SQLite database
     * @return true on success
     */
    bool save_bm25_index(const std::string& persistence_path);

    /**
     * @brief Perform hybrid search
     *
     * Combines vector similarity and BM25 keyword search.
     *
     * @param query_text Text query for BM25
     * @param query_vector Vector query for similarity search
     * @param top_k Number of results to return
     * @return Vector of hybrid search results
     */
    std::vector<HybridSearchResult> search(
        const std::string& query_text,
        const std::vector<float>& query_vector,
        size_t top_k
    );

    /**
     * @brief Perform BM25-only search
     *
     * @param query_text Text query
     * @param top_k Number of results to return
     * @return Vector of results with BM25 scores
     */
    std::vector<HybridSearchResult> search_bm25_only(
        const std::string& query_text,
        size_t top_k
    );

    /**
     * @brief Set vector search results provider
     *
     * This function allows injection of vector search results.
     * In production, this would integrate with SimilaritySearchService.
     *
     * @param provider Function that performs vector search
     */
    using VectorSearchProvider = std::function<
        std::vector<SearchResult>(const std::vector<float>&, size_t)
    >;

    void set_vector_search_provider(VectorSearchProvider provider) {
        vector_search_provider_ = provider;
    }

    /**
     * @brief Check if BM25 index is ready
     */
    bool is_bm25_index_ready() const {
        return bm25_scorer_.get_document_count() > 0;
    }

    /**
     * @brief Get hybrid search configuration
     */
    const HybridSearchConfig& get_config() const {
        return config_;
    }

    /**
     * @brief Update hybrid search configuration
     */
    void set_config(const HybridSearchConfig& config) {
        config_ = config;
        score_fusion_.set_config(FusionConfig(config.rrf_k, config.alpha));
    }

    /**
     * @brief Get BM25 index statistics
     *
     * @param total_docs Output: total documents
     * @param total_terms Output: total terms
     * @param avg_doc_length Output: average document length
     */
    void get_bm25_stats(
        size_t& total_docs,
        size_t& total_terms,
        double& avg_doc_length
    ) const;

private:
    std::string database_id_;
    HybridSearchConfig config_;

    // BM25 components
    BM25Scorer bm25_scorer_;
    InvertedIndex inverted_index_;

    // Score fusion
    ScoreFusion score_fusion_;

    // Vector search provider (injected)
    VectorSearchProvider vector_search_provider_;

    /**
     * @brief Perform vector similarity search
     *
     * Uses injected provider or returns empty if not set.
     *
     * @param query_vector Query vector
     * @param top_k Number of results
     * @return Vector search results
     */
    std::vector<SearchResult> perform_vector_search(
        const std::vector<float>& query_vector,
        size_t top_k
    );

    /**
     * @brief Perform BM25 keyword search
     *
     * @param query_text Query text
     * @param top_k Number of results
     * @return BM25 search results
     */
    std::vector<SearchResult> perform_bm25_search(
        const std::string& query_text,
        size_t top_k
    );

    /**
     * @brief Convert SearchResult to HybridSearchResult
     */
    HybridSearchResult convert_to_hybrid_result(
        const SearchResult& result,
        double vector_score,
        double bm25_score
    ) const;

    /**
     * @brief Apply score fusion to results
     *
     * @param vector_results Results from vector search
     * @param bm25_results Results from BM25 search
     * @return Fused results
     */
    std::vector<SearchResult> apply_fusion(
        const std::vector<SearchResult>& vector_results,
        const std::vector<SearchResult>& bm25_results
    );
};

} // namespace search
} // namespace jadedb
