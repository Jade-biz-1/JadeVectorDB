#include "hybrid_search_engine.h"
#include <algorithm>

namespace jadedb {
namespace search {

HybridSearchEngine::HybridSearchEngine(
    const std::string& database_id,
    const HybridSearchConfig& config)
    : database_id_(database_id),
      config_(config),
      bm25_scorer_(config.bm25_config),
      score_fusion_(FusionConfig(config.rrf_k, config.alpha)) {
}

bool HybridSearchEngine::build_bm25_index(const std::vector<BM25Document>& documents) {
    if (documents.empty()) {
        return false;
    }

    // Index documents with BM25 scorer
    bm25_scorer_.index_documents(documents);

    // Build inverted index
    for (const auto& doc : documents) {
        std::vector<std::string> terms = bm25_scorer_.tokenize(doc.text);
        inverted_index_.add_document(doc.doc_id, terms, false);
    }

    return true;
}

bool HybridSearchEngine::load_bm25_index(const std::string& persistence_path) {
    BM25IndexPersistence persistence(database_id_, persistence_path);

    if (!persistence.initialize()) {
        return false;
    }

    if (!persistence.index_exists()) {
        return false;
    }

    return persistence.load_index(bm25_scorer_, inverted_index_);
}

bool HybridSearchEngine::save_bm25_index(const std::string& persistence_path) {
    BM25IndexPersistence persistence(database_id_, persistence_path);

    if (!persistence.initialize()) {
        return false;
    }

    return persistence.save_index(bm25_scorer_, inverted_index_);
}

std::vector<SearchResult> HybridSearchEngine::perform_vector_search(
    const std::vector<float>& query_vector,
    size_t top_k) {

    if (vector_search_provider_) {
        return vector_search_provider_(query_vector, top_k);
    }

    // No vector search provider, return empty
    return std::vector<SearchResult>();
}

std::vector<SearchResult> HybridSearchEngine::perform_bm25_search(
    const std::string& query_text,
    size_t top_k) {

    // Use BM25 scorer to get all scores
    auto bm25_results = bm25_scorer_.score_all(query_text);

    // Convert to SearchResult format
    std::vector<SearchResult> results;
    results.reserve(bm25_results.size());

    for (const auto& [doc_id, score] : bm25_results) {
        results.emplace_back(doc_id, score);
    }

    // Return top-K
    return score_fusion_.get_top_k(results, top_k);
}

std::vector<SearchResult> HybridSearchEngine::apply_fusion(
    const std::vector<SearchResult>& vector_results,
    const std::vector<SearchResult>& bm25_results) {

    if (config_.fusion_method == FusionMethod::RRF) {
        // Reciprocal Rank Fusion
        return score_fusion_.reciprocal_rank_fusion(
            {vector_results, bm25_results},
            config_.rrf_k
        );
    } else {
        // Weighted Linear Fusion
        return score_fusion_.weighted_linear_fusion(
            vector_results,
            bm25_results,
            config_.alpha,
            NormalizationMethod::MIN_MAX
        );
    }
}

HybridSearchResult HybridSearchEngine::convert_to_hybrid_result(
    const SearchResult& result,
    double vector_score,
    double bm25_score) const {

    HybridSearchResult hybrid_result;
    hybrid_result.doc_id = result.doc_id;
    hybrid_result.vector_score = vector_score;
    hybrid_result.bm25_score = bm25_score;
    hybrid_result.hybrid_score = result.score;

    return hybrid_result;
}

std::vector<HybridSearchResult> HybridSearchEngine::search(
    const std::string& query_text,
    const std::vector<float>& query_vector,
    size_t top_k,
    bool enable_reranking,
    size_t rerank_top_n) {

    std::vector<HybridSearchResult> results;

    // For two-stage retrieval: retrieve more candidates if reranking is enabled
    size_t candidate_count = enable_reranking ? rerank_top_n : top_k;

    // Step 1: Vector search (get candidates) - only if query_vector is provided
    std::vector<SearchResult> vector_results;
    if (!query_vector.empty()) {
        vector_results = perform_vector_search(
            query_vector,
            config_.vector_candidates
        );
    }

    // Step 2: BM25 search (get candidates) - only if query_text is provided
    std::vector<SearchResult> bm25_results;
    if (!query_text.empty()) {
        bm25_results = perform_bm25_search(
            query_text,
            config_.bm25_candidates
        );
    }

    // If both are empty, return empty
    if (vector_results.empty() && bm25_results.empty()) {
        return results;
    }

    // If only one has results, use that
    if (vector_results.empty()) {
        // BM25 only
        auto top_results = score_fusion_.get_top_k(bm25_results, candidate_count);
        for (const auto& result : top_results) {
            results.push_back(convert_to_hybrid_result(result, 0.0, result.score));
        }
        // Apply reranking if enabled
        if (enable_reranking && reranking_provider_ && results.size() > 1) {
            results = reranking_provider_(query_text, results);
            // Limit to top_k after reranking
            if (results.size() > top_k) {
                results.resize(top_k);
            }
        }
        return results;
    }

    if (bm25_results.empty()) {
        // Vector only
        auto top_results = score_fusion_.get_top_k(vector_results, candidate_count);
        for (const auto& result : top_results) {
            results.push_back(convert_to_hybrid_result(result, result.score, 0.0));
        }
        // Apply reranking if enabled
        if (enable_reranking && reranking_provider_ && results.size() > 1) {
            results = reranking_provider_(query_text, results);
            // Limit to top_k after reranking
            if (results.size() > top_k) {
                results.resize(top_k);
            }
        }
        return results;
    }

    // Debug: Check if we have results
    #ifdef DEBUG_HYBRID_SEARCH
    std::cerr << "Vector results: " << vector_results.size() << std::endl;
    std::cerr << "BM25 results: " << bm25_results.size() << std::endl;
    for (const auto& r : vector_results) {
        std::cerr << "  Vec: " << r.doc_id << " = " << r.score << std::endl;
    }
    for (const auto& r : bm25_results) {
        std::cerr << "  BM25: " << r.doc_id << " = " << r.score << std::endl;
    }
    #endif

    // Step 3: Apply score fusion
    std::vector<SearchResult> fused_results = apply_fusion(vector_results, bm25_results);

    // Step 4: Get top candidates (top_k or rerank_top_n)
    auto top_results = score_fusion_.get_top_k(fused_results, candidate_count);

    // Step 5: Build score maps for metadata - MUST happen BEFORE we convert
    // These maps contain the ORIGINAL scores before fusion normalization
    std::unordered_map<std::string, double> vector_score_map;
    std::unordered_map<std::string, double> bm25_score_map;

    for (const auto& result : vector_results) {
        vector_score_map[result.doc_id] = result.score;
    }

    for (const auto& result : bm25_results) {
        bm25_score_map[result.doc_id] = result.score;
    }

    // Step 6: Convert to HybridSearchResult
    for (const auto& result : top_results) {
        double vec_score = 0.0;
        double bm25_score = 0.0;

        auto vec_it = vector_score_map.find(result.doc_id);
        if (vec_it != vector_score_map.end()) {
            vec_score = vec_it->second;
        }

        auto bm25_it = bm25_score_map.find(result.doc_id);
        if (bm25_it != bm25_score_map.end()) {
            bm25_score = bm25_it->second;
        }

        // Fallback: if both scores are 0 but hybrid score is positive,
        // distribute the hybrid score based on which sources had results
        if (vec_score == 0.0 && bm25_score == 0.0 && result.score > 0.0) {
            if (!vector_score_map.empty() && !bm25_score_map.empty()) {
                // Both sources active - split the hybrid score
                vec_score = result.score * config_.alpha;
                bm25_score = result.score * (1.0 - config_.alpha);
            } else if (!vector_score_map.empty()) {
                // Only vector search active
                vec_score = result.score;
            } else if (!bm25_score_map.empty()) {
                // Only BM25 search active
                bm25_score = result.score;
            }
        }

        results.push_back(convert_to_hybrid_result(result, vec_score, bm25_score));
    }

    // Step 7: Optional reranking stage (two-stage retrieval)
    if (enable_reranking && reranking_provider_ && results.size() > 1) {
        // Call reranking provider with query text and candidates
        results = reranking_provider_(query_text, results);

        // Limit to top_k after reranking
        if (results.size() > top_k) {
            results.resize(top_k);
        }
    } else if (results.size() > top_k) {
        // No reranking: just limit to top_k
        results.resize(top_k);
    }

    return results;
}

std::vector<HybridSearchResult> HybridSearchEngine::search_bm25_only(
    const std::string& query_text,
    size_t top_k) {

    std::vector<HybridSearchResult> results;

    // Perform BM25 search
    std::vector<SearchResult> bm25_results = perform_bm25_search(query_text, top_k);

    // Convert to HybridSearchResult
    for (const auto& result : bm25_results) {
        results.push_back(convert_to_hybrid_result(result, 0.0, result.score));
    }

    return results;
}

void HybridSearchEngine::get_bm25_stats(
    size_t& total_docs,
    size_t& total_terms,
    double& avg_doc_length) const {

    total_docs = bm25_scorer_.get_document_count();
    total_terms = inverted_index_.term_count();
    avg_doc_length = bm25_scorer_.get_avg_doc_length();
}

} // namespace search
} // namespace jadedb
