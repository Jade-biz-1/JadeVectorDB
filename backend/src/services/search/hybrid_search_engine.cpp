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
    size_t top_k) {

    std::vector<HybridSearchResult> results;

    // Step 1: Vector search (get candidates)
    std::vector<SearchResult> vector_results = perform_vector_search(
        query_vector,
        config_.vector_candidates
    );

    // Step 2: BM25 search (get candidates)
    std::vector<SearchResult> bm25_results = perform_bm25_search(
        query_text,
        config_.bm25_candidates
    );

    // If both are empty, return empty
    if (vector_results.empty() && bm25_results.empty()) {
        return results;
    }

    // If only one has results, use that
    if (vector_results.empty()) {
        // BM25 only
        auto top_results = score_fusion_.get_top_k(bm25_results, top_k);
        for (const auto& result : top_results) {
            results.push_back(convert_to_hybrid_result(result, 0.0, result.score));
        }
        return results;
    }

    if (bm25_results.empty()) {
        // Vector only
        auto top_results = score_fusion_.get_top_k(vector_results, top_k);
        for (const auto& result : top_results) {
            results.push_back(convert_to_hybrid_result(result, result.score, 0.0));
        }
        return results;
    }

    // Step 3: Apply score fusion
    std::vector<SearchResult> fused_results = apply_fusion(vector_results, bm25_results);

    // Step 4: Get top-K
    auto top_results = score_fusion_.get_top_k(fused_results, top_k);

    // Step 5: Build score maps for metadata
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

        results.push_back(convert_to_hybrid_result(result, vec_score, bm25_score));
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
