#include "../src/services/search/score_fusion.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace jadedb::search;

// Test utilities
void assert_true(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << std::endl;
        assert(false);
    }
}

void assert_equals(size_t expected, size_t actual, const std::string& test_name) {
    if (expected != actual) {
        std::cerr << "FAIL: " << test_name << std::endl;
        std::cerr << "  Expected: " << expected << ", Got: " << actual << std::endl;
        assert(false);
    }
}

void assert_near(double expected, double actual, double epsilon, const std::string& test_name) {
    if (std::abs(expected - actual) > epsilon) {
        std::cerr << "FAIL: " << test_name << std::endl;
        std::cerr << "  Expected: " << expected << ", Got: " << actual << std::endl;
        assert(false);
    }
}

// Test 1: RRF Basic
void test_rrf_basic() {
    ScoreFusion fusion;

    // Create two result lists
    std::vector<SearchResult> vector_results = {
        {"doc1", 0.9},
        {"doc2", 0.8},
        {"doc3", 0.7}
    };

    std::vector<SearchResult> bm25_results = {
        {"doc2", 5.0},
        {"doc3", 4.0},
        {"doc1", 3.0}
    };

    std::vector<std::vector<SearchResult>> results_list = {vector_results, bm25_results};

    auto fused = fusion.reciprocal_rank_fusion(results_list, 60);

    assert_equals(3, fused.size(), "RRF Basic - Result count");

    // doc2 should rank high (rank 2 in vector, rank 1 in BM25)
    // doc1 should rank next (rank 1 in vector, rank 3 in BM25)
    // doc3 should rank last (rank 3 in vector, rank 2 in BM25)

    assert_true(fused[0].doc_id == "doc2" || fused[0].doc_id == "doc1",
                "RRF Basic - Top result is doc2 or doc1");

    std::cout << "✓ test_rrf_basic passed" << std::endl;
}

// Test 2: RRF with K parameter
void test_rrf_with_k() {
    ScoreFusion fusion;

    std::vector<SearchResult> results1 = {{"doc1", 1.0}, {"doc2", 0.9}};
    std::vector<SearchResult> results2 = {{"doc2", 1.0}, {"doc1", 0.9}};

    // Test with different k values
    auto fused_k60 = fusion.reciprocal_rank_fusion({results1, results2}, 60);
    auto fused_k10 = fusion.reciprocal_rank_fusion({results1, results2}, 10);

    assert_equals(2, fused_k60.size(), "RRF K - Result count k=60");
    assert_equals(2, fused_k10.size(), "RRF K - Result count k=10");

    // Scores should be different with different k
    assert_true(fused_k60[0].score != fused_k10[0].score, "RRF K - Scores differ with k");

    std::cout << "✓ test_rrf_with_k passed" << std::endl;
}

// Test 3: Min-Max Normalization
void test_normalize_min_max() {
    ScoreFusion fusion;

    std::vector<SearchResult> results = {
        {"doc1", 10.0},
        {"doc2", 5.0},
        {"doc3", 2.0}
    };

    fusion.normalize_min_max(results);

    // After normalization: doc1=1.0, doc2=0.375, doc3=0.0
    assert_near(1.0, results[0].score, 0.01, "Min-Max - Max score");
    assert_near(0.0, results[2].score, 0.01, "Min-Max - Min score");
    assert_true(results[1].score > 0.0 && results[1].score < 1.0, "Min-Max - Middle score in range");

    std::cout << "✓ test_normalize_min_max passed" << std::endl;
}

// Test 4: Min-Max with Same Scores
void test_normalize_min_max_same_scores() {
    ScoreFusion fusion;

    std::vector<SearchResult> results = {
        {"doc1", 5.0},
        {"doc2", 5.0},
        {"doc3", 5.0}
    };

    fusion.normalize_min_max(results);

    // All scores should be 1.0 when all are the same
    for (const auto& result : results) {
        assert_near(1.0, result.score, 0.01, "Min-Max Same - All scores 1.0");
    }

    std::cout << "✓ test_normalize_min_max_same_scores passed" << std::endl;
}

// Test 5: Z-Score Normalization
void test_normalize_z_score() {
    ScoreFusion fusion;

    std::vector<SearchResult> results = {
        {"doc1", 10.0},
        {"doc2", 5.0},
        {"doc3", 2.0}
    };

    fusion.normalize_z_score(results);

    // After z-score: mean should be close to 0
    double sum = 0.0;
    for (const auto& result : results) {
        sum += result.score;
    }
    double mean = sum / results.size();

    assert_near(0.0, mean, 0.01, "Z-Score - Mean close to 0");

    std::cout << "✓ test_normalize_z_score passed" << std::endl;
}

// Test 6: Weighted Linear Fusion
void test_weighted_linear_fusion() {
    ScoreFusion fusion;

    std::vector<SearchResult> vector_results = {
        {"doc1", 0.9},
        {"doc2", 0.7},
        {"doc3", 0.5}
    };

    std::vector<SearchResult> bm25_results = {
        {"doc2", 10.0},
        {"doc1", 8.0},
        {"doc3", 6.0}
    };

    // Alpha = 0.7 means 70% vector, 30% BM25
    auto fused = fusion.weighted_linear_fusion(
        vector_results, bm25_results, 0.7, NormalizationMethod::MIN_MAX);

    assert_equals(3, fused.size(), "Weighted Linear - Result count");

    // All results should have scores between 0 and 1
    for (const auto& result : fused) {
        assert_true(result.score >= 0.0 && result.score <= 1.0,
                    "Weighted Linear - Scores in [0, 1]");
    }

    std::cout << "✓ test_weighted_linear_fusion passed" << std::endl;
}

// Test 7: Weighted Linear with Alpha=1.0
void test_weighted_linear_alpha_extremes() {
    ScoreFusion fusion;

    std::vector<SearchResult> vector_results = {
        {"doc1", 1.0},
        {"doc2", 0.5}
    };

    std::vector<SearchResult> bm25_results = {
        {"doc2", 10.0},
        {"doc1", 1.0}
    };

    // Alpha = 1.0 means 100% vector, 0% BM25
    auto fused_alpha1 = fusion.weighted_linear_fusion(
        vector_results, bm25_results, 1.0, NormalizationMethod::MIN_MAX);

    // doc1 should rank higher (has higher vector score)
    assert_true(fused_alpha1[0].doc_id == "doc1", "Alpha 1.0 - doc1 first");

    // Alpha = 0.0 means 0% vector, 100% BM25
    auto fused_alpha0 = fusion.weighted_linear_fusion(
        vector_results, bm25_results, 0.0, NormalizationMethod::MIN_MAX);

    // doc2 should rank higher (has higher BM25 score)
    assert_true(fused_alpha0[0].doc_id == "doc2", "Alpha 0.0 - doc2 first");

    std::cout << "✓ test_weighted_linear_alpha_extremes passed" << std::endl;
}

// Test 8: Merge Results
void test_merge_results() {
    ScoreFusion fusion;

    std::vector<SearchResult> results1 = {
        {"doc1", 0.9},
        {"doc2", 0.8}
    };

    std::vector<SearchResult> results2 = {
        {"doc2", 0.95},  // Higher score for doc2
        {"doc3", 0.7}
    };

    auto merged = fusion.merge_results({results1, results2});

    assert_equals(3, merged.size(), "Merge - Result count");

    // Check that doc2 has the higher score (0.95)
    bool found_doc2 = false;
    for (const auto& result : merged) {
        if (result.doc_id == "doc2") {
            found_doc2 = true;
            assert_near(0.95, result.score, 0.01, "Merge - doc2 has max score");
        }
    }
    assert_true(found_doc2, "Merge - doc2 found");

    std::cout << "✓ test_merge_results passed" << std::endl;
}

// Test 9: Get Top K
void test_get_top_k() {
    ScoreFusion fusion;

    std::vector<SearchResult> results = {
        {"doc1", 0.9},
        {"doc2", 0.8},
        {"doc3", 0.7},
        {"doc4", 0.6},
        {"doc5", 0.5}
    };

    auto top3 = fusion.get_top_k(results, 3);

    assert_equals(3, top3.size(), "Top K - Result count");
    assert_true(top3[0].doc_id == "doc1", "Top K - First result");
    assert_true(top3[1].doc_id == "doc2", "Top K - Second result");
    assert_true(top3[2].doc_id == "doc3", "Top K - Third result");

    std::cout << "✓ test_get_top_k passed" << std::endl;
}

// Test 10: Get Top K with K > Size
void test_get_top_k_larger() {
    ScoreFusion fusion;

    std::vector<SearchResult> results = {
        {"doc1", 0.9},
        {"doc2", 0.8}
    };

    auto top10 = fusion.get_top_k(results, 10);

    assert_equals(2, top10.size(), "Top K Larger - Returns all results");

    std::cout << "✓ test_get_top_k_larger passed" << std::endl;
}

// Test 11: Empty Results
void test_empty_results() {
    ScoreFusion fusion;

    std::vector<SearchResult> empty;

    fusion.normalize_min_max(empty);
    fusion.normalize_z_score(empty);

    auto top_k = fusion.get_top_k(empty, 10);
    assert_equals(0, top_k.size(), "Empty - Top K returns empty");

    std::cout << "✓ test_empty_results passed" << std::endl;
}

// Test 12: Single Result
void test_single_result() {
    ScoreFusion fusion;

    std::vector<SearchResult> single = {{"doc1", 5.0}};

    fusion.normalize_min_max(single);
    assert_near(1.0, single[0].score, 0.01, "Single - Min-max normalizes to 1.0");

    std::cout << "✓ test_single_result passed" << std::endl;
}

// Test 13: Fusion Config
void test_fusion_config() {
    FusionConfig config(30, 0.5);
    ScoreFusion fusion(config);

    const FusionConfig& retrieved_config = fusion.get_config();

    assert_equals(30, retrieved_config.rrf_k, "Config - RRF k");
    assert_near(0.5, retrieved_config.alpha, 0.01, "Config - Alpha");

    // Update config
    FusionConfig new_config(100, 0.8);
    fusion.set_config(new_config);

    const FusionConfig& updated_config = fusion.get_config();
    assert_equals(100, updated_config.rrf_k, "Config - Updated RRF k");
    assert_near(0.8, updated_config.alpha, 0.01, "Config - Updated alpha");

    std::cout << "✓ test_fusion_config passed" << std::endl;
}

// Test 14: Documents Only in One Source
void test_documents_only_in_one_source() {
    ScoreFusion fusion;

    std::vector<SearchResult> vector_results = {
        {"doc1", 0.9},
        {"doc2", 0.7}
    };

    std::vector<SearchResult> bm25_results = {
        {"doc3", 10.0},
        {"doc4", 8.0}
    };

    // No overlap between sources
    auto fused = fusion.weighted_linear_fusion(
        vector_results, bm25_results, 0.7, NormalizationMethod::MIN_MAX);

    assert_equals(4, fused.size(), "No Overlap - All docs included");

    std::cout << "✓ test_documents_only_in_one_source passed" << std::endl;
}

int main() {
    std::cout << "Running ScoreFusion Unit Tests..." << std::endl;
    std::cout << "==================================" << std::endl;

    try {
        test_rrf_basic();
        test_rrf_with_k();
        test_normalize_min_max();
        test_normalize_min_max_same_scores();
        test_normalize_z_score();
        test_weighted_linear_fusion();
        test_weighted_linear_alpha_extremes();
        test_merge_results();
        test_get_top_k();
        test_get_top_k_larger();
        test_empty_results();
        test_single_result();
        test_fusion_config();
        test_documents_only_in_one_source();

        std::cout << "==================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
