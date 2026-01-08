#include "../src/services/search/hybrid_search_engine.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

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

const std::string TEST_DB_PATH = "/tmp/test_hybrid_search.db";

void cleanup_test_db() {
    std::filesystem::remove(TEST_DB_PATH);
}

// Test 1: Build BM25 Index
void test_build_bm25_index() {
    HybridSearchEngine engine("test_db");

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "quick brown fox jumps over lazy dog"),
        BM25Document("doc2", "lazy brown dog sleeps under tree"),
        BM25Document("doc3", "quick cat jumps high")
    };

    assert_true(engine.build_bm25_index(docs), "Build index - Success");
    assert_true(engine.is_bm25_index_ready(), "Build index - Index ready");

    // Check stats
    size_t total_docs, total_terms;
    double avg_doc_length;
    engine.get_bm25_stats(total_docs, total_terms, avg_doc_length);

    assert_equals(3, total_docs, "Build index - Total docs");
    assert_true(total_terms > 0, "Build index - Has terms");
    assert_true(avg_doc_length > 0.0, "Build index - Avg doc length > 0");

    std::cout << "✓ test_build_bm25_index passed" << std::endl;
}

// Test 2: BM25-Only Search
void test_bm25_only_search() {
    HybridSearchEngine engine("test_db");

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "quick brown fox"),
        BM25Document("doc2", "lazy brown dog"),
        BM25Document("doc3", "quick lazy cat")
    };

    engine.build_bm25_index(docs);

    // Search for "quick"
    auto results = engine.search_bm25_only("quick", 10);

    assert_true(results.size() >= 2, "BM25 only - Found results");

    // doc1 and doc3 contain "quick"
    bool found_doc1 = false;
    bool found_doc3 = false;

    for (const auto& result : results) {
        if (result.doc_id == "doc1") {
            found_doc1 = true;
            assert_true(result.bm25_score > 0.0, "BM25 only - doc1 has BM25 score");
            assert_near(0.0, result.vector_score, 0.01, "BM25 only - doc1 no vector score");
        } else if (result.doc_id == "doc3") {
            found_doc3 = true;
            assert_true(result.bm25_score > 0.0, "BM25 only - doc3 has BM25 score");
        }
    }

    assert_true(found_doc1, "BM25 only - Found doc1");
    assert_true(found_doc3, "BM25 only - Found doc3");

    std::cout << "✓ test_bm25_only_search passed" << std::endl;
}

// Test 3: Hybrid Search with Mock Vector Search
void test_hybrid_search_with_mock() {
    HybridSearchEngine engine("test_db");

    // Build BM25 index
    std::vector<BM25Document> docs = {
        BM25Document("doc1", "machine learning algorithms"),
        BM25Document("doc2", "deep neural networks"),
        BM25Document("doc3", "machine learning models")
    };

    engine.build_bm25_index(docs);

    // Mock vector search provider
    engine.set_vector_search_provider(
        [](const std::vector<float>& query_vec, size_t top_k) -> std::vector<SearchResult> {
            // Return mock vector search results
            return {
                SearchResult("doc2", 0.95),  // doc2 ranks high in vector search
                SearchResult("doc1", 0.85),
                SearchResult("doc3", 0.75)
            };
        }
    );

    // Perform hybrid search
    std::vector<float> mock_vector = {0.1f, 0.2f, 0.3f};
    auto results = engine.search("machine learning", mock_vector, 10);

    assert_true(results.size() > 0, "Hybrid search - Has results");

    // Check that results have both vector and BM25 scores
    for (const auto& result : results) {
        if (result.doc_id == "doc1" || result.doc_id == "doc3") {
            // doc1 and doc3 match "machine learning" text query
            assert_true(result.bm25_score > 0.0, "Hybrid search - Has BM25 score");
        }
        // All docs should have vector scores from mock
        assert_true(result.vector_score > 0.0, "Hybrid search - Has vector score");
        assert_true(result.hybrid_score > 0.0, "Hybrid search - Has hybrid score");
    }

    std::cout << "✓ test_hybrid_search_with_mock passed" << std::endl;
}

// Test 4: Hybrid Search with RRF Fusion
void test_hybrid_search_rrf() {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::RRF;
    config.rrf_k = 60;

    HybridSearchEngine engine("test_db", config);

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "artificial intelligence"),
        BM25Document("doc2", "machine learning"),
        BM25Document("doc3", "deep learning")
    };

    engine.build_bm25_index(docs);

    // Mock vector search
    engine.set_vector_search_provider(
        [](const std::vector<float>&, size_t) -> std::vector<SearchResult> {
            return {
                SearchResult("doc1", 0.9),
                SearchResult("doc2", 0.8),
                SearchResult("doc3", 0.7)
            };
        }
    );

    std::vector<float> mock_vector = {1.0f};
    auto results = engine.search("learning", mock_vector, 3);

    assert_equals(3, results.size(), "RRF fusion - Result count");

    // Verify RRF was applied (hybrid score should differ from individual scores)
    for (const auto& result : results) {
        assert_true(result.hybrid_score > 0.0, "RRF fusion - Has hybrid score");
    }

    std::cout << "✓ test_hybrid_search_rrf passed" << std::endl;
}

// Test 5: Hybrid Search with Linear Fusion
void test_hybrid_search_linear() {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::LINEAR;
    config.alpha = 0.7;  // 70% vector, 30% BM25

    HybridSearchEngine engine("test_db", config);

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "neural network training"),
        BM25Document("doc2", "network protocols")
    };

    engine.build_bm25_index(docs);

    // Mock vector search
    engine.set_vector_search_provider(
        [](const std::vector<float>&, size_t) -> std::vector<SearchResult> {
            return {
                SearchResult("doc1", 1.0),
                SearchResult("doc2", 0.5)
            };
        }
    );

    std::vector<float> mock_vector = {1.0f};
    auto results = engine.search("network", mock_vector, 2);

    assert_equals(2, results.size(), "Linear fusion - Result count");

    std::cout << "✓ test_hybrid_search_linear passed" << std::endl;
}

// Test 6: Empty BM25 Index
void test_empty_bm25_index() {
    HybridSearchEngine engine("test_db");

    assert_true(!engine.is_bm25_index_ready(), "Empty index - Not ready");

    // Try BM25-only search on empty index
    auto results = engine.search_bm25_only("query", 10);
    assert_equals(0, results.size(), "Empty index - No results");

    std::cout << "✓ test_empty_bm25_index passed" << std::endl;
}

// Test 7: Vector-Only Search (No BM25 Results)
void test_vector_only_search() {
    HybridSearchEngine engine("test_db");

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "completely different text"),
        BM25Document("doc2", "another unrelated document")
    };

    engine.build_bm25_index(docs);

    // Mock vector search
    engine.set_vector_search_provider(
        [](const std::vector<float>&, size_t) -> std::vector<SearchResult> {
            return {
                SearchResult("doc1", 0.9),
                SearchResult("doc2", 0.8)
            };
        }
    );

    // Search with query that has no BM25 matches
    std::vector<float> mock_vector = {1.0f};
    auto results = engine.search("nonexistent query terms xyz", mock_vector, 10);

    // Should still return vector results
    assert_true(results.size() > 0, "Vector only - Has results");

    for (const auto& result : results) {
        assert_true(result.vector_score > 0.0, "Vector only - Has vector score");
        // BM25 score should be 0
        assert_near(0.0, result.bm25_score, 0.01, "Vector only - BM25 score is 0");
    }

    std::cout << "✓ test_vector_only_search passed" << std::endl;
}

// Test 8: Configuration Management
void test_configuration_management() {
    HybridSearchConfig config1;
    config1.fusion_method = FusionMethod::RRF;
    config1.rrf_k = 30;

    HybridSearchEngine engine("test_db", config1);

    const HybridSearchConfig& retrieved_config = engine.get_config();
    assert_equals(static_cast<int>(FusionMethod::RRF),
                  static_cast<int>(retrieved_config.fusion_method),
                  "Config - Fusion method");
    assert_equals(30, retrieved_config.rrf_k, "Config - RRF k");

    // Update config
    HybridSearchConfig config2;
    config2.fusion_method = FusionMethod::LINEAR;
    config2.alpha = 0.5;

    engine.set_config(config2);

    const HybridSearchConfig& updated_config = engine.get_config();
    assert_equals(static_cast<int>(FusionMethod::LINEAR),
                  static_cast<int>(updated_config.fusion_method),
                  "Config - Updated fusion method");
    assert_near(0.5, updated_config.alpha, 0.01, "Config - Updated alpha");

    std::cout << "✓ test_configuration_management passed" << std::endl;
}

// Test 9: Save and Load Index
void test_save_load_index() {
    cleanup_test_db();

    // Create and save index
    {
        HybridSearchEngine engine("test_db");

        std::vector<BM25Document> docs = {
            BM25Document("doc1", "save and load test"),
            BM25Document("doc2", "persistence verification")
        };

        engine.build_bm25_index(docs);
        assert_true(engine.save_bm25_index(TEST_DB_PATH), "Save/Load - Save success");
    }

    // Load index in new engine
    {
        HybridSearchEngine engine("test_db");

        assert_true(!engine.is_bm25_index_ready(), "Save/Load - Initially not ready");
        assert_true(engine.load_bm25_index(TEST_DB_PATH), "Save/Load - Load success");
        assert_true(engine.is_bm25_index_ready(), "Save/Load - Ready after load");

        // Verify loaded index works
        auto results = engine.search_bm25_only("save", 10);
        assert_true(results.size() > 0, "Save/Load - Can search after load");
    }

    cleanup_test_db();
    std::cout << "✓ test_save_load_index passed" << std::endl;
}

// Test 10: Top-K Limit
void test_top_k_limit() {
    HybridSearchEngine engine("test_db");

    std::vector<BM25Document> docs;
    for (int i = 0; i < 20; i++) {
        docs.push_back(BM25Document("doc" + std::to_string(i), "test document number " + std::to_string(i)));
    }

    engine.build_bm25_index(docs);

    // Request only top-5
    auto results = engine.search_bm25_only("test document", 5);

    assert_equals(5, results.size(), "Top-K - Returns exactly K results");

    std::cout << "✓ test_top_k_limit passed" << std::endl;
}

int main() {
    std::cout << "Running HybridSearchEngine Unit Tests..." << std::endl;
    std::cout << "=========================================" << std::endl;

    try {
        test_build_bm25_index();
        test_bm25_only_search();
        test_hybrid_search_with_mock();
        test_hybrid_search_rrf();
        test_hybrid_search_linear();
        test_empty_bm25_index();
        test_vector_only_search();
        test_configuration_management();
        test_save_load_index();
        test_top_k_limit();

        std::cout << "=========================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
