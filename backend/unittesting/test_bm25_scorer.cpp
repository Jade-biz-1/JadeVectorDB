#include "../src/services/search/bm25_scorer.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

using namespace jadedb::search;

// Test utilities
void assert_near(double a, double b, double epsilon, const std::string& test_name) {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "FAIL: " << test_name << std::endl;
        std::cerr << "  Expected: " << a << ", Got: " << b << std::endl;
        assert(false);
    }
}

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

// Test 1: Tokenization - Basic
void test_tokenization_basic() {
    BM25Scorer scorer;

    std::string text = "The quick brown fox jumps over the lazy dog";
    auto tokens = scorer.tokenize(text);

    // "the" appears twice but is a stop word and should be filtered
    // Expected tokens: quick, brown, fox, jumps, over, lazy, dog (7 tokens)
    assert_equals(7, tokens.size(), "Tokenization - Basic count");

    // Check specific tokens
    assert_true(std::find(tokens.begin(), tokens.end(), "quick") != tokens.end(),
                "Tokenization - Contains 'quick'");
    assert_true(std::find(tokens.begin(), tokens.end(), "brown") != tokens.end(),
                "Tokenization - Contains 'brown'");
    assert_true(std::find(tokens.begin(), tokens.end(), "the") == tokens.end(),
                "Tokenization - Stop word 'the' removed");

    std::cout << "✓ test_tokenization_basic passed" << std::endl;
}

// Test 2: Tokenization - Edge cases
void test_tokenization_edge_cases() {
    BM25Scorer scorer;

    // Empty string
    auto tokens1 = scorer.tokenize("");
    assert_equals(0, tokens1.size(), "Tokenization - Empty string");

    // Only stop words
    auto tokens2 = scorer.tokenize("the and or");
    assert_equals(0, tokens2.size(), "Tokenization - Only stop words");

    // Punctuation and special characters
    auto tokens3 = scorer.tokenize("Product-A, Product-B! configuration?");
    assert_true(tokens3.size() >= 2, "Tokenization - Handles punctuation");
    assert_true(std::find(tokens3.begin(), tokens3.end(), "configuration") != tokens3.end(),
                "Tokenization - Extracts 'configuration'");

    // Numbers and alphanumeric
    auto tokens4 = scorer.tokenize("XYZ-100 ABC-200 product123");
    assert_true(tokens4.size() >= 3, "Tokenization - Handles alphanumeric");

    // Case insensitivity
    auto tokens5 = scorer.tokenize("Product PRODUCT product");
    assert_true(tokens5.size() > 0, "Tokenization - Case insensitive");
    for (const auto& token : tokens5) {
        assert_true(token == "product", "Tokenization - All lowercase");
    }

    std::cout << "✓ test_tokenization_edge_cases passed" << std::endl;
}

// Test 3: BM25 Scoring - Basic
void test_bm25_scoring_basic() {
    BM25Scorer scorer;

    // Create test documents
    std::vector<BM25Document> docs = {
        BM25Document("doc1", "Product-A configuration and setup procedure"),
        BM25Document("doc2", "Product-B installation manual"),
        BM25Document("doc3", "Product-A troubleshooting guide for configuration")
    };

    scorer.index_documents(docs);

    // Test document count
    assert_equals(3, scorer.get_document_count(), "BM25 - Document count");

    // Test query that matches doc1 and doc3
    double score1 = scorer.score("Product-A configuration", "doc1");
    double score2 = scorer.score("Product-A configuration", "doc2");
    double score3 = scorer.score("Product-A configuration", "doc3");

    // doc2 should have lowest score (no match)
    assert_true(score1 > score2, "BM25 - doc1 score > doc2 score");
    assert_true(score3 > score2, "BM25 - doc3 score > doc2 score");

    // doc1 and doc3 both match, scores should be positive
    assert_true(score1 > 0.0, "BM25 - doc1 has positive score");
    assert_true(score3 > 0.0, "BM25 - doc3 has positive score");

    std::cout << "✓ test_bm25_scoring_basic passed" << std::endl;
}

// Test 4: IDF Calculation
void test_idf_calculation() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "common word rare unique"),
        BM25Document("doc2", "common word another"),
        BM25Document("doc3", "common word test")
    };

    scorer.index_documents(docs);

    // "common" and "word" appear in all 3 documents
    double idf_common = scorer.get_idf("common");
    double idf_word = scorer.get_idf("word");

    // "rare" appears in only 1 document
    double idf_rare = scorer.get_idf("rare");

    // "nonexistent" doesn't appear in any document
    double idf_nonexistent = scorer.get_idf("nonexistent");

    // Rare terms should have higher IDF
    assert_true(idf_rare > idf_common, "IDF - Rare term has higher IDF");
    assert_true(idf_rare > idf_word, "IDF - Rare term has higher IDF than common");

    // IDF for common terms should be similar
    assert_near(idf_common, idf_word, 0.01, "IDF - Common terms have similar IDF");

    std::cout << "✓ test_idf_calculation passed" << std::endl;
}

// Test 5: Score All Documents
void test_score_all_documents() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "quick brown fox"),
        BM25Document("doc2", "lazy brown dog"),
        BM25Document("doc3", "quick lazy cat"),
        BM25Document("doc4", "fast brown horse")
    };

    scorer.index_documents(docs);

    auto results = scorer.score_all("brown quick");

    // Should return results in descending score order
    assert_true(results.size() >= 2, "ScoreAll - Returns multiple results");

    // Scores should be in descending order
    for (size_t i = 1; i < results.size(); i++) {
        assert_true(results[i-1].second >= results[i].second,
                    "ScoreAll - Results sorted by score descending");
    }

    // doc1 should rank high (has both "brown" and "quick")
    bool found_doc1 = false;
    for (const auto& [doc_id, score] : results) {
        if (doc_id == "doc1") {
            found_doc1 = true;
            assert_true(score > 0.0, "ScoreAll - doc1 has positive score");
        }
    }
    assert_true(found_doc1, "ScoreAll - doc1 found in results");

    std::cout << "✓ test_score_all_documents passed" << std::endl;
}

// Test 6: Empty Query
void test_empty_query() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "test document"),
        BM25Document("doc2", "another document")
    };

    scorer.index_documents(docs);

    // Empty query should return 0 score
    double score = scorer.score("", "doc1");
    assert_near(0.0, score, 0.001, "Empty query - Returns 0 score");

    auto results = scorer.score_all("");
    assert_equals(0, results.size(), "Empty query - score_all returns empty");

    std::cout << "✓ test_empty_query passed" << std::endl;
}

// Test 7: Nonexistent Document
void test_nonexistent_document() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "test document")
    };

    scorer.index_documents(docs);

    // Query for document that doesn't exist
    double score = scorer.score("test", "doc999");
    assert_near(0.0, score, 0.001, "Nonexistent doc - Returns 0 score");

    std::cout << "✓ test_nonexistent_document passed" << std::endl;
}

// Test 8: BM25 Parameters (k1 and b)
void test_bm25_parameters() {
    // Test with different k1 and b values
    BM25Config config1(1.5, 0.75);  // Default
    BM25Config config2(2.0, 0.5);   // Different parameters

    BM25Scorer scorer1(config1);
    BM25Scorer scorer2(config2);

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "short doc"),
        BM25Document("doc2", "this is a much longer document with many more words in it")
    };

    scorer1.index_documents(docs);
    scorer2.index_documents(docs);

    // Scores should differ due to different parameters
    double score1_doc1 = scorer1.score("document", "doc1");
    double score2_doc1 = scorer2.score("document", "doc1");

    // Different parameters may yield different scores
    // (not guaranteed to be different, but testing parameter acceptance)
    assert_true(score1_doc1 >= 0.0, "BM25 Params - scorer1 valid");
    assert_true(score2_doc1 >= 0.0, "BM25 Params - scorer2 valid");

    std::cout << "✓ test_bm25_parameters passed" << std::endl;
}

// Test 9: Custom Stop Words
void test_custom_stop_words() {
    BM25Scorer scorer;

    // Add custom stop words
    scorer.add_stop_words({"product", "configuration"});

    auto tokens = scorer.tokenize("product configuration setup");

    // "product" and "configuration" should be filtered out
    assert_true(std::find(tokens.begin(), tokens.end(), "product") == tokens.end(),
                "Custom stop words - 'product' filtered");
    assert_true(std::find(tokens.begin(), tokens.end(), "configuration") == tokens.end(),
                "Custom stop words - 'configuration' filtered");
    assert_true(std::find(tokens.begin(), tokens.end(), "setup") != tokens.end(),
                "Custom stop words - 'setup' kept");

    std::cout << "✓ test_custom_stop_words passed" << std::endl;
}

// Test 10: Clear Index
void test_clear_index() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "test document"),
        BM25Document("doc2", "another document")
    };

    scorer.index_documents(docs);
    assert_equals(2, scorer.get_document_count(), "Clear - Initial count");

    scorer.clear();
    assert_equals(0, scorer.get_document_count(), "Clear - Count after clear");
    assert_near(0.0, scorer.get_avg_doc_length(), 0.001, "Clear - Avg length reset");

    // Scoring after clear should return 0
    double score = scorer.score("test", "doc1");
    assert_near(0.0, score, 0.001, "Clear - Score after clear");

    std::cout << "✓ test_clear_index passed" << std::endl;
}

// Test 11: Average Document Length
void test_average_doc_length() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "one two"),           // 2 tokens
        BM25Document("doc2", "one two three four") // 4 tokens
    };

    scorer.index_documents(docs);

    // Average should be (2 + 4) / 2 = 3.0
    double avg_len = scorer.get_avg_doc_length();
    assert_near(3.0, avg_len, 0.1, "Average doc length calculation");

    std::cout << "✓ test_average_doc_length passed" << std::endl;
}

// Test 12: Exact Match Scoring
void test_exact_match_scoring() {
    BM25Scorer scorer;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "XYZ-100 manual"),
        BM25Document("doc2", "ABC-200 manual"),
        BM25Document("doc3", "general manual documentation")
    };

    scorer.index_documents(docs);

    // Query for exact code should score highest on matching doc
    double score1 = scorer.score("xyz 100", "doc1");
    double score2 = scorer.score("xyz 100", "doc2");
    double score3 = scorer.score("xyz 100", "doc3");

    assert_true(score1 > score2, "Exact match - doc1 > doc2");
    assert_true(score1 > score3, "Exact match - doc1 > doc3");

    std::cout << "✓ test_exact_match_scoring passed" << std::endl;
}

int main() {
    std::cout << "Running BM25Scorer Unit Tests..." << std::endl;
    std::cout << "=================================" << std::endl;

    try {
        test_tokenization_basic();
        test_tokenization_edge_cases();
        test_bm25_scoring_basic();
        test_idf_calculation();
        test_score_all_documents();
        test_empty_query();
        test_nonexistent_document();
        test_bm25_parameters();
        test_custom_stop_words();
        test_clear_index();
        test_average_doc_length();
        test_exact_match_scoring();

        std::cout << "=================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
