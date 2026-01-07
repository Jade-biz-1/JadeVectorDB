#include "../src/services/search/inverted_index.h"
#include <iostream>
#include <cassert>
#include <algorithm>

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

// Test 1: Add Document and Basic Lookup
void test_add_document_basic() {
    InvertedIndex index;

    std::vector<std::string> terms = {"quick", "brown", "fox"};
    index.add_document("doc1", terms);

    assert_equals(1, index.document_count(), "Add document - Document count");
    assert_equals(3, index.term_count(), "Add document - Term count");

    assert_true(index.contains_term("quick"), "Add document - Contains 'quick'");
    assert_true(index.contains_term("brown"), "Add document - Contains 'brown'");
    assert_true(index.contains_term("fox"), "Add document - Contains 'fox'");
    assert_true(!index.contains_term("dog"), "Add document - Does not contain 'dog'");

    std::cout << "✓ test_add_document_basic passed" << std::endl;
}

// Test 2: Multiple Documents
void test_multiple_documents() {
    InvertedIndex index;

    index.add_document("doc1", {"quick", "brown", "fox"});
    index.add_document("doc2", {"lazy", "brown", "dog"});
    index.add_document("doc3", {"quick", "lazy", "cat"});

    assert_equals(3, index.document_count(), "Multiple docs - Document count");
    assert_equals(6, index.term_count(), "Multiple docs - Term count");

    // "brown" appears in doc1 and doc2
    assert_equals(2, index.get_document_frequency("brown"), "Multiple docs - 'brown' df");

    // "quick" appears in doc1 and doc3
    assert_equals(2, index.get_document_frequency("quick"), "Multiple docs - 'quick' df");

    // "fox" appears only in doc1
    assert_equals(1, index.get_document_frequency("fox"), "Multiple docs - 'fox' df");

    std::cout << "✓ test_multiple_documents passed" << std::endl;
}

// Test 3: Get Postings
void test_get_postings() {
    InvertedIndex index;

    index.add_document("doc1", {"test", "test", "word"});  // "test" appears twice
    index.add_document("doc2", {"test", "another"});

    const PostingsList& postings = index.get_postings("test");

    assert_equals(2, postings.postings.size(), "Get postings - Posting count for 'test'");

    // Check term frequency
    bool found_doc1 = false;
    bool found_doc2 = false;

    for (const auto& posting : postings.postings) {
        if (posting.doc_id == "doc1") {
            found_doc1 = true;
            assert_equals(2, posting.term_frequency, "Get postings - doc1 term freq");
        } else if (posting.doc_id == "doc2") {
            found_doc2 = true;
            assert_equals(1, posting.term_frequency, "Get postings - doc2 term freq");
        }
    }

    assert_true(found_doc1, "Get postings - Found doc1");
    assert_true(found_doc2, "Get postings - Found doc2");

    std::cout << "✓ test_get_postings passed" << std::endl;
}

// Test 4: Store Positions
void test_store_positions() {
    InvertedIndex index;

    std::vector<std::string> terms = {"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
    index.add_document("doc1", terms, true);  // store_positions = true

    const PostingsList& postings = index.get_postings("the");
    assert_equals(1, postings.postings.size(), "Store positions - Posting count");

    const Posting& posting = postings.postings[0];
    assert_equals(2, posting.term_frequency, "Store positions - Term frequency");
    assert_equals(2, posting.positions.size(), "Store positions - Position count");

    // "the" appears at positions 0 and 6
    assert_true(std::find(posting.positions.begin(), posting.positions.end(), 0) != posting.positions.end(),
                "Store positions - Position 0");
    assert_true(std::find(posting.positions.begin(), posting.positions.end(), 6) != posting.positions.end(),
                "Store positions - Position 6");

    std::cout << "✓ test_store_positions passed" << std::endl;
}

// Test 5: Add Document with Frequencies
void test_add_with_frequencies() {
    InvertedIndex index;

    std::unordered_map<std::string, int> freq_map = {
        {"term1", 3},
        {"term2", 5},
        {"term3", 1}
    };

    index.add_document_with_frequencies("doc1", freq_map);

    assert_equals(1, index.document_count(), "Add with freq - Document count");
    assert_equals(3, index.term_count(), "Add with freq - Term count");

    const PostingsList& postings1 = index.get_postings("term1");
    assert_equals(3, postings1.postings[0].term_frequency, "Add with freq - term1 frequency");

    const PostingsList& postings2 = index.get_postings("term2");
    assert_equals(5, postings2.postings[0].term_frequency, "Add with freq - term2 frequency");

    std::cout << "✓ test_add_with_frequencies passed" << std::endl;
}

// Test 6: Remove Document
void test_remove_document() {
    InvertedIndex index;

    index.add_document("doc1", {"quick", "brown", "fox"});
    index.add_document("doc2", {"lazy", "brown", "dog"});

    assert_equals(2, index.document_count(), "Remove doc - Initial count");
    assert_equals(2, index.get_document_frequency("brown"), "Remove doc - Initial 'brown' df");

    bool removed = index.remove_document("doc1");
    assert_true(removed, "Remove doc - Removal successful");

    assert_equals(1, index.document_count(), "Remove doc - Count after removal");
    assert_equals(1, index.get_document_frequency("brown"), "Remove doc - 'brown' df after removal");

    // "fox" and "quick" should be completely removed
    assert_true(!index.contains_term("fox"), "Remove doc - 'fox' removed");
    assert_true(!index.contains_term("quick"), "Remove doc - 'quick' removed");

    // "brown" should still exist (in doc2)
    assert_true(index.contains_term("brown"), "Remove doc - 'brown' still exists");

    std::cout << "✓ test_remove_document passed" << std::endl;
}

// Test 7: Remove Nonexistent Document
void test_remove_nonexistent() {
    InvertedIndex index;

    index.add_document("doc1", {"test"});

    bool removed = index.remove_document("doc999");
    assert_true(!removed, "Remove nonexistent - Returns false");

    assert_equals(1, index.document_count(), "Remove nonexistent - Count unchanged");

    std::cout << "✓ test_remove_nonexistent passed" << std::endl;
}

// Test 8: Clear Index
void test_clear() {
    InvertedIndex index;

    index.add_document("doc1", {"quick", "brown", "fox"});
    index.add_document("doc2", {"lazy", "brown", "dog"});

    assert_equals(2, index.document_count(), "Clear - Initial doc count");
    assert_equals(5, index.term_count(), "Clear - Initial term count");

    index.clear();

    assert_equals(0, index.document_count(), "Clear - Doc count after clear");
    assert_equals(0, index.term_count(), "Clear - Term count after clear");
    assert_true(index.empty(), "Clear - Index is empty");

    std::cout << "✓ test_clear passed" << std::endl;
}

// Test 9: Get All Terms
void test_get_all_terms() {
    InvertedIndex index;

    index.add_document("doc1", {"alpha", "beta", "gamma"});
    index.add_document("doc2", {"delta", "beta"});

    std::vector<std::string> all_terms = index.get_all_terms();

    assert_equals(4, all_terms.size(), "Get all terms - Count");

    // Check all expected terms are present
    assert_true(std::find(all_terms.begin(), all_terms.end(), "alpha") != all_terms.end(),
                "Get all terms - Contains 'alpha'");
    assert_true(std::find(all_terms.begin(), all_terms.end(), "beta") != all_terms.end(),
                "Get all terms - Contains 'beta'");
    assert_true(std::find(all_terms.begin(), all_terms.end(), "gamma") != all_terms.end(),
                "Get all terms - Contains 'gamma'");
    assert_true(std::find(all_terms.begin(), all_terms.end(), "delta") != all_terms.end(),
                "Get all terms - Contains 'delta'");

    std::cout << "✓ test_get_all_terms passed" << std::endl;
}

// Test 10: Get All Documents
void test_get_all_documents() {
    InvertedIndex index;

    index.add_document("doc1", {"test"});
    index.add_document("doc2", {"test"});
    index.add_document("doc3", {"test"});

    std::vector<std::string> all_docs = index.get_all_documents();

    assert_equals(3, all_docs.size(), "Get all docs - Count");

    assert_true(std::find(all_docs.begin(), all_docs.end(), "doc1") != all_docs.end(),
                "Get all docs - Contains doc1");
    assert_true(std::find(all_docs.begin(), all_docs.end(), "doc2") != all_docs.end(),
                "Get all docs - Contains doc2");
    assert_true(std::find(all_docs.begin(), all_docs.end(), "doc3") != all_docs.end(),
                "Get all docs - Contains doc3");

    std::cout << "✓ test_get_all_documents passed" << std::endl;
}

// Test 11: Index Statistics
void test_index_stats() {
    InvertedIndex index;

    index.add_document("doc1", {"term1", "term2", "term3"});
    index.add_document("doc2", {"term1", "term2"});
    index.add_document("doc3", {"term1"});

    InvertedIndexStats stats = index.get_stats();

    assert_equals(3, stats.total_documents, "Stats - Total documents");
    assert_equals(3, stats.total_terms, "Stats - Total terms");
    assert_equals(6, stats.total_postings, "Stats - Total postings");

    // avg_postings_per_term = 6 / 3 = 2.0
    assert_true(stats.avg_postings_per_term > 1.9 && stats.avg_postings_per_term < 2.1,
                "Stats - Avg postings per term");

    assert_true(stats.memory_bytes > 0, "Stats - Memory usage > 0");

    std::cout << "✓ test_index_stats passed" << std::endl;
}

// Test 12: Empty Index Operations
void test_empty_index() {
    InvertedIndex index;

    assert_true(index.empty(), "Empty index - is empty");
    assert_equals(0, index.document_count(), "Empty index - doc count");
    assert_equals(0, index.term_count(), "Empty index - term count");

    const PostingsList& postings = index.get_postings("nonexistent");
    assert_equals(0, postings.postings.size(), "Empty index - postings for nonexistent term");

    assert_equals(0, index.get_document_frequency("nonexistent"), "Empty index - df for nonexistent");

    std::cout << "✓ test_empty_index passed" << std::endl;
}

// Test 13: Large Document
void test_large_document() {
    InvertedIndex index;

    // Create a document with many terms
    std::vector<std::string> large_doc;
    for (int i = 0; i < 1000; i++) {
        large_doc.push_back("term" + std::to_string(i % 100));  // 100 unique terms, repeated
    }

    index.add_document("large_doc", large_doc);

    assert_equals(1, index.document_count(), "Large doc - Document count");
    assert_equals(100, index.term_count(), "Large doc - Term count");

    // Each term should appear 10 times
    const PostingsList& postings = index.get_postings("term0");
    assert_equals(1, postings.postings.size(), "Large doc - Postings count");
    assert_equals(10, postings.postings[0].term_frequency, "Large doc - Term frequency");

    std::cout << "✓ test_large_document passed" << std::endl;
}

// Test 14: Duplicate Documents
void test_duplicate_documents() {
    InvertedIndex index;

    // Add same document twice (should be treated as updates in real usage)
    index.add_document("doc1", {"test", "word"});
    index.add_document("doc1", {"another", "word"});

    // Document count should still be 1 (same doc_id)
    assert_equals(1, index.document_count(), "Duplicate docs - Document count");

    // Both sets of terms should be indexed
    // (In real usage, you'd remove old doc first, but index allows duplicates)
    assert_true(index.contains_term("test"), "Duplicate docs - Contains 'test'");
    assert_true(index.contains_term("word"), "Duplicate docs - Contains 'word'");
    assert_true(index.contains_term("another"), "Duplicate docs - Contains 'another'");

    std::cout << "✓ test_duplicate_documents passed" << std::endl;
}

// Test 15: Memory Estimation
void test_memory_estimation() {
    InvertedIndex index;

    size_t initial_memory = index.estimate_memory_usage();

    // Add documents
    for (int i = 0; i < 10; i++) {
        std::vector<std::string> terms = {"term1", "term2", "term3"};
        index.add_document("doc" + std::to_string(i), terms);
    }

    size_t after_memory = index.estimate_memory_usage();

    assert_true(after_memory > initial_memory, "Memory estimation - Increased after adding docs");

    std::cout << "✓ test_memory_estimation passed" << std::endl;
}

int main() {
    std::cout << "Running InvertedIndex Unit Tests..." << std::endl;
    std::cout << "====================================" << std::endl;

    try {
        test_add_document_basic();
        test_multiple_documents();
        test_get_postings();
        test_store_positions();
        test_add_with_frequencies();
        test_remove_document();
        test_remove_nonexistent();
        test_clear();
        test_get_all_terms();
        test_get_all_documents();
        test_index_stats();
        test_empty_index();
        test_large_document();
        test_duplicate_documents();
        test_memory_estimation();

        std::cout << "====================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
