#include "../src/services/search/bm25_index_persistence.h"
#include <iostream>
#include <cassert>
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

const std::string TEST_DB_PATH = "/tmp/test_bm25_persistence.db";

void cleanup_test_db() {
    std::filesystem::remove(TEST_DB_PATH);
}

// Test 1: Initialize and Create Schema
void test_initialize() {
    cleanup_test_db();

    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    assert_true(persistence.initialize(), "Initialize - Success");

    // Check if database file was created
    assert_true(std::filesystem::exists(TEST_DB_PATH), "Initialize - DB file created");

    cleanup_test_db();
    std::cout << "✓ test_initialize passed" << std::endl;
}

// Test 2: Save and Load Index
void test_save_load_index() {
    cleanup_test_db();

    // Create BM25 scorer and inverted index
    BM25Scorer scorer;
    InvertedIndex index;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "quick brown fox"),
        BM25Document("doc2", "lazy brown dog"),
        BM25Document("doc3", "quick lazy cat")
    };

    scorer.index_documents(docs);

    // Add to inverted index
    for (const auto& doc : docs) {
        std::vector<std::string> terms = scorer.tokenize(doc.text);
        index.add_document(doc.doc_id, terms, false);
    }

    // Save to persistence
    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    persistence.initialize();

    assert_true(persistence.save_index(scorer, index), "Save - Success");

    // Load into new instances
    BM25Scorer loaded_scorer;
    InvertedIndex loaded_index;

    assert_true(persistence.load_index(loaded_scorer, loaded_index), "Load - Success");

    // Verify loaded data
    assert_true(loaded_index.document_count() > 0, "Load - Documents loaded");
    assert_true(loaded_index.term_count() > 0, "Load - Terms loaded");

    cleanup_test_db();
    std::cout << "✓ test_save_load_index passed" << std::endl;
}

// Test 3: Index Exists
void test_index_exists() {
    cleanup_test_db();

    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    persistence.initialize();

    // Initially should not exist
    assert_true(!persistence.index_exists(), "Index exists - Initially false");

    // Save an index
    BM25Scorer scorer;
    InvertedIndex index;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "test document")
    };
    scorer.index_documents(docs);
    index.add_document("doc1", scorer.tokenize("test document"), false);

    persistence.save_index(scorer, index);

    // Now should exist
    assert_true(persistence.index_exists(), "Index exists - After save");

    cleanup_test_db();
    std::cout << "✓ test_index_exists passed" << std::endl;
}

// Test 4: Clear Index
void test_clear_index() {
    cleanup_test_db();

    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    persistence.initialize();

    // Save an index
    BM25Scorer scorer;
    InvertedIndex index;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "test document"),
        BM25Document("doc2", "another document")
    };
    scorer.index_documents(docs);

    for (const auto& doc : docs) {
        index.add_document(doc.doc_id, scorer.tokenize(doc.text), false);
    }

    persistence.save_index(scorer, index);
    assert_true(persistence.index_exists(), "Clear - Index exists before clear");

    // Clear the index
    assert_true(persistence.clear_index(), "Clear - Clear success");
    assert_true(!persistence.index_exists(), "Clear - Index does not exist after clear");

    cleanup_test_db();
    std::cout << "✓ test_clear_index passed" << std::endl;
}

// Test 5: Get Index Stats
void test_get_index_stats() {
    cleanup_test_db();

    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    persistence.initialize();

    // Save an index with known stats
    BM25Scorer scorer;
    InvertedIndex index;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "one two three"),
        BM25Document("doc2", "four five six")
    };
    scorer.index_documents(docs);

    for (const auto& doc : docs) {
        index.add_document(doc.doc_id, scorer.tokenize(doc.text), false);
    }

    persistence.save_index(scorer, index);

    // Get stats
    size_t total_docs = 0;
    size_t total_terms = 0;
    double avg_doc_length = 0.0;

    assert_true(persistence.get_index_stats(total_docs, total_terms, avg_doc_length),
                "Get stats - Success");

    assert_equals(2, total_docs, "Get stats - Total docs");
    assert_true(total_terms > 0, "Get stats - Total terms > 0");
    assert_true(avg_doc_length > 0.0, "Get stats - Avg doc length > 0");

    cleanup_test_db();
    std::cout << "✓ test_get_index_stats passed" << std::endl;
}

// Test 6: Get Last Update Time
void test_get_last_update_time() {
    cleanup_test_db();

    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    persistence.initialize();

    // Initially should be empty
    std::string timestamp = persistence.get_last_update_time();
    assert_true(timestamp.empty(), "Last update time - Initially empty");

    // Save an index
    BM25Scorer scorer;
    InvertedIndex index;

    std::vector<BM25Document> docs = {
        BM25Document("doc1", "test document")
    };
    scorer.index_documents(docs);
    index.add_document("doc1", scorer.tokenize("test document"), false);

    persistence.save_index(scorer, index);

    // Now should have timestamp
    timestamp = persistence.get_last_update_time();
    assert_true(!timestamp.empty(), "Last update time - Not empty after save");

    cleanup_test_db();
    std::cout << "✓ test_get_last_update_time passed" << std::endl;
}

// Test 7: Multiple Databases
void test_multiple_databases() {
    cleanup_test_db();

    BM25IndexPersistence persistence1("db1", TEST_DB_PATH);
    BM25IndexPersistence persistence2("db2", TEST_DB_PATH);

    persistence1.initialize();
    persistence2.initialize();

    // Save index for db1
    BM25Scorer scorer1;
    InvertedIndex index1;
    std::vector<BM25Document> docs1 = {
        BM25Document("doc1", "database one content")
    };
    scorer1.index_documents(docs1);
    index1.add_document("doc1", scorer1.tokenize("database one content"), false);
    persistence1.save_index(scorer1, index1);

    // Save index for db2
    BM25Scorer scorer2;
    InvertedIndex index2;
    std::vector<BM25Document> docs2 = {
        BM25Document("doc2", "database two content")
    };
    scorer2.index_documents(docs2);
    index2.add_document("doc2", scorer2.tokenize("database two content"), false);
    persistence2.save_index(scorer2, index2);

    // Both should exist
    assert_true(persistence1.index_exists(), "Multiple DBs - db1 exists");
    assert_true(persistence2.index_exists(), "Multiple DBs - db2 exists");

    // Clear db1
    persistence1.clear_index();

    // db1 should not exist, db2 should still exist
    assert_true(!persistence1.index_exists(), "Multiple DBs - db1 cleared");
    assert_true(persistence2.index_exists(), "Multiple DBs - db2 still exists");

    cleanup_test_db();
    std::cout << "✓ test_multiple_databases passed" << std::endl;
}

// Test 8: Empty Index Save/Load
void test_empty_index() {
    cleanup_test_db();

    BM25IndexPersistence persistence("test_db", TEST_DB_PATH);
    persistence.initialize();

    // Save empty index
    BM25Scorer scorer;
    InvertedIndex index;

    persistence.save_index(scorer, index);

    // Load empty index
    BM25Scorer loaded_scorer;
    InvertedIndex loaded_index;

    persistence.load_index(loaded_scorer, loaded_index);

    assert_equals(0, loaded_index.document_count(), "Empty index - Doc count");
    assert_equals(0, loaded_index.term_count(), "Empty index - Term count");

    cleanup_test_db();
    std::cout << "✓ test_empty_index passed" << std::endl;
}

int main() {
    std::cout << "Running BM25IndexPersistence Unit Tests..." << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        test_initialize();
        test_save_load_index();
        test_index_exists();
        test_clear_index();
        test_get_index_stats();
        test_get_last_update_time();
        test_multiple_databases();
        test_empty_index();

        std::cout << "===========================================" << std::endl;
        std::cout << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
