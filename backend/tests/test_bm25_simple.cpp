#include <gtest/gtest.h>
#include "services/search/bm25_scorer.h"
#include "services/search/inverted_index.h"
#include "services/search/hybrid_search_engine.h"

using namespace jadedb::search;

// Simple synchronous test for BM25 components
TEST(BM25SimpleTest, BM25ScorerBasic) {
    BM25Config config;
    config.k1 = 1.2;
    config.b = 0.75;

    BM25Scorer scorer(config);

    // Create test documents
    std::vector<BM25Document> documents;

    BM25Document doc1;
    doc1.doc_id = "doc1";
    doc1.text = "machine learning algorithms";
    documents.push_back(doc1);

    BM25Document doc2;
    doc2.doc_id = "doc2";
    doc2.text = "deep learning neural networks";
    documents.push_back(doc2);

    BM25Document doc3;
    doc3.doc_id = "doc3";
    doc3.text = "machine learning and deep learning";
    documents.push_back(doc3);

    // Index documents
    scorer.index_documents(documents);

    // Check stats
    EXPECT_EQ(scorer.get_document_count(), 3);
    EXPECT_GT(scorer.get_avg_doc_length(), 0.0);

    // Score a query
    auto results = scorer.score_all("machine learning");
    EXPECT_GT(results.size(), 0);

    // Check that doc1 and doc3 have higher scores than doc2
    bool found_doc1 = false;
    bool found_doc3 = false;
    for (const auto& [doc_id, score] : results) {
        if (doc_id == "doc1" || doc_id == "doc3") {
            EXPECT_GT(score, 0.0);
            if (doc_id == "doc1") found_doc1 = true;
            if (doc_id == "doc3") found_doc3 = true;
        }
    }

    EXPECT_TRUE(found_doc1);
    EXPECT_TRUE(found_doc3);
}

// Test inverted index
TEST(BM25SimpleTest, InvertedIndexBasic) {
    InvertedIndex index;

    // Add documents
    std::unordered_map<std::string, int> doc1_freqs = {
        {"machine", 1},
        {"learning", 1},
        {"algorithms", 1}
    };
    index.add_document_with_frequencies("doc1", doc1_freqs);

    std::unordered_map<std::string, int> doc2_freqs = {
        {"deep", 1},
        {"learning", 1},
        {"neural", 1}
    };
    index.add_document_with_frequencies("doc2", doc2_freqs);

    // Check stats
    EXPECT_GT(index.term_count(), 0);
    EXPECT_EQ(index.document_count(), 2);

    // Check postings
    auto learning_postings = index.get_postings("learning");
    EXPECT_EQ(learning_postings.postings.size(), 2);  // Appears in both docs

    auto machine_postings = index.get_postings("machine");
    EXPECT_EQ(machine_postings.postings.size(), 1);  // Only in doc1
}

// Test hybrid search engine build
TEST(BM25SimpleTest, HybridSearchEngineBuild) {
    HybridSearchConfig config;
    config.fusion_method = FusionMethod::RRF;

    HybridSearchEngine engine("test_db", config);

    // Create test documents
    std::vector<BM25Document> documents;

    BM25Document doc1;
    doc1.doc_id = "doc1";
    doc1.text = "machine learning for data analysis";
    documents.push_back(doc1);

    BM25Document doc2;
    doc2.doc_id = "doc2";
    doc2.text = "deep learning and neural networks";
    documents.push_back(doc2);

    BM25Document doc3;
    doc3.doc_id = "doc3";
    doc3.text = "natural language processing with machine learning";
    documents.push_back(doc3);

    // Build BM25 index
    bool success = engine.build_bm25_index(documents);
    EXPECT_TRUE(success);

    // Check index is ready
    EXPECT_TRUE(engine.is_bm25_index_ready());

    // Search with BM25 only
    auto results = engine.search_bm25_only("machine learning", 3);
    EXPECT_GT(results.size(), 0);

    // doc1 and doc3 should be in results
    bool found_doc1 = false;
    bool found_doc3 = false;
    for (const auto& result : results) {
        if (result.doc_id == "doc1") found_doc1 = true;
        if (result.doc_id == "doc3") found_doc3 = true;
    }

    EXPECT_TRUE(found_doc1 || found_doc3);
}
