# Hybrid Search Specification

**Feature**: Hybrid Search (Vector + Keyword BM25)
**Version**: 1.0
**Date**: January 6, 2026
**Status**: Proposed

---

## Executive Summary

Implement hybrid search combining vector similarity search with BM25 keyword search to improve retrieval accuracy, especially for exact matches (model numbers, part codes, technical terms) while maintaining semantic search capabilities.

---

## Motivation

### Problem Statement

Pure vector similarity search, while excellent for semantic matching, has limitations:

1. **Exact Match Weakness**: Poor at finding exact identifiers like "Product-A" when query is "ProductA"
2. **Keyword Mismatch**: May miss documents with exact keyword matches but different semantic context
3. **Domain-Specific Terms**: Specialized terminology often requires exact string matching
4. **Alphanumeric Codes**: Product codes and identifiers don't embed well semantically

### Industry Adoption

Leading vector databases support hybrid search:
- **Weaviate**: BM25 + vector fusion
- **Qdrant**: Hybrid search with score normalization
- **Milvus**: Dense + sparse vector support
- **Pinecone**: Metadata filtering + vector search

### Use Cases for JadeVectorDB

1. **RAG Applications**: Document search with identifiers and codes
2. **E-commerce**: Product search with SKUs and descriptions
3. **Code Search**: Exact function names + semantic similarity
4. **Knowledge Base**: Exact terminology + contextual relevance

---

## Technical Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Search Pipeline                   │
└─────────────────────────────────────────────────────────────┘

User Query: "Product-A configuration setup procedure"
     │
     ├─────────────────────────────────────────┐
     │                                         │
     ▼                                         ▼
┌──────────────────────┐            ┌──────────────────────┐
│  Vector Search Path  │            │  Keyword Search Path │
│                      │            │                      │
│  1. Embed query      │            │  1. Tokenize query   │
│  2. HNSW search      │            │  2. BM25 scoring     │
│  3. Cosine similarity│            │  3. Inverted index   │
│  4. Get top-K        │            │  4. Get top-K        │
│     vectors          │            │     documents        │
└──────────────────────┘            └──────────────────────┘
     │                                         │
     │    Results with                  Results with      │
     │    similarity scores             BM25 scores       │
     │                                                     │
     └──────────────────┬─────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │  Score Fusion        │
              │                      │
              │  - Normalize scores  │
              │  - Combine with α    │
              │  - Re-rank           │
              │  - Return top-K      │
              └──────────────────────┘
                        │
                        ▼
              Final Ranked Results
```

### Components

#### 1. BM25 Scoring Engine

**Purpose**: Keyword-based relevance scoring

**Algorithm**: BM25 (Best Matching 25)
```
BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) /
                         (f(qi, d) + k1 × (1 - b + b × |d| / avgdl))

where:
- q = query terms
- d = document
- IDF = Inverse Document Frequency
- f(qi, d) = term frequency of qi in d
- |d| = document length
- avgdl = average document length
- k1 = term saturation parameter (default: 1.5)
- b = length normalization parameter (default: 0.75)
```

**Implementation**:
```cpp
// src/search/bm25_scorer.h

#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace jadevectordb {
namespace search {

struct BM25Config {
    float k1 = 1.5f;        // Term saturation
    float b = 0.75f;        // Length normalization
    size_t max_tokens = 512; // Max tokens per doc
};

class BM25Scorer {
public:
    BM25Scorer(const BM25Config& config = BM25Config());

    // Build inverted index from documents
    void index_documents(
        const std::vector<std::string>& doc_ids,
        const std::vector<std::string>& doc_texts
    );

    // Score documents for a query
    std::vector<std::pair<std::string, float>> score(
        const std::string& query,
        size_t top_k = 10
    );

    // Get IDF for a term
    float get_idf(const std::string& term) const;

private:
    BM25Config config_;

    // Inverted index: term -> [(doc_id, term_freq)]
    std::unordered_map<std::string, std::vector<std::pair<std::string, int>>> inverted_index_;

    // Document frequencies
    std::unordered_map<std::string, int> doc_freq_;

    // Document lengths
    std::unordered_map<std::string, int> doc_lengths_;

    float avg_doc_length_;
    size_t total_docs_;

    // Tokenization
    std::vector<std::string> tokenize(const std::string& text);

    // IDF calculation
    float calculate_idf(const std::string& term) const;

    // BM25 score for single document
    float score_document(
        const std::string& doc_id,
        const std::vector<std::string>& query_terms
    );
};

} // namespace search
} // namespace jadevectordb
```

#### 2. Inverted Index

**Purpose**: Fast keyword lookup

**Structure**:
```cpp
// Token -> PostingsList
std::unordered_map<std::string, PostingsList> inverted_index_;

struct PostingsList {
    std::vector<Posting> postings;
};

struct Posting {
    std::string doc_id;
    int term_frequency;
    std::vector<int> positions; // Optional: for phrase search
};
```

**Storage**:
- In-memory for fast access
- Persisted to disk (SQLite or memory-mapped file)
- Incremental updates on document insertion

#### 3. Score Fusion

**Purpose**: Combine vector and BM25 scores

**Methods**:

**a) Reciprocal Rank Fusion (RRF)** - Recommended
```
RRF(d) = Σ 1 / (k + rank_vector(d)) + 1 / (k + rank_bm25(d))

where k = 60 (standard constant)
```

**b) Weighted Linear Combination**
```
hybrid_score(d) = α × norm_vector_score(d) + (1 - α) × norm_bm25_score(d)

where α ∈ [0, 1] (default: 0.7 for vector-heavy)
```

**c) Convex Combination (Configurable)**
```cpp
struct HybridSearchConfig {
    enum FusionMethod {
        RRF,                    // Reciprocal Rank Fusion
        LINEAR,                 // Weighted linear
        MULTIPLICATIVE          // Score multiplication
    };

    FusionMethod fusion_method = RRF;
    float alpha = 0.7f;         // Weight for vector search (LINEAR mode)
    int rrf_k = 60;             // RRF constant
    bool normalize_scores = true;
};
```

**Implementation**:
```cpp
// src/search/hybrid_search.h

#pragma once
#include "bm25_scorer.h"
#include "../services/similarity_search_service.h"

namespace jadevectordb {
namespace search {

struct HybridResult {
    std::string vector_id;
    float vector_score;
    float bm25_score;
    float hybrid_score;
    std::unordered_map<std::string, std::string> metadata;
};

class HybridSearchEngine {
public:
    HybridSearchEngine(
        std::shared_ptr<SimilaritySearchService> vector_search,
        std::shared_ptr<BM25Scorer> bm25_scorer,
        const HybridSearchConfig& config = HybridSearchConfig()
    );

    // Main hybrid search interface
    std::vector<HybridResult> search(
        const std::string& database_id,
        const std::vector<float>& query_vector,
        const std::string& query_text,
        size_t top_k = 10,
        const MetadataFilter& filter = {}
    );

    // Update configuration
    void set_config(const HybridSearchConfig& config);

private:
    std::shared_ptr<SimilaritySearchService> vector_search_;
    std::shared_ptr<BM25Scorer> bm25_scorer_;
    HybridSearchConfig config_;

    // Score fusion
    std::vector<HybridResult> fuse_results(
        const std::vector<SearchResult>& vector_results,
        const std::vector<std::pair<std::string, float>>& bm25_results,
        size_t top_k
    );

    // Normalization
    std::vector<float> normalize_scores(const std::vector<float>& scores);

    // RRF fusion
    std::vector<HybridResult> reciprocal_rank_fusion(
        const std::vector<SearchResult>& vector_results,
        const std::vector<std::pair<std::string, float>>& bm25_results,
        size_t top_k
    );

    // Linear fusion
    std::vector<HybridResult> linear_fusion(
        const std::vector<SearchResult>& vector_results,
        const std::vector<std::pair<std::string, float>>& bm25_results,
        size_t top_k
    );
};

} // namespace search
} // namespace jadevectordb
```

---

## API Design

### REST API Endpoints

#### 1. Hybrid Search

**Endpoint**: `POST /v1/databases/{database_id}/search/hybrid`

**Request**:
```json
{
  "query_vector": [0.1, 0.2, ...],
  "query_text": "Product-A configuration setup procedure",
  "top_k": 10,
  "fusion_method": "rrf",
  "alpha": 0.7,
  "metadata_filter": {
    "category": "type-a"
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "vector_id": "doc_123_chunk_45",
      "vector_score": 0.89,
      "bm25_score": 12.5,
      "hybrid_score": 0.92,
      "metadata": {
        "doc_name": "Product_A_Guide.pdf",
        "page_num": 45,
        "text": "To configure the Product-A system..."
      }
    }
  ],
  "total_results": 127,
  "search_time_ms": 15
}
```

#### 2. Configure Hybrid Search

**Endpoint**: `PUT /v1/databases/{database_id}/search/hybrid/config`

**Request**:
```json
{
  "fusion_method": "linear",
  "alpha": 0.6,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "normalize_scores": true
}
```

#### 3. Build BM25 Index

**Endpoint**: `POST /v1/databases/{database_id}/search/bm25/build`

**Request**:
```json
{
  "text_field": "text",
  "incremental": false
}
```

---

## Storage Requirements

### Inverted Index Storage

**Per Document**:
- Tokenized text: ~500 tokens average
- Unique terms: ~200 terms/doc
- Posting entry: 24 bytes (doc_id: 16B + freq: 4B + offset: 4B)

**For 50,000 documents**:
- Total unique terms: ~100,000
- Average postings per term: 50 docs
- Storage: 100K × 50 × 24 = ~120 MB

**With Compression** (using variable-length encoding):
- Compressed storage: ~40-60 MB

### Index Structure on Disk

**SQLite Table**:
```sql
CREATE TABLE bm25_index (
    term TEXT PRIMARY KEY,
    doc_frequency INTEGER,
    postings BLOB  -- Compressed posting list
);

CREATE TABLE bm25_metadata (
    doc_id TEXT PRIMARY KEY,
    doc_length INTEGER,
    indexed_at TIMESTAMP
);

CREATE TABLE bm25_config (
    database_id TEXT PRIMARY KEY,
    k1 REAL,
    b REAL,
    avg_doc_length REAL,
    total_docs INTEGER
);
```

---

## Performance Targets

| Operation | Target | Expected |
|-----------|--------|----------|
| BM25 index build (50K docs) | <5 min | 3-4 min |
| BM25 query (50K docs) | <10ms | 5-8ms |
| Vector search | <20ms | 10-15ms |
| **Hybrid search total** | **<30ms** | **20-25ms** |
| Index size (50K docs) | <100 MB | 50-70 MB (compressed) |
| Memory usage | <200 MB | 100-150 MB |

---

## Implementation Phases

### Phase 1: BM25 Core (Week 1)
- [ ] Implement tokenization (lowercase, stop words, stemming optional)
- [ ] Build inverted index data structure
- [ ] Implement BM25 scoring algorithm
- [ ] Add basic keyword search

### Phase 2: Index Persistence (Week 2)
- [ ] Design SQLite schema for inverted index
- [ ] Implement index serialization/deserialization
- [ ] Add incremental index updates
- [ ] Create index rebuild functionality

### Phase 3: Hybrid Search Engine (Week 3)
- [ ] Implement score normalization
- [ ] Add RRF fusion method
- [ ] Add linear fusion method
- [ ] Create HybridSearchEngine class

### Phase 4: API Integration (Week 4)
- [ ] Add REST endpoints for hybrid search
- [ ] Add configuration endpoints
- [ ] Integrate with existing search infrastructure
- [ ] Add CLI support

### Phase 5: Testing & Optimization (Week 5)
- [ ] Unit tests for BM25 scorer
- [ ] Integration tests for hybrid search
- [ ] Benchmark performance
- [ ] Optimize index compression
- [ ] Memory profiling

### Phase 6: Documentation (Week 6)
- [ ] API documentation
- [ ] Usage examples
- [ ] Best practices guide
- [ ] Migration guide for existing users

---

## Testing Strategy

### Unit Tests

```cpp
// tests/test_bm25_scorer.cpp

TEST(BM25ScorerTest, TokenizationBasic) {
    BM25Scorer scorer;
    auto tokens = scorer.tokenize("Hello World Test");
    EXPECT_EQ(tokens.size(), 3);
    EXPECT_EQ(tokens[0], "hello");
}

TEST(BM25ScorerTest, ScoreRelevance) {
    BM25Scorer scorer;
    scorer.index_documents(
        {"doc1", "doc2"},
        {"hello world", "hello universe"}
    );

    auto results = scorer.score("hello world", 2);
    EXPECT_GT(results[0].second, results[1].second); // doc1 ranks higher
}

TEST(HybridSearchTest, RRFFusion) {
    // Test RRF produces reasonable combined rankings
}
```

### Integration Tests

```cpp
TEST(HybridSearchIntegration, EndToEnd) {
    // 1. Create database
    // 2. Store vectors with text metadata
    // 3. Build BM25 index
    // 4. Run hybrid search
    // 5. Verify results include both vector and keyword matches
}
```

### Benchmark Tests

```cpp
BENCHMARK(BM25_Index_Build_50K_Docs);
BENCHMARK(BM25_Query_Latency);
BENCHMARK(Hybrid_Search_Latency);
```

---

## Configuration

### Database-Level Config

```json
{
  "database_id": "product_docs",
  "hybrid_search": {
    "enabled": true,
    "bm25": {
      "k1": 1.5,
      "b": 0.75,
      "min_term_length": 2,
      "max_term_length": 50,
      "stopwords": ["the", "a", "an", "is", "are"],
      "stemming": false
    },
    "fusion": {
      "method": "rrf",
      "alpha": 0.7,
      "rrf_k": 60
    },
    "index": {
      "auto_build": true,
      "incremental_update": true,
      "compression": true
    }
  }
}
```

---

## Migration Plan

### For Existing Users

**Step 1: Opt-in Feature**
- Hybrid search disabled by default
- Users explicitly enable via config

**Step 2: Build Index**
```bash
# CLI command to build BM25 index for existing database
jade-db hybrid-search build --database product_docs
```

**Step 3: Test**
```bash
# Test hybrid search
jade-db hybrid-search query --database product_docs \
  --text "Product-A configuration" --top-k 5
```

**Step 4: Switch Endpoints**
- Replace `/v1/databases/{id}/search` with `/v1/databases/{id}/search/hybrid`
- Or use auto-detect: if `query_text` provided, use hybrid mode

---

## Open Questions

1. **Stemming**: Include Porter Stemmer or keep tokens as-is?
   - **Recommendation**: Optional, disabled by default

2. **Stopwords**: Which language(s) to support?
   - **Recommendation**: Start with English, configurable list

3. **Phrase Search**: Support exact phrases like "reset procedure"?
   - **Recommendation**: Phase 2 feature

4. **Index Rebuild**: How to handle schema changes?
   - **Recommendation**: Version inverted index schema

---

## Success Metrics

### Quantitative
- BM25 query latency: <10ms for 50K docs
- Hybrid search latency: <30ms total
- Index build time: <5 min for 50K docs
- Memory overhead: <200 MB

### Qualitative
- Improved retrieval for exact matches (model numbers, codes)
- Better user satisfaction in RAG applications
- Competitive with Weaviate/Qdrant hybrid search

---

## References

1. [BM25 Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
2. [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)
3. [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)
4. [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

**Next Steps**:
1. Review and approve specification
2. Create implementation tasks in TasksTracking
3. Begin Phase 1: BM25 Core development
4. Set up benchmarking infrastructure
