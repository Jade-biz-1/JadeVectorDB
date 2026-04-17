# Exercise 8: Advanced Features

## Learning Objectives

By the end of this exercise, you will be able to:
- Generate vector embeddings from text using the CLI
- Build a BM25 keyword index and run hybrid (vector + keyword) searches
- Tune hybrid search fusion methods and weights
- Rerank search results using a cross-encoder model
- Use standalone reranking on arbitrary document sets
- Query analytics statistics for database usage insights

## Prerequisites

- Completed Exercise 4: Index Management
- JadeVectorDB running at `http://localhost:8080`
- A database with text metadata already populated (we'll create one below)

## Scenario

You're building a product search system where users type natural-language queries like _"lightweight laptop for travel"_. Pure vector search misses keyword-heavy queries; pure keyword search misses semantic meaning. This exercise shows how to combine both — and then rerank to get the most relevant results at the top.

---

## Part 1: Embedding Generation

### Step 1: Generate an Embedding from Text

The `generate-embedding` command converts text into a vector using the configured embedding model. This is useful for testing and for generating query vectors without an external embedding service:

```bash
jade-db generate-embedding \
  --text "lightweight laptop for travel"
```

The response contains the embedding vector you can use directly in a search.

**Generate with explicit model and input type:**
```bash
jade-db generate-embedding \
  --text "What is a vector database?" \
  --input-type query \
  --model default \
  --provider default
```

**Use the output as a search query vector:**
```bash
# Generate embedding and extract the vector
QUERY_VECTOR=$(jade-db generate-embedding \
  --text "lightweight laptop for travel" | jq -c '.embedding')

# Use it in a similarity search
jade-db search \
  --database-id my_products \
  --query-vector "$QUERY_VECTOR" \
  --top-k 5
```

✅ **Checkpoint:** You should see a JSON response containing a `embedding` array. Note its length — this is your model's embedding dimension.

---

## Part 2: Hybrid Search

Hybrid search combines **vector similarity** (semantic understanding) with **BM25 keyword matching** (exact term relevance). It outperforms either approach alone when queries mix semantic intent with specific keywords.

### Step 2: Create a Database with Text Metadata

Hybrid search requires a `text` field in your vector metadata so the BM25 index can index it:

```bash
jade-db create-db \
  --name product_search \
  --description "Hybrid search demo" \
  --dimension 8 \
  --index-type HNSW
```

Store vectors that include a `text` field in their metadata:

```bash
jade-db store \
  --database-id product_search \
  --vector-id laptop_001 \
  --values "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --metadata '{"text": "UltraBook Pro lightweight laptop for travel and business", "category": "laptop", "price": 1299.99}'

jade-db store \
  --database-id product_search \
  --vector-id phone_001 \
  --values "[0.22, 0.89, 0.45, 0.31, 0.67, 0.88, 0.12, 0.54]" \
  --metadata '{"text": "SmartPhone X flagship mobile phone with pro camera", "category": "phone", "price": 999.99}'

jade-db store \
  --database-id product_search \
  --vector-id tablet_001 \
  --values "[0.55, 0.44, 0.71, 0.62, 0.38, 0.29, 0.83, 0.47]" \
  --metadata '{"text": "ProTab 12 tablet for drawing creative professionals", "category": "tablet", "price": 849.99}'

jade-db store \
  --database-id product_search \
  --vector-id laptop_002 \
  --values "[0.81, 0.15, 0.19, 0.74, 0.61, 0.47, 0.88, 0.29]" \
  --metadata '{"text": "ThinBook Air ultra-portable laptop 800g featherweight", "category": "laptop", "price": 1149.99}'

jade-db store \
  --database-id product_search \
  --vector-id headphones_001 \
  --values "[0.12, 0.33, 0.88, 0.21, 0.76, 0.54, 0.41, 0.67]" \
  --metadata '{"text": "NoiseCancel Pro wireless headphones for travel", "category": "audio", "price": 349.99}'
```

---

### Step 3: Build the BM25 Index

Before hybrid search works, you must build the keyword index:

```bash
jade-db hybrid-build \
  --database-id product_search \
  --text-field text
```

Check the build status:
```bash
jade-db hybrid-status --database-id product_search
```

Wait until the status shows `ready` before running searches.

---

### Step 4: Run a Hybrid Search (Text Query)

Search with a natural-language text query — no vector required:

```bash
jade-db hybrid-search \
  --database-id product_search \
  --query-text "lightweight laptop for travel" \
  --top-k 3
```

Both laptops and the headphones (which also has "travel" in its text) should rank highly.

---

### Step 5: Hybrid Search with Both Text and Vector

Provide both a text query and a vector for the strongest signal:

```bash
jade-db hybrid-search \
  --database-id product_search \
  --query-text "lightweight laptop" \
  --query-vector "[0.83, 0.14, 0.20, 0.76, 0.63, 0.45, 0.89, 0.30]" \
  --top-k 3
```

---

### Step 6: Tune the Fusion Method

JadeVectorDB supports two score fusion strategies:

**RRF (Reciprocal Rank Fusion)** — default, rank-based, robust across score scales:
```bash
jade-db hybrid-search \
  --database-id product_search \
  --query-text "laptop travel" \
  --fusion-method rrf \
  --top-k 3
```

**Weighted Linear** — `--alpha` controls vector vs. keyword balance (0.0 = keyword only, 1.0 = vector only):
```bash
# Lean heavily on vector similarity (alpha=0.8)
jade-db hybrid-search \
  --database-id product_search \
  --query-text "laptop travel" \
  --fusion-method weighted_linear \
  --alpha 0.8 \
  --top-k 3

# Balance both equally (alpha=0.5)
jade-db hybrid-search \
  --database-id product_search \
  --query-text "laptop travel" \
  --fusion-method weighted_linear \
  --alpha 0.5 \
  --top-k 3
```

**Task:** Run both fusion methods for the same query and compare the result ranking. Which gives more relevant results for your data?

| `--alpha` | Behaviour |
|-----------|-----------|
| `0.9` | Almost pure vector search |
| `0.7` | Default — vector-leaning balanced |
| `0.5` | Equal weight |
| `0.2` | Keyword-leaning |

---

### Step 7: Hybrid Search with Metadata Filters

Combine hybrid search with a metadata filter to narrow the result set:

```bash
jade-db hybrid-search \
  --database-id product_search \
  --query-text "lightweight portable" \
  --filters '{"category": "laptop"}' \
  --top-k 5
```

---

### Step 8: Rebuild the BM25 Index

After adding new vectors, rebuild the BM25 index to include them:

```bash
# Add a new product
jade-db store \
  --database-id product_search \
  --vector-id watch_001 \
  --values "[0.34, 0.67, 0.55, 0.43, 0.81, 0.22, 0.76, 0.38]" \
  --metadata '{"text": "SmartWatch Ultra fitness tracker for travel athletes", "category": "watch", "price": 499.99}'

# Rebuild to include the new entry
jade-db hybrid-rebuild \
  --database-id product_search \
  --text-field text

# Or use incremental build for large databases (faster, adds only new entries)
jade-db hybrid-build \
  --database-id product_search \
  --text-field text \
  --incremental
```

---

## Part 3: Reranking

Reranking uses a cross-encoder model to re-score an initial candidate set. It's slower than vector search but more accurate — use it as a final pass over the top-N results.

### Step 9: Search with Built-in Reranking

`rerank-search` performs a vector search and then reranks the results in one step:

```bash
jade-db rerank-search \
  --database-id product_search \
  --query-text "laptop for business travel" \
  --top-k 5
```

Combine with a vector for better initial retrieval:
```bash
jade-db rerank-search \
  --database-id product_search \
  --query-text "laptop for business travel" \
  --query-vector "[0.83, 0.14, 0.20, 0.76, 0.63, 0.45, 0.89, 0.30]" \
  --top-k 5
```

---

### Step 10: Standalone Reranking

Use `rerank` to reorder any set of documents against a query — independent of JadeVectorDB search. This is useful when you already have candidates from multiple sources:

```bash
jade-db rerank \
  --query "lightweight laptop for travel" \
  --documents '[
    {"id": "laptop_001", "text": "UltraBook Pro lightweight laptop for travel and business"},
    {"id": "phone_001",  "text": "SmartPhone X flagship mobile phone with pro camera"},
    {"id": "laptop_002", "text": "ThinBook Air ultra-portable laptop 800g featherweight"},
    {"id": "headphones_001", "text": "NoiseCancel Pro wireless headphones for travel"}
  ]' \
  --top-k 3
```

The results come back ordered by relevance to the query, with a score for each document.

---

## Part 4: Analytics

### Step 11: View Database Usage Statistics

Query analytics to understand how a database is being used over time:

```bash
jade-db analytics-stats \
  --database-id product_search
```

Default granularity is hourly. Switch to daily or weekly for longer time windows:

```bash
jade-db analytics-stats \
  --database-id product_search \
  --granularity daily
```

Query a specific time range (ISO 8601 format):
```bash
jade-db analytics-stats \
  --database-id product_search \
  --granularity hourly \
  --start-time "2026-04-01T00:00:00Z" \
  --end-time "2026-04-16T23:59:59Z"
```

The response includes metrics such as query counts, average latency, and vector operation counts — useful for capacity planning and detecting traffic spikes.

---

## Putting It All Together

Here's a complete pipeline that combines embedding generation, hybrid search, and reranking:

```bash
#!/bin/bash
# advanced_search.sh <query-text>
# Full pipeline: embed → hybrid search → rerank

DB_ID="product_search"
QUERY="$1"
TOP_K=10

if [[ -z "$QUERY" ]]; then
  echo "Usage: $0 \"your search query\""
  exit 1
fi

echo "=== Step 1: Generate embedding ==="
QUERY_VECTOR=$(jade-db generate-embedding --text "$QUERY" | jq -c '.embedding')
echo "Embedding dimension: $(echo $QUERY_VECTOR | jq 'length')"

echo ""
echo "=== Step 2: Hybrid search (top $TOP_K candidates) ==="
CANDIDATES=$(jade-db hybrid-search \
  --database-id "$DB_ID" \
  --query-text "$QUERY" \
  --query-vector "$QUERY_VECTOR" \
  --fusion-method rrf \
  --top-k "$TOP_K")
echo "$CANDIDATES" | jq '[.results[] | {id, score}]'

echo ""
echo "=== Step 3: Rerank top results ==="
# Build document list from hybrid search results for standalone reranking
DOCS=$(echo "$CANDIDATES" | jq -c '[.results[] | {id: .id, text: .metadata.text}]')

jade-db rerank \
  --query "$QUERY" \
  --documents "$DOCS" \
  --top-k 3
```

Run it:
```bash
bash advanced_search.sh "lightweight laptop for travel"
```

---

## Verification

Expected outcomes after completing all steps:
- ✅ Generated an embedding vector from text
- ✅ Built a BM25 index on the `product_search` database
- ✅ Ran hybrid search with text-only, text+vector, and filtered queries
- ✅ Compared RRF vs. weighted linear fusion results
- ✅ Used `rerank-search` for combined search + reranking
- ✅ Used standalone `rerank` on a custom document set
- ✅ Queried analytics stats at hourly and daily granularity

---

## Challenges (Optional)

1. **Challenge 1:** Build the BM25 index on a large database (from Exercise 2's 100 synthetic vectors), run the same query with `--fusion-method rrf` and `weighted_linear`, and record which fusion method returns the more relevant results
2. **Challenge 2:** Write a script that runs `analytics-stats --granularity daily` each day and appends the output to a CSV log file for trend analysis
3. **Challenge 3:** Compare `jade-db search` (pure vector), `jade-db hybrid-search` (hybrid), and `jade-db rerank-search` (hybrid + rerank) for the same query — measure the response time of each and note the quality difference

---

## Best Practices

1. **Build the BM25 index before hybrid searches** — queries will fail or fall back to vector-only if the index hasn't been built yet
2. **Use `--incremental` for large databases** — full rebuilds on millions of vectors are slow; incremental updates after each ingestion batch are faster
3. **Start with RRF fusion** — it's robust and doesn't require tuning; only switch to `weighted_linear` when you have query logs to calibrate `--alpha`
4. **Use reranking selectively** — reranking is more expensive than vector search; apply it to the top 10–20 candidates rather than the entire result set
5. **Monitor with `analytics-stats`** — set up a daily cron job to log analytics and alert on unusual query volume or latency spikes

---

## Next Steps

You've now completed all eight CLI exercises. From here:

- **Deploy to production** — see `docs/deployment_guide.md` for Docker and Kubernetes setup
- **EnterpriseRAG** — the `EnterpriseRAG/` directory shows a full RAG application built on JadeVectorDB
- **Distributed cluster** — see `tutorials/cli/exercises/` and `cli/distributed/` for multi-node operations

## See Also

- `docs/cli-documentation.md` — full CLI reference with all 40+ commands
- `tutorials/web/` — interactive web tutorial covering the same topics in a browser-based playground
- `cli/python/README.md` — Python CLI installation and configuration guide
