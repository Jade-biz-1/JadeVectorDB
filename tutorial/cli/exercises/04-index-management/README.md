# Exercise 4: Index Management

## Learning Objectives

By the end of this exercise, you will be able to:
- Understand different vector index algorithms (HNSW, IVF, LSH, FLAT)
- Choose the right index type for your use case
- Configure index parameters for optimal performance
- Compare search performance across index types
- Benchmark index build times and search speeds
- Understand trade-offs between accuracy and speed

## Prerequisites

- Completed Exercise 1: CLI Basics
- Completed Exercise 2: Batch Operations
- Understanding of performance concepts (latency, throughput)
- JadeVectorDB running at `http://localhost:8080`

## Introduction to Vector Indexes

Vector indexes are data structures that enable fast similarity search. Different index types offer different trade-offs:

| Index Type | Speed | Accuracy | Memory | Build Time | Best For |
|------------|-------|----------|--------|------------|----------|
| **FLAT** | Slow | 100% | Low | Fast | Small datasets (<10K vectors) |
| **HNSW** | Fast | 95-99% | High | Medium | General purpose, medium-large datasets |
| **IVF** | Medium | 90-95% | Medium | Fast | Very large datasets (>1M vectors) |
| **LSH** | Very Fast | 80-90% | Low | Fast | Massive datasets, approximate search OK |

## Exercise Steps

### Step 1: Create Databases with Different Index Types

Create 4 databases with identical dimensions but different index types:

```bash
# FLAT Index - Exact search (baseline)
jade-db create-db \
  --name index_flat \
  --description "Baseline with FLAT index" \
  --dimension 8 \
  --index-type FLAT

# HNSW Index - Hierarchical Navigable Small World
jade-db create-db \
  --name index_hnsw \
  --description "HNSW index for balanced performance" \
  --dimension 8 \
  --index-type HNSW

# IVF Index - Inverted File Index
jade-db create-db \
  --name index_ivf \
  --description "IVF index for large datasets" \
  --dimension 8 \
  --index-type IVF

# LSH Index - Locality Sensitive Hashing
jade-db create-db \
  --name index_lsh \
  --description "LSH index for approximate search" \
  --dimension 8 \
  --index-type LSH
```

✅ **Checkpoint:** Verify all 4 databases were created.

### Step 2: Import Identical Data to All Databases

Create a script to import the same products to all databases:

```bash
#!/bin/bash

DATABASES=("index_flat" "index_hnsw" "index_ivf" "index_lsh")

for db in "${DATABASES[@]}"; do
  echo "Importing to $db..."

  cat ../../sample-data/products.json | jq -c '.[]' | while read product; do
    id=$(echo $product | jq -r '.id')
    embedding=$(echo $product | jq -c '.embedding')
    metadata=$(echo $product | jq -c 'del(.id, .embedding)')

    jade-db store \
      --database-id "$db" \
      --vector-id "$id" \
      --values "$embedding" \
      --metadata "$metadata"
  done

  echo "✓ Completed $db"
done
```

### Step 3: Benchmark Search Performance

**Task:** Measure search latency across all index types.

Create a benchmark script:

```bash
#!/bin/bash

QUERY_VECTOR="[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]"
DATABASES=("index_flat" "index_hnsw" "index_ivf" "index_lsh")

echo "Index Type | Search Time (ms)"
echo "-----------|----------------"

for db in "${DATABASES[@]}"; do
  start=$(date +%s%N)

  jade-db search \
    --database-id "$db" \
    --query-vector "$QUERY_VECTOR" \
    --top-k 5 > /dev/null 2>&1

  end=$(date +%s%N)
  duration=$(( (end - start) / 1000000 ))

  echo "$db | $duration ms"
done
```

**Expected Results:**
```
Index Type    | Search Time (ms)
--------------|----------------
index_flat    | 5-10 ms (exact, but slowest)
index_hnsw    | 1-3 ms (fast, high accuracy)
index_ivf     | 2-4 ms (medium speed)
index_lsh     | 1-2 ms (fastest, lower accuracy)
```

### Step 4: Compare Search Accuracy

**Concept:** Compare results from approximate indexes against FLAT (exact search).

```bash
#!/bin/bash

QUERY="[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]"

echo "=== FLAT (Exact) Results ==="
jade-db search --database-id index_flat --query-vector "$QUERY" --top-k 5

echo ""
echo "=== HNSW Results ==="
jade-db search --database-id index_hnsw --query-vector "$QUERY" --top-k 5

echo ""
echo "=== IVF Results ==="
jade-db search --database-id index_ivf --query-vector "$QUERY" --top-k 5

echo ""
echo "=== LSH Results ==="
jade-db search --database-id index_lsh --query-vector "$QUERY" --top-k 5
```

**Task:** Compare the result IDs and similarity scores. Calculate recall:
```
Recall = (# of results matching FLAT) / (# of total results)
```

### Step 5: Test with Different Dataset Sizes

**Task:** Create databases with varying amounts of data.

```bash
#!/bin/bash

# Function to generate synthetic vectors
generate_vectors() {
  count=$1
  db_name=$2

  for i in $(seq 1 $count); do
    vector="["
    for d in {1..8}; do
      val=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf "%.3f", rand() }')
      vector="$vector$val"
      [ $d -lt 8 ] && vector="$vector, "
    done
    vector="$vector]"

    jade-db store \
      --database-id "$db_name" \
      --vector-id "vec_$i" \
      --values "$vector" \
      --metadata "{\"index\": $i}" \
      > /dev/null 2>&1
  done
}

# Create and test with 100, 500, 1000 vectors
for size in 100 500 1000; do
  echo "Testing with $size vectors..."

  # Create HNSW database
  jade-db create-db \
    --name "hnsw_$size" \
    --dimension 8 \
    --index-type HNSW \
    > /dev/null 2>&1

  # Import vectors
  echo "  Importing vectors..."
  generate_vectors $size "hnsw_$size"

  # Benchmark search
  echo "  Benchmarking search..."
  start=$(date +%s%N)
  jade-db search \
    --database-id "hnsw_$size" \
    --query-vector "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]" \
    --top-k 10 \
    > /dev/null 2>&1
  end=$(date +%s%N)
  duration=$(( (end - start) / 1000000 ))

  echo "  ✓ $size vectors: $duration ms"
done
```

### Step 6: HNSW Parameter Tuning

**Concept:** HNSW has two key parameters:
- `M`: Number of connections per layer (higher = more accurate, more memory)
- `ef_construction`: Size of dynamic candidate list (higher = better quality, slower build)

```bash
# Create HNSW databases with different M values
jade-db create-db \
  --name hnsw_m_8 \
  --dimension 8 \
  --index-type HNSW \
  --index-params '{"M": 8, "ef_construction": 100}'

jade-db create-db \
  --name hnsw_m_16 \
  --dimension 8 \
  --index-type HNSW \
  --index-params '{"M": 16, "ef_construction": 100}'

jade-db create-db \
  --name hnsw_m_32 \
  --dimension 8 \
  --index-type HNSW \
  --index-params '{"M": 32, "ef_construction": 100}'
```

**Parameter Guidelines:**
- **Small M (8-16):** Less memory, faster build, slightly lower accuracy
- **Medium M (16-32):** Balanced (recommended for most use cases)
- **Large M (32-64):** High accuracy, more memory, slower build

### Step 7: IVF Parameter Tuning

**Concept:** IVF parameters:
- `nlist`: Number of clusters (typically √N for N vectors)
- `nprobe`: Number of clusters to search (higher = more accurate, slower)

```bash
# Create IVF databases with different nlist values
jade-db create-db \
  --name ivf_nlist_10 \
  --dimension 8 \
  --index-type IVF \
  --index-params '{"nlist": 10, "nprobe": 1}'

jade-db create-db \
  --name ivf_nlist_50 \
  --dimension 8 \
  --index-type IVF \
  --index-params '{"nlist": 50, "nprobe": 3}'

jade-db create-db \
  --name ivf_nlist_100 \
  --dimension 8 \
  --index-type IVF \
  --index-params '{"nlist": 100, "nprobe": 5}'
```

**Parameter Guidelines:**
- **nlist:** For N vectors, use nlist = √N to 4√N
- **nprobe:** Start with 1-5, increase for better accuracy

### Step 8: Measure Index Build Time

**Task:** Compare how long it takes to build different indexes.

```bash
#!/bin/bash

echo "Building 1000-vector databases..."
echo ""

# FLAT index (no build time, just storage)
echo "FLAT Index:"
start=$(date +%s)
jade-db create-db --name build_flat --dimension 8 --index-type FLAT > /dev/null 2>&1
for i in {1..1000}; do
  jade-db store \
    --database-id build_flat \
    --vector-id "v$i" \
    --values "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]" \
    --metadata "{}" > /dev/null 2>&1
done
end=$(date +%s)
echo "  Build time: $((end - start)) seconds"

# HNSW index
echo "HNSW Index:"
start=$(date +%s)
jade-db create-db --name build_hnsw --dimension 8 --index-type HNSW > /dev/null 2>&1
for i in {1..1000}; do
  jade-db store \
    --database-id build_hnsw \
    --vector-id "v$i" \
    --values "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]" \
    --metadata "{}" > /dev/null 2>&1
done
end=$(date +%s)
echo "  Build time: $((end - start)) seconds"

# Continue for IVF and LSH...
```

### Step 9: Memory Usage Comparison

**Task:** Estimate memory usage for different index types.

```bash
# Check database sizes (if exposed by API)
for db in index_flat index_hnsw index_ivf index_lsh; do
  echo "Database: $db"
  jade-db stats --database-id "$db" | grep -i "memory\|size"
  echo ""
done
```

**Typical Memory Usage (for same data):**
- FLAT: 1x (baseline)
- HNSW: 1.2-2x (depends on M parameter)
- IVF: 1.1-1.5x
- LSH: 0.8-1.2x

### Step 10: Choose the Right Index for Your Use Case

**Decision Matrix:**

```bash
# Use Case 1: Small dataset (<10K), need exact results
# ✅ FLAT
jade-db create-db --name small_exact --dimension 128 --index-type FLAT

# Use Case 2: Medium dataset (10K-1M), balanced performance
# ✅ HNSW
jade-db create-db --name medium_balanced --dimension 128 --index-type HNSW

# Use Case 3: Large dataset (>1M), willing to trade accuracy
# ✅ IVF
jade-db create-db --name large_fast --dimension 128 --index-type IVF

# Use Case 4: Massive dataset (>10M), approximate results OK
# ✅ LSH
jade-db create-db --name massive_approx --dimension 128 --index-type LSH
```

## Advanced Challenges

### Challenge 1: Accuracy vs Speed Trade-off Analysis

Create a comprehensive benchmark comparing all indexes:

```bash
#!/bin/bash

# Run 100 queries on each index type
# Measure: avg latency, recall@10, recall@50

QUERIES=100
databases=("index_flat" "index_hnsw" "index_ivf" "index_lsh")

for db in "${databases[@]}"; do
  total_time=0

  for i in $(seq 1 $QUERIES); do
    # Generate random query
    query="["
    for d in {1..8}; do
      val=$(awk 'BEGIN { srand(); printf "%.3f", rand() }')
      query="$query$val"
      [ $d -lt 8 ] && query="$query, "
    done
    query="$query]"

    # Measure search time
    start=$(date +%s%N)
    jade-db search --database-id "$db" --query-vector "$query" --top-k 10 > /dev/null 2>&1
    end=$(date +%s%N)

    duration=$(( (end - start) / 1000000 ))
    total_time=$((total_time + duration))
  done

  avg_time=$((total_time / QUERIES))
  echo "$db: Average latency = $avg_time ms"
done
```

### Challenge 2: Parameter Optimization

Find the optimal HNSW parameters for your use case:

```bash
# Test different M values: 8, 16, 32, 64
# Test different ef_construction values: 50, 100, 200, 400

# Create performance matrix
for M in 8 16 32 64; do
  for ef in 50 100 200 400; do
    db_name="hnsw_m${M}_ef${ef}"
    echo "Testing M=$M, ef_construction=$ef..."

    # Create, populate, and benchmark
    # (implementation left as exercise)
  done
done
```

### Challenge 3: Hybrid Index Strategy

For very large datasets, use a hybrid approach:

```bash
# Strategy: Use IVF for bulk filtering, HNSW for refinement

# 1. Coarse search with IVF (fast, less accurate)
jade-db search \
  --database-id ivf_database \
  --query-vector "[...]" \
  --top-k 100 \
  > candidates.json

# 2. Refine candidates with HNSW (slower, more accurate)
# Extract candidate IDs and re-rank
cat candidates.json | jq -r '.[].id' | while read id; do
  # Re-score with HNSW
  # (implementation depends on API capabilities)
done
```

## Performance Benchmarking Best Practices

### 1. Warm-up Queries
```bash
# Run warm-up queries before benchmarking
for i in {1..10}; do
  jade-db search \
    --database-id test_db \
    --query-vector "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]" \
    --top-k 10 > /dev/null 2>&1
done
```

### 2. Multiple Runs
```bash
# Run each benchmark multiple times and average
RUNS=10
total=0

for run in $(seq 1 $RUNS); do
  start=$(date +%s%N)
  # ... benchmark code ...
  end=$(date +%s%N)
  total=$((total + (end - start)))
done

avg=$((total / RUNS / 1000000))
echo "Average: $avg ms"
```

### 3. Measure Percentiles
```bash
# Track p50, p95, p99 latencies
# Store all measurements and calculate percentiles
```

## Index Selection Guidelines

### FLAT Index
**When to use:**
- Dataset size < 10,000 vectors
- Need 100% accuracy
- Simplicity more important than speed
- Prototyping/development

**Avoid when:**
- Dataset > 100,000 vectors
- Real-time search required
- Memory constraints exist

### HNSW Index
**When to use:**
- Dataset size: 10K - 10M vectors
- Need high accuracy (>95%)
- Can afford higher memory usage
- General-purpose production use

**Parameters:**
- M: 16-32 (recommended)
- ef_construction: 100-200

### IVF Index
**When to use:**
- Dataset size > 1M vectors
- Can tolerate 90-95% accuracy
- Memory constrained
- Batch search workloads

**Parameters:**
- nlist: √N to 4√N
- nprobe: 1-10

### LSH Index
**When to use:**
- Dataset size > 10M vectors
- Can tolerate 80-90% accuracy
- Extremely low latency required
- Approximation acceptable

## Verification

Run the verification script:

```bash
bash verify.sh
```

Expected outcomes:
- ✅ All 4 index types created successfully
- ✅ Search performance measured for each index
- ✅ Accuracy comparison completed
- ✅ Parameter tuning experiments conducted

## Next Steps

After completing this exercise:
- **Exercise 5:** Advanced Workflows - Production monitoring, backup, and automation

## Troubleshooting

**Index build takes too long:**
- Reduce ef_construction for HNSW
- Reduce nlist for IVF
- Use smaller dataset for testing

**Poor search accuracy:**
- Increase M for HNSW
- Increase nprobe for IVF
- Consider using FLAT for small datasets

**High memory usage:**
- Reduce M for HNSW
- Use IVF or LSH instead
- Quantize vectors (if supported)

## Resources

- HNSW Paper: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- IVF Paper: "Inverted Multi-Index: Quantization and approximate nearest neighbor search"
- LSH Paper: "Locality-sensitive hashing scheme based on p-stable distributions"
