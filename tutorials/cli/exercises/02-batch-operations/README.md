# Exercise 2: Batch Operations

## Learning Objectives

By the end of this exercise, you will be able to:
- Import multiple vectors efficiently using loops
- Use the built-in `import` and `export` commands for production workloads
- Use the sample data files provided
- Process JSON data for batch imports
- Handle errors during batch operations
- Monitor progress during large imports

## Prerequisites

- Completed Exercise 1: CLI Basics
- Understanding of shell scripting (loops, conditionals)
- JadeVectorDB running at `http://localhost:8080`

## Exercise Steps

### Step 1: Create a Batch Import Database

Create a new database for batch operations:

```bash
jade-db create-db \
  --name batch_products \
  --description "Database for batch import exercise" \
  --dimension 8 \
  --index-type HNSW
```

### Step 2: Import All Products from Sample Data

**Task:** Write a script to import all 8 products from `../../sample-data/products.json`

**Approach 1: Manual Loop (Beginner)**
```bash
# Import each product one by one
for product in laptop_001 laptop_002 phone_001 phone_002 tablet_001 tablet_002 watch_001 headphones_001; do
  echo "Importing $product..."
  # Extract data from JSON and import
  # YOUR CODE HERE
done
```

**Approach 2: JSON Parsing (Advanced)**
```bash
# Use jq to parse JSON and loop through products
cat ../../sample-data/products.json | jq -c '.[]' | while read product; do
  id=$(echo $product | jq -r '.id')
  embedding=$(echo $product | jq -c '.embedding')
  metadata=$(echo $product | jq -c 'del(.id, .embedding)')

  echo "Importing $id..."
  jade-db store \
    --database-id batch_products \
    --vector-id "$id" \
    --values "$embedding" \
    --metadata "$metadata"
done
```

### Step 3: Add Error Handling

**Task:** Modify your import script to:
1. Check if each import succeeds
2. Log failures to a file
3. Continue on errors instead of stopping
4. Report summary at the end

**Hint:**
```bash
SUCCESSFUL=0
FAILED=0

# In your loop:
if jade-db store ...; then
  ((SUCCESSFUL++))
else
  ((FAILED++))
  echo "Failed: $id" >> failed_imports.log
fi

# After loop:
echo "Summary: $SUCCESSFUL succeeded, $FAILED failed"
```

### Step 4: Monitor Progress

**Task:** Add progress indicators to your batch import script

```bash
TOTAL=8
CURRENT=0

# In your loop:
((CURRENT++))
echo "[$CURRENT/$TOTAL] Importing $id..."
```

### Step 5: Generate Synthetic Data

**Task:** Create a script that generates and imports 100 synthetic vectors

**Requirements:**
- Vector IDs: `synthetic_001` through `synthetic_100`
- Random 8-dimensional vectors (values between 0 and 1)
- Metadata with: `{"type": "synthetic", "index": N, "timestamp": "..."}`

**Hint for generating random vectors:**
```bash
# Generate random vector
vector="["
for i in {1..8}; do
  value=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); printf "%.2f", rand() }')
  vector="$vector$value"
  if [ $i -lt 8 ]; then
    vector="$vector, "
  fi
done
vector="$vector]"
```

### Step 6: Using the Built-in Import/Export Commands

The manual loop approach above is great for understanding how batch operations work. For production workloads, JadeVectorDB provides built-in `import` and `export` commands with automatic progress tracking and error reporting.

**Import from JSON (the fast way):**
```bash
jade-db import \
  --database-id batch_products \
  --file ../../sample-data/products.json \
  --format json \
  --batch-size 100
```

The `--format` flag is optional — the CLI auto-detects JSON or CSV from the file extension.

**Import from CSV:**
```bash
jade-db import \
  --database-id batch_products \
  --file vectors.csv \
  --batch-size 50
```

**Export all vectors to JSON:**
```bash
jade-db export \
  --database-id batch_products \
  --file backup.json
```

**Export to CSV:**
```bash
jade-db export \
  --database-id batch_products \
  --file backup.csv \
  --format csv
```

**Export a specific subset of vectors:**
```bash
jade-db export \
  --database-id batch_products \
  --file subset.json \
  --vector-ids "laptop_001,laptop_002,phone_001"
```

**Task:** Use `jade-db import` to re-import the sample data into a fresh database and verify the count matches your manual import from Step 2:

```bash
# Create a fresh database to compare
jade-db create-db \
  --name batch_products_v2 \
  --description "Testing built-in import" \
  --dimension 8 \
  --index-type HNSW

# Import using the built-in command
jade-db import \
  --database-id batch_products_v2 \
  --file ../../sample-data/products.json

# Verify the result
jade-db list-vectors --database-id batch_products_v2
```

**When to use each approach:**

| Approach | Use when |
|----------|----------|
| Manual loop (`jade-db store`) | Learning, custom transformation logic, conditional imports |
| `jade-db import` | Production data loads, large files, CI/CD pipelines |
| `jade-db export` | Backups, migrations, sharing datasets |

---

### Step 7: Batch Search Testing

After importing all vectors:

**Task:** Perform 10 different searches and compare results

```bash
# Test searches with different query vectors
for i in {1..10}; do
  echo "Test search $i..."
  # Generate random query vector and search
  # YOUR CODE HERE
done
```

### Step 8: Performance Comparison

**Task:** Measure the time taken to import:
- 10 vectors
- 50 vectors
- 100 vectors

```bash
start_time=$(date +%s)

# Your batch import code

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Import took $duration seconds"
```

## Verification

Run the verification script:

```bash
bash verify.sh
```

Expected outcomes:
- ✅ All 8 products from sample data imported successfully (manual loop)
- ✅ All 8 products imported again using `jade-db import` (built-in command)
- ✅ 100 synthetic vectors created and imported
- ✅ Export produces a valid JSON/CSV file
- ✅ Error handling works correctly
- ✅ Performance metrics collected

## Challenges (Optional)

1. **Challenge 1:** Parallelize batch imports using background processes (`&`)
2. **Challenge 2:** Create a progress bar visualization
3. **Challenge 3:** Implement retry logic for failed imports
4. **Challenge 4:** Create a batch delete script to clean up test data

## Best Practices Learned

1. **Always validate data** before importing
2. **Log errors** for debugging
3. **Use progress indicators** for long operations
4. **Handle failures gracefully** without stopping the entire process
5. **Measure performance** to understand system limits
6. **Use built-in `import`/`export` for production** — they handle batching, progress, and error reporting automatically
7. **Tune `--batch-size`** based on vector dimension: larger dimensions → smaller batch size (e.g. 1024-dim → batch-size 25–50)

## Next Steps

Move on to:
- **Exercise 3:** Metadata Filtering for advanced searches
- **Exercise 4:** Index Management for performance optimization

## Troubleshooting

**Out of memory errors:**
- Reduce batch size
- Add delays between imports: `sleep 0.1`

**Rate limiting errors:**
- Add longer delays between requests
- Reduce concurrent requests

**JSON parsing errors:**
- Install `jq` if not available: `sudo apt-get install jq`
- Validate JSON syntax with `jq . file.json`
