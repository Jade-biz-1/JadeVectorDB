# Exercise 1: CLI Basics

## Learning Objectives

By the end of this exercise, you will be able to:
- Set up your CLI environment
- Create a vector database
- Store vectors with metadata
- Retrieve vectors by ID
- Perform basic similarity searches
- Check system health

## Prerequisites

- JadeVectorDB running at `http://localhost:8080`
- One of the CLI tools installed (Python, Shell, or JavaScript)
- API key (default: `mykey123` for local development)

## Exercise Steps

### Step 1: Environment Setup

Set up environment variables to avoid repetitive typing:

```bash
export JADE_DB_URL=http://localhost:8080
export JADE_DB_API_KEY=mykey123
```

### Step 2: Check System Health

Before starting, verify that JadeVectorDB is running:

**Python CLI:**
```bash
jade-db health
```

**Shell CLI:**
```bash
bash cli/shell/scripts/jade-db.sh health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Step 3: Create Your First Database

Create a database to store product embeddings:

**Task:** Create a database named `my_products` with:
- Dimension: 8
- Index type: HNSW
- Description: "My first product database"

**Python CLI:**
```bash
jade-db create-db \
  --name my_products \
  --description "My first product database" \
  --dimension 8 \
  --index-type HNSW
```

**Shell CLI:**
```bash
bash cli/shell/scripts/jade-db.sh create-db \
  my_products \
  "My first product database" \
  8 \
  HNSW
```

✅ **Checkpoint:** Save the database ID returned in the response. You'll need it for the next steps.

### Step 4: Store Your First Vector

Store a laptop product vector:

**Task:** Store a vector with:
- Vector ID: `laptop_001`
- Values: `[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]`
- Metadata: Product information

**Python CLI:**
```bash
jade-db store \
  --database-id my_products \
  --vector-id laptop_001 \
  --values "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --metadata '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99, "in_stock": true}'
```

**Shell CLI:**
```bash
bash cli/shell/scripts/jade-db.sh \
  --database-id my_products \
  store laptop_001 \
  '[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]' \
  '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99, "in_stock": true}'
```

✅ **Checkpoint:** Verify you receive a success response.

### Step 5: Store Additional Vectors

**Task:** Store at least 2 more product vectors from `../../sample-data/products.json`

Choose products with different categories (e.g., phone, tablet) to make search results more interesting.

### Step 6: Retrieve a Vector

Retrieve the laptop vector you stored:

**Python CLI:**
```bash
jade-db retrieve \
  --database-id my_products \
  --vector-id laptop_001
```

✅ **Checkpoint:** Verify the response includes both the vector values and metadata.

### Step 7: Perform a Similarity Search

Search for products similar to a query vector:

**Task:** Find the top 3 products similar to `[0.83, 0.14, 0.24, 0.76, 0.63, 0.42, 0.89, 0.32]`

**Python CLI:**
```bash
jade-db search \
  --database-id my_products \
  --query-vector "[0.83, 0.14, 0.24, 0.76, 0.63, 0.42, 0.89, 0.32]" \
  --top-k 3
```

✅ **Checkpoint:** The laptop should be in the top results since the query vector is very similar.

### Step 8: Search with Threshold

**Task:** Perform the same search but only return results with similarity > 0.9

**Hint:** Add `--threshold 0.9` parameter

## Verification

Run the verification script to check your work:

```bash
bash verify.sh
```

## Challenges (Optional)

1. **Challenge 1:** Create a second database with different dimensions (e.g., 128)
2. **Challenge 2:** Store 10 vectors programmatically using a loop
3. **Challenge 3:** Find products that are NOT similar to a laptop by using a very different query vector

## Next Steps

Once you've completed this exercise, move on to:
- **Exercise 2:** Batch Operations for efficient data import
- **Exercise 3:** Metadata Filtering for advanced searches

## Troubleshooting

**Database not found error:**
- Make sure you're using the correct database ID
- List all databases with `jade-db list-db`

**Dimension mismatch error:**
- Verify your vectors have exactly 8 values
- Check that there are no extra spaces in the array

**Connection refused:**
- Ensure JadeVectorDB is running
- Check the URL and port are correct
