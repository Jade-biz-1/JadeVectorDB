# Exercise 3: Metadata Filtering

## Learning Objectives

By the end of this exercise, you will be able to:
- Understand metadata filtering concepts
- Combine vector similarity with metadata filters
- Use complex filter conditions (AND, OR operations)
- Perform range queries on numeric fields
- Filter by array-type metadata (tags, categories)
- Optimize filtered search performance

## Prerequisites

- Completed Exercise 1: CLI Basics
- Completed Exercise 2: Batch Operations
- Understanding of JSON and metadata structures
- JadeVectorDB running at `http://localhost:8080`

## Introduction to Metadata Filtering

Metadata filtering allows you to narrow down similarity search results based on additional criteria beyond vector similarity. This is crucial for real-world applications where you need to:

- Search only within a specific category
- Filter by price range
- Find items with specific tags
- Match boolean conditions (in_stock, featured, etc.)

**Example Use Cases:**
- "Find similar laptops under $1000"
- "Search for products in the 'electronics' category"
- "Find documents tagged with 'urgent' and 'customer'"

## Exercise Steps

### Step 1: Create a Database with Rich Metadata

Create a database specifically designed for metadata filtering:

```bash
jade-db create-db \
  --name filtered_products \
  --description "Products database with rich metadata for filtering" \
  --dimension 8 \
  --index-type HNSW
```

### Step 2: Import Products with Diverse Metadata

Import all products from the sample data:

```bash
# Using the sample data file
cat ../../sample-data/products.json | jq -c '.[]' | while read product; do
  id=$(echo $product | jq -r '.id')
  name=$(echo $product | jq -r '.name')
  category=$(echo $product | jq -r '.category')
  brand=$(echo $product | jq -r '.brand')
  price=$(echo $product | jq -r '.price')
  in_stock=$(echo $product | jq -r '.in_stock')
  tags=$(echo $product | jq -c '.tags')
  embedding=$(echo $product | jq -c '.embedding')

  echo "Importing $id ($name)..."
  jade-db store \
    --database-id filtered_products \
    --vector-id "$id" \
    --values "$embedding" \
    --metadata "{\"name\": \"$name\", \"category\": \"$category\", \"brand\": \"$brand\", \"price\": $price, \"in_stock\": $in_stock, \"tags\": $tags}"
done
```

✅ **Checkpoint:** Verify all 8 products were imported successfully.

### Step 3: Basic Category Filtering

**Task:** Search for laptops only, regardless of similarity.

**Concept:** Use metadata filters to restrict search to a specific category.

```bash
# Search for products similar to a laptop vector, but only return laptops
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --top-k 5 \
  --filter '{"category": "laptop"}'
```

**Expected:** Only laptop products in results, even if other categories are more similar.

### Step 4: Price Range Filtering

**Task:** Find products under $500.

**Concept:** Use range queries for numeric metadata.

```bash
# Search for affordable products (under $500)
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --top-k 5 \
  --filter '{"price": {"$lt": 500}}'
```

**Price Range Operators:**
- `$lt`: Less than
- `$lte`: Less than or equal
- `$gt`: Greater than
- `$gte`: Greater than or equal
- `$eq`: Equal to
- `$ne`: Not equal to

### Step 5: Boolean Filtering

**Task:** Find only in-stock products.

```bash
# Search for products that are currently in stock
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.15, 0.87, 0.92, 0.21, 0.48, 0.76, 0.33, 0.81]" \
  --top-k 5 \
  --filter '{"in_stock": true}'
```

✅ **Checkpoint:** Results should exclude any out-of-stock items.

### Step 6: Tag-Based Filtering

**Task:** Find products tagged as "premium".

**Concept:** Filter by array-type metadata fields.

```bash
# Search for premium products
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --top-k 5 \
  --filter '{"tags": {"$contains": "premium"}}'
```

**Array Operators:**
- `$contains`: Array contains value
- `$in`: Value is in array
- `$all`: Array contains all specified values

### Step 7: Complex AND Conditions

**Task:** Find in-stock laptops under $1500.

**Concept:** Combine multiple filter conditions with AND logic.

```bash
# Search for affordable, available laptops
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --top-k 5 \
  --filter '{
    "$and": [
      {"category": "laptop"},
      {"in_stock": true},
      {"price": {"$lt": 1500}}
    ]
  }'
```

### Step 8: Complex OR Conditions

**Task:** Find products that are either laptops OR tablets.

```bash
# Search for portable computing devices
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.70, 0.40, 0.35, 0.75, 0.70, 0.50, 0.80, 0.42]" \
  --top-k 5 \
  --filter '{
    "$or": [
      {"category": "laptop"},
      {"category": "tablet"}
    ]
  }'
```

### Step 9: Nested AND/OR Conditions

**Task:** Find (TechCorp products OR products under $500) AND in-stock.

**Concept:** Complex nested filter logic.

```bash
# Search for affordable TechCorp products or any cheap in-stock items
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.65, 0.50, 0.45, 0.70, 0.68, 0.52, 0.75, 0.48]" \
  --top-k 5 \
  --filter '{
    "$and": [
      {"in_stock": true},
      {
        "$or": [
          {"brand": "TechCorp"},
          {"price": {"$lt": 500}}
        ]
      }
    ]
  }'
```

### Step 10: Multi-Tag Filtering

**Task:** Find products tagged with BOTH "wireless" AND "premium".

```bash
# Search for premium wireless products
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.45, 0.38, 0.51, 0.62, 0.43, 0.71, 0.58, 0.39]" \
  --top-k 5 \
  --filter '{"tags": {"$all": ["wireless", "premium"]}}'
```

### Step 11: Exclusion Filtering

**Task:** Find products that are NOT phones.

```bash
# Search for non-phone products
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.60, 0.55, 0.50, 0.70, 0.65, 0.55, 0.72, 0.52]" \
  --top-k 5 \
  --filter '{"category": {"$ne": "phone"}}'
```

### Step 12: Combining Filters with Similarity Threshold

**Task:** Find in-stock laptops with similarity > 0.85.

**Concept:** Use both metadata filters AND similarity thresholds.

```bash
# Search for very similar in-stock laptops
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --top-k 5 \
  --threshold 0.85 \
  --filter '{
    "$and": [
      {"category": "laptop"},
      {"in_stock": true}
    ]
  }'
```

## Advanced Challenges

### Challenge 1: Price Range by Category
Find phones between $400 and $900:

```bash
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.15, 0.87, 0.92, 0.21, 0.48, 0.76, 0.33, 0.81]" \
  --top-k 5 \
  --filter '{
    "$and": [
      {"category": "phone"},
      {"price": {"$gte": 400}},
      {"price": {"$lte": 900}}
    ]
  }'
```

### Challenge 2: Multi-Brand Filtering
Find products from either TechCorp OR MobilePlus:

```bash
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.70, 0.50, 0.60, 0.65, 0.60, 0.58, 0.68, 0.55]" \
  --top-k 5 \
  --filter '{
    "$or": [
      {"brand": "TechCorp"},
      {"brand": "MobilePlus"}
    ]
  }'
```

### Challenge 3: Complex Business Logic
Find featured products (tagged "flagship" or "premium") that are either:
- Laptops under $1200, OR
- Phones over $800

```bash
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.75, 0.35, 0.45, 0.70, 0.65, 0.60, 0.78, 0.48]" \
  --top-k 5 \
  --filter '{
    "$and": [
      {
        "$or": [
          {"tags": {"$contains": "flagship"}},
          {"tags": {"$contains": "premium"}}
        ]
      },
      {
        "$or": [
          {
            "$and": [
              {"category": "laptop"},
              {"price": {"$lt": 1200}}
            ]
          },
          {
            "$and": [
              {"category": "phone"},
              {"price": {"$gt": 800}}
            ]
          }
        ]
      }
    ]
  }'
```

## Performance Considerations

### Filter First, Then Search
When using strict filters, JadeVectorDB can optimize by:
1. Filtering the dataset first (smaller search space)
2. Then performing similarity search on filtered results

**Best Practices:**
```bash
# GOOD: Specific filters reduce search space
--filter '{"category": "laptop", "in_stock": true}'

# LESS EFFICIENT: Very broad OR conditions
--filter '{"$or": [{"price": {"$gt": 0}}, {"price": {"$lt": 10000}}]}'
```

### Indexed Metadata Fields
Some metadata fields may be indexed for faster filtering:
- Category (typically indexed)
- Brand (typically indexed)
- Boolean flags (in_stock, featured)

**Price range queries** may be slower on large datasets.

## Verification

Run the verification script:

```bash
bash verify.sh
```

Expected outcomes:
- ✅ Category filtering works correctly
- ✅ Price range queries return expected results
- ✅ Boolean filtering excludes out-of-stock items
- ✅ Tag-based filtering finds tagged products
- ✅ Complex AND/OR conditions work as expected

## Real-World Use Cases

### E-Commerce Product Search
```bash
# "Find similar wireless headphones under $300 that are in stock"
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.45, 0.38, 0.51, 0.62, 0.43, 0.71, 0.58, 0.39]" \
  --top-k 10 \
  --filter '{
    "$and": [
      {"category": "audio"},
      {"tags": {"$contains": "wireless"}},
      {"price": {"$lt": 300}},
      {"in_stock": true}
    ]
  }'
```

### Inventory Management
```bash
# "Find all premium products that are out of stock"
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.75, 0.35, 0.45, 0.70, 0.65, 0.60, 0.78, 0.48]" \
  --top-k 20 \
  --filter '{
    "$and": [
      {"tags": {"$contains": "premium"}},
      {"in_stock": false}
    ]
  }'
```

### Price Optimization
```bash
# "Find similar products from competitors"
jade-db search \
  --database-id filtered_products \
  --query-vector "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --top-k 5 \
  --filter '{"brand": {"$ne": "TechCorp"}}'
```

## Common Pitfalls

### ❌ Pitfall 1: Overly Restrictive Filters
```bash
# May return 0 results if no products match ALL conditions
--filter '{
  "$and": [
    {"category": "laptop"},
    {"price": {"$lt": 100}},  # Too low for laptops!
    {"tags": {"$contains": "premium"}}
  ]
}'
```

**Solution:** Relax some conditions or use OR logic.

### ❌ Pitfall 2: Incorrect JSON Syntax
```bash
# WRONG: Missing quotes around field names
--filter '{category: "laptop"}'

# CORRECT:
--filter '{"category": "laptop"}'
```

### ❌ Pitfall 3: Type Mismatches
```bash
# WRONG: Comparing number to string
--filter '{"price": "500"}'  # String instead of number

# CORRECT:
--filter '{"price": 500}'
```

## Next Steps

After completing this exercise:
- **Exercise 4:** Index Management - Optimize performance with different index types
- **Exercise 5:** Advanced Workflows - Production patterns and automation

## Troubleshooting

**No results returned:**
- Check that filters match existing data
- Verify JSON syntax with `echo '{"category": "laptop"}' | jq .`
- Try relaxing filter conditions

**Unexpected results:**
- Review the actual metadata in stored vectors
- Check for type mismatches (string vs number)
- Verify filter operator syntax

**Slow queries:**
- Reduce the search space with more specific filters
- Consider index optimization (Exercise 4)
- Check if metadata fields are indexed
