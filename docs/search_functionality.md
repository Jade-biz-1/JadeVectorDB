# Search Functionality Documentation

## Overview

The JadeVectorDB provides multiple search algorithms to find similar vectors efficiently. The system supports exact and approximate similarity search methods with various filtering capabilities.

## Search Algorithms

### 1. Cosine Similarity Search

Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity regardless of magnitude.

**Formula:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**Characteristics:**
- Output range: [-1, 1] (or [0, 1] for non-negative vectors)
- Higher values indicate greater similarity
- Invariant to vector magnitude

**Usage Example:**
```cpp
Vector query_vector = /* your query */;
SearchParams params;
params.top_k = 10;  // Return top 10 results
params.threshold = 0.7;  // Minimum similarity threshold
params.include_metadata = true;  // Include metadata in results

auto results = search_service->similarity_search(db_id, query_vector, params);
```

### 2. Euclidean Distance Search

Euclidean distance measures the straight-line distance between two vectors in Euclidean space.

**Formula:**
```
d = √(Σ(Ai - Bi)²)
```

**Characteristics:**
- Output range: [0, ∞)
- Lower values indicate greater similarity
- Sensitive to vector magnitude

**Usage Example:**
```cpp
auto results = search_service->euclidean_search(db_id, query_vector, params);
```

### 3. Dot Product Search

Dot product measures the product of vector magnitudes and the cosine of the angle between them.

**Formula:**
```
A · B = Σ(Ai × Bi)
```

**Characteristics:**
- Output range: [-∞, ∞]
- Higher values indicate greater similarity
- Sensitive to vector magnitude

**Usage Example:**
```cpp
auto results = search_service->dot_product_search(db_id, query_vector, params);
```

## Search Parameters

### SearchParams Structure

```cpp
struct SearchParams {
    int top_k = 10;                    // Number of nearest neighbors to return
    float threshold = 0.0f;            // Minimum similarity threshold
    bool include_vector_data = false;  // Include vector values in results
    bool include_metadata = false;     // Include metadata in results
    std::vector<std::string> filter_tags;  // Tags to filter by
    std::string filter_owner;          // Owner to filter by
    std::string filter_category;       // Category to filter by
    float filter_min_score = 0.0f;     // Minimum score filter
    float filter_max_score = 1.0f;     // Maximum score filter
};
```

## Metadata Filtering

The system supports filtering search results based on vector metadata using a flexible filtering system.

### Available Filter Operators

- `EQUALS`: Exact match
- `NOT_EQUALS`: Inverse match
- `GREATER_THAN`: Numerical comparison
- `GREATER_THAN_OR_EQUAL`: Numerical comparison
- `LESS_THAN`: Numerical comparison
- `LESS_THAN_OR_EQUAL`: Numerical comparison
- `IN`: Check if value is in a list
- `NOT_IN`: Check if value is not in a list

### Complex Filters

Multiple conditions can be combined using AND/OR logic:

```cpp
ComplexFilter filter;
filter.combination = FilterCombination::AND;  // or FilterCombination::OR

FilterCondition condition1;
condition1.field = "metadata.category";
condition1.op = FilterOperator::EQUALS;
condition1.value = "technology";
filter.conditions.push_back(condition1);

FilterCondition condition2;
condition2.field = "metadata.score";
condition2.op = FilterOperator::GREATER_THAN;
condition2.value = "0.7";
filter.conditions.push_back(condition2);
```

## Performance Characteristics

### Time Complexity

- **Linear Search**: O(n×d) where n is the number of vectors and d is the vector dimension
- **With Filtering**: Additional O(n) for metadata filtering

### Performance Benchmarks

The system is designed to meet the following performance targets:

- **Similarity searches**: Return results for 1 million vectors in under 50ms with 95% accuracy
- **Filtered similarity searches**: Return results in under 150 milliseconds for complex queries with multiple metadata filters

## API Endpoints

### POST /v1/databases/{databaseId}/search

Perform similarity search with the specified algorithm and parameters.

**Request Body:**
```json
{
  "queryVector": [0.1, 0.2, 0.3, ...],
  "topK": 10,
  "threshold": 0.7,
  "includeMetadata": true,
  "includeVectorData": false,
  "filters": {
    // Filter conditions here
  }
}
```

**Response:**
```json
[
  {
    "vectorId": "string",
    "similarityScore": float,
    "vector": {  // Included if includeVectorData is true
      "id": "string",
      "values": [float],
      "metadata": {}
    }
  }
]
```

### POST /v1/databases/{databaseId}/search/advanced

Perform similarity search with advanced filtering capabilities.

**Request Body:**
```json
{
  "queryVector": [0.1, 0.2, 0.3, ...],
  "topK": 10,
  "threshold": 0.7,
  "filters": {
    "combination": "AND",  // "AND" or "OR"
    "conditions": [
      {
        "field": "metadata.category",
        "op": "EQUALS",
        "value": "technology"
      }
    ]
  }
}
```

## Best Practices

1. **Set appropriate thresholds** to filter out low-quality results
2. **Use metadata filtering** to narrow down the search space before similarity computation
3. **Optimize top_k values** based on your application needs
4. **Consider pre-filtering** when possible to improve search performance
5. **Use the right algorithm** for your specific use case:
   - Cosine similarity for directional similarity
   - Euclidean distance for spatial proximity
   - Dot product when magnitude matters

## Error Handling

Common error responses include:

- `400 Bad Request`: Invalid query parameters or vector format
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Database or vector not found
- `500 Internal Server Error`: Server-side processing error

## Examples

### Basic Similarity Search

```cpp
// Create a query vector
Vector query_vector;
query_vector.id = "my_query";
query_vector.values = {0.5, 0.3, 0.8, 0.1};  // Example 4-dimensional vector

// Set search parameters
SearchParams params;
params.top_k = 5;
params.threshold = 0.6;

// Perform the search
auto results = search_service->similarity_search("my_database_id", query_vector, params);

// Process results
for (const auto& result : results.value()) {
    std::cout << "Vector ID: " << result.vector_id 
              << ", Similarity: " << result.similarity_score << std::endl;
}
```

### Filtered Search

```cpp
// Set up filtering parameters
SearchParams params;
params.top_k = 10;
params.include_metadata = true;

// The filtering is handled internally based on the parameters
auto results = search_service->similarity_search("my_database_id", query_vector, params);
```