# Similarity Search Functionality Documentation

## Overview

The JadeVectorDB similarity search functionality provides high-performance vector similarity search capabilities supporting multiple algorithms and advanced filtering options. The system implements cosine similarity, Euclidean distance, and dot product algorithms with K-nearest neighbor (KNN) search capabilities.

## Search Algorithms

### 1. Cosine Similarity
- Measures the cosine of the angle between two vectors
- Formula: `cos(θ) = (A·B) / (||A|| ||B||)`
- Range: [-1, 1], where 1 means vectors are identical in direction
- Use case: Text similarity, document comparison

### 2. Euclidean Distance
- Measures the straight-line distance between two points
- Formula: `d = √[(x₁-x₂)² + (y₁-y₂)² + ...]`
- Range: [0, ∞], where 0 means identical vectors
- Use case: Spatial similarity, geometric relationships

### 3. Dot Product
- Measures the product of vector magnitudes and cosine of angle
- Formula: `A·B = Σ(Aᵢ * Bᵢ)`
- Range: [-∞, ∞]
- Use case: Neural network similarity, unnormalized similarity

## Search Parameters

The search functionality supports the following parameters:

```cpp
struct SearchParams {
    int top_k = 10;  // Number of nearest neighbors to return
    float threshold = 0.0f;  // Minimum similarity threshold
    bool include_vector_data = false;  // Include vector values in results
    bool include_metadata = false;  // Include metadata in results
    std::vector<std::string> filter_tags;  // Tags to filter by
    std::string filter_owner;  // Owner to filter by
    std::string filter_category;  // Category to filter by
    float filter_min_score = 0.0f;  // Minimum score filter
    float filter_max_score = 1.0f;  // Maximum score filter
};
```

## API Endpoints

### Basic Similarity Search
- **Endpoint**: `POST /v1/databases/{databaseId}/search`
- **Description**: Performs cosine similarity search
- **Request Body**:
  ```json
  {
    "queryVector": [0.1, 0.2, 0.3, ...],
    "topK": 10,
    "threshold": 0.5,
    "includeMetadata": true,
    "includeVectorData": false,
    "filters": {
      "tags": ["tag1", "tag2"],
      "owner": "user1",
      "category": "documents",
      "minScore": 0.0,
      "maxScore": 1.0
    }
  }
  ```

### Advanced Similarity Search
- **Endpoint**: `POST /v1/databases/{databaseId}/search/advanced`
- **Description**: Performs advanced search with metadata filtering
- **Request Body**: Same as basic search

## Performance Optimizations

The search implementation includes several performance optimizations:

1. **Top-K Selection**: Uses heap-based selection for efficient K-nearest neighbor retrieval
2. **Early Termination**: Optimized algorithms that can terminate early when possible
3. **Memory Efficiency**: Optimized data structures to reduce memory footprint
4. **SIMD Instructions**: Optimized calculations using SIMD when available

## Metrics and Monitoring

The system provides the following metrics for monitoring search performance:

- `search_requests_total`: Total number of search requests
- `search_results_total`: Total number of search results returned
- `search_request_duration_seconds`: Histogram of search request durations
- `active_searches`: Current number of active searches

## Quality Validation

The system includes search result quality validation to ensure accuracy:

- Brute-force comparison against ground truth for small datasets
- Tolerance-based validation for approximate algorithms
- Performance validation against defined benchmarks

## Security Considerations

- All search endpoints require valid API key authentication
- Granular permissions control via role-based access
- Request rate limiting to prevent abuse
- Input validation to prevent injection attacks

## Performance Benchmarks

The system targets the following performance benchmarks:

- Sub-50ms response times for datasets up to 10 million vectors
- 95% accuracy compared to brute-force search
- Support for 10,000+ vectors per second ingestion
- 1000+ concurrent users support