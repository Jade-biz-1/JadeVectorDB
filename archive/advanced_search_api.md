# Advanced Search API Endpoints Documentation

## Overview

The Advanced Search API provides sophisticated vector similarity search capabilities with complex metadata filtering, multi-condition queries, and enhanced result options. This API extends the basic search functionality with more powerful filtering and querying mechanisms.

## API Base URL

```
https://your-jadevectordb-host.com/v1
```

## Authentication

All API requests require a valid API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
# or
Authorization: ApiKey YOUR_API_KEY
```

## API Endpoints

### 1. Basic Similarity Search

**Endpoint:** `POST /databases/{databaseId}/search`

**Description:** Performs a basic similarity search using cosine similarity by default.

#### Request Parameters
- `{databaseId}` (path): The ID of the database to search in

#### Request Body
```json
{
  "queryVector": [float, float, ...],
  "topK": int,
  "threshold": float,
  "includeMetadata": boolean,
  "includeVectorData": boolean
}
```

**Fields:**
- `queryVector`: Array of floats representing the query vector
- `topK`: Number of nearest neighbors to return (default: 10)
- `threshold`: Minimum similarity threshold [0.0, 1.0] (default: 0.0)
- `includeMetadata`: Include metadata in results (default: false)
- `includeVectorData`: Include vector values in results (default: false)

#### Example Request
```bash
curl -X POST \
  https://your-jadevectordb-host.com/v1/databases/my_db_id/search \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "queryVector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "topK": 5,
    "threshold": 0.7,
    "includeMetadata": true
  }'
```

#### Example Response
```json
{
  "results": [
    {
      "vectorId": "vec_123",
      "similarityScore": 0.92,
      "vector": {
        "id": "vec_123",
        "metadata": {
          "category": "technology",
          "tags": ["ai", "ml"],
          "score": 0.85
        }
      }
    }
  ]
}
```

### 2. Advanced Search with Filtering

**Endpoint:** `POST /databases/{databaseId}/search/advanced`

**Description:** Performs similarity search with complex metadata filtering capabilities.

#### Request Parameters
- `{databaseId}` (path): The ID of the database to search in

#### Request Body
```json
{
  "queryVector": [float, float, ...],
  "topK": int,
  "threshold": float,
  "includeMetadata": boolean,
  "includeVectorData": boolean,
  "filters": {
    "combination": "AND" | "OR",
    "conditions": [
      {
        "field": "metadata.category",
        "op": "EQUALS" | "NOT_EQUALS" | "GREATER_THAN" | "GREATER_THAN_OR_EQUAL" | "LESS_THAN" | "LESS_THAN_OR_EQUAL" | "IN" | "NOT_IN",
        "value": "value_to_compare"
      }
    ]
  }
}
```

**Fields:**
- `queryVector`: Array of floats representing the query vector
- `topK`: Number of nearest neighbors to return (default: 10)
- `threshold`: Minimum similarity threshold [0.0, 1.0] (default: 0.0)
- `includeMetadata`: Include metadata in results (default: false)
- `includeVectorData`: Include vector values in results (default: false)
- `filters`: Object containing filtering conditions
  - `combination`: How to combine multiple conditions ("AND" or "OR")
  - `conditions`: Array of filter conditions
    - `field`: The metadata field to filter on (e.g., "metadata.category", "metadata.tags")
    - `op`: The comparison operator
    - `value`: The value to compare against

#### Filter Operators

| Operator | Description | Value Type |
|----------|-------------|------------|
| `EQUALS` | Exact match | string, number |
| `NOT_EQUALS` | Inverse match | string, number |
| `GREATER_THAN` | Numerical comparison | number |
| `GREATER_THAN_OR_EQUAL` | Numerical comparison | number |
| `LESS_THAN` | Numerical comparison | number |
| `LESS_THAN_OR_EQUAL` | Numerical comparison | number |
| `IN` | Check if value is in a list | string, number |
| `NOT_IN` | Check if value is not in a list | string, number |

#### Example Request
```bash
curl -X POST \
  https://your-jadevectordb-host.com/v1/databases/my_db_id/search/advanced \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "queryVector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "topK": 10,
    "threshold": 0.6,
    "includeMetadata": true,
    "filters": {
      "combination": "AND",
      "conditions": [
        {
          "field": "metadata.category",
          "op": "EQUALS",
          "value": "technology"
        },
        {
          "field": "metadata.score",
          "op": "GREATER_THAN_OR_EQUAL",
          "value": "0.7"
        },
        {
          "field": "metadata.tags",
          "op": "IN",
          "value": "ai"
        }
      ]
    }
  }'
```

#### Example Response
```json
{
  "results": [
    {
      "vectorId": "vec_456",
      "similarityScore": 0.88,
      "vector": {
        "id": "vec_456",
        "metadata": {
          "category": "technology",
          "tags": ["ai", "ml", "research"],
          "score": 0.85,
          "created_at": "2023-10-15T10:30:00Z"
        }
      }
    },
    {
      "vectorId": "vec_789",
      "similarityScore": 0.82,
      "vector": {
        "id": "vec_789",
        "metadata": {
          "category": "technology",
          "tags": ["ai", "deep_learning"],
          "score": 0.78,
          "created_at": "2023-10-16T14:20:00Z"
        }
      }
    }
  ]
}
```

### 3. Euclidean Distance Search

**Endpoint:** `POST /databases/{databaseId}/search/euclidean`

**Description:** Performs similarity search using Euclidean distance metric.

#### Request Body
Same as basic search but optimized for Euclidean distance:
```json
{
  "queryVector": [float, float, ...],
  "topK": int,
  "threshold": float,
  "includeMetadata": boolean,
  "includeVectorData": boolean
}
```

### 4. Dot Product Search

**Endpoint:** `POST /databases/{databaseId}/search/dotproduct`

**Description:** Performs similarity search using dot product metric.

#### Request Body
Same as basic search but optimized for dot product:
```json
{
  "queryVector": [float, float, ...],
  "topK": int,
  "threshold": float,
  "includeMetadata": boolean,
  "includeVectorData": boolean
}
```

## Common Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (invalid API key) |
| 404 | Database or vector not found |
| 429 | Rate limited (too many requests) |
| 500 | Internal server error |

## Request/Response Examples

### Example 1: Search with Multiple Conditions

```json
{
  "queryVector": [0.5, 0.2, 0.8, 0.1],
  "topK": 5,
  "threshold": 0.65,
  "includeMetadata": true,
  "filters": {
    "combination": "AND",
    "conditions": [
      {
        "field": "metadata.category",
        "op": "EQUALS",
        "value": "finance"
      },
      {
        "field": "metadata.score",
        "op": "GREATER_THAN",
        "value": "0.5"
      }
    ]
  }
}
```

### Example 2: Search with OR Combination

```json
{
  "queryVector": [0.3, 0.9, 0.1, 0.7],
  "topK": 8,
  "threshold": 0.5,
  "includeMetadata": false,
  "filters": {
    "combination": "OR",
    "conditions": [
      {
        "field": "metadata.tags",
        "op": "IN",
        "value": "ai"
      },
      {
        "field": "metadata.tags",
        "op": "IN",
        "value": "blockchain"
      }
    ]
  }
}
```

### Example 3: Search with Range Filter

```json
{
  "queryVector": [0.6, 0.4, 0.3, 0.7],
  "topK": 10,
  "threshold": 0.4,
  "includeMetadata": true,
  "filters": {
    "combination": "AND",
    "conditions": [
      {
        "field": "metadata.score",
        "op": "GREATER_THAN_OR_EQUAL",
        "value": "0.6"
      },
      {
        "field": "metadata.score",
        "op": "LESS_THAN_OR_EQUAL",
        "value": "0.9"
      }
    ]
  }
}
```

## Performance Considerations

1. **Vector Dimension**: Higher dimensional vectors require more processing time
2. **Database Size**: Search time scales with the number of vectors in the database
3. **Filtering Complexity**: Complex filters with multiple conditions may impact performance
4. **Result Size**: Larger topK values will take more time to return

## Error Responses

All error responses follow the same format:

```json
{
  "error": "Error message",
  "code": "error_code",
  "timestamp": "ISO 8601 timestamp"
}
```

### Common Error Codes
- `INVALID_ARGUMENT`: Request contains invalid parameters
- `NOT_FOUND`: Requested resource does not exist
- `RESOURCE_EXHAUSTED`: Request exceeds rate limits
- `INTERNAL_ERROR`: Server-side error occurred
- `UNAUTHENTICATED`: Failed to authenticate request

## Best Practices

1. **Set appropriate thresholds** to optimize results and performance
2. **Use filters effectively** to narrow down the search space
3. **Consider vector dimension** when designing your search strategy
4. **Monitor API usage** to stay within rate limits
5. **Optimize topK values** based on actual requirements