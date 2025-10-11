# Advanced Search API Documentation

## Overview

The JadeVectorDB Advanced Search API provides powerful vector similarity search capabilities with rich metadata filtering options. This API extends the basic similarity search functionality with complex filtering, advanced query composition, and customizable result formatting.

## Base URL

```
https://api.jadevectordb.com/v1
```

## Authentication

All API requests require authentication using an API key:

```
Authorization: Bearer YOUR_API_KEY
```

or

```
Authorization: ApiKey YOUR_API_KEY
```

## Endpoints

### Advanced Similarity Search

#### POST `/databases/{databaseId}/search/advanced`

Performs advanced similarity search with complex metadata filtering and result customization options.

**Path Parameters:**
- `databaseId` (string, required): The unique identifier of the database to search in

**Request Headers:**
```
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

```json
{
  "queryVector": [0.1, 0.2, 0.3, 0.4],
  "searchParams": {
    "topK": 10,
    "threshold": 0.0,
    "includeVectorData": false,
    "includeMetadata": true
  },
  "filters": {
    "simpleFilters": [
      {
        "field": "metadata.owner",
        "operator": "equals",
        "value": "user1"
      },
      {
        "field": "metadata.category",
        "operator": "in",
        "values": ["documents", "images"]
      },
      {
        "field": "metadata.score",
        "operator": "greaterThanOrEqual",
        "value": "0.8"
      }
    ],
    "complexFilters": [
      {
        "combination": "and",
        "conditions": [
          {
            "field": "metadata.tags",
            "operator": "contains",
            "value": "important"
          },
          {
            "field": "metadata.status",
            "operator": "equals",
            "value": "active"
          }
        ]
      }
    ]
  },
  "sortBy": [
    {
      "field": "metadata.score",
      "order": "desc"
    },
    {
      "field": "metadata.created_at",
      "order": "asc"
    }
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "vectorId": "vector_123",
      "similarityScore": 0.95,
      "vectorData": {
        "id": "vector_123",
        "values": [0.1, 0.2, 0.3, 0.4],
        "metadata": {
          "owner": "user1",
          "category": "documents",
          "tags": ["important", "review"],
          "score": 0.95,
          "status": "active",
          "created_at": "2025-01-01T00:00:00Z",
          "updated_at": "2025-01-02T00:00:00Z",
          "custom": {
            "project": "project-alpha",
            "department": "engineering"
          }
        }
      }
    }
  ],
  "totalResults": 1,
  "searchTimeMs": 15.2
}
```

### Filter Operators

The following operators are supported for filtering:

| Operator | Description | Example |
|----------|-------------|---------|
| `equals` | Exact match | `"value" == "target"` |
| `notEquals` | Not equal | `"value" != "target"` |
| `greaterThan` | Greater than | `"score" > 0.8` |
| `greaterThanOrEqual` | Greater than or equal | `"score" >= 0.8` |
| `lessThan` | Less than | `"score" < 0.9` |
| `lessThanOrEqual` | Less than or equal | `"score" <= 0.9` |
| `contains` | String/array contains | `"tags" contains "important"` |
| `notContains` | String/array does not contain | `"tags" not contains "spam"` |
| `in` | Value in list | `"owner" in ["user1", "user2"]` |
| `notIn` | Value not in list | `"category" not in ["spam", "trash"]` |
| `exists` | Field exists | `"custom.field" exists` |
| `notExists` | Field does not exist | `"custom.field" not exists` |
| `matchesRegex` | Regular expression match | `"text" matches "/pattern/i"` |

### Complex Filter Combinations

Complex filters allow combining multiple conditions with logical operators:

```json
{
  "combination": "and",  // or "or"
  "conditions": [
    {
      "field": "metadata.owner",
      "operator": "equals",
      "value": "user1"
    }
  ],
  "nestedFilters": [
    {
      "combination": "or",
      "conditions": [
        {
          "field": "metadata.category",
          "operator": "equals",
          "value": "documents"
        },
        {
          "field": "metadata.category",
          "operator": "equals",
          "value": "images"
        }
      ]
    }
  ]
}
```

## Request Parameters

### Query Vector
- `queryVector` (array of floats, required): The vector to search for similar vectors

### Search Parameters
- `topK` (integer, optional, default: 10): Maximum number of results to return
- `threshold` (float, optional, default: 0.0): Minimum similarity score threshold (0.0 to 1.0)
- `includeVectorData` (boolean, optional, default: false): Whether to include vector values in results
- `includeMetadata` (boolean, optional, default: true): Whether to include metadata in results

### Filters
- `simpleFilters` (array, optional): Simple filter conditions
- `complexFilters` (array, optional): Complex filter combinations

### Sorting
- `sortBy` (array, optional): Fields to sort results by

## Response Fields

- `results` (array): Array of search results
- `totalResults` (integer): Total number of results found
- `searchTimeMs` (float): Time taken for the search in milliseconds

### Result Object
- `vectorId` (string): Unique identifier of the matching vector
- `similarityScore` (float): Similarity score (0.0 to 1.0)
- `vectorData` (object): Vector data if requested (see Vector Model)

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Database not found
- `500 Internal Server Error`: Server-side error

## Rate Limits

The API implements rate limiting to ensure fair usage:
- Free tier: 100 requests per minute
- Paid tiers: Higher limits based on subscription

Rate limit responses include:
```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

## Examples

### Basic Advanced Search

```bash
curl -X POST "https://api.jadevectordb.com/v1/databases/my_database/search/advanced" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queryVector": [0.5, 0.6, 0.7, 0.8],
    "searchParams": {
      "topK": 5,
      "threshold": 0.7
    },
    "filters": {
      "simpleFilters": [
        {
          "field": "metadata.owner",
          "operator": "equals",
          "value": "user1"
        }
      ]
    }
  }'
```

### Complex Filter Search

```bash
curl -X POST "https://api.jadevectordb.com/v1/databases/my_database/search/advanced" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queryVector": [0.1, 0.2, 0.3, 0.4],
    "searchParams": {
      "topK": 10,
      "includeMetadata": true
    },
    "filters": {
      "complexFilters": [
        {
          "combination": "and",
          "conditions": [
            {
              "field": "metadata.score",
              "operator": "greaterThanOrEqual",
              "value": "0.8"
            }
          ],
          "nestedFilters": [
            {
              "combination": "or",
              "conditions": [
                {
                  "field": "metadata.category",
                  "operator": "equals",
                  "value": "documents"
                },
                {
                  "field": "metadata.tags",
                  "operator": "contains",
                  "value": "important"
                }
              ]
            }
          ]
        }
      ]
    }
  }'
```

## Best Practices

1. **Use appropriate thresholds**: Set meaningful thresholds to reduce unnecessary results
2. **Limit topK**: Keep topK reasonable (typically < 100) for optimal performance
3. **Cache results**: Cache search results when appropriate to reduce API calls
4. **Handle rate limits**: Implement exponential backoff for rate limit handling
5. **Validate inputs**: Always validate query vectors and filter parameters before sending
6. **Use specific filters**: The more specific your filters, the better the performance