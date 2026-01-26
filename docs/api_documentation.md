# JadeVectorDB API Documentation

## Overview

The JadeVectorDB API provides programmatic access to vector database operations including database management, vector storage, and similarity search. The API supports both REST and gRPC interfaces.

## Authentication

All API requests require authentication using either a JWT token or an API key in the Authorization header:

```
Authorization: Bearer {jwt-token}
# or
Authorization: Bearer {api-key}
```

### Authentication Lifecycle

**1. User Registration** → **2. Login** → **3. Token-based Access** → **4. Token Refresh/Logout**

| Step | Endpoint | Description |
|------|----------|-------------|
| Register | `POST /v1/auth/register` | Create user account with username/password |
| Login | `POST /v1/auth/login` | Authenticate and receive JWT token |
| Access | Any protected endpoint | Include token in `Authorization: Bearer {token}` header |
| Logout | `POST /v1/auth/logout` | Invalidate current session token |

**Token Expiration**: JWT tokens expire after 24 hours by default. Users must re-authenticate after expiration.

**API Keys**: For machine-to-machine access, use API keys instead of JWT tokens. API keys don't expire unless revoked.

**Default Users (Development/Test only)**:
| Username | Password | Roles |
|----------|----------|-------|
| admin | admin123 | admin, developer, user |
| dev | dev123 | developer, user |
| test | test123 | tester, user |

*Note: Default users are only seeded when `JADEVECTORDB_ENV` is set to `development`, `test`, or `local`.*

### Authentication Endpoints

#### POST /v1/auth/register
- **Description**: Register a new user account
- **Authentication**: Not required
- **Request Body**:
```json
{
  "username": "string",
  "password": "string",
  "email": "string (optional)",
  "roles": ["string"] (optional, defaults to ["user"])
}
```
- **Response** (201 Created):
```json
{
  "success": true,
  "userId": "string",
  "username": "string",
  "message": "User registered successfully"
}
```

#### POST /v1/auth/login
- **Description**: Authenticate user and receive JWT token
- **Authentication**: Not required
- **Request Body**:
```json
{
  "username": "string",
  "password": "string"
}
```
- **Response** (200 OK):
```json
{
  "success": true,
  "userId": "string",
  "username": "string",
  "token": "string (JWT token)",
  "expiresAt": "integer (Unix timestamp)",
  "message": "Login successful"
}
```

#### POST /v1/auth/logout
- **Description**: Revoke current JWT token and end session
- **Authentication**: Required (Bearer token)
- **Response** (200 OK):
```json
{
  "success": true,
  "message": "Logout successful"
}
```

#### POST /v1/auth/forgot-password
- **Description**: Request password reset token
- **Authentication**: Not required
- **Request Body**:
```json
{
  "email": "string (or username)"
}
```
- **Response** (200 OK):
```json
{
  "success": true,
  "message": "If the account exists, a password reset link has been sent"
}
```

#### POST /v1/auth/reset-password
- **Description**: Reset password using reset token
- **Authentication**: Not required
- **Request Body**:
```json
{
  "token": "string (reset token)",
  "newPassword": "string"
}
```
- **Response** (200 OK):
```json
{
  "success": true,
  "message": "Password reset successfully"
}
```

### User Management Endpoints

#### GET /v1/users
- **Description**: List all users (admin only)
- **Authentication**: Required (admin role)
- **Response** (200 OK):
```json
{
  "success": true,
  "users": [
    {
      "userId": "string",
      "username": "string",
      "email": "string",
      "roles": ["string"],
      "status": "active|inactive",
      "createdAt": "timestamp",
      "lastLogin": "timestamp"
    }
  ],
  "count": "integer"
}
```

#### POST /v1/users
- **Description**: Create new user (admin only)
- **Authentication**: Required (admin role)
- **Request Body**:
```json
{
  "username": "string",
  "password": "string",
  "email": "string",
  "roles": ["string"]
}
```
- **Response** (201 Created):
```json
{
  "success": true,
  "userId": "string",
  "username": "string",
  "message": "User created successfully"
}
```

#### PUT /v1/users/{userId}
- **Description**: Update user details
- **Authentication**: Required (admin or own account)
- **Request Body**:
```json
{
  "email": "string (optional)",
  "roles": ["string"] (optional, admin only),
  "status": "string (optional, admin only)"
}
```
- **Response** (200 OK):
```json
{
  "success": true,
  "message": "User updated successfully"
}
```

#### DELETE /v1/users/{userId}
- **Description**: Delete user account
- **Authentication**: Required (admin only)
- **Response** (200 OK):
```json
{
  "success": true,
  "message": "User deleted successfully"
}
```

### API Key Management Endpoints

#### GET /v1/apikeys
- **Description**: List all API keys for authenticated user
- **Authentication**: Required (Bearer token)
- **Response** (200 OK):
```json
{
  "success": true,
  "apiKeys": [
    {
      "keyId": "string",
      "name": "string",
      "description": "string",
      "permissions": ["string"],
      "createdAt": "timestamp",
      "lastUsed": "timestamp",
      "status": "active|revoked"
    }
  ],
  "count": "integer"
}
```

#### POST /v1/apikeys
- **Description**: Create new API key
- **Authentication**: Required (Bearer token)
- **Request Body**:
```json
{
  "userId": "string",
  "description": "string (optional)",
  "permissions": ["string"] (optional)
}
```
- **Response** (201 Created):
```json
{
  "success": true,
  "apiKey": "string (the actual key - save this!)",
  "keyId": "string",
  "message": "API key created successfully"
}
```

**Note**: The API key is only returned once during creation. Store it securely.

#### DELETE /v1/apikeys/{keyId}
- **Description**: Revoke an API key
- **Authentication**: Required (Bearer token or admin)
- **Response** (200 OK):
```json
{
  "success": true,
  "message": "API key revoked successfully"
}
```

## API Endpoints

### Health and Monitoring

#### GET /health
- **Description**: Basic health check endpoint
- **Authentication**: Optional
- **Response**:
```json
{
  "status": "healthy",
  "timestamp": "1700000000000",
  "version": "1.0.0",
  "checks": {
    "database": "ok",
    "storage": "ok",
    "network": "ok"
  }
}
```
- **Example Usage**:
```bash
curl -X GET http://localhost:8080/health
```

#### GET /status
- **Description**: Detailed system status
- **Authentication**: Required
- **Permissions**: monitoring:read
- **Response**:
```json
{
  "status": "operational",
  "timestamp": "1700000000000",
  "uptime": "2h 34m 12s",
  "version": "1.0.0",
  "system": {
    "cpu_usage": 15.3,
    "memory_usage": 45.7,
    "disk_usage": 67.2,
    "network_io": "placeholder_network_io"
  },
  "performance": {
    "avg_query_time_ms": 2.5,
    "qps": 1250,
    "active_connections": 42,
    "total_vectors": 1000000
  }
}
```
- **Example Usage**:
```bash
curl -X GET http://localhost:8080/status \
  -H "Authorization: Bearer <your-api-key>"
```

#### GET /v1/databases/{databaseId}/status
- **Description**: Database-specific status
- **Authentication**: Required
- **Permissions**: monitoring:read
- **Response**:
```json
{
  "databaseId": "example_db_id",
  "status": "online",
  "timestamp": "1700000000000",
  "metrics": {
    "vector_count": 50000,
    "index_count": 3,
    "storage_used_mb": 1024.5,
    "avg_query_time_ms": 1.8,
    "qps": 850
  },
  "indexes": {
    "hnsw_index_1": "ready",
    "ivf_index_1": "ready",
    "flat_index_1": "ready"
  }
}
```
- **Example Usage**:
```bash
curl -X GET http://localhost:8080/v1/databases/my_database_id/status \
  -H "Authorization: Bearer <your-api-key>"
```

### Database Management

#### POST /v1/databases
- **Description**: Create a new database
- **Authentication**: Required
- **Permissions**: database:create
- **Request Body**:
```json
{
  "name": "string",
  "description": "string",
  "vectorDimension": "integer",
  "indexType": "string (FLAT|HNSW|IVF|LSH)",
  "config": {
    "key": "value"
  }
}
```
- **Response**:
```json
{
  "databaseId": "string",
  "status": "success"
}
```

#### GET /v1/databases
- **Description**: List all databases
- **Authentication**: Required
- **Permissions**: database:list
- **Query Parameters**:
  - `name`: Filter by database name
  - `owner`: Filter by owner
  - `limit`: Maximum results to return (max 1000)
  - `offset`: Offset for pagination
- **Response**:
```json
[
  {
    "databaseId": "string",
    "name": "string",
    "description": "string",
    "vectorDimension": "integer",
    "indexType": "string",
    "created_at": "timestamp",
    "updated_at": "timestamp"
  }
]
```

#### GET /v1/databases/{databaseId}
- **Description**: Get database details
- **Authentication**: Required
- **Permissions**: database:read
- **Response**:
```json
{
  "databaseId": "string",
  "name": "string",
  "description": "string",
  "vectorDimension": "integer",
  "indexType": "string",
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

#### PUT /v1/databases/{databaseId}
- **Description**: Update database configuration
- **Authentication**: Required
- **Permissions**: database:update
- **Request Body**:
```json
{
  "name": "string",
  "description": "string",
  "config": {
    "key": "value"
  }
}
```
- **Response**:
```json
{
  "status": "success",
  "message": "Database updated successfully"
}
```

#### DELETE /v1/databases/{databaseId}
- **Description**: Delete a database
- **Authentication**: Required
- **Permissions**: database:delete
- **Response**:
```json
{
  "status": "success",
  "message": "Database deleted successfully"
}
```

### Vector Operations

#### POST /v1/databases/{databaseId}/vectors
- **Description**: Store a vector
- **Authentication**: Required
- **Permissions**: vector:add
- **Request Body**:
```json
{
  "id": "string",
  "values": ["float_array"],
  "metadata": {
    "key": "value"
  }
}
```
- **Response**:
```json
{
  "status": "success",
  "vectorId": "string"
}
```

#### GET /v1/databases/{databaseId}/vectors/{vectorId}
- **Description**: Retrieve a vector
- **Authentication**: Required
- **Permissions**: vector:read
- **Response**:
```json
{
  "id": "string",
  "values": ["float_array"],
  "metadata": {
    "key": "value"
  }
}
```

#### PUT /v1/databases/{databaseId}/vectors/{vectorId}
- **Description**: Update a vector
- **Authentication**: Required
- **Permissions**: vector:update
- **Request Body**:
```json
{
  "values": ["float_array"],
  "metadata": {
    "key": "value"
  }
}
```
- **Response**:
```json
{
  "status": "success"
}
```

#### DELETE /v1/databases/{databaseId}/vectors/{vectorId}
- **Description**: Delete a vector
- **Authentication**: Required
- **Permissions**: vector:delete
- **Response**:
```json
{
  "status": "success"
}
```

#### POST /v1/databases/{databaseId}/vectors/batch
- **Description**: Batch store vectors
- **Authentication**: Required
- **Permissions**: vector:add
- **Request Body**:
```json
{
  "vectors": [
    {
      "id": "string",
      "values": ["float_array"],
      "metadata": {
        "key": "value"
      }
    }
  ]
}
```
- **Response**:
```json
{
  "status": "success",
  "count": "integer"
}
```

#### POST /v1/databases/{databaseId}/vectors/batch-get
- **Description**: Batch retrieve vectors
- **Authentication**: Required
- **Permissions**: vector:read
- **Request Body**:
```json
{
  "vectorIds": ["string_array"]
}
```
- **Response**:
```json
[
  {
    "id": "string",
    "values": ["float_array"],
    "metadata": {
      "key": "value"
    }
  }
]
```

### Search Operations

#### POST /v1/databases/{databaseId}/search
- **Description**: Perform similarity search to find vectors most similar to the query vector
- **Authentication**: Required
- **Permissions**: search:execute
- **Request Body**:
```json
{
  "queryVector": [0.1, 0.2, 0.3, ...],  // Array of floats representing the query vector
  "topK": 10,                            // Number of most similar vectors to return (default: 10)
  "threshold": 0.0,                      // Minimum similarity score threshold (default: 0.0)
  "includeMetadata": true,               // Include vector metadata in results (default: false)
  "includeVectorData": false             // Include vector values in results (default: false)
}
```
- **Response** (200 OK):
```json
{
  "results": [
    {
      "vectorId": "string",              // ID of the matching vector
      "similarityScore": 0.95,           // Similarity score (0.0 to 1.0 for cosine similarity)
      "vector": {                        // Included based on request parameters
        "id": "string",                  // Vector ID
        "values": [0.1, 0.2, ...],       // Vector values (only if includeVectorData is true)
        "metadata": {                    // Vector metadata (only if includeMetadata is true)
          "category": "string",
          "tags": ["tag1", "tag2"],
          "owner": "string",
          "custom_field": "value"
        }
      }
    }
  ],
  "count": 10                            // Number of results returned
}
```
- **Notes**:
  - Results are ordered by similarity score (highest first)
  - The `vector` object structure depends on `includeMetadata` and `includeVectorData` parameters
  - Similarity scores range from 0.0 (least similar) to 1.0 (most similar) for cosine similarity

#### POST /v1/databases/{databaseId}/search/advanced
- **Description**: Advanced similarity search with metadata filtering capabilities
- **Authentication**: Required
- **Permissions**: search:execute
- **Request Body**:
```json
{
  "queryVector": [0.1, 0.2, 0.3, ...],   // Array of floats representing the query vector
  "topK": 10,                             // Number of most similar vectors to return (default: 10)
  "threshold": 0.0,                       // Minimum similarity score threshold (default: 0.0)
  "includeMetadata": true,                // Include vector metadata in results (default: false)
  "includeVectorData": false,             // Include vector values in results (default: false)
  "filters": {                            // Optional metadata filters
    "combination": "AND",                 // Combine conditions with "AND" or "OR" (default: "AND")
    "conditions": [
      {
        "field": "metadata.category",    // Metadata field to filter on
        "op": "EQUALS",                   // Filter operator (see below)
        "value": "technology"             // Value to match
      },
      {
        "field": "metadata.score",
        "op": "GREATER_THAN",
        "value": "0.7"
      }
    ]
  }
}
```
- **Filter Operators**:
  - `EQUALS`: Exact match
  - `NOT_EQUALS`: Inverse match
  - `GREATER_THAN`: Numerical comparison (>)
  - `GREATER_THAN_OR_EQUAL`: Numerical comparison (>=)
  - `LESS_THAN`: Numerical comparison (<)
  - `LESS_THAN_OR_EQUAL`: Numerical comparison (<=)
  - `IN`: Check if value is in a list
  - `NOT_IN`: Check if value is not in a list

- **Response** (200 OK):
```json
{
  "results": [
    {
      "vectorId": "string",              // ID of the matching vector
      "similarityScore": 0.95,           // Similarity score (0.0 to 1.0 for cosine similarity)
      "vector": {                        // Included based on request parameters
        "id": "string",                  // Vector ID
        "values": [0.1, 0.2, ...],       // Vector values (only if includeVectorData is true)
        "metadata": {                    // Vector metadata (only if includeMetadata is true)
          "category": "technology",      // Matches filter condition
          "score": 0.85,                 // Matches filter condition
          "tags": ["ai", "ml"],
          "owner": "user123"
        }
      }
    }
  ],
  "count": 10,                           // Number of results returned
  "filtersApplied": true                 // Indicates whether filters were applied
}
```
- **Notes**:
  - Results are first filtered by metadata conditions, then ranked by similarity score
  - Filters are applied before similarity search for better performance
  - Multiple conditions can be combined with AND/OR logic
  - Filter values are always strings and are type-coerced when comparing

#### POST /v1/databases/{databaseId}/search/hybrid
- **Description**: Hybrid search combining vector similarity and keyword (BM25) search
- **Authentication**: Required
- **Permissions**: search:execute
- **Request Body**:
```json
{
  "queryVector": [0.1, 0.2, 0.3, ...],   // Array of floats for vector search (optional)
  "queryText": "machine learning",       // Text query for BM25 keyword search (optional)
  "topK": 10,                             // Number of results to return (default: 10)
  "fusionMethod": "rrf",                  // "rrf" (Reciprocal Rank Fusion) or "linear" (default: "rrf")
  "alpha": 0.7,                           // Weight for vector scores in linear fusion (0.0-1.0, default: 0.7)
  "k": 60,                                // RRF k parameter (default: 60)
  "includeMetadata": true,                // Include vector metadata in results (default: true)
  "includeVectorData": false,             // Include vector values in results (default: false)
  "includeScores": true                   // Include individual scores (vector, BM25) (default: true)
}
```
- **Fusion Methods**:
  - `rrf` (Reciprocal Rank Fusion): Rank-based fusion that combines results based on their positions in the ranked lists. Formula: `1 / (k + rank)`, where k is typically 60.
  - `linear` (Weighted Linear): Score-based fusion that combines normalized scores using weighted average. Formula: `alpha * vector_score + (1 - alpha) * bm25_score`

- **Notes on Query Parameters**:
  - At least one of `queryVector` or `queryText` must be provided
  - If only `queryVector` is provided, performs vector-only search
  - If only `queryText` is provided, performs BM25-only search
  - If both are provided, performs true hybrid search with score fusion

- **Response** (200 OK):
```json
{
  "count": 10,                            // Number of results returned
  "fusionMethod": "rrf",                  // Fusion method used
  "results": [
    {
      "vectorId": "doc_123",              // ID of the matching vector/document
      "hybridScore": 0.95,                // Combined hybrid score
      "vectorScore": 0.92,                // Vector similarity score (if includeScores is true)
      "bm25Score": 0.88,                  // BM25 keyword score (if includeScores is true)
      "vector": {                         // Included based on request parameters
        "id": "doc_123",
        "values": [0.1, 0.2, ...],        // Only if includeVectorData is true
        "metadata": {                     // Only if includeMetadata is true
          "source": "Machine learning document",
          "category": "AI",
          "tags": ["ml", "algorithms"],
          "owner": "researcher1"
        }
      }
    }
  ]
}
```

- **Use Cases**:
  - **Semantic + Keyword Search**: Find documents that are semantically similar AND contain specific keywords
  - **Enhanced Recall**: Combine the strengths of both vector similarity (semantic understanding) and keyword matching (exact term matching)
  - **Ranking Diversity**: Get diverse results by fusing different ranking signals

- **Best Practices**:
  - Use `fusionMethod: "rrf"` for general-purpose hybrid search (more robust to score scale differences)
  - Use `fusionMethod: "linear"` when you want fine control over the balance between vector and keyword signals
  - Adjust `alpha` parameter (0.0 to 1.0) to control the weight: higher values favor vector similarity, lower values favor keyword matching
  - For alpha: 0.7 is a good default (70% vector, 30% BM25), adjust based on your use case
  - BM25 index must be built before performing hybrid or BM25-only search

- **Notes**:
  - Results are ordered by hybrid score (highest first)
  - Hybrid scores combine both vector similarity and BM25 keyword relevance
  - The BM25 index must be populated with document text for keyword search to work
  - Vector search uses the existing similarity search service (HNSW/IVF indexes)

#### POST /v1/databases/{databaseId}/search/rerank
- **Description**: Hybrid search with optional cross-encoder reranking for improved relevance
- **Authentication**: Required
- **Permissions**: search:execute
- **Request Body**:
```json
{
  "queryVector": [0.1, 0.2, 0.3, ...],   // Array of floats (optional)
  "queryText": "machine learning",       // Text query (optional)
  "topK": 10,                            // Number of results to return
  "enableReranking": true,               // Enable cross-encoder reranking (default: false)
  "rerankTopN": 100,                     // Number of candidates to rerank (default: 100)
  "fusionMethod": "rrf",                 // Fusion method: "rrf" or "linear"
  "alpha": 0.7,                          // Weight for linear fusion (0.0-1.0)
  "metadataFilter": {                    // Optional metadata filters
    "category": "research",
    "status": "published"
  }
}
```
- **Response**:
```json
{
  "results": [
    {
      "id": "vector-id",
      "vectorScore": 0.85,        // Cosine similarity score
      "bm25Score": 0.72,          // BM25 keyword score
      "hybridScore": 0.80,        // Fused score (RRF or linear)
      "rerankScore": 0.92,        // Cross-encoder relevance score
      "combinedScore": 0.87,      // Final score (hybrid + rerank)
      "metadata": {
        "source": "document text for this vector...",
        "category": "research"
      }
    }
  ],
  "timings": {
    "retrievalMs": 15,     // Time for hybrid search retrieval
    "rerankingMs": 150,    // Time for cross-encoder reranking
    "totalMs": 165         // Total query time
  }
}
```

- **Best Practices**:
  - **Two-Stage Retrieval**: Set `rerankTopN` > `topK` (e.g., retrieve 100, rerank, return top 10)
  - **Performance**: Reranking adds 50-300ms latency depending on `rerankTopN` and hardware
  - **When to Rerank**: Best for critical queries where precision is more important than latency
  - **Model Selection**: Default model is `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good quality)
  - **Score Combination**: Weighted average of hybrid score (30%) and rerank score (70%) by default

- **Notes**:
  - Reranking uses a cross-encoder model running in a Python subprocess
  - The subprocess is started automatically on first use and cached for subsequent requests
  - Results are ordered by `combinedScore` when reranking is enabled
  - Without reranking, behavior is identical to `/search/hybrid` endpoint
  - Reranking significantly improves precision@5 (typically +15-25%) at the cost of latency

#### POST /v1/rerank
- **Description**: Standalone reranking endpoint for arbitrary query-document pairs (no database required)
- **Authentication**: Required
- **Permissions**: search:execute
- **Request Body**:
```json
{
  "query": "machine learning algorithms",
  "documents": [
    {"id": "doc1", "text": "Machine learning is a subset of AI..."},
    {"id": "doc2", "text": "Neural networks are computing systems..."},
    {"id": "doc3", "text": "Data science involves statistical analysis..."}
  ],
  "topK": 5                // Optional: number of results to return
}
```
- **Response**:
```json
{
  "results": [
    {
      "id": "doc1",
      "score": 0.92,     // Cross-encoder relevance score (0-1)
      "rank": 1          // Rank position after reranking
    },
    {
      "id": "doc2",
      "score": 0.78,
      "rank": 2
    }
  ],
  "latencyMs": 145
}
```

- **Use Cases**:
  - Rerank search results from external systems
  - Compare query relevance across arbitrary documents
  - A/B testing different ranking strategies
  - Post-processing results from other vector databases

- **Notes**:
  - This endpoint does not require a database to be created
  - Documents must include `id` and `text` fields
  - Results are ordered by score (descending)
  - Suitable for up to ~200 documents per request

#### GET /v1/databases/{databaseId}/reranking/config
- **Description**: Get current reranking configuration
- **Authentication**: Required
- **Permissions**: database:read
- **Response**:
```json
{
  "modelName": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "batchSize": 32,
  "scoreThreshold": 0.0,
  "combineScores": true,
  "rerankWeight": 0.7,
  "statistics": {
    "totalRequests": 1523,
    "failedRequests": 2,
    "avgLatencyMs": 145.2,
    "totalDocumentsReranked": 152300
  }
}
```

#### PUT /v1/databases/{databaseId}/reranking/config
- **Description**: Update reranking configuration for a database
- **Authentication**: Required
- **Permissions**: database:update
- **Request Body**:
```json
{
  "modelName": "cross-encoder/ms-marco-MiniLM-L-12-v2",  // Cross-encoder model
  "batchSize": 32,                                        // Batch size for inference
  "scoreThreshold": 0.0,                                  // Min score to include
  "combineScores": true,                                  // Combine with hybrid score
  "rerankWeight": 0.7                                     // Weight for rerank score (0-1)
}
```
- **Response**:
```json
{
  "success": true,
  "config": {
    "modelName": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "batchSize": 32,
    "scoreThreshold": 0.0,
    "combineScores": true,
    "rerankWeight": 0.7
  }
}
```

- **Available Models**:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (default): Fast, good quality, 80MB
  - `cross-encoder/ms-marco-MiniLM-L-12-v2`: Better quality, slower, 130MB
  - `cross-encoder/ms-marco-TinyBERT-L-2-v2`: Fastest, lower quality, 40MB
  - Any Hugging Face cross-encoder model compatible with `sentence-transformers`

- **Notes**:
  - Configuration changes require restarting the reranking subprocess
  - Model downloads happen automatically on first use
  - `rerankWeight` controls score fusion: 1.0 = only rerank score, 0.0 = only hybrid score
  - `scoreThreshold` filters out low-relevance results after reranking

### Index Management

#### POST /v1/databases/{databaseId}/indexes
- **Description**: Create an index
- **Authentication**: Required
- **Permissions**: index:create
- **Request Body**:
```json
{
  "type": "string (FLAT|HNSW|IVF|LSH)",
  "name": "string",
  "description": "string",
  "parameters": {
    "key": "value"
  }
}
```
- **Response**:
```json
{
  "indexId": "string",
  "databaseId": "string",
  "type": "string",
  "parameters": {
    "key": "value"
  },
  "status": "created",
  "createdAt": "timestamp"
}
```

#### GET /v1/databases/{databaseId}/indexes
- **Description**: List indexes
- **Authentication**: Required
- **Permissions**: index:read
- **Response**:
```json
[
  {
    "indexId": "string",
    "databaseId": "string",
    "type": "string",
    "status": "string",
    "parameters": {
      "key": "value"
    },
    "createdAt": "timestamp",
    "updatedAt": "timestamp"
  }
]
```

#### PUT /v1/databases/{databaseId}/indexes/{indexId}
- **Description**: Update index configuration
- **Authentication**: Required
- **Permissions**: index:update
- **Request Body**:
```json
{
  "parameters": {
    "key": "value"
  }
}
```
- **Response**:
```json
{
  "indexId": "string",
  "databaseId": "string",
  "status": "updated",
  "updatedAt": "timestamp"
}
```

#### DELETE /v1/databases/{databaseId}/indexes/{indexId}
- **Description**: Delete an index
- **Authentication**: Required
- **Permissions**: index:delete
- **Response**:
```json
{
  "indexId": "string",
  "databaseId": "string",
  "status": "deleted",
  "deletedAt": "timestamp"
}
```

### Embedding Generation

#### POST /v1/embeddings/generate
- **Description**: Generate embedding from input
- **Authentication**: Required
- **Permissions**: embedding:generate
- **Request Body**:
```json
{
  "input": "string",
  "input_type": "string (text|image)",
  "model": "string",
  "provider": "string"
}
```
- **Response**:
```json
{
  "input": "string",
  "input_type": "string",
  "model": "string",
  "provider": "string",
  "embedding": ["float_array"],
  "dimension": "integer",
  "status": "success",
  "generated_at": "timestamp"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `201`: Created
- `400`: Bad Request - Invalid input
- `401`: Unauthorized - Invalid API key
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource does not exist
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error

### Error Response Format

All error responses follow this format:

```json
{
  "error": "error_message"
}
```

## Rate Limits

The API implements rate limiting per API key with default limits:

- Database operations: 100 requests per minute
- Vector operations: 1000 requests per minute
- Search operations: 500 requests per minute
- Embedding generation: 10 requests per minute

### Distributed System Management

#### GET /v1/cluster/status
- **Description**: Get cluster status and member information
- **Authentication**: Required
- **Permissions**: cluster:read
- **Response**:
```json
{
  "cluster_id": "string",
  "nodes": [
    {
      "node_id": "string",
      "address": "string",
      "status": "active|inactive",
      "role": "master|worker",
      "health": "healthy|degraded|unhealthy"
    }
  ],
  "total_nodes": "integer",
  "healthy_nodes": "integer"
}
```

#### GET /v1/performance/metrics
- **Description**: Get performance metrics for the system
- **Authentication**: Required
- **Permissions**: metrics:read
- **Response**:
```json
{
  "cpu_usage": "float",
  "memory_usage": "float",
  "disk_usage": "float",
  "query_latency_p50": "float",
  "query_latency_p95": "float",
  "query_latency_p99": "float",
  "queries_per_second": "float",
  "timestamp": "timestamp"
}
```

#### GET /v1/alerts
- **Description**: Get active alerts and notifications
- **Authentication**: Required
- **Permissions**: alerts:read
- **Response**:
```json
{
  "alerts": [
    {
      "alert_id": "string",
      "severity": "info|warning|error|critical",
      "message": "string",
      "timestamp": "timestamp",
      "acknowledged": "boolean"
    }
  ],
  "total_alerts": "integer"
}
```

*Note: Cluster, performance, and alert endpoints are currently stub implementations and will be fully implemented in future releases.*

## SDKs and Client Libraries

For convenience, the following client libraries are available:

- Python client (pip install jadevectordb)
- JavaScript/TypeScript client (npm install @jadevectordb/client)
- Go client (go get github.com/jadevectordb/client-go)

## OpenAPI Specification

For detailed API specification, see the OpenAPI 3.0 document at `/spec/openapi.yaml` or in the `contracts/` directory of the repository.