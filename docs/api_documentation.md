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

## SDKs and Client Libraries

For convenience, the following client libraries are available:

- Python client (pip install jadevectordb)
- JavaScript/TypeScript client (npm install @jadevectordb/client)
- Go client (go get github.com/jadevectordb/client-go)

## OpenAPI Specification

For detailed API specification, see the OpenAPI 3.0 document at `/spec/openapi.yaml` or in the `contracts/` directory of the repository.