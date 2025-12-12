# JadeVectorDB Documentation

## Overview

JadeVectorDB is a high-performance distributed vector database designed for similarity search with support for metadata filtering, advanced indexing algorithms, and horizontal scaling. Built in C++20 for maximum performance, it provides an efficient solution for AI/ML applications requiring fast vector similarity search capabilities.

## Architecture

### Core Components

1. **Vector Storage Service**: Manages storing, retrieving, and managing vector embeddings with metadata
2. **Similarity Search Service**: Implements various similarity algorithms (cosine, Euclidean, dot product)
3. **Database Service**: Handles database creation, configuration, and management
4. **Index Service**: Supports multiple indexing algorithms (HNSW, IVF, LSH, FLAT) for optimized search
5. **API Layer**: Provides REST and gRPC interfaces
6. **Security Layer**: Implements authentication, authorization, and audit logging
7. **Monitoring & Metrics**: Provides comprehensive observability

### Data Model

- **Vector**: Contains ID, float values, and metadata
- **Database**: Contains configuration for vector dimensions, index types, etc.
- **Index**: Configuration for similarity search optimization
- **Embedding Model**: Defines how embeddings are generated and processed

## Quickstart Guide

### Prerequisites

- C++20 compatible compiler (GCC 11+ or Clang 12+)
- CMake 3.20+
- Dependencies: Eigen, OpenBLAS/BLIS, FlatBuffers, Apache Arrow, gRPC, Google Test

### Installation

#### Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-organization/JadeVectorDB.git
cd JadeVectorDB

# Build Docker image
docker build -t jadevectordb .

# Run the service
docker run -p 8080:8080 jadevectordb
```

#### Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-organization/JadeVectorDB.git
cd JadeVectorDB

# Build the project
cd backend
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run the service
./jadevectordb
```

### Basic Usage

#### Creating a Database

```bash
curl -X POST "http://localhost:8080/v1/databases" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_database",
    "description": "My first vector database",
    "vectorDimension": 128,
    "indexType": "HNSW"
  }'
```

#### Storing Vectors

```bash
curl -X POST "http://localhost:8080/v1/databases/{database-id}/vectors" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "vector-1",
    "values": [0.1, 0.2, 0.3, ..., 0.128],
    "metadata": {
      "category": "image",
      "tags": ["tag1", "tag2"]
    }
  }'
```

#### Performing Similarity Search

```bash
curl -X POST "http://localhost:8080/v1/databases/{database-id}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "queryVector": [0.15, 0.25, 0.35, ..., 0.118],
    "topK": 5,
    "threshold": 0.5
  }'
```

#### Advanced Search with Filters

```bash
curl -X POST "http://localhost:8080/v1/databases/{database-id}/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "queryVector": [0.15, 0.25, 0.35, ..., 0.118],
    "topK": 10,
    "filters": {
      "category": "image",
      "tags": {"$in": ["tag1", "tag2"]}
    }
  }'
```

## API Documentation

### Authentication

All API requests require an API key in the Authorization header:

```
Authorization: Bearer {api-key}
# or
Authorization: ApiKey {api-key}
```

#### Default Users (Development/Test only)

For local development and testing, JadeVectorDB provides default user accounts:

| Username | Password | Roles |
|----------|----------|-------|
| admin | admin123 | admin, developer, user |
| dev | dev123 | developer, user |
| test | test123 | tester, user |

*Note: Default users are only seeded when `JADE_ENV` is set to `development`, `test`, or `local`.*

**Login Example:**
```bash
curl -X POST "http://localhost:8080/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `201`: Created
- `400`: Bad Request - Invalid input
- `401`: Unauthorized - Invalid API key
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource does not exist
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error

### Endpoints

#### Database Management

- `POST /v1/databases` - Create a new database
- `GET /v1/databases` - List all databases
- `GET /v1/databases/{databaseId}` - Get database details
- `PUT /v1/databases/{databaseId}` - Update database configuration
- `DELETE /v1/databases/{databaseId}` - Delete a database

#### Vector Operations

- `POST /v1/databases/{databaseId}/vectors` - Store a vector
- `GET /v1/databases/{databaseId}/vectors/{vectorId}` - Retrieve a vector
- `PUT /v1/databases/{databaseId}/vectors/{vectorId}` - Update a vector
- `DELETE /v1/databases/{databaseId}/vectors/{vectorId}` - Delete a vector
- `POST /v1/databases/{databaseId}/vectors/batch` - Batch store vectors
- `POST /v1/databases/{databaseId}/vectors/batch-get` - Batch retrieve vectors

#### Search Operations

- `POST /v1/databases/{databaseId}/search` - Similarity search
- `POST /v1/databases/{databaseId}/search/advanced` - Advanced search with metadata filters

#### Index Management

- `POST /v1/databases/{databaseId}/indexes` - Create an index
- `GET /v1/databases/{databaseId}/indexes` - List indexes
- `PUT /v1/databases/{databaseId}/indexes/{indexId}` - Update index configuration
- `DELETE /v1/databases/{databaseId}/indexes/{indexId}` - Delete an index

## Performance Optimization

### Index Selection

- **FLAT**: Best for small datasets (< 10K vectors), exact search
- **HNSW**: Good for most use cases, great balance of speed and accuracy
- **IVF**: Good for very large datasets, controllable trade-off between speed and accuracy
- **LSH**: Good for very high dimensional data, approximate search

### Configuration Best Practices

1. **Vector Dimensions**: Match your embedding model's output dimensions exactly
2. **Index Parameters**: Tune index-specific parameters based on your query patterns
3. **Memory Configuration**: Ensure sufficient memory for active indexes
4. **Batch Operations**: Use batch operations for efficient bulk loading

## Security

### Authentication

- API keys are required for all requests
- Role-based access control (RBAC) restricts operations by user role
- Rate limiting prevents abuse

### Data Protection

- Data at rest encryption (optional)
- Transport Layer Security (TLS) for all API calls
- Audit logging for all operations

## Monitoring and Observability

### Metrics

The system exposes metrics in Prometheus format:

- `GET /metrics` - Prometheus metrics endpoint

Key metrics include:
- Query latency (p50, p95, p99)
- Request rate (queries per second)
- Error rate
- Memory and CPU usage
- Index build progress

### Health Checks

- `GET /health` - Basic health status
- `GET /status` - Detailed system status
- `GET /v1/databases/{databaseId}/status` - Database-specific status

## Troubleshooting

### Common Issues

1. **High Query Latency**:
   - Check if the appropriate index is built
   - Verify system resources (CPU, memory, disk I/O)
   - Consider increasing cache size or rebuilding index

2. **Memory Issues**:
   - Reduce index cache size
   - Limit the number of active databases
   - Use memory-mapped files for large datasets

3. **Connection Problems**:
   - Verify API key is correct
   - Check firewall and network settings
   - Ensure service is running at expected endpoint

### Performance Tuning

- Monitor query patterns and optimize index parameters accordingly
- Use bulk operations for large imports
- Adjust batch sizes based on system capabilities

## Advanced Features

### Lifecycle Management

Configure data retention policies:

```bash
curl -X PUT "http://localhost:8080/v1/databases/{databaseId}/lifecycle" \
  -H "Content-Type: application/json" \
  -d '{
    "maxAgeDays": 30,
    "archiveOnExpire": true,
    "enableCleanup": true
  }'
```

### Embedding Generation

Generate embeddings directly in the database:

```bash
curl -X POST "http://localhost:8080/v1/embeddings/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Text to convert to embedding",
    "input_type": "text",
    "model": "default_model",
    "provider": "default_provider"
  }'
```

For complete API documentation, see the OpenAPI specification in the repository.