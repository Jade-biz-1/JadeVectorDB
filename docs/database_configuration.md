# Database Configuration Documentation

## Overview

This document provides comprehensive documentation for configuring databases in JadeVectorDB. The system supports flexible database configuration with various options for vector dimensions, indexing, sharding, replication, and metadata schema management.

## Database Creation Parameters

### Required Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `name` | string | Unique name for the database | 1-255 characters, unique per system |
| `vectorDimension` | integer | Dimension of vectors in this database | 1-4096 |

### Optional Parameters

| Parameter | Type | Description | Default Value | Constraints |
|-----------|------|-------------|---------------|-------------|
| `description` | string | Human-readable description of the database | "" | 0-1000 characters |
| `indexType` | string | Type of index for similarity search | "HNSW" | "HNSW", "IVF", "LSH", "FLAT" |
| `indexParameters` | object | Configuration parameters for the selected index type | {} | Varies by index type |
| `sharding` | object | Sharding configuration | {"strategy":"hash","numShards":1} | See Sharding Configuration |
| `replication` | object | Replication configuration | {"factor":1,"sync":true} | See Replication Configuration |
| `embeddingModels` | array | List of embedding models associated with this database | [] | See Embedding Models |
| `metadataSchema` | object | Schema definition for vector metadata | {} | See Metadata Schema |
| `retentionPolicy` | object | Data retention policy | null | See Retention Policy |
| `accessControl` | object | Access control configuration | {"roles":[],"defaultPermissions":[]} | See Access Control |

## Index Configuration

### HNSW (Hierarchical Navigable Small World)

Best for single-node deployments or when maximum accuracy is prioritized.

**Parameters:**
```json
{
  "M": 16,              // Number of connections per element
  "efConstruction": 200, // Construction time ef parameter
  "efSearch": 64        // Search time ef parameter
}
```

### IVF (Inverted File)

Recommended for large, distributed deployments due to superior query routing and scalability.

**Parameters:**
```json
{
  "nlist": 100,         // Number of clusters
  "nprobe": 10          // Number of clusters to search
}
```

### LSH (Locality Sensitive Hashing)

Available for specialized use cases where build time and memory usage are primary constraints.

**Parameters:**
```json
{
  "numTables": 10,      // Number of hash tables
  "numFunctions": 5     // Number of hash functions per table
}
```

### FLAT (Flat Index)

Baseline for comparison, stores vectors as-is without additional indexing.

**Parameters:**
```json
{
  // No additional parameters for FLAT index
}
```

## Sharding Configuration

Controls how data is distributed across nodes in a cluster.

### Properties

| Property | Type | Description | Default | Constraints |
|----------|------|-------------|---------|-------------|
| `strategy` | string | Sharding strategy | "hash" | "hash", "range", "vector" |
| `numShards` | integer | Number of shards | 1 | 1-1000 |

### Strategies

1. **Hash-based Sharding**: Distributes vectors based on a hash of the vector ID
2. **Range-based Sharding**: Distributes vectors based on ranges of vector IDs
3. **Vector-based Sharding**: Distributes vectors based on vector similarity (clustering)

## Replication Configuration

Controls data redundancy and availability.

### Properties

| Property | Type | Description | Default | Constraints |
|----------|------|-------------|---------|-------------|
| `factor` | integer | Number of replicas | 1 | 1-10 |
| `sync` | boolean | Whether to use synchronous replication | true | true/false |

## Embedding Models

Associates embedding models with the database for automatic vector generation.

### Properties

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| `name` | string | Name of the embedding model | Yes |
| `version` | string | Version of the model | Yes |
| `provider` | string | Provider of the model | Yes |
| `inputType` | string | Type of input data | Yes |
| `outputDimension` | integer | Dimension of output vectors | Yes |
| `parameters` | object | Model-specific parameters | No |
| `status` | string | Status of the model | Yes |

## Metadata Schema

Defines the structure and validation rules for vector metadata.

### Example Schema
```json
{
  "author": "string",
  "category": "string",
  "score": "float",
  "tags": "array",
  "created_at": "datetime",
  "is_public": "boolean"
}
```

### Supported Types

| Type | Description | Validation |
|------|-------------|------------|
| `string` | Text data | Length constraints |
| `integer` | Whole numbers | Range constraints |
| `float` | Decimal numbers | Range constraints |
| `boolean` | True/False values | - |
| `array` | Lists of values | Element type constraints |
| `datetime` | Date/time values | ISO 8601 format |
| `object` | Nested objects | Nested schema support |

## Retention Policy

Controls how long data is kept and what happens when it expires.

### Properties

| Property | Type | Description | Default |
|----------|------|-------------|---------|
| `maxAgeDays` | integer | Maximum age of data in days | null (no limit) |
| `archiveOnExpire` | boolean | Archive data instead of deleting | false |

## Access Control

Defines permissions and roles for database access.

### Properties

| Property | Type | Description | Default |
|----------|------|-------------|---------|
| `roles` | array | Roles that can access this database | [] |
| `defaultPermissions` | array | Default permissions for users | [] |

### Supported Permissions

| Permission | Description |
|------------|-------------|
| `database:read` | Read database configuration |
| `database:update` | Update database configuration |
| `database:delete` | Delete the database |
| `vector:add` | Add vectors to the database |
| `vector:read` | Read vectors from the database |
| `vector:update` | Update vectors in the database |
| `vector:delete` | Delete vectors from the database |
| `search:execute` | Execute similarity searches |
| `index:create` | Create indexes |
| `index:update` | Update indexes |
| `index:delete` | Delete indexes |

## API Examples

### Create Database
```bash
curl -X POST "http://localhost:8080/v1/databases" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "product_recommendations",
    "description": "Product recommendation vectors",
    "vectorDimension": 768,
    "indexType": "HNSW",
    "indexParameters": {
      "M": 16,
      "efConstruction": 200,
      "efSearch": 64
    },
    "sharding": {
      "strategy": "hash",
      "numShards": 4
    },
    "replication": {
      "factor": 3,
      "sync": true
    },
    "embeddingModels": [
      {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "version": "1.0",
        "provider": "huggingface",
        "inputType": "text",
        "outputDimension": 384,
        "status": "active"
      }
    ],
    "metadataSchema": {
      "product_id": "string",
      "category": "string",
      "price": "float",
      "tags": "array"
    },
    "retentionPolicy": {
      "maxAgeDays": 365,
      "archiveOnExpire": true
    },
    "accessControl": {
      "roles": ["admin", "user"],
      "defaultPermissions": ["vector:add", "vector:read", "search:execute"]
    }
  }'
```

### List Databases
```bash
curl -X GET "http://localhost:8080/v1/databases?limit=10&offset=0" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Get Database Details
```bash
curl -X GET "http://localhost:8080/v1/databases/DATABASE_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Update Database Configuration
```bash
curl -X PUT "http://localhost:8080/v1/databases/DATABASE_ID" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Updated product recommendation vectors",
    "indexParameters": {
      "M": 32,
      "efConstruction": 400,
      "efSearch": 100
    }
  }'
```

### Delete Database
```bash
curl -X DELETE "http://localhost:8080/v1/databases/DATABASE_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Best Practices

1. **Choose Appropriate Vector Dimensions**: Match your embedding model's output dimension
2. **Select the Right Index Type**: 
   - Use HNSW for single-node deployments prioritizing accuracy
   - Use IVF for large, distributed deployments prioritizing scalability
   - Use LSH for use cases prioritizing fast build times and low memory usage
3. **Configure Sharding Wisely**: 
   - Start with 1 shard for small datasets
   - Increase shards as your data grows
   - Consider your query patterns when choosing sharding strategy
4. **Set Appropriate Replication**: 
   - Use factor 1 for development/testing
   - Use factor 3+ for production environments
   - Consider synchronous vs asynchronous replication based on consistency requirements
5. **Define Metadata Schema**: Plan your metadata structure upfront to ensure consistency
6. **Implement Retention Policies**: Set appropriate data lifecycles to manage storage costs
7. **Configure Access Control**: Use role-based access control to secure your data

## Performance Considerations

1. **Index Selection**: Different indexes have different performance characteristics:
   - HNSW: High accuracy, moderate memory usage
   - IVF: Lower memory usage, good for large datasets
   - LSH: Fast build times, lower accuracy
   - FLAT: Baseline performance, highest memory usage

2. **Sharding**: 
   - More shards can improve query parallelism
   - Too many shards can increase management overhead
   - Consider your cluster size when determining shard count

3. **Replication**: 
   - Higher replication factors improve availability
   - Higher factors increase storage and network overhead
   - Synchronous replication provides stronger consistency but higher latency

4. **Metadata Schema**: 
   - Well-defined schemas improve query performance
   - Complex nested objects may impact performance
   - Consider indexing frequently queried metadata fields