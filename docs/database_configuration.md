# Database Configuration Documentation

## Overview

JadeVectorDB supports multiple database instances, each with configurable parameters for vector dimensions, indexing algorithms, metadata schemas, and other operational settings. This document describes how to create, configure, manage, and optimize database instances.

## Database Configuration Parameters

### Core Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Unique name for the database |
| `description` | string | No | "" | Human-readable description |
| `vectorDimension` | integer | Yes | - | Dimension of vectors stored in this database (1-4096) |
| `indexType` | string | No | "HNSW" | Indexing algorithm to use ("HNSW", "IVF", "LSH", "FLAT") |

### Indexing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indexParameters` | object | See below | Configuration parameters for the selected index type |

#### HNSW Index Parameters
```json
{
  "M": 16,
  "efConstruction": 200,
  "efSearch": 64
}
```

#### IVF Index Parameters
```json
{
  "nlist": 100,
  "nprobe": 10
}
```

#### LSH Index Parameters
```json
{
  "nBuckets": 1000,
  "nProbes": 5
}
```

### Sharding Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sharding.strategy` | string | "hash" | Sharding strategy ("hash", "range", "vector-based") |
| `sharding.numShards` | integer | 1 | Number of shards for distributed storage |

### Replication Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `replication.factor` | integer | 1 | Number of replicas per shard |
| `replication.sync` | boolean | false | Whether to use synchronous replication |

### Retention Policy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retentionPolicy.maxAgeDays` | integer | null | Maximum age in days before archival/deletion |
| `retentionPolicy.archiveOnExpire` | boolean | true | Whether to archive instead of delete expired data |

### Access Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accessControl.roles` | array | ["admin", "user", "auditor"] | Available roles in the system |
| `accessControl.defaultPermissions` | array | ["read", "search"] | Default permissions for users |

## Creating a Database

### API Endpoint
```
POST /v1/databases
```

### Example Request
```bash
curl -X POST \
  https://your-jadevectordb-host.com/v1/databases \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my_vectors_db",
    "description": "Database for storing document embeddings",
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
      "factor": 2,
      "sync": false
    },
    "metadataSchema": {
      "category": "string",
      "tags": "array<string>",
      "score": "float",
      "created_at": "timestamp",
      "custom_data": "object"
    },
    "retentionPolicy": {
      "maxAgeDays": 365,
      "archiveOnExpire": true
    }
  }'
```

### Response
```json
{
  "databaseId": "db_random_id_12345",
  "status": "created",
  "message": "Database created successfully"
}
```

## Managing Databases

### List All Databases
```
GET /v1/databases
```

#### Example Request
```bash
curl -X GET \
  https://your-jadevectordb-host.com/v1/databases \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

#### Example Response
```json
[
  {
    "databaseId": "db_random_id_12345",
    "name": "my_vectors_db",
    "description": "Database for storing document embeddings",
    "vectorDimension": 768,
    "indexType": "HNSW",
    "status": "ready",
    "created_at": "2023-10-10T10:00:00Z",
    "updated_at": "2023-10-10T10:00:00Z"
  }
]
```

### Get Database Configuration
```
GET /v1/databases/{databaseId}
```

#### Example Request
```bash
curl -X GET \
  https://your-jadevectordb-host.com/v1/databases/db_random_id_12345 \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

#### Example Response
```json
{
  "databaseId": "db_random_id_12345",
  "name": "my_vectors_db",
  "description": "Database for storing document embeddings",
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
    "factor": 2,
    "sync": false
  },
  "metadataSchema": {
    "category": "string",
    "tags": "array<string>",
    "score": "float",
    "created_at": "timestamp",
    "custom_data": "object"
  },
  "retentionPolicy": {
    "maxAgeDays": 365,
    "archiveOnExpire": true
  },
  "accessControl": {
    "roles": ["admin", "user", "auditor"],
    "defaultPermissions": ["read", "search"]
  },
  "status": "ready",
  "created_at": "2023-10-10T10:00:00Z",
  "updated_at": "2023-10-10T10:00:00Z"
}
```

### Update Database Configuration
```
PUT /v1/databases/{databaseId}
```

#### Example Request
```bash
curl -X PUT \
  https://your-jadevectordb-host.com/v1/databases/db_random_id_12345 \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Updated description for document embeddings database",
    "indexParameters": {
      "M": 32,
      "efConstruction": 400,
      "efSearch": 128
    },
    "retentionPolicy": {
      "maxAgeDays": 730,
      "archiveOnExpire": true
    }
  }'
```

### Delete Database
```
DELETE /v1/databases/{databaseId}
```

#### Example Request
```bash
curl -X DELETE \
  https://your-jadevectordb-host.com/v1/databases/db_random_id_12345 \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

## Configuration Guidelines

### Vector Dimension Selection

- **Small dimension (1-64)**: Suitable for simple features or compressed embeddings
- **Medium dimension (65-512)**: Good balance for many ML models
- **Large dimension (513-1024)**: Standard for transformer models like BERT
- **Very large dimension (1025-4096)**: High fidelity embeddings, more storage/compute intensive

### Index Type Selection

#### HNSW (Hierarchical Navigable Small World)
- **Best for**: High accuracy, fast query performance
- **Use when**: Query performance is critical
- **Trade-offs**: Higher memory usage, slower build times

#### IVF (Inverted File)
- **Best for**: Large datasets, good balance of accuracy and speed
- **Use when**: Dataset is large and search performance is important
- **Trade-offs**: Requires parameter tuning, approximate results

#### LSH (Locality Sensitive Hashing)
- **Best for**: Extremely large datasets, when speed is more important than accuracy
- **Use when**: Need very fast search with acceptable accuracy trade-offs
- **Trade-offs**: Lower accuracy, good for pre-filtering

#### FLAT (Linear Search)
- **Best for**: Small datasets, exact results required
- **Use when**: Dataset is small (< 10k vectors) or exact search is needed
- **Trade-offs**: Slow performance on large datasets

### Sharding Strategies

#### Hash-based Sharding
- Distributes vectors uniformly across shards
- Good for general-purpose workloads
- Load is balanced across shards

#### Range-based Sharding
- Shards based on vector ID ranges
- Good when there are known access patterns
- May lead to uneven load distribution

#### Vector-based Sharding
- Shards based on vector similarity/clustering
- Good for similarity search workloads
- Reduces network overhead for search operations

### Performance Optimization Tips

1. **Index Parameter Tuning**:
   - Increase `efSearch` for higher accuracy (at cost of speed)
   - Decrease `efSearch` for faster queries (with potential accuracy loss)
   - Adjust `M` and `efConstruction` based on dataset characteristics

2. **Sharding Considerations**:
   - Start with fewer shards and scale as needed
   - Monitor shard balance and performance
   - Consider your query patterns when selecting sharding strategy

3. **Memory Management**:
   - Monitor memory usage with different index types
   - HNSW indexes may require more memory than IVF
   - Consider vector compression for memory efficiency

4. **Replication**:
   - Use replication for high availability requirements
   - Consider network latency when selecting replication factor
   - Synchronous replication provides stronger consistency but lower performance

## Metadata Schema Definition

You can define a schema for metadata fields to enable better filtering and validation:

```json
{
  "metadataSchema": {
    "category": {
      "type": "string",
      "enum": ["finance", "technology", "healthcare"]
    },
    "tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    },
    "owner": {
      "type": "string",
      "pattern": "^user_[a-zA-Z0-9]+$"
    }
  }
}
```

## Health and Monitoring

Each database reports its status through the API:

- `ready`: Database is operational and accepting requests
- `initializing`: Database is being created or updated
- `degraded`: Database is operational but with reduced performance
- `failed`: Database is not operational

## Troubleshooting

### Common Issues

1. **"Vector dimension mismatch"**: Ensure all vectors stored in the database have the same dimension as specified in database configuration.

2. **"Index build failed"**: Check the index parameters and ensure sufficient memory is available.

3. **"Database not found"**: Verify the database ID is correct and the database hasn't been deleted.

4. **Performance issues**: 
   - Review index parameters
   - Check sharding configuration
   - Monitor system resources

### Recommended Monitoring

- Database storage usage
- Query response times
- Index build times
- Error rates
- Resource utilization (CPU, memory, disk, network)

## Examples

### Example 1: Creating a High-Performance Search Database

```json
{
  "name": "high_perf_search",
  "description": "Database optimized for fast similarity search",
  "vectorDimension": 128,
  "indexType": "HNSW",
  "indexParameters": {
    "M": 32,
    "efConstruction": 400,
    "efSearch": 200
  },
  "sharding": {
    "strategy": "hash",
    "numShards": 8
  },
  "replication": {
    "factor": 2,
    "sync": false
  },
  "metadataSchema": {
    "category": "string",
    "tags": "array<string>",
    "score": "float"
  }
}
```

### Example 2: Creating a Large-Scale Archival Database

```json
{
  "name": "archival_vectors",
  "description": "Database for long-term vector storage with archival",
  "vectorDimension": 768,
  "indexType": "IVF",
  "indexParameters": {
    "nlist": 1000,
    "nprobe": 50
  },
  "sharding": {
    "strategy": "range",
    "numShards": 16
  },
  "retentionPolicy": {
    "maxAgeDays": 1095,
    "archiveOnExpire": true
  }
}
```