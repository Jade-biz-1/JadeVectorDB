# Quickstart Guide for JadeVectorDB

## Overview

This guide will help you get started with JadeVectorDB, a high-performance distributed vector database. You'll learn how to set up the database, create your first vector database instance, and perform basic operations using our various interfaces: the web UI, CLI tools, or directly through the API.

JadeVectorDB features:
- **Backend**: High-performance C++ microservices
- **Web UI**: Next.js-based interface with shadcn UI components
- **CLI**: Tools with both Python and shell script integration
- **API**: Comprehensive REST API for programmatic access

## Prerequisites

- Linux server (Ubuntu 20.04+, CentOS 8+, or RHEL 8+)
- Docker and Docker Compose
- At least 8GB RAM and 2 CPU cores
- Network access to pull Docker images

## Installation

### Option 1: Docker Compose (Recommended for Development)

1. Create a directory for your JadeVectorDB deployment:
```bash
mkdir jadevectordb-deployment && cd jadevectordb-deployment
```

2. Create a `docker-compose.yml` file with the following content:
```yaml
version: '3.8'

services:
  jadevectordb-master:
    image: jadevectordb/server:latest
    container_name: jadevectordb-master
    ports:
      - "8080:8080"
    environment:
      - NODE_TYPE=master
      - API_KEY=your-secret-api-key
      - CLUSTER_NAME=jade-cluster
    volumes:
      - ./data/master:/var/lib/jadevectordb
    networks:
      - jade-network

  jadevectordb-worker-1:
    image: jadevectordb/server:latest
    container_name: jadevectordb-worker-1
    environment:
      - NODE_TYPE=worker
      - API_KEY=your-secret-api-key
      - CLUSTER_NAME=jade-cluster
      - MASTER_HOST=jadevectordb-master
    volumes:
      - ./data/worker-1:/var/lib/jadevectordb
    depends_on:
      - jadevectordb-master
    networks:
      - jade-network

  jadevectordb-worker-2:
    image: jadevectordb/server:latest
    container_name: jadevectordb-worker-2
    environment:
      - NODE_TYPE=worker
      - API_KEY=your-secret-api-key
      - CLUSTER_NAME=jade-cluster
      - MASTER_HOST=jadevectordb-master
    volumes:
      - ./data/worker-2:/var/lib/jadevectordb
    depends_on:
      - jadevectordb-master
    networks:
      - jade-network

networks:
  jade-network:
    driver: bridge
```

3. Start the cluster:
```bash
docker-compose up -d
```

4. Verify that the cluster is running:
```bash
curl -H "X-API-Key: your-secret-api-key" http://localhost:8080/health
```

### Option 2: Kubernetes (Production)

1. Add the JadeVectorDB Helm repository:
```bash
helm repo add jadevectordb https://charts.jadevectordb.com
helm repo update
```

2. Install with default settings:
```bash
helm install my-jadevectordb jadevectordb/jadevectordb \
  --set api.key=your-secret-api-key \
  --set cluster.replicaCount=3
```

## Basic Usage

### 1. Create a Vector Database

To create a database for 768-dimensional vectors using HNSW indexing:

```bash
curl -X POST http://localhost:8080/api/v1/databases \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "name": "my-document-embeddings",
    "description": "Database for document embeddings",
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
      "sync": true
    }
  }'
```

Note: The create database request will return a `databaseId`. Use that ID in the following requests in place of `{your-database-id}`.

### 2. Store Vectors

Add a vector with metadata:

```bash
curl -X POST http://localhost:8080/api/v1/databases/{your-database-id}/vectors \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "id": "vec-001",
    "values": [0.1, 0.2, 0.3, ..., 0.9],  // 768-dimensional vector
    "metadata": {
      "source": "document",
      "category": "research-paper",
      "tags": ["AI", "ML"],
      "author": "Jane Doe"
    }
  }'
```

### 3. Perform Similarity Search

Find the 5 most similar vectors to a query vector:

```bash
curl -X POST http://localhost:8080/api/v1/databases/{your-database-id}/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "queryVector": [0.15, 0.25, 0.35, ..., 0.95],
    "topK": 5,
    "threshold": 0.8,
    "includeMetadata": true
  }'
```

### 4. Advanced Search with Filters

Search with metadata filters:

```bash
curl -X POST http://localhost:8080/api/v1/databases/{your-database-id}/search/advanced \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "queryVector": [0.15, 0.25, 0.35, ..., 0.95],
    "topK": 5,
    "filters": {
      "metadata.category": "research-paper",
      "metadata.tags": ["ML"]
    },
    "includeMetadata": true
  }'
```

## Client Libraries

### Python Client

Install the Python client:
```bash
pip install jadevectordb-client
```

Example usage:
```python
from jadevectordb import JadeVectorDB

# Initialize client
client = JadeVectorDB(
    host="localhost",
    port=8080,
    api_key="your-secret-api-key"
)

# Create database
db = client.create_database(
    name="my-docs",
    dimensions=768,
    index_type="HNSW"
)

# Store a vector
vector_id = db.store(
    values=[0.1, 0.2, 0.3, ...],
    metadata={"category": "research", "author": "Jane Doe"}
)

# Search for similar vectors
results = db.search(
    query=[0.15, 0.25, 0.35, ...],
    top_k=5,
    filters={"metadata.category": "research"}
)

for result in results:
    print(f"ID: {result.id}, Similarity: {result.similarity}")
```

## Configuration

### Environment Variables

The following environment variables can be used to configure the JadeVectorDB server:

- `NODE_TYPE`: Either "master" or "worker" (default: "master")
- `API_KEY`: API key for authentication (required)
- `CLUSTER_NAME`: Name of the cluster (default: "default-cluster")
- `MASTER_HOST`: Host of the master node (for worker nodes)
- `HTTP_PORT`: Port for HTTP API (default: 8080)
- `GRPC_PORT`: Port for gRPC API (default: 9090)
- `DATA_DIR`: Directory for data storage (default: /var/lib/jadevectordb)
- `MAX_CONNECTIONS`: Maximum number of concurrent connections (default: 1000)

## Monitoring

Check the health of your cluster:
```bash
curl http://localhost:8080/health
```

Get detailed status:
```bash
curl http://localhost:8080/api/v1/status
```

Get database-specific status:
```bash
curl http://localhost:8080/api/v1/databases/{your-database-id}/status
```

## Web UI (Next.js + shadcn)

The JadeVectorDB Web UI provides a comprehensive dashboard for managing your vector databases:

1. Start the Web UI:
```bash
cd frontend
npm install
npm run dev
```

2. Access the UI at `http://localhost:3000`
3. Connect to your backend by providing the API endpoint and key

The UI provides:
- Cluster monitoring and management
- Database creation and configuration
- Vector data exploration with visualization
- Query interface with metadata filtering
- User and role management

## CLI Tools (Python + Shell)

JadeVectorDB provides CLI tools in both Python and shell script formats:

### Python CLI
```bash
pip install jadevectordb-cli

# Example usage:
jadevectordb cluster status
jadevectordb database create --name my-db --dimension 768
jadevectordb search --db my-db --vector "[0.1, 0.2, 0.3]"
```

### Shell CLI
```bash
# Example usage:
./cli/shell/bin/jade-db cluster status
./cli/shell/bin/jade-db database create --name my-db --dimension 768
./cli/shell/bin/jade-db search --db my-db --vector "[0.1, 0.2, 0.3]"
```

## Next Steps

1. Explore the [full API documentation](contracts/vector-db-api.yaml)
2. Review the [data model](data-model.md) for advanced schema design
3. Check the [architecture decisions](research/architecture_decisions.md) for implementation details
4. Learn about [embedding model integration](#) for automatic vector generation
5. Review [security best practices](#) for production deployment