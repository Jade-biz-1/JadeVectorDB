# JadeVectorDB Python CLI Examples

This document provides comprehensive examples for using the JadeVectorDB Python CLI.

## Setup

First, install the Python CLI package:

```bash
pip install -e ./cli/python
```

## Examples

### 1. Database Operations

#### Create a Database
```bash
# Create a new database with default settings
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name my_database

# Create a database with custom settings
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name product_embeddings --description "Database for product embeddings" --dimension 768 --index-type HNSW
```

#### List Databases
```bash
# List all databases
jade-db --url http://localhost:8080 --api-key mykey123 list-dbs
```

### 2. Vector Operations

#### Store a Vector
```bash
# Store a vector with minimal information
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id my_db --vector-id vec_1 --values "[0.1, 0.2, 0.3, 0.4]"

# Store a vector with metadata
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id my_db --vector-id vec_2 --values "[0.5, 0.6, 0.7, 0.8]" --metadata '{"category": "electronics", "price": 299.99}'
```

#### Retrieve a Vector
```bash
# Retrieve a vector by ID
jade-db --url http://localhost:8080 --api-key mykey123 retrieve --database-id my_db --vector-id vec_1
```

### 3. Search Operations

#### Perform Similarity Search
```bash
# Simple search - find 5 most similar vectors
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id my_db --query-vector "[0.25, 0.35, 0.45, 0.55]" --top-k 5

# Search with similarity threshold
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id my_db --query-vector "[0.25, 0.35, 0.45, 0.55]" --top-k 10 --threshold 0.7
```

### 4. System Operations

#### Check Health Status
```bash
# Get system health
jade-db --url http://localhost:8080 health
```

#### Check System Status
```bash
# Get detailed system status
jade-db --url http://localhost:8080 status
```

### 5. Complete Workflow Example

Here's a complete example of creating a database, storing vectors, and performing a search:

```bash
# 1. Create a database for product embeddings
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name product_db --dimension 4 --index-type HNSW

# 2. Store some product vectors
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id product_db --vector-id laptop_1 --values "[0.8, 0.2, 0.1, 0.9]" --metadata '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99}'
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id product_db --vector-id phone_1 --values "[0.9, 0.1, 0.8, 0.2]" --metadata '{"name": "SmartPhone X", "category": "phone", "price": 899.99}'
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id product_db --vector-id tablet_1 --values "[0.7, 0.3, 0.6, 0.4]" --metadata '{"name": "Tablet Max", "category": "tablet", "price": 649.99}'

# 3. Perform a search to find similar products
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id product_db --query-vector "[0.85, 0.15, 0.75, 0.25]" --top-k 5
```