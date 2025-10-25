# JadeVectorDB Shell CLI Examples

This document provides comprehensive examples for using the JadeVectorDB Shell CLI.

## Setup

The shell CLI is located at `cli/shell/scripts/jade-db.sh`. You can use it directly:

```bash
bash cli/shell/scripts/jade-db.sh [options] [command] [args...]
```

Or make it executable and use it as a command:

```bash
chmod +x cli/shell/scripts/jade-db.sh
./cli/shell/scripts/jade-db.sh [options] [command] [args...]
```

## Examples

### 1. Database Operations

#### Create a Database
```bash
# Create a new database with default settings
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 create-db my_database

# Create a database with custom settings
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 create-db product_embeddings "Database for product embeddings" 768 HNSW
```

#### List Databases
```bash
# List all databases
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 list-dbs
```

#### Get Database Details
```bash
# Get details for a specific database
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 get-db my_database_id
```

### 2. Vector Operations

#### Store a Vector
```bash
# Store a vector with minimal information
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id my_db store vec_1 '[0.1, 0.2, 0.3, 0.4]'

# Store a vector with metadata
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id my_db store vec_2 '[0.5, 0.6, 0.7, 0.8]' '{"category": "electronics", "price": 299.99}'
```

#### Retrieve a Vector
```bash
# Retrieve a vector by ID
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id my_db retrieve vec_1
```

#### Delete a Vector
```bash
# Delete a vector by ID
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id my_db delete vec_1
```

### 3. Search Operations

#### Perform Similarity Search
```bash
# Simple search - find 5 most similar vectors
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id my_db search '[0.25, 0.35, 0.45, 0.55]' 5

# Search with similarity threshold
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id my_db search '[0.25, 0.35, 0.45, 0.55]' 10 0.7
```

### 4. System Operations

#### Check Health Status
```bash
# Get system health
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 health
```

#### Check System Status
```bash
# Get detailed system status
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 status
```

### 5. Complete Workflow Example

Here's a complete example of creating a database, storing vectors, and performing a search:

```bash
# 1. Create a database for product embeddings
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 create-db product_db "Product embeddings database" 4 HNSW

# 2. Store some product vectors
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_db store laptop_1 '[0.8, 0.2, 0.1, 0.9]' '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99}'
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_db store phone_1 '[0.9, 0.1, 0.8, 0.2]' '{"name": "SmartPhone X", "category": "phone", "price": 899.99}'
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_db store tablet_1 '[0.7, 0.3, 0.6, 0.4]' '{"name": "Tablet Max", "category": "tablet", "price": 649.99}'

# 3. Perform a search to find similar products
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_db search '[0.85, 0.15, 0.75, 0.25]' 5
```