# JadeVectorDB JavaScript CLI

A command-line interface for interacting with JadeVectorDB using JavaScript/Node.js.

## Installation

First, install the required dependencies:

```bash
npm install
```

## Usage

### Global Options
- `--url <url>`: JadeVectorDB API URL (default: http://localhost:8080)
- `--api-key <key>`: API key for authentication

### Database Operations

#### Create a Database
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 database create --name mydb --description "My test database" --dimension 128 --index-type HNSW
```

#### List Databases
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 database list
```

#### Get Database Details
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 database get mydatabaseid
```

### Vector Operations

#### Store a Vector
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector store --database-id mydb --vector-id vector1 --values "[0.1, 0.2, 0.3]" --metadata '{"category":"test"}'
```

#### Retrieve a Vector
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector retrieve --database-id mydb --vector-id vector1
```

#### Delete a Vector
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector delete --database-id mydb --vector-id vector1
```

### Search Operations

#### Search for Similar Vectors
```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey123 search --database-id mydb --query-vector "[0.15, 0.25, 0.35]" --top-k 5 --threshold 0.5
```

### System Operations

#### Check System Health
```bash
node bin/jade-db.js --url http://localhost:8080 health
```

#### Check System Status
```bash
node bin/jade-db.js --url http://localhost:8080 status
```

## Examples

### Creating a Database
```bash
jade-db --url http://localhost:8080 database create --name "product_embeddings" --description "Database for product embeddings" --dimension 768 --index-type "HNSW"
```

### Storing a Vector
```bash
jade-db --url http://localhost:8080 --api-key mykey123 vector store --database-id product_embeddings --vector-id product_123 --values "[0.1, 0.5, 0.3, 0.9, 0.2]" --metadata '{"category":"electronics", "brand":"tech_brand"}'
```

### Performing a Search
```bash
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id product_embeddings --query-vector "[0.12, 0.48, 0.32, 0.88, 0.21]" --top-k 3
```

## Environment Variables

You can set environment variables for common options:
- `JADE_DB_URL` - Sets the default API URL
- `JADE_DB_API_KEY` - Sets the default API key

## Development

Run the CLI directly:
```bash
node bin/jade-db.js [command]
```

Run tests:
```bash
npm test
```