# Advanced JadeVectorDB CLI Tutorial

This tutorial covers advanced features of the JadeVectorDB CLI, including batch operations, metadata filtering, index management, and embedding generation.

## Prerequisites

Before starting this tutorial, you should have completed the basic tutorial and understand:

- How to create databases
- How to store and retrieve vectors
- How to perform similarity searches

## Tutorial Overview

This tutorial covers:

1. Batch vector operations
2. Advanced search with metadata filtering
3. Index management
4. Embedding generation (if supported)
5. Database lifecycle management

## Step 1: Batch Vector Operations

Instead of storing vectors one by one, you can use batch operations for better performance when importing large datasets.

### Python CLI Batch Example
```bash
# First, create a database for batch operations
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name batch_db --dimension 4 --index-type HNSW

# The Python client supports batch operations, but the CLI needs to be called multiple times
# In a real scenario, you'd use the Python client directly:
```

For batch operations, you might want to use the Python client directly, but you can also use shell scripting to import multiple vectors:

```bash
# Create a file with vectors to import (example import script)
# This would be a real-world batch import script
for i in {1..100}; do
  jade-db --url http://localhost:8080 --api-key mykey123 store --database-id batch_db --vector-id "vector_$i" --values "[0.$i, 0.$((100-i)), 0.5, 0.7]" --metadata "{\"id\": $i, \"category\": \"batch\", \"timestamp\": \"$(date -Iseconds)\"}"
done
```

### Shell CLI Batch Example
```bash
# Create and run a batch import script
cat > batch_import.sh << 'EOF'
#!/bin/bash
for i in {1..10}; do
  echo "Storing vector_$i"
  bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id batch_db store "vector_$i" "[0.$i, 0.$((10-i)), 0.5, 0.7]" "{\"id\": $i, \"category\": \"batch\"}"
done
EOF

chmod +x batch_import.sh
./batch_import.sh
```

## Step 2: Advanced Search with Metadata Filtering

JadeVectorDB supports filtering search results based on metadata values. This is particularly useful for narrowing down search results to specific categories or conditions.

### Creating a Database with Product Metadata
```bash
# Create a database for products with more detailed metadata
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name advanced_products --dimension 5 --index-type HNSW
```

### Storing Products with Complex Metadata
```bash
# Store products with detailed metadata
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id advanced_products --vector-id laptop_1 --values "[0.9, 0.1, 0.2, 0.8, 0.7]" --metadata '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99, "brand": "TechCorp", "in_stock": true, "tags": ["premium", "portable"]}'
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id advanced_products --vector-id laptop_2 --values "[0.85, 0.15, 0.25, 0.75, 0.65]" --metadata '{"name": "PowerBook Air", "category": "laptop", "price": 999.99, "brand": "TechCorp", "in_stock": true, "tags": ["affordable", "lightweight"]}'
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id advanced_products --vector-id phone_1 --values "[0.7, 0.8, 0.9, 0.1, 0.5]" --metadata '{"name": "SmartPhone X", "category": "phone", "price": 899.99, "brand": "MobilePlus", "in_stock": false, "tags": ["flagship", "camera"]}'
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id advanced_products --vector-id tablet_1 --values "[0.6, 0.7, 0.4, 0.9, 0.8]" --metadata '{"name": "Tablet Pro", "category": "tablet", "price": 749.99, "brand": "TechCorp", "in_stock": true, "tags": ["pro", "stylus"]}'
```

## Step 3: Index Management

Advanced index management allows you to optimize your database performance. Different index types have different performance characteristics for various use cases.

### Creating Different Index Types

You can create databases with different index types, and the Python CLI supports full index management after database creation:

```bash
# HNSW: Good for general purpose, balanced performance
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name hnsw_db --dimension 128 --index-type HNSW

# IVF: Good for very large datasets with adjustable parameters
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name ivf_db --dimension 128 --index-type IVF

# FLAT: Exact search, good for small datasets
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name flat_db --dimension 128 --index-type FLAT
```

### Managing Indexes (Python CLI)

The Python CLI provides commands for creating, listing, and deleting indexes on existing databases:

```bash
# Create an index with custom parameters
jade-db --url http://localhost:8080 --api-key mykey123 create-index \
  --database-id hnsw_db --index-type HNSW \
  --name my_hnsw_index --parameters '{"M": 16, "ef_construction": 200}'

# List indexes on a database
jade-db --url http://localhost:8080 --api-key mykey123 list-indexes --database-id hnsw_db

# Delete an index
jade-db --url http://localhost:8080 --api-key mykey123 delete-index \
  --database-id hnsw_db --index-id my_hnsw_index
```

## Step 4: Lifecycle Management

Managing the lifecycle of your database includes monitoring, backup/restore, and cleanup operations.

### Monitoring Database Status

```bash
# Check overall system health
jade-db --url http://localhost:8080 --api-key mykey123 health

# Check detailed system status
jade-db --url http://localhost:8080 --api-key mykey123 status
```

## Step 5: Performance Considerations

### Searching with Performance Parameters

You can adjust search parameters to balance accuracy and performance:

```bash
# Fast search (fewer results)
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id advanced_products --query-vector "[0.88, 0.12, 0.22, 0.78, 0.68]" --top-k 3

# More comprehensive search
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id advanced_products --query-vector "[0.88, 0.12, 0.22, 0.78, 0.68]" --top-k 20
```

## Step 6: Using Environment Variables

To avoid repeatedly typing your URL and API key, you can use environment variables:

### For Python CLI
```bash
export JADEVECTORDB_URL=http://localhost:8080
export JADEVECTORDB_API_KEY=mykey123

# Now you can run commands without specifying URL and key
jade-db create-db --name env_db --dimension 4
```

### Using Shell Scripts for Complex Operations

Create a shell script to perform complex operations:

```bash
cat > product_workflow.sh << 'EOF'
#!/bin/bash

# Setup
DB_NAME="workflow_db"
API_KEY="mykey123"
BASE_URL="http://localhost:8080"

# Create a database
echo "Creating database..."
jade-db --url $BASE_URL --api-key $API_KEY create-db --name $DB_NAME --dimension 4 --index-type HNSW

# Store multiple related items
echo "Storing related items..."
jade-db --url $BASE_URL --api-key $API_KEY store --database-id $DB_NAME --vector-id "cat_1" --values "[0.9, 0.1, 0.8, 0.2]" --metadata '{"name": "Cat", "category": "animal", "type": "mammal"}'
jade-db --url $BASE_URL --api-key $API_KEY store --database-id $DB_NAME --vector-id "dog_1" --values "[0.85, 0.15, 0.75, 0.25]" --metadata '{"name": "Dog", "category": "animal", "type": "mammal"}'
jade-db --url $BASE_URL --api-key $API_KEY store --database-id $DB_NAME --vector-id "bird_1" --values "[0.2, 0.9, 0.1, 0.8]" --metadata '{"name": "Bird", "category": "animal", "type": "avian"}'

# Search for similar items to cat
echo "Searching for animals similar to cat..."
jade-db --url $BASE_URL --api-key $API_KEY search --database-id $DB_NAME --query-vector "[0.88, 0.12, 0.78, 0.22]" --top-k 5

echo "Workflow completed!"
EOF

chmod +x product_workflow.sh
./product_workflow.sh
```

## Step 7: Error Handling and Debugging

Understanding error messages and handling common issues:

### Common Error Types and Solutions

1. **Connection errors**: Check that the database server is running and accessible at the specified URL
2. **Authentication errors**: Verify your API key is correct and has required permissions
3. **Dimension mismatch errors**: Ensure vector dimensions match the database configuration
4. **Memory errors**: For large operations, consider breaking them into smaller batches

## Best Practices

### Performance Best Practices
1. Use appropriate index types for your use case
2. Batch operations when storing multiple vectors
3. Use filters to narrow search results when possible
4. Monitor resource usage during large operations

### Security Best Practices
1. Store API keys securely (environment variables or secure storage)
2. Use HTTPS in production environments
3. Regularly rotate API keys
4. Implement rate limiting for client access

## Conclusion

You've now completed the advanced tutorial covering batch operations, metadata filtering, index management, and lifecycle operations. You're ready to use JadeVectorDB CLI in production environments with complex workflows.

## Next Steps

- Explore **API key management** (`create-api-key`, `list-api-keys`, `revoke-api-key`)
- Try **advanced search** with metadata filtering (`advanced-search`)
- Use **hybrid search** combining vector similarity with BM25 keyword search (`hybrid-search`)
- Explore **reranking** for improved search quality (`rerank-search`, `rerank`)
- Generate **embeddings** directly from text (`generate-embedding`)
- View **analytics** and **audit logs** (`analytics-stats`, `audit-log`)
- Set up monitoring and alerting for your production deployments
- Implement backup and disaster recovery procedures for production use