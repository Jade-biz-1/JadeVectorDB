# JadeVectorDB CLI Tutorial

Welcome to the JadeVectorDB CLI tutorial! This tutorial will guide you through the basics of using the JadeVectorDB command-line interface to store, manage, and search vector embeddings.

## Prerequisites

Before starting this tutorial, you'll need:

1. A running instance of JadeVectorDB (typically available at http://localhost:8080)
2. API key for authentication (if required by your server setup)
3. One of the CLI implementations installed (Python, Shell, or JavaScript)

## Introduction to Vector Databases

Vector databases store and search high-dimensional vectors. Each vector represents some data (like text, images, or other features) in a high-dimensional space. The database can find similar vectors using similarity search.

In JadeVectorDB:
- **Databases** store collections of vectors
- **Vectors** have IDs, values, and optional metadata
- **Search** finds the most similar vectors to a query vector

## Tutorial Overview

This tutorial covers:

1. Setting up your CLI environment
2. Creating your first database
3. Storing vectors with metadata
4. Performing similarity searches
5. Managing your vector data

## Step 1: Setting Up Your Environment

### Python CLI Setup
```bash
pip install -e cli/python
```

### Shell CLI Setup
The shell CLI is ready to use directly:
```bash
bash cli/shell/scripts/jade-db.sh --help
```

### JavaScript CLI Setup
```bash
cd cli/js
npm install
node bin/jade-db.js --help
```

## Step 2: Creating Your First Database

Let's start by creating a database to store product embeddings:

### Python CLI
```bash
jade-db --url http://localhost:8080 --api-key mykey123 create-db --name product_embeddings --description "Product embedding vectors" --dimension 4 --index-type HNSW
```

### Shell CLI
```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 create-db product_embeddings "Product embedding vectors" 4 HNSW
```

### JavaScript CLI
```bash
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 database create --name product_embeddings --description "Product embedding vectors" --dimension 4 --index-type HNSW
```

You should see a response with the database ID. Make note of this ID as you'll need it for subsequent commands.

## Step 3: Storing Vectors

Now let's store some example product vectors in your database:

### Python CLI
```bash
# Store a laptop vector
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id product_embeddings --vector-id laptop_1 --values "[0.8, 0.2, 0.1, 0.9]" --metadata '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99}'

# Store a phone vector
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id product_embeddings --vector-id phone_1 --values "[0.9, 0.1, 0.8, 0.2]" --metadata '{"name": "SmartPhone X", "category": "phone", "price": 899.99}'

# Store a tablet vector
jade-db --url http://localhost:8080 --api-key mykey123 store --database-id product_embeddings --vector-id tablet_1 --values "[0.7, 0.3, 0.6, 0.4]" --metadata '{"name": "Tablet Max", "category": "tablet", "price": 649.99}'
```

### Shell CLI
```bash
# Store a laptop vector
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_embeddings store laptop_1 '[0.8, 0.2, 0.1, 0.9]' '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99}'

# Store a phone vector
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_embeddings store phone_1 '[0.9, 0.1, 0.8, 0.2]' '{"name": "SmartPhone X", "category": "phone", "price": 899.99}'

# Store a tablet vector
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_embeddings store tablet_1 '[0.7, 0.3, 0.6, 0.4]' '{"name": "Tablet Max", "category": "tablet", "price": 649.99}'
```

### JavaScript CLI
```bash
# Store a laptop vector
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector store --database-id product_embeddings --vector-id laptop_1 --values "[0.8, 0.2, 0.1, 0.9]" --metadata '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99}'

# Store a phone vector
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector store --database-id product_embeddings --vector-id phone_1 --values "[0.9, 0.1, 0.8, 0.2]" --metadata '{"name": "SmartPhone X", "category": "phone", "price": 899.99}'

# Store a tablet vector
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector store --database-id product_embeddings --vector-id tablet_1 --values "[0.7, 0.3, 0.6, 0.4]" --metadata '{"name": "Tablet Max", "category": "tablet", "price": 649.99}'
```

## Step 4: Retrieving Vectors

Let's check that our vectors were stored correctly:

### Python CLI
```bash
jade-db --url http://localhost:8080 --api-key mykey123 retrieve --database-id product_embeddings --vector-id laptop_1
```

### Shell CLI
```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_embeddings retrieve laptop_1
```

### JavaScript CLI
```bash
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 vector retrieve --database-id product_embeddings --vector-id laptop_1
```

## Step 5: Performing Similarity Search

Now let's perform a similarity search. Imagine we have a vector representing a new product and want to find similar existing products:

### Python CLI
```bash
# Search for products similar to [0.85, 0.15, 0.75, 0.25] (might be similar to the phone we stored)
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id product_embeddings --query-vector "[0.85, 0.15, 0.75, 0.25]" --top-k 3
```

### Shell CLI
```bash
# Search for products similar to [0.85, 0.15, 0.75, 0.25]
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_embeddings search '[0.85, 0.15, 0.75, 0.25]' 3
```

### JavaScript CLI
```bash
# Search for products similar to [0.85, 0.15, 0.75, 0.25]
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 search --database-id product_embeddings --query-vector "[0.85, 0.15, 0.75, 0.25]" --top-k 3
```

## Step 6: Advanced Search with Threshold

You can also set a minimum similarity threshold to filter results:

### Python CLI
```bash
jade-db --url http://localhost:8080 --api-key mykey123 search --database-id product_embeddings --query-vector "[0.85, 0.15, 0.75, 0.25]" --top-k 5 --threshold 0.7
```

### Shell CLI
```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 --database-id product_embeddings search '[0.85, 0.15, 0.75, 0.25]' 5 0.7
```

### JavaScript CLI
```bash
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 search --database-id product_embeddings --query-vector "[0.85, 0.15, 0.75, 0.25]" --top-k 5 --threshold 0.7
```

## Step 7: System Health and Status

Check the health and status of your JadeVectorDB instance:

### Python CLI
```bash
jade-db --url http://localhost:8080 health
jade-db --url http://localhost:8080 status
```

### Shell CLI
```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 health
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 status
```

### JavaScript CLI
```bash
node cli/js/bin/jade-db.js --url http://localhost:8080 health
node cli/js/bin/jade-db.js --url http://localhost:8080 status
```

## Common Troubleshooting

### Error: Connection Refused
- Make sure JadeVectorDB is running at the specified URL
- Check that the port is correct (default is 8080)

### Error: Unauthorized
- Verify your API key is correct
- Check that your API key has the necessary permissions

### Error: Database not found
- Verify the database ID is correct
- Check that you're using the right database ID in your commands

## Next Steps

Congratulations! You've completed the basic tutorial. Now you can:

1. Try more complex examples with larger vector dimensions
2. Experiment with different index types (HNSW, IVF, LSH, FLAT)
3. Explore batch operations for storing multiple vectors at once
4. Learn about metadata filtering for more sophisticated searches

For more examples, check out the examples directory in the repository.