#!/bin/bash

# JadeVectorDB Quick Start Script
# This script demonstrates the basic operations of JadeVectorDB CLI
# It creates a database, stores vectors, and performs a similarity search

set -e  # Exit on error

echo "=================================="
echo "JadeVectorDB Quick Start Tutorial"
echo "=================================="
echo ""

# Configuration
API_URL="${JADEVECTORDB_URL:-http://localhost:8080}"
API_KEY="${JADEVECTORDB_API_KEY:-your-api-key}"
DB_NAME="quickstart_db"

echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Database: $DB_NAME"
echo ""

# Step 1: Check server health
echo "Step 1: Checking server health..."
jade-db --url "$API_URL" health
echo "✓ Server is healthy"
echo ""

# Step 2: Create a database
echo "Step 2: Creating database '$DB_NAME'..."
jade-db --url "$API_URL" --api-key "$API_KEY" create-db \
  --name "$DB_NAME" \
  --description "Quick start tutorial database" \
  --dimension 4 \
  --index-type HNSW
echo "✓ Database created"
echo ""

# Step 3: Store sample vectors
echo "Step 3: Storing sample product vectors..."

echo "  Storing laptop vector..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id laptop_1 \
  --values "[0.8, 0.2, 0.1, 0.9]" \
  --metadata '{"name": "UltraBook Pro", "category": "laptop", "price": 1299.99}'

echo "  Storing phone vector..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id phone_1 \
  --values "[0.9, 0.1, 0.8, 0.2]" \
  --metadata '{"name": "SmartPhone X", "category": "phone", "price": 899.99}'

echo "  Storing tablet vector..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id tablet_1 \
  --values "[0.7, 0.3, 0.6, 0.4]" \
  --metadata '{"name": "Tablet Max", "category": "tablet", "price": 649.99}'

echo "✓ All vectors stored"
echo ""

# Step 4: Retrieve a vector
echo "Step 4: Retrieving laptop vector..."
jade-db --url "$API_URL" --api-key "$API_KEY" retrieve \
  --database-id "$DB_NAME" \
  --vector-id laptop_1
echo ""

# Step 5: Perform similarity search
echo "Step 5: Searching for products similar to [0.85, 0.15, 0.75, 0.25]..."
jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id "$DB_NAME" \
  --query-vector "[0.85, 0.15, 0.75, 0.25]" \
  --top-k 3
echo ""

# Step 6: List all databases
echo "Step 6: Listing all databases..."
jade-db --url "$API_URL" --api-key "$API_KEY" list-dbs
echo ""

echo "=================================="
echo "Quick Start Tutorial Completed! ✓"
echo "=================================="
echo ""
echo "Next steps:"
echo "  - Try modifying the vectors and search queries"
echo "  - Explore the advanced tutorial (../advanced.md)"
echo "  - Check out batch-import.py for importing larger datasets"
echo ""
echo "To clean up, delete the database:"
echo "  jade-db --url $API_URL --api-key $API_KEY delete-db --database-id $DB_NAME"
echo ""
