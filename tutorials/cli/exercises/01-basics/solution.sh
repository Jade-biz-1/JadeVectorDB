#!/bin/bash

# Exercise 1: CLI Basics - Solution
# This is the complete solution for reference

# Setup
export JADE_DB_URL=http://localhost:8080
export JADE_DB_API_KEY=mykey123

echo "========================================="
echo "Exercise 1: CLI Basics - Solution"
echo "========================================="
echo ""

# Step 1: Check system health
echo "Step 1: Checking system health..."
jade-db health

# Step 2: Create a database
echo ""
echo "Step 2: Creating database..."
jade-db create-db \
  --name my_products \
  --description "My first product database" \
  --dimension 8 \
  --index-type HNSW

# Step 3: Store first vector (laptop)
echo ""
echo "Step 3: Storing laptop vector..."
jade-db store \
  --database-id my_products \
  --vector-id laptop_001 \
  --values "[0.85, 0.12, 0.22, 0.78, 0.65, 0.43, 0.91, 0.33]" \
  --metadata '{"name": "UltraBook Pro 15", "category": "laptop", "brand": "TechCorp", "price": 1299.99, "in_stock": true, "tags": ["premium", "portable", "business"]}'

# Step 4: Store second vector (phone)
echo ""
echo "Step 4: Storing phone vector..."
jade-db store \
  --database-id my_products \
  --vector-id phone_001 \
  --values "[0.15, 0.87, 0.92, 0.21, 0.48, 0.76, 0.33, 0.81]" \
  --metadata '{"name": "SmartPhone X Pro", "category": "phone", "brand": "MobilePlus", "price": 899.99, "in_stock": false, "tags": ["flagship", "camera", "5G"]}'

# Step 5: Store third vector (tablet)
echo ""
echo "Step 5: Storing tablet vector..."
jade-db store \
  --database-id my_products \
  --vector-id tablet_001 \
  --values "[0.62, 0.55, 0.48, 0.72, 0.81, 0.59, 0.67, 0.54]" \
  --metadata '{"name": "Tablet Pro 12", "category": "tablet", "brand": "TechCorp", "price": 749.99, "in_stock": true, "tags": ["pro", "stylus", "creative"]}'

# Step 6: Retrieve a vector
echo ""
echo "Step 6: Retrieving laptop vector..."
jade-db retrieve \
  --database-id my_products \
  --vector-id laptop_001

# Step 7: Perform similarity search
echo ""
echo "Step 7: Performing similarity search..."
jade-db search \
  --database-id my_products \
  --query-vector "[0.83, 0.14, 0.24, 0.76, 0.63, 0.42, 0.89, 0.32]" \
  --top-k 3

# Step 8: Search with threshold
echo ""
echo "Step 8: Searching with threshold (similarity > 0.9)..."
jade-db search \
  --database-id my_products \
  --query-vector "[0.83, 0.14, 0.24, 0.76, 0.63, 0.42, 0.89, 0.32]" \
  --top-k 3 \
  --threshold 0.9

echo ""
echo "========================================="
echo "Solution complete!"
echo "========================================="
