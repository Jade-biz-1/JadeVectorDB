#!/bin/bash

# JadeVectorDB Complete Workflow Demo
# This script demonstrates a complete real-world workflow including:
# - Database creation with multiple databases
# - Storing related vectors across databases
# - Cross-database queries
# - Database management operations

set -e  # Exit on error

echo "================================================"
echo "JadeVectorDB Complete Workflow Demo"
echo "================================================"
echo ""
echo "This demo simulates a multi-tenant product catalog"
echo "with separate databases for different stores."
echo ""

# Configuration
API_URL="${JADEVECTORDB_URL:-http://localhost:8080}"
API_KEY="${JADEVECTORDB_API_KEY:-your-api-key}"

# Pause function for readability
pause() {
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

echo "Configuration:"
echo "  API URL: $API_URL"
echo ""

# Step 1: Check system status
echo "STEP 1: Checking system health and status"
echo "==========================================="
jade-db --url "$API_URL" health
echo ""
jade-db --url "$API_URL" status
pause

# Step 2: Create multiple databases for different stores
echo "STEP 2: Creating databases for different stores"
echo "================================================"

echo "Creating Electronics Store database..."
jade-db --url "$API_URL" --api-key "$API_KEY" create-db \
  --name electronics_store \
  --description "Electronics product embeddings" \
  --dimension 8 \
  --index-type HNSW

echo ""
echo "Creating Clothing Store database..."
jade-db --url "$API_URL" --api-key "$API_KEY" create-db \
  --name clothing_store \
  --description "Clothing product embeddings" \
  --dimension 8 \
  --index-type HNSW

echo ""
echo "Creating Books Store database..."
jade-db --url "$API_URL" --api-key "$API_KEY" create-db \
  --name books_store \
  --description "Book embeddings" \
  --dimension 8 \
  --index-type HNSW

echo ""
echo "✓ All store databases created"
pause

# Step 3: List all databases
echo "STEP 3: Listing all databases"
echo "=============================="
jade-db --url "$API_URL" --api-key "$API_KEY" list-dbs
pause

# Step 4: Populate Electronics Store
echo "STEP 4: Populating Electronics Store"
echo "====================================="

echo "Adding laptops..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id electronics_store \
  --vector-id laptop_001 \
  --values "[0.9, 0.1, 0.2, 0.8, 0.7, 0.3, 0.5, 0.6]" \
  --metadata '{"name": "UltraBook Pro 15", "category": "laptop", "brand": "TechCorp", "price": 1299.99, "in_stock": true}'

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id electronics_store \
  --vector-id laptop_002 \
  --values "[0.85, 0.15, 0.25, 0.75, 0.65, 0.35, 0.55, 0.58]" \
  --metadata '{"name": "Gaming Laptop X", "category": "laptop", "brand": "GameTech", "price": 1599.99, "in_stock": true}'

echo ""
echo "Adding smartphones..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id electronics_store \
  --vector-id phone_001 \
  --values "[0.7, 0.8, 0.9, 0.1, 0.5, 0.6, 0.4, 0.3]" \
  --metadata '{"name": "SmartPhone X Pro", "category": "phone", "brand": "MobilePlus", "price": 999.99, "in_stock": true}'

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id electronics_store \
  --vector-id phone_002 \
  --values "[0.65, 0.75, 0.85, 0.15, 0.55, 0.65, 0.45, 0.35]" \
  --metadata '{"name": "Budget Phone Y", "category": "phone", "brand": "ValueTech", "price": 299.99, "in_stock": false}'

echo ""
echo "✓ Electronics store populated"
pause

# Step 5: Populate Clothing Store
echo "STEP 5: Populating Clothing Store"
echo "=================================="

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id clothing_store \
  --vector-id shirt_001 \
  --values "[0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.9, 0.1]" \
  --metadata '{"name": "Cotton T-Shirt", "category": "shirt", "size": "M", "color": "blue", "price": 29.99}'

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id clothing_store \
  --vector-id jeans_001 \
  --values "[0.4, 0.6, 0.3, 0.7, 0.5, 0.5, 0.8, 0.2]" \
  --metadata '{"name": "Denim Jeans", "category": "pants", "size": "32", "color": "dark blue", "price": 59.99}'

echo ""
echo "✓ Clothing store populated"
pause

# Step 6: Populate Books Store
echo "STEP 6: Populating Books Store"
echo "==============================="

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id books_store \
  --vector-id book_001 \
  --values "[0.2, 0.3, 0.9, 0.8, 0.1, 0.4, 0.7, 0.6]" \
  --metadata '{"title": "Vector Databases Explained", "author": "Jane Smith", "genre": "technology", "price": 39.99, "pages": 350}'

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id books_store \
  --vector-id book_002 \
  --values "[0.25, 0.35, 0.85, 0.75, 0.15, 0.45, 0.65, 0.55]" \
  --metadata '{"title": "Machine Learning Basics", "author": "John Doe", "genre": "technology", "price": 49.99, "pages": 420}'

echo ""
echo "✓ Books store populated"
pause

# Step 7: Search in Electronics Store
echo "STEP 7: Searching Electronics Store"
echo "===================================="
echo "Finding laptops similar to [0.88, 0.12, 0.22, 0.78, 0.68, 0.32, 0.52, 0.59]"
echo ""

jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id electronics_store \
  --query-vector "[0.88, 0.12, 0.22, 0.78, 0.68, 0.32, 0.52, 0.59]" \
  --top-k 3

pause

# Step 8: Search in Clothing Store
echo "STEP 8: Searching Clothing Store"
echo "================================="
echo "Finding items similar to [0.35, 0.65, 0.25, 0.75, 0.45, 0.55, 0.85, 0.15]"
echo ""

jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id clothing_store \
  --query-vector "[0.35, 0.65, 0.25, 0.75, 0.45, 0.55, 0.85, 0.15]" \
  --top-k 2

pause

# Step 9: Retrieve specific items
echo "STEP 9: Retrieving Specific Products"
echo "====================================="

echo "Retrieving laptop_001 from electronics store..."
jade-db --url "$API_URL" --api-key "$API_KEY" retrieve \
  --database-id electronics_store \
  --vector-id laptop_001

echo ""
echo "Retrieving book_001 from books store..."
jade-db --url "$API_URL" --api-key "$API_KEY" retrieve \
  --database-id books_store \
  --vector-id book_001

pause

# Step 10: Get database info
echo "STEP 10: Getting Database Information"
echo "======================================"

echo "Electronics Store Info:"
jade-db --url "$API_URL" --api-key "$API_KEY" get-db \
  --database-id electronics_store

pause

# Step 11: Cleanup demonstration
echo "STEP 11: Cleanup (Optional)"
echo "==========================="
echo ""
echo "To clean up all demo databases, run:"
echo ""
echo "  jade-db --url $API_URL --api-key $API_KEY delete-db --database-id electronics_store"
echo "  jade-db --url $API_URL --api-key $API_KEY delete-db --database-id clothing_store"
echo "  jade-db --url $API_URL --api-key $API_KEY delete-db --database-id books_store"
echo ""

read -p "Do you want to delete the demo databases now? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deleting databases..."
    jade-db --url "$API_URL" --api-key "$API_KEY" delete-db --database-id electronics_store
    jade-db --url "$API_URL" --api-key "$API_KEY" delete-db --database-id clothing_store
    jade-db --url "$API_URL" --api-key "$API_KEY" delete-db --database-id books_store
    echo "✓ All demo databases deleted"
else
    echo "Databases preserved for further exploration"
fi

echo ""
echo "================================================"
echo "Complete Workflow Demo Finished! ✓"
echo "================================================"
echo ""
echo "You've learned:"
echo "  - Multi-database management"
echo "  - Organizing vectors by domain"
echo "  - Cross-database operations"
echo "  - Complete CRUD workflow"
echo ""
echo "Next steps:"
echo "  - Try the advanced tutorial (../advanced.md)"
echo "  - Explore metadata filtering"
echo "  - Experiment with different index types"
echo ""
