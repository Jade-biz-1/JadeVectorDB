#!/bin/bash

# JadeVectorDB Product Recommendation Demo
# This script demonstrates a practical use case: product recommendations
# using similarity search on product embeddings

set -e  # Exit on error

echo "====================================================="
echo "JadeVectorDB Product Recommendation System Demo"
echo "====================================================="
echo ""

# Configuration
API_URL="${JADEVECTORDB_URL:-http://localhost:8080}"
API_KEY="${JADEVECTORDB_API_KEY:-your-api-key}"
DB_NAME="product_catalog"

echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Database: $DB_NAME"
echo ""

# Create product catalog database
echo "Setting up product catalog database..."
jade-db --url "$API_URL" --api-key "$API_KEY" create-db \
  --name "$DB_NAME" \
  --description "Product recommendation catalog" \
  --dimension 16 \
  --index-type HNSW

echo "✓ Database created"
echo ""

# Populate with diverse products
echo "Populating product catalog..."
echo ""

# High-end laptops
echo "Adding premium laptops..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "laptop_premium_001" \
  --values "[0.95, 0.05, 0.10, 0.90, 0.85, 0.15, 0.20, 0.80, 0.75, 0.25, 0.30, 0.70, 0.65, 0.35, 0.40, 0.60]" \
  --metadata '{"name": "UltraBook Pro 16", "category": "laptop", "brand": "TechCorp", "price": 2499.99, "rating": 4.8, "tags": ["premium", "professional", "high-performance"]}'

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "laptop_premium_002" \
  --values "[0.92, 0.08, 0.12, 0.88, 0.82, 0.18, 0.22, 0.78, 0.72, 0.28, 0.32, 0.68, 0.62, 0.38, 0.42, 0.58]" \
  --metadata '{"name": "Developer Elite X1", "category": "laptop", "brand": "DevTech", "price": 2299.99, "rating": 4.7, "tags": ["premium", "developer", "powerful"]}'

# Mid-range laptops
echo "Adding mid-range laptops..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "laptop_midrange_001" \
  --values "[0.70, 0.30, 0.25, 0.75, 0.65, 0.35, 0.40, 0.60, 0.55, 0.45, 0.50, 0.50, 0.45, 0.55, 0.60, 0.40]" \
  --metadata '{"name": "Everyday Laptop Plus", "category": "laptop", "brand": "ValueTech", "price": 899.99, "rating": 4.3, "tags": ["affordable", "reliable", "everyday"]}'

# Gaming laptops
echo "Adding gaming laptops..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "laptop_gaming_001" \
  --values "[0.88, 0.12, 0.85, 0.15, 0.90, 0.10, 0.80, 0.20, 0.75, 0.25, 0.82, 0.18, 0.78, 0.22, 0.72, 0.28]" \
  --metadata '{"name": "Gaming Beast RTX", "category": "laptop", "brand": "GameTech", "price": 1899.99, "rating": 4.6, "tags": ["gaming", "high-fps", "rgb"]}'

# Smartphones
echo "Adding smartphones..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "phone_flagship_001" \
  --values "[0.15, 0.85, 0.20, 0.80, 0.25, 0.75, 0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55, 0.50, 0.50]" \
  --metadata '{"name": "SmartPhone X Pro Max", "category": "phone", "brand": "MobilePlus", "price": 1199.99, "rating": 4.5, "tags": ["flagship", "camera", "5g"]}'

jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "phone_budget_001" \
  --values "[0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55, 0.50, 0.50, 0.55, 0.45, 0.60, 0.40, 0.65, 0.35]" \
  --metadata '{"name": "Budget Smart Y", "category": "phone", "brand": "ValueMobile", "price": 299.99, "rating": 4.0, "tags": ["budget", "reliable", "value"]}'

# Tablets
echo "Adding tablets..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "tablet_pro_001" \
  --values "[0.55, 0.45, 0.40, 0.60, 0.50, 0.50, 0.55, 0.45, 0.60, 0.40, 0.65, 0.35, 0.70, 0.30, 0.75, 0.25]" \
  --metadata '{"name": "Tablet Pro 12.9", "category": "tablet", "brand": "TechCorp", "price": 999.99, "rating": 4.7, "tags": ["professional", "creative", "stylus"]}'

# Monitors
echo "Adding monitors..."
jade-db --url "$API_URL" --api-key "$API_KEY" store \
  --database-id "$DB_NAME" \
  --vector-id "monitor_4k_001" \
  --values "[0.60, 0.40, 0.50, 0.50, 0.65, 0.35, 0.70, 0.30, 0.75, 0.25, 0.80, 0.20, 0.85, 0.15, 0.90, 0.10]" \
  --metadata '{"name": "4K UltraWide Monitor", "category": "monitor", "brand": "DisplayTech", "price": 799.99, "rating": 4.6, "tags": ["4k", "ultrawide", "professional"]}'

echo "✓ Product catalog populated with 8 products"
echo ""
echo "====================================================="
echo ""

# Demo 1: Find products similar to premium laptop
echo "DEMO 1: Recommendations for users viewing 'UltraBook Pro 16'"
echo "=============================================================="
echo ""
echo "Searching for products similar to premium laptop..."
echo "Query vector: [0.95, 0.05, 0.10, 0.90, 0.85, 0.15, 0.20, 0.80, 0.75, 0.25, 0.30, 0.70, 0.65, 0.35, 0.40, 0.60]"
echo ""

jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id "$DB_NAME" \
  --query-vector "[0.95, 0.05, 0.10, 0.90, 0.85, 0.15, 0.20, 0.80, 0.75, 0.25, 0.30, 0.70, 0.65, 0.35, 0.40, 0.60]" \
  --top-k 3

echo ""
read -p "Press Enter to continue to next demo..."
echo ""

# Demo 2: Find products similar to gaming laptop
echo "DEMO 2: Recommendations for gaming enthusiasts"
echo "==============================================="
echo ""
echo "Searching for products similar to gaming laptop..."
echo "Query vector: [0.88, 0.12, 0.85, 0.15, 0.90, 0.10, 0.80, 0.20, 0.75, 0.25, 0.82, 0.18, 0.78, 0.22, 0.72, 0.28]"
echo ""

jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id "$DB_NAME" \
  --query-vector "[0.88, 0.12, 0.85, 0.15, 0.90, 0.10, 0.80, 0.20, 0.75, 0.25, 0.82, 0.18, 0.78, 0.22, 0.72, 0.28]" \
  --top-k 3

echo ""
read -p "Press Enter to continue to next demo..."
echo ""

# Demo 3: Find products similar to smartphone
echo "DEMO 3: Mobile device recommendations"
echo "======================================"
echo ""
echo "Searching for products similar to flagship smartphone..."
echo "Query vector: [0.15, 0.85, 0.20, 0.80, 0.25, 0.75, 0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55, 0.50, 0.50]"
echo ""

jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id "$DB_NAME" \
  --query-vector "[0.15, 0.85, 0.20, 0.80, 0.25, 0.75, 0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55, 0.50, 0.50]" \
  --top-k 3

echo ""
read -p "Press Enter to continue to next demo..."
echo ""

# Demo 4: Budget-conscious search
echo "DEMO 4: Budget-friendly alternatives"
echo "====================================="
echo ""
echo "Searching for budget-friendly products..."
echo "Query vector biased toward mid-range/budget: [0.50, 0.50, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65, 0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55]"
echo ""

jade-db --url "$API_URL" --api-key "$API_KEY" search \
  --database-id "$DB_NAME" \
  --query-vector "[0.50, 0.50, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65, 0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55]" \
  --top-k 4

echo ""
echo "====================================================="
echo "Product Recommendation Demo Completed! ✓"
echo "====================================================="
echo ""
echo "Use cases demonstrated:"
echo "  ✓ Similar product recommendations"
echo "  ✓ Category-specific suggestions"
echo "  ✓ Budget-aware recommendations"
echo "  ✓ Cross-selling opportunities"
echo ""
echo "Real-world applications:"
echo "  - E-commerce product recommendations"
echo "  - Content discovery systems"
echo "  - Similar item suggestions"
echo "  - Personalized shopping experiences"
echo ""
echo "To clean up, delete the database:"
echo "  jade-db --url $API_URL --api-key $API_KEY delete-db --database-id $DB_NAME"
echo ""
