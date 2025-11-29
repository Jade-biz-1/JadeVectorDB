#!/bin/bash

# Exercise 1: CLI Basics - Practice Script
# Complete the TODOs to finish this exercise

# Setup
export JADE_DB_URL=http://localhost:8080
export JADE_DB_API_KEY=mykey123

echo "========================================="
echo "Exercise 1: CLI Basics"
echo "========================================="
echo ""

# TODO 1: Check system health
echo "Step 1: Checking system health..."
# HINT: Use 'jade-db health' or the shell equivalent
# YOUR CODE HERE:


# TODO 2: Create a database
echo ""
echo "Step 2: Creating database..."
# HINT: Create a database named 'my_products' with dimension 8 and index type HNSW
# YOUR CODE HERE:


# TODO 3: Store first vector (laptop)
echo ""
echo "Step 3: Storing laptop vector..."
# HINT: Use vector ID 'laptop_001' with values from the README
# YOUR CODE HERE:


# TODO 4: Store second vector (phone)
echo ""
echo "Step 4: Storing phone vector..."
# HINT: Use data from ../../sample-data/products.json for phone_001
# YOUR CODE HERE:


# TODO 5: Store third vector (tablet)
echo ""
echo "Step 5: Storing tablet vector..."
# HINT: Use data from ../../sample-data/products.json for tablet_001
# YOUR CODE HERE:


# TODO 6: Retrieve a vector
echo ""
echo "Step 6: Retrieving laptop vector..."
# HINT: Retrieve vector with ID 'laptop_001'
# YOUR CODE HERE:


# TODO 7: Perform similarity search
echo ""
echo "Step 7: Performing similarity search..."
# HINT: Search for top 3 similar vectors to [0.83, 0.14, 0.24, 0.76, 0.63, 0.42, 0.89, 0.32]
# YOUR CODE HERE:


# TODO 8: Search with threshold
echo ""
echo "Step 8: Searching with threshold..."
# HINT: Same search as above but only results with similarity > 0.9
# YOUR CODE HERE:


echo ""
echo "========================================="
echo "Exercise complete! Run verify.sh to check your work."
echo "========================================="
