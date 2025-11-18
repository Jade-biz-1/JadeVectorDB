#!/usr/bin/env python3
"""
JadeVectorDB Batch Import Example

This script demonstrates how to efficiently import large numbers of vectors
using the Python client library.
"""

import os
import sys
import random
import time
from jadevectordb import JadeVectorDB, Vector

# Configuration
API_URL = os.environ.get('JADEVECTORDB_URL', 'http://localhost:8080')
API_KEY = os.environ.get('JADEVECTORDB_API_KEY', 'your-api-key')
DB_NAME = "batch_import_db"
VECTOR_DIMENSION = 128
NUM_VECTORS = 1000

def generate_sample_vector(dimension):
    """Generate a random normalized vector."""
    vector = [random.random() for _ in range(dimension)]
    # Normalize the vector
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [x / magnitude for x in vector]

def main():
    print("=" * 60)
    print("JadeVectorDB Batch Import Tutorial")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  API URL: {API_URL}")
    print(f"  Database: {DB_NAME}")
    print(f"  Vector Dimension: {VECTOR_DIMENSION}")
    print(f"  Number of Vectors: {NUM_VECTORS}")
    print()

    # Initialize client
    print("Initializing JadeVectorDB client...")
    client = JadeVectorDB(API_URL, api_key=API_KEY)
    print("✓ Client initialized")
    print()

    # Create database
    print(f"Creating database '{DB_NAME}'...")
    try:
        db_response = client.create_database(
            name=DB_NAME,
            vector_dimension=VECTOR_DIMENSION,
            index_type="HNSW",
            description="Batch import tutorial database"
        )
        print(f"✓ Database created: {db_response.get('databaseId', DB_NAME)}")
    except Exception as e:
        print(f"Note: Database may already exist ({e})")
    print()

    # Generate and batch store vectors
    print(f"Generating {NUM_VECTORS} sample vectors...")
    vectors = []
    categories = ["electronics", "clothing", "books", "home", "toys"]

    for i in range(NUM_VECTORS):
        vector = Vector(
            id=f"item_{i:06d}",
            values=generate_sample_vector(VECTOR_DIMENSION),
            metadata={
                "item_id": i,
                "category": random.choice(categories),
                "price": round(random.uniform(10.0, 1000.0), 2),
                "in_stock": random.choice([True, False]),
                "rating": round(random.uniform(1.0, 5.0), 1)
            }
        )
        vectors.append(vector)

    print(f"✓ Generated {NUM_VECTORS} vectors")
    print()

    # Batch import in chunks
    BATCH_SIZE = 100
    print(f"Importing vectors in batches of {BATCH_SIZE}...")
    start_time = time.time()

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (NUM_VECTORS + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            client.batch_store_vectors(DB_NAME, batch)
            print(f"  ✓ Batch {batch_num}/{total_batches} ({len(batch)} vectors) imported")
        except Exception as e:
            print(f"  ✗ Error importing batch {batch_num}: {e}")
            continue

    elapsed_time = time.time() - start_time
    vectors_per_second = NUM_VECTORS / elapsed_time

    print()
    print("Import Statistics:")
    print(f"  Total vectors: {NUM_VECTORS}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Import rate: {vectors_per_second:.2f} vectors/second")
    print()

    # Perform a sample search
    print("Performing sample similarity search...")
    query_vector = generate_sample_vector(VECTOR_DIMENSION)

    try:
        results = client.search(
            database_id=DB_NAME,
            query_vector=query_vector,
            top_k=5
        )

        print(f"✓ Found {len(results.get('results', []))} similar items:")
        for idx, result in enumerate(results.get('results', [])[:5], 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)
            print(f"  {idx}. {result.get('vectorId')} - "
                  f"Category: {metadata.get('category')}, "
                  f"Price: ${metadata.get('price')}, "
                  f"Score: {score:.4f}")
    except Exception as e:
        print(f"✗ Search error: {e}")

    print()
    print("=" * 60)
    print("Batch Import Tutorial Completed! ✓")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  - Query the database with different search vectors")
    print("  - Experiment with metadata filtering")
    print("  - Try different batch sizes for optimal performance")
    print()
    print("To clean up, delete the database:")
    print(f"  jade-db --url {API_URL} --api-key {API_KEY} delete-db --database-id {DB_NAME}")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nImport interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
