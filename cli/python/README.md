# JadeVectorDB Python Client Library

The JadeVectorDB Python client library provides a convenient interface for interacting with the JadeVectorDB vector database from Python applications.

## Installation

```bash
pip install jadevectordb
```

## Usage

### Initialize the Client

```python
from jadevectordb import JadeVectorDB

# Initialize with server URL and optional API key
client = JadeVectorDB(base_url="http://localhost:8080", api_key="your-api-key")
```

### Create a Database

```python
# Create a new vector database
db_id = client.create_database(
    name="my-documents",
    description="Vector database for document embeddings",
    vector_dimension=768,  # For BERT-based models
    index_type="HNSW"
)
print(f"Created database with ID: {db_id}")
```

### Store a Vector

```python
from jadevectordb import Vector

# Store a single vector with metadata
success = client.store_vector(
    database_id="my-documents",
    vector_id="doc-123",
    values=[0.1, 0.2, 0.3, 0.4],  # Vector embedding
    metadata={
        "category": "article",
        "title": "Introduction to Vector Databases",
        "author": "John Doe"
    }
)
```

### Batch Store Vectors

```python
from jadevectordb import Vector

# Create a list of vectors to store
vectors = [
    Vector(id="doc-1", values=[0.1, 0.2, 0.3], metadata={"category": "tech"}),
    Vector(id="doc-2", values=[0.4, 0.5, 0.6], metadata={"category": "science"}),
    Vector(id="doc-3", values=[0.7, 0.8, 0.9], metadata={"category": "research"})
]

# Store all vectors in a single request
success = client.batch_store_vectors(
    database_id="my-documents",
    vectors=vectors
)
```

### Retrieve a Vector

```python
# Retrieve a specific vector
vector = client.retrieve_vector(
    database_id="my-documents",
    vector_id="doc-123"
)

if vector:
    print(f"Vector ID: {vector.id}")
    print(f"Vector Values: {vector.values}")
    print(f"Metadata: {vector.metadata}")
else:
    print("Vector not found")
```

### Perform Similarity Search

```python
# Perform a similarity search
query_vector = [0.15, 0.25, 0.35]  # Example query vector
results = client.search(
    database_id="my-documents",
    query_vector=query_vector,
    top_k=5,  # Return top 5 most similar vectors
    threshold=0.7  # Minimum similarity threshold
)

for result in results:
    print(f"ID: {result['id']}, Similarity: {result['similarity']}")
```

### Get Database Information

```python
# Get information about a specific database
database_info = client.get_database(db_id)
print(database_info)

# List all databases
all_databases = client.list_databases()
for db in all_databases:
    print(f"Database: {db['name']} ({db['databaseId']})")
```

## API Reference

### JadeVectorDB Class

The main client class for interacting with the JadeVectorDB API.

#### Constructor

`JadeVectorDB(base_url, api_key=None)`

- `base_url`: The base URL for the JadeVectorDB API (e.g., "http://localhost:8080")
- `api_key`: Optional API key for authentication

#### Methods

- `create_database(name, description="", vector_dimension=128, index_type="HNSW", **kwargs)`: Create a new vector database
- `get_database(database_id)`: Get information about a specific database
- `list_databases()`: Get a list of all databases
- `store_vector(database_id, vector_id, values, metadata=None)`: Store a vector in the database
- `batch_store_vectors(database_id, vectors)`: Store multiple vectors in the database
- `retrieve_vector(database_id, vector_id)`: Retrieve a vector from the database
- `delete_vector(database_id, vector_id)`: Delete a vector from the database
- `search(database_id, query_vector, top_k=10, threshold=None, filters=None)`: Perform similarity search
- `get_status()`: Get system status information
- `get_health()`: Get system health information