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

#### Database & Vector Operations

- `update_database(database_id, name=None, description=None, vector_dimension=None, index_type=None)`: Update a database's configuration
- `list_vectors(database_id, limit=50, offset=0)`: List vectors with pagination
- `update_vector(database_id, vector_id, values, metadata=None)`: Update a vector's values and/or metadata
- `batch_get_vectors(database_id, vector_ids)`: Retrieve multiple vectors by ID in one request

#### Search & Reranking

- `advanced_search(database_id, query_vector, top_k=10, threshold=None, filters=None, include_metadata=True, include_values=False)`: Advanced similarity search with metadata/value inclusion options
- `rerank_search(database_id, query_text, query_vector=None, top_k=10, enable_reranking=True, rerank_top_n=100)`: Search with reranking
- `rerank(query, documents, top_k=None)`: Standalone document reranking
- `get_reranking_config(database_id)`: Get reranking configuration
- `update_reranking_config(database_id, model_name=None, batch_size=None, score_threshold=None, combine_scores=None, rerank_weight=None)`: Update reranking configuration

#### Index Management

- `create_index(database_id, index_type, name=None, parameters=None)`: Create an index on a database
- `list_indexes(database_id)`: List all indexes for a database
- `update_index(database_id, index_id, parameters=None)`: Update an index's parameters
- `delete_index(database_id, index_id)`: Delete an index

#### Embeddings

- `generate_embeddings(text, input_type="text", model="default", provider="default")`: Generate vector embeddings from text

#### API Key Management

- `create_api_key(user_id, description="", permissions=None, validity_days=0)`: Create a new API key
- `list_api_keys(user_id=None)`: List API keys, optionally filtered by user
- `revoke_api_key(key_id)`: Revoke an API key

#### Security & Audit

- `get_audit_log(user_id=None, event_type=None, limit=100)`: Get audit log entries
- `get_sessions(user_id)`: Get active sessions for a user
- `get_audit_stats()`: Get audit statistics summary

#### Analytics

- `get_analytics_stats(database_id, start_time=None, end_time=None, granularity="hourly")`: Get analytics statistics
- `get_analytics_queries(database_id, start_time=None, end_time=None, limit=100, offset=0)`: Get query analytics
- `get_analytics_patterns(database_id, start_time=None, end_time=None, min_count=2, limit=100)`: Get query patterns
- `get_analytics_insights(database_id, start_time=None, end_time=None)`: Get analytics insights
- `get_analytics_trending(database_id, start_time=None, end_time=None, min_growth=0.5, limit=100)`: Get trending queries
- `submit_analytics_feedback(database_id, query_id, user_id=None, rating=None, feedback_text=None, clicked_result_id=None, clicked_rank=None)`: Submit search feedback
- `export_analytics(database_id, format="json", start_time=None, end_time=None)`: Export analytics data

#### Password Management

- `change_password(user_id, old_password, new_password)`: Change a user's password
- `admin_reset_password(user_id, new_password)`: Admin reset a user's password

### Usage Examples

#### Advanced Search with Filters

```python
results = client.advanced_search(
    database_id="my-db",
    query_vector=[0.1, 0.2, 0.3],
    top_k=5,
    filters={"category": "tech"},
    include_metadata=True,
    include_values=True
)
```

#### API Key Management

```python
# Create an API key
key_info = client.create_api_key(
    user_id="user-123",
    description="Production API key",
    permissions=["read", "write"],
    validity_days=90
)
print(f"API Key: {key_info['key']}")

# List keys for a user
keys = client.list_api_keys(user_id="user-123")

# Revoke a key
client.revoke_api_key(key_id="key-456")
```

#### Index Management

```python
# Create an HNSW index
index = client.create_index(
    database_id="my-db",
    index_type="HNSW",
    name="my-hnsw-index",
    parameters={"M": 16, "efConstruction": 200}
)

# List indexes
indexes = client.list_indexes(database_id="my-db")

# Delete an index
client.delete_index(database_id="my-db", index_id="idx-123")
```

#### Generate Embeddings

```python
result = client.generate_embeddings(
    text="What is a vector database?",
    model="default"
)
embedding = result['embedding']
```

#### Analytics

```python
# Get analytics stats
stats = client.get_analytics_stats(
    database_id="my-db",
    granularity="daily"
)

# Submit feedback on a search query
client.submit_analytics_feedback(
    database_id="my-db",
    query_id="q-123",
    rating=5,
    feedback_text="Great results!"
)
```