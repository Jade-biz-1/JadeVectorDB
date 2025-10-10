# Quickstart Guide for JadeVectorDB API

This guide provides a quick walk-through of the essential JadeVectorDB API endpoints. You will learn how to create a database, store vectors, and perform a similarity search using `curl`.

## Prerequisites

- A running instance of JadeVectorDB.
- `curl` installed on your machine.
- An API key for authentication.

## Basic Usage

The following examples assume your JadeVectorDB instance is running at `http://localhost:8080` and you have an API key.

### 1. Create a Vector Database

First, let's create a new database for 128-dimensional vectors.

```bash
curl -X POST http://localhost:8080/api/v1/databases \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "name": "quickstart-db",
    "vectorDimension": 128,
    "indexType": "HNSW"
  }'
```

The response will contain the details of the newly created database, including its unique `databaseId`.

**Example Response:**
```json
{
  "databaseId": "db_1234567890abcdef",
  "name": "quickstart-db",
  "vectorDimension": 128,
  "indexType": "HNSW",
  ...
}
```

**Copy the `databaseId` from the response for the next steps.**

### 2. Store Vectors

Now, let's add some vectors to our new database. Replace `{databaseId}` with the ID you copied from the previous step.

```bash
# Store two vectors in a batch request
curl -X POST http://localhost:8080/api/v1/databases/{databaseId}/vectors/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "vectors": [
      {
        "id": "vec-001",
        "values": [0.11, 0.22, ..., 0.88],
        "metadata": { "genre": "sci-fi" }
      },
      {
        "id": "vec-002",
        "values": [0.99, 0.88, ..., 0.11],
        "metadata": { "genre": "fantasy" }
      }
    ]
  }'
```

### 3. Perform a Similarity Search

Finally, let's find vectors similar to a query vector.

```bash
curl -X POST http://localhost:8080/api/v1/databases/{databaseId}/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "queryVector": [0.12, 0.23, ..., 0.89],
    "topK": 2,
    "includeMetadata": true
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "id": "vec-001",
      "similarity": 0.987,
      "metadata": { "genre": "sci-fi" }
    },
    {
      "id": "vec-002",
      "similarity": 0.456,
      "metadata": { "genre": "fantasy" }
    }
  ]
}
```

## Next Steps

This guide covered the very basics. To learn more, explore the full [API documentation](contracts/vector-db-api.yaml).
