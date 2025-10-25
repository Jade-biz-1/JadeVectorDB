# JadeVectorDB cURL Command Generation

The JadeVectorDB CLI includes built-in support for generating cURL commands for all operations. This feature allows you to see the exact API calls being made and use them directly in your own scripts or applications.

## Usage

To generate cURL commands instead of executing operations directly, use the `--curl-only` flag with any CLI command.

### Python CLI

```bash
# Generate cURL command for creating a database
jade-db --curl-only --url http://localhost:8080 create-db --name mydb --description "My database" --dimension 128

# Generate cURL command for storing a vector
jade-db --curl-only --url http://localhost:8080 store --database-id mydb --vector-id vec1 --values "[0.1,0.2,0.3]" --metadata '{"category":"test"}'

# Generate cURL command for similarity search
jade-db --curl-only --url http://localhost:8080 search --database-id mydb --query-vector "[0.15,0.25,0.35]" --top-k 5 --threshold 0.7
```

### Shell Script CLI

```bash
# Generate cURL command for listing databases
./jade-db.sh --curl-only --url http://localhost:8080 list-dbs

# Generate cURL command for retrieving a vector
./jade-db.sh --curl-only --url http://localhost:8080 retrieve --database-id mydb --vector-id vec1

# Generate cURL command for getting system status
./jade-db.sh --curl-only --url http://localhost:8080 status
```

## Benefits

1. **API Transparency**: See exactly what API calls are being made
2. **Direct cURL Usage**: Copy and paste generated commands for direct API interaction
3. **Educational Value**: Learn the underlying API while using familiar CLI syntax
4. **Scripting Integration**: Easily integrate cURL commands into shell scripts
5. **Debugging Aid**: Troubleshoot issues by examining actual API requests

## Examples

### Create Database
```bash
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "name": "my-database",
    "description": "My test database",
    "vectorDimension": 128,
    "indexType": "HNSW"
  }'
```

### Store Vector
```bash
curl -X POST http://localhost:8080/v1/databases/my-database/vectors \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "id": "vector-1",
    "values": [0.1, 0.2, 0.3],
    "metadata": {
      "category": "test",
      "tags": ["example", "tutorial"]
    }
  }'
```

### Similarity Search
```bash
curl -X POST http://localhost:8080/v1/databases/my-database/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "queryVector": [0.15, 0.25, 0.35],
    "topK": 5,
    "threshold": 0.7
  }'
```

## Supported Operations

All CLI operations support cURL command generation:
- Database management (create, list, get, update, delete)
- Vector operations (store, retrieve, update, delete, batch operations)
- Search operations (similarity search, advanced search with filters)
- Index management (create, list, update, delete)
- Embedding generation (text and image embeddings)
- System monitoring (status, health)

## Authentication

When using API keys, they will be included in the generated cURL commands as Authorization headers:

```bash
-H "Authorization: Bearer YOUR_API_KEY"
```

## Usage Tips

1. **Copy and Paste**: Generated cURL commands can be copied and pasted directly into your terminal
2. **Modify as Needed**: Edit the generated commands to change parameters or add additional options
3. **Save for Later**: Store generated commands in scripts for repeated use
4. **Share with Others**: Share cURL commands with teammates who may not be familiar with the CLI