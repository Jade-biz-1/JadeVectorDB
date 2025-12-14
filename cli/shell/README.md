# JadeVectorDB Shell CLI

A lightweight bash-based command-line interface for interacting with JadeVectorDB. Perfect for system administration, automation, and environments where Python or Node.js are not available.

## Features

- ✅ **No Dependencies** - Pure bash implementation, works anywhere
- ✅ **cURL Generation** - Generate cURL commands for API learning and debugging
- ✅ **Scriptable** - Easy integration into shell scripts and CI/CD pipelines
- ✅ **Portable** - Works on any system with bash and curl
- ✅ **Lightweight** - Minimal resource footprint

## Installation

No installation required! Just make the script executable:

```bash
chmod +x cli/shell/scripts/jade-db.sh
```

Or use the compiled binary:

```bash
chmod +x cli/shell/bin/jade-db
```

## Usage

### Using the Script

```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 [command] [options]
```

### Using the Binary

```bash
./cli/shell/bin/jade-db --url http://localhost:8080 [command] [options]
```

### Global Options

- `--url <url>` - JadeVectorDB API URL (required)
- `--api-key <key>` - API key for authentication
- `--curl-only` - Generate cURL command instead of executing

## Commands

### Database Management

#### Create a Database

```bash
bash jade-db.sh --url http://localhost:8080 create-db my_database "My database description" 768 HNSW
```

**Parameters:**
1. Database name (required)
2. Description (optional, default: "")
3. Vector dimension (optional, default: 128)
4. Index type (optional, default: HNSW)

#### List Databases

```bash
bash jade-db.sh --url http://localhost:8080 list-dbs
```

#### Get Database Details

```bash
bash jade-db.sh --url http://localhost:8080 get-db my_database_id
```

#### Delete a Database

```bash
bash jade-db.sh --url http://localhost:8080 delete-db my_database_id
```

### Vector Operations

#### Store a Vector

```bash
bash jade-db.sh --url http://localhost:8080 --database-id my_db store vec_1 '[0.1, 0.2, 0.3]'
```

With metadata:

```bash
bash jade-db.sh --url http://localhost:8080 --database-id my_db store vec_1 '[0.1, 0.2, 0.3]' '{"category":"test"}'
```

#### Retrieve a Vector

```bash
bash jade-db.sh --url http://localhost:8080 --database-id my_db retrieve vec_1
```

#### Delete a Vector

```bash
bash jade-db.sh --url http://localhost:8080 --database-id my_db delete vec_1
```

### Search Operations

#### Similarity Search

```bash
bash jade-db.sh --url http://localhost:8080 --database-id my_db search '[0.15, 0.25, 0.35]' 5 0.7
```

**Parameters:**
1. Query vector (required)
2. Top K results (optional, default: 10)
3. Similarity threshold (optional, default: 0.0)

### System Operations

#### Health Check

```bash
bash jade-db.sh --url http://localhost:8080 health
```

#### System Status

```bash
bash jade-db.sh --url http://localhost:8080 status
```

## cURL Command Generation

Generate cURL commands for any operation using `--curl-only`:

```bash
# Generate cURL for database creation
bash jade-db.sh --curl-only --url http://localhost:8080 create-db my_db "Test database" 768 HNSW

# Output:
curl -X POST http://localhost:8080/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{"name":"my_db","description":"Test database","dimension":768,"index_type":"HNSW"}'
```

This is useful for:
- Learning the JadeVectorDB REST API
- Debugging API calls
- Creating standalone curl scripts
- API documentation examples

## Environment Variables

Set default values using environment variables:

```bash
export JADE_DB_URL=http://localhost:8080
export JADE_DB_API_KEY=your-api-key

# Now you can omit --url and --api-key
bash jade-db.sh list-dbs
```

## Examples

### Complete Workflow

```bash
# Set environment variables
export JADE_DB_URL=http://localhost:8080

# Create database
bash jade-db.sh create-db products "Product embeddings" 384 HNSW

# Store vectors
bash jade-db.sh --database-id products store prod_1 '[0.1, 0.2, 0.3]' '{"category":"electronics"}'
bash jade-db.sh --database-id products store prod_2 '[0.4, 0.5, 0.6]' '{"category":"clothing"}'
bash jade-db.sh --database-id products store prod_3 '[0.7, 0.8, 0.9]' '{"category":"books"}'

# Search for similar products
bash jade-db.sh --database-id products search '[0.15, 0.25, 0.35]' 3

# Retrieve specific vector
bash jade-db.sh --database-id products retrieve prod_1

# Check system health
bash jade-db.sh health
```

### CI/CD Integration

```bash
#!/bin/bash
# deploy.sh - Automated vector database setup

set -e

JADE_URL="http://jadevectordb.production:8080"
API_KEY="${JADE_API_KEY}"

# Create database
bash jade-db.sh --url "$JADE_URL" --api-key "$API_KEY" \
  create-db embeddings "Production embeddings" 768 HNSW

# Import vectors from file
while IFS=',' read -r id vector metadata; do
  bash jade-db.sh --url "$JADE_URL" --api-key "$API_KEY" \
    --database-id embeddings store "$id" "$vector" "$metadata"
done < vectors.csv

# Verify health
bash jade-db.sh --url "$JADE_URL" --api-key "$API_KEY" health
```

### Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Check cluster health every 5 minutes

while true; do
  STATUS=$(bash jade-db.sh --url http://localhost:8080 health | jq -r '.status')

  if [ "$STATUS" != "healthy" ]; then
    echo "ALERT: JadeVectorDB is unhealthy!" | mail -s "Alert" admin@example.com
  fi

  sleep 300
done
```

## Comparison with Other CLIs

| Feature | Shell CLI | Python CLI | JavaScript CLI |
|---------|-----------|------------|----------------|
| **Dependencies** | None (bash + curl) | Python 3.8+ | Node.js 14+ |
| **Installation** | chmod +x | pip install | npm install |
| **Performance** | Fast | Fast | Fast |
| **cURL Generation** | ✅ Yes | ✅ Yes | ❌ No |
| **JSON Parsing** | jq (optional) | Built-in | Built-in |
| **Best For** | Scripts, CI/CD, System admin | Data science, Python apps | Web dev, Node.js apps |

## Troubleshooting

### Command Not Found

```bash
# Ensure script is executable
chmod +x cli/shell/scripts/jade-db.sh

# Or use bash explicitly
bash cli/shell/scripts/jade-db.sh [command]
```

### Connection Refused

```bash
# Verify JadeVectorDB is running
curl http://localhost:8080/health

# Check firewall rules
telnet localhost 8080
```

### Invalid JSON Response

```bash
# Use jq for pretty printing
bash jade-db.sh --url http://localhost:8080 list-dbs | jq '.'

# Check raw response
bash jade-db.sh --url http://localhost:8080 health
```

## File Structure

```
cli/shell/
├── README.md              # This file
├── bin/
│   └── jade-db           # Compiled binary
└── scripts/
    └── jade-db.sh        # Main shell script
```

## Development

### Testing

```bash
# Run basic tests
bash cli/tests/test_cli_curl.sh

# Test cURL generation
bash jade-db.sh --curl-only --url http://localhost:8080 list-dbs
```

### Contributing

When updating the shell CLI:
1. Maintain POSIX compliance where possible
2. Test on bash, zsh, and sh
3. Add error handling for all operations
4. Update this README with new commands
5. Keep dependencies minimal (bash + curl only)

## Related Documentation

- [Main CLI Documentation](../README.md)
- [CLI Examples](../../examples/cli/shell-examples.md)
- [Python CLI](../python/README.md)
- [JavaScript CLI](../js/README.md)
- [Distributed CLI](../distributed/README.md)
- [CLI Tutorial](../../tutorials/cli/README.md)

## Support

For issues:
- Check [Troubleshooting](#troubleshooting) section above
- See [Main CLI README](../README.md)
- GitHub Issues: [JadeVectorDB Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
