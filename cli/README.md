# JadeVectorDB CLI Tools

Command-line interface tools for interacting with the JadeVectorDB vector database system.

## Overview

JadeVectorDB provides **four different CLI implementations** to suit various environments, use cases, and user preferences:

1. **[Python CLI](#python-cli)** (`cli/python/`) - Full-featured CLI for data science and Python environments
2. **[Shell CLI](#shell-cli)** (`cli/shell/`) - Lightweight bash CLI for system administration and automation
3. **[JavaScript CLI](#javascript-cli)** (`cli/js/`) - Node.js CLI for web development environments
4. **[Distributed CLI](#distributed-cli)** (`cli/distributed/`) - Specialized CLI for managing distributed clusters

## Quick Start

### Python CLI

```bash
# Install
pip install -e cli/python

# Use
jade-db --url http://localhost:8080 create-db --name my-database
```

### Shell CLI

```bash
# Use directly (no installation needed)
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 create-db my-database
```

### JavaScript CLI

```bash
# Install dependencies
cd cli/js && npm install

# Use
node bin/jade-db.js --url http://localhost:8080 database create --name my-database
```

### Distributed CLI

```bash
# For cluster management
python cli/distributed/cluster_cli.py --host localhost --port 8080 status
```

## CLI Implementations

### Python CLI

**Location:** [`cli/python/`](python/README.md)

**Best For:**
- Data science workflows
- Python-based applications
- Programmatic API usage
- Integration with Jupyter notebooks

**Features:**
- ✅ Full API coverage
- ✅ Python client library included
- ✅ cURL command generation
- ✅ Rich error messages
- ✅ Batch operations support

**Installation:**

```bash
pip install -e cli/python
# Or from PyPI (when published)
pip install jadevectordb
```

**Example:**

```bash
jade-db --url http://localhost:8080 --api-key mykey \
  create-db --name documents --dimension 768 --index-type HNSW
```

[Full Python CLI Documentation →](python/README.md)

### Shell CLI

**Location:** [`cli/shell/`](shell/README.md)

**Best For:**
- Shell scripts and automation
- CI/CD pipelines
- System administration
- Environments without Python/Node.js

**Features:**
- ✅ Zero dependencies (bash + curl)
- ✅ cURL command generation
- ✅ Portable and lightweight
- ✅ Easy integration into scripts

**No Installation Required:**

```bash
chmod +x cli/shell/scripts/jade-db.sh
```

**Example:**

```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 \
  create-db products "Product embeddings" 384 HNSW
```

[Full Shell CLI Documentation →](shell/README.md)

### JavaScript CLI

**Location:** [`cli/js/`](js/README.md)

**Best For:**
- Node.js environments
- Web development workflows
- JavaScript-based automation
- NPM-based tooling

**Features:**
- ✅ Commander.js based
- ✅ Colorized output
- ✅ Promise-based API
- ✅ Environment variable support

**Installation:**

```bash
cd cli/js
npm install
```

**Example:**

```bash
node bin/jade-db.js --url http://localhost:8080 --api-key mykey \
  database create --name products --dimension 768
```

[Full JavaScript CLI Documentation →](js/README.md)

### Distributed CLI

**Location:** [`cli/distributed/`](distributed/README.md)

**Best For:**
- Cluster management
- DevOps and SRE tasks
- Multi-node operations
- Production monitoring

**Features:**
- ✅ Cluster status and diagnostics
- ✅ Node management (add/remove)
- ✅ Shard operations and migrations
- ✅ JSON/table/compact output formats
- ✅ Performance metrics

**Installation:**

```bash
pip install requests
```

**Example:**

```bash
python cli/distributed/cluster_cli.py --host localhost --port 8080 status
```

[Full Distributed CLI Documentation →](distributed/README.md)

## Feature Comparison

| Feature | Python | Shell | JavaScript | Distributed |
|---------|--------|-------|------------|-------------|
| **Database Operations** | ✅ | ✅ | ✅ | ❌ |
| **Vector Operations** | ✅ | ✅ | ✅ | ❌ |
| **Search Operations** | ✅ | ✅ | ✅ | ❌ |
| **Cluster Management** | ❌ | ❌ | ❌ | ✅ |
| **cURL Generation** | ✅ | ✅ | ❌ | ❌ |
| **Installation Required** | Yes (pip) | No | Yes (npm) | Yes (pip) |
| **Dependencies** | Python 3.8+ | bash, curl | Node.js 14+ | Python 3.8+ |
| **Client Library** | ✅ Included | ❌ | ❌ | ❌ |
| **Output Formats** | JSON | JSON | JSON | JSON/Table/Compact |

## Common Operations

### Database Management

#### Create a Database

```bash
# Python
jade-db --url http://localhost:8080 create-db --name my-db --dimension 768

# Shell
bash jade-db.sh --url http://localhost:8080 create-db my-db "Description" 768 HNSW

# JavaScript
node bin/jade-db.js --url http://localhost:8080 database create --name my-db --dimension 768
```

#### List Databases

```bash
# Python
jade-db --url http://localhost:8080 list-dbs

# Shell
bash jade-db.sh --url http://localhost:8080 list-dbs

# JavaScript
node bin/jade-db.js --url http://localhost:8080 database list
```

### Vector Operations

#### Store a Vector

```bash
# Python
jade-db --url http://localhost:8080 store --database-id db1 --vector-id vec1 --values "[0.1, 0.2, 0.3]"

# Shell
bash jade-db.sh --url http://localhost:8080 --database-id db1 store vec1 '[0.1, 0.2, 0.3]'

# JavaScript
node bin/jade-db.js --url http://localhost:8080 vector store --database-id db1 --vector-id vec1 --values "[0.1, 0.2, 0.3]"
```

### Cluster Management

```bash
# Distributed CLI only
python cli/distributed/cluster_cli.py --host localhost status
python cli/distributed/cluster_cli.py --host localhost nodes
python cli/distributed/cluster_cli.py --host localhost shards
```

## cURL Command Generation

Both Python and Shell CLIs support `--curl-only` flag to generate cURL commands:

```bash
# Python CLI
jade-db --curl-only --url http://localhost:8080 create-db --name my-db

# Shell CLI
bash jade-db.sh --curl-only --url http://localhost:8080 create-db my-db
```

**Output:**
```bash
curl -X POST http://localhost:8080/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{"name":"my-db","description":"","dimension":128,"index_type":"HNSW"}'
```

This is useful for:
- Learning the JadeVectorDB API
- Debugging API calls
- Creating standalone scripts
- Sharing API examples

[See cURL Commands Guide](curl_commands.md)

## Configuration

### Environment Variables

All CLIs support these environment variables:

```bash
export JADE_DB_URL=http://localhost:8080
export JADE_DB_API_KEY=your-api-key

# Now you can omit --url and --api-key flags
jade-db list-dbs
bash jade-db.sh list-dbs
node bin/jade-db.js database list
```

## Examples

### Complete Workflow

```bash
# 1. Create a database
jade-db --url http://localhost:8080 create-db --name documents --dimension 768

# 2. Store vectors
jade-db --url http://localhost:8080 store \
  --database-id documents \
  --vector-id doc1 \
  --values "[0.1, 0.2, 0.3]" \
  --metadata '{"title":"Example"}'

# 3. Search for similar vectors
jade-db --url http://localhost:8080 search \
  --database-id documents \
  --query-vector "[0.15, 0.25, 0.35]" \
  --top-k 5

# 4. Check system health
jade-db --url http://localhost:8080 health
```

### Cluster Management

```bash
# Check cluster status
python cli/distributed/cluster_cli.py --host master-node status

# Add a new worker node
python cli/distributed/cluster_cli.py --host master-node add-node \
  --node-id worker-003 --address 192.168.1.13:8080

# Monitor shard distribution
python cli/distributed/cluster_cli.py --host master-node shards
```

## Testing

The `cli/tests/` directory contains automated tests for all CLI implementations:

```bash
# Run Python CLI tests
cd cli/tests
pytest test_curl_generation.py -v

# Run Shell CLI tests
bash test_cli_curl.sh
```

[See Testing Documentation →](tests/README.md)

## Directory Structure

```
cli/
├── README.md                 # This file
├── python/                   # Python CLI implementation
│   ├── README.md
│   ├── setup.py
│   ├── jadevectordb/
│   │   ├── cli.py           # CLI commands
│   │   ├── client.py        # API client library
│   │   └── curl_generator.py
│   └── requirements.txt
├── shell/                    # Shell CLI implementation
│   ├── README.md
│   ├── scripts/
│   │   └── jade-db.sh       # Main shell script
│   └── bin/
│       └── jade-db          # Compiled binary
├── js/                       # JavaScript CLI implementation
│   ├── README.md
│   ├── package.json
│   ├── bin/
│   │   └── jade-db.js       # CLI entry point
│   └── src/
│       └── api.js           # API client functions
├── distributed/              # Distributed cluster CLI
│   ├── README.md
│   └── cluster_cli.py       # Cluster management commands
└── tests/                    # CLI test suite
    ├── README.md
    ├── test_curl_generation.py
    └── test_cli_curl.sh
```

## Troubleshooting

### Python CLI: Command Not Found

```bash
# Ensure installation was successful
pip install -e cli/python

# Verify jade-db is in PATH
which jade-db

# Or use python -m
python -m jadevectordb.cli --url http://localhost:8080 list-dbs
```

### Shell CLI: Permission Denied

```bash
# Make script executable
chmod +x cli/shell/scripts/jade-db.sh

# Or use bash explicitly
bash cli/shell/scripts/jade-db.sh [command]
```

### JavaScript CLI: Module Not Found

```bash
# Install dependencies
cd cli/js
npm install

# Verify node_modules exists
ls node_modules/
```

### Connection Errors

```bash
# Verify JadeVectorDB server is running
curl http://localhost:8080/health

# Check firewall rules
telnet localhost 8080

# Use correct URL
jade-db --url http://localhost:8080 health  # ✓ Correct
jade-db --url localhost:8080 health         # ✗ Missing http://
```

## Additional Resources

### Documentation

- [CLI Examples](../examples/cli/README.md) - Usage examples for all CLIs
- [CLI Tutorials](../tutorials/cli/README.md) - Step-by-step tutorials
- [CLI Documentation](../docs/cli-documentation.md) - Comprehensive API reference
- [API Documentation](../docs/api_documentation.md) - REST API reference

### Guides

- [Python CLI Guide](python/README.md)
- [Shell CLI Guide](shell/README.md)
- [JavaScript CLI Guide](js/README.md)
- [Distributed CLI Guide](distributed/README.md)
- [cURL Commands Guide](curl_commands.md)

### Related

- [Main Documentation](../README.md)
- [Deployment Guide](../docs/DOCKER_DEPLOYMENT.md)
- [Distributed Deployment](../docs/distributed_deployment_guide.md)

## Contributing

When adding new CLI features:

1. **Implement across all CLIs** - Maintain feature parity where applicable
2. **Add tests** - Update `cli/tests/` with new test cases
3. **Update documentation** - Update this README and individual CLI READMEs
4. **Add examples** - Update `examples/cli/` with usage examples
5. **Update tutorials** - Add to `tutorials/cli/` if appropriate

## Support

For CLI-related issues:
- Check [Troubleshooting](#troubleshooting) section above
- See individual CLI README files for detailed help
- GitHub Issues: [JadeVectorDB Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
- Documentation: [docs/cli-documentation.md](../docs/cli-documentation.md)
