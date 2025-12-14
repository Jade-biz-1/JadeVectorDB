# JadeVectorDB CLI Documentation

This document provides comprehensive documentation for all CLI implementations of JadeVectorDB.

## Overview

JadeVectorDB offers four command-line interface implementations to suit different environments and user preferences:

- **Python CLI**: Full-featured Python-based CLI for data science environments
- **Shell CLI**: Lightweight bash-based CLI for system administration
- **JavaScript CLI**: Node.js-based CLI for web development environments
- **Distributed CLI**: Specialized CLI for cluster management and distributed operations

The first three CLIs provide the same core database and vector functionality through a consistent interface. The Distributed CLI focuses exclusively on cluster management operations.

## Common Functionality

All CLI implementations support these operations:

### Database Management
- Create databases with specific configurations
- List all databases
- Get database details
- Delete databases

### Vector Operations
- Store vectors with optional metadata
- Retrieve vectors by ID
- Delete vectors
- Update vectors (where supported)

### Search Operations
- Similarity search with configurable parameters
- Search with metadata filters
- Threshold-based result filtering

### System Operations
- Health checks
- Status information
- Resource monitoring

## Implementation-Specific Details

### Python CLI (`/cli/python/`)

#### Installation
```bash
pip install -e cli/python
```

#### Usage
```bash
jade-db --url http://localhost:8080 --api-key mykey123 [command] [options]
```

#### Features
- Full API coverage
- Batch operations support in the client library
- Integration with Python data science workflows
- Extensive error handling and validation

#### Commands
- `create-db` - Create a new database
- `list-dbs` - List all databases
- `store` - Store a vector
- `retrieve` - Retrieve a vector
- `search` - Perform similarity search
- `health` - Get system health
- `status` - Get system status

### Shell CLI (`/cli/shell/`)

#### Usage
```bash
bash cli/shell/scripts/jade-db.sh --url http://localhost:8080 --api-key mykey123 [command] [arguments]
```

#### Features
- Lightweight implementation
- No external dependencies beyond standard shell tools
- Direct HTTP requests using curl
- Simple and fast for basic operations

#### Commands
- `create-db NAME [DESCRIPTION] [DIMENSION] [INDEX_TYPE]` - Create a new database
- `list-dbs` - List all databases
- `get-db ID` - Get database details
- `store ID VALUES [METADATA]` - Store a vector
- `retrieve ID` - Retrieve a vector
- `delete ID` - Delete a vector
- `search QUERY_VECTOR [TOP_K] [THRESHOLD]` - Search for similar vectors
- `health` - Get system health
- `status` - Get system status

### JavaScript CLI (`/cli/js/`)

#### Installation
```bash
cd cli/js
npm install
```

#### Usage
```bash
node cli/js/bin/jade-db.js --url http://localhost:8080 --api-key mykey123 [command] [options]
```

#### Features
- Modern command-line interface using commander.js
- Comprehensive error handling and validation
- Consistent with Node.js ecosystem tools
- Extensible architecture for additional features

#### Commands
- `database create` - Create a new database
- `database list` - List all databases  
- `database get <id>` - Get database details
- `vector store` - Store a vector
- `vector retrieve` - Retrieve a vector
- `vector delete` - Delete a vector
- `search` - Perform similarity search
- `health` - Get system health
- `status` - Get system status

### Distributed CLI (`/cli/distributed/`)

#### Installation
```bash
pip install requests
```

#### Usage
```bash
python cli/distributed/cluster_cli.py --host <master-host> --port <port> [command] [options]
```

#### Features
- Cluster-wide status monitoring
- Node management (add, remove, health checks)
- Shard operations and migrations
- Multiple output formats (JSON, table, compact)
- Diagnostics and troubleshooting tools

#### Commands
- `status` - Get cluster status
- `diagnostics` - Comprehensive cluster diagnostics
- `metrics` - Performance and resource metrics
- `nodes` - List all cluster nodes
- `add-node` - Add a new node to cluster
- `remove-node` - Remove a node from cluster
- `node-health` - Check specific node health
- `shards` - List all shards
- `shard-status` - Get shard status
- `migrate-shard` - Migrate shard between nodes

#### Output Formats
The Distributed CLI supports multiple output formats via `--format`:
- `json` - Machine-readable JSON output
- `table` - Human-readable formatted tables (default)
- `compact` - Condensed single-line output

**Example**:
```bash
python cluster_cli.py --host localhost --port 8080 status --format json
```

## Common Parameters

All implementations accept these common parameters:

- `--url <url>`: JadeVectorDB API URL (default: http://localhost:8080)
- `--api-key <key>`: API key for authentication (if required)

## Configuration

### Environment Variables

All CLIs support these environment variables:
- `JADEVECTORDB_URL` - Sets the default API URL (default: http://localhost:8080)
- `JADEVECTORDB_API_KEY` - Sets the default API key

**Shell CLI only**:
- `JADEVECTORDB_DATABASE_ID` - Sets the default database ID for vector operations

**Example usage**:
```bash
export JADEVECTORDB_URL="http://localhost:8080"
export JADEVECTORDB_API_KEY="your-api-key"
jade-db list-dbs  # No need to specify --url or --api-key
```

### API Endpoints

All CLIs interact with these API endpoints:

- `POST /v1/databases` - Create database
- `GET /v1/databases` - List databases
- `GET /v1/databases/{id}` - Get database
- `POST /v1/databases/{id}/vectors` - Store vector
- `GET /v1/databases/{id}/vectors/{vectorId}` - Retrieve vector
- `DELETE /v1/databases/{id}/vectors/{vectorId}` - Delete vector
- `POST /v1/databases/{id}/search` - Search vectors
- `GET /health` - Health check
- `GET /status` - System status

## Error Handling

All CLIs provide detailed error messages that include:
- Specific error description
- HTTP status code
- Potential solutions

Common error codes and meanings:
- 400: Bad request (check parameters)
- 401: Unauthorized (check API key)
- 404: Resource not found (check IDs)
- 500: Server error (check server logs)

## Performance Considerations

- For bulk operations, consider using the client libraries directly rather than the CLI
- Use appropriate index types for your data size and query patterns
- Monitor resource usage during large operations
- Use connection pooling for repeated operations

## Security Best Practices

- Rotate API keys regularly
- Use HTTPS in production environments
- Store API keys securely (environment variables, not in scripts)
- Implement rate limiting for client access
- Validate input data before sending to the server

## Development

To contribute to any CLI implementation:

1. Make changes in the appropriate `/cli/{python|shell|js}/` directory
2. Test changes against a local JadeVectorDB instance
3. Update examples and documentation as needed
4. Ensure all implementations provide consistent functionality

## Tutorial Examples

JadeVectorDB provides ready-to-run executable tutorial scripts in `/tutorials/cli/examples/`:

### Quick Start (quick-start.sh)
- **Duration**: 2 minutes
- **Level**: Beginner
- Covers: Database creation, vector storage, retrieval, and search
- **Run**: `cd tutorial/cli/examples && ./quick-start.sh`

### Batch Import (batch-import.py)
- **Duration**: 5 minutes
- **Level**: Intermediate
- Covers: High-performance batch operations (1,000 vectors)
- **Run**: `cd tutorial/cli/examples && ./batch-import.py`

### Workflow Demo (workflow-demo.sh)
- **Duration**: 5-10 minutes
- **Level**: Intermediate
- Covers: Multi-database management, cross-database operations
- **Run**: `cd tutorial/cli/examples && ./workflow-demo.sh`

### Product Search Demo (product-search-demo.sh)
- **Duration**: 10 minutes
- **Level**: Advanced
- Covers: Real-world e-commerce recommendation system
- **Run**: `cd tutorials/cli/examples && ./product-search-demo.sh`

See `/tutorials/cli/examples/README.md` for detailed information on each example.

## Specification Compliance

This section documents how the current CLI implementations align with the original specification requirements defined in `specs/002-check-if-we/spec.md`.

### Feature Coverage Matrix

| Requirement | Python CLI | Shell CLI | JavaScript CLI | Distributed CLI | Status |
|-------------|-----------|-----------|----------------|----------------|--------|
| **UI-014: Administrative Operations** |  |  |  |  |  |
| `cluster status` | ❌ | ❌ | ❌ | ✅ | Implemented |
| `database create --name <db> --dimension <dim>` | ✅ | ✅ | ✅ | ❌ | Implemented |
| `user add <email> --role <role>` | ❌ | ❌ | ❌ | ❌ | **Missing** |
| **UI-015: Data Operations** |  |  |  |  |  |
| `search <db> --vector "[...]"` | ✅ | ✅ | ✅ | ❌ | Implemented |
| `import <db> --file <path>` | ❌ | ❌ | ❌ | ❌ | **Missing** |
| **UI-016: Output Formats** |  |  |  |  |  |
| JSON output | ✅ | ✅ | ✅ | ✅ | Implemented |
| YAML output | ❌ | ❌ | ❌ | ❌ | **Missing** |
| Table output | ❌ | ❌ | ❌ | ✅ | Partial |
| **Core Database Operations** |  |  |  |  |  |
| Database creation | ✅ | ✅ | ✅ | ❌ | Implemented |
| Database listing | ✅ | ✅ | ✅ | ❌ | Implemented |
| Database details | ✅ | ✅ | ✅ | ❌ | Implemented |
| **Core Vector Operations** |  |  |  |  |  |
| Vector storage | ✅ | ✅ | ✅ | ❌ | Implemented |
| Vector retrieval | ✅ | ✅ | ✅ | ❌ | Implemented |
| Vector deletion | ✅ | ✅ | ✅ | ❌ | Implemented |
| Similarity search | ✅ | ✅ | ✅ | ❌ | Implemented |
| **System Operations** |  |  |  |  |  |
| Health check | ✅ | ✅ | ✅ | ❌ | Implemented |
| Status check | ✅ | ✅ | ✅ | ❌ | Implemented |
| **Cluster Management** |  |  |  |  |  |
| Node management | ❌ | ❌ | ❌ | ✅ | Implemented |
| Shard operations | ❌ | ❌ | ❌ | ✅ | Implemented |
| Cluster diagnostics | ❌ | ❌ | ❌ | ✅ | Implemented |

### Missing Features

The following specification requirements are currently not implemented:

#### 1. User Management CLI (UI-014)
**Requirement**: `jade-db user add <email> --role <role>`

**Status**: Not implemented

**Impact**: Administrators cannot manage users through CLI. User management must be performed through:
- Direct API calls
- Web UI (when available)
- Backend database administration

**Workaround**:
```bash
# Using curl to add a user
curl -X POST http://localhost:8080/api/v1/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "email": "user@example.com",
    "role": "developer"
  }'
```

**Future Implementation**: Planned for CLI version 2.0. Will include:
- `jade-db user add <email> --role <role>`
- `jade-db user list`
- `jade-db user update <email> --role <new-role>`
- `jade-db user delete <email>`
- `jade-db user show <email>`

#### 2. Bulk Data Import (UI-015)
**Requirement**: `jade-db import <db_name> --file <path_to_data>`

**Status**: Not implemented

**Impact**: Users cannot perform bulk imports through CLI. Large datasets must be:
- Imported programmatically using the Python client library
- Uploaded through individual API calls
- Processed using custom scripts

**Workaround**:
```python
# Using the Python client library for batch import
from jadevectordb import JadeVectorDBClient
import json

client = JadeVectorDBClient(url="http://localhost:8080", api_key="your-key")

# Load vectors from file
with open('vectors.json', 'r') as f:
    vectors = json.load(f)

# Batch insert
for batch in chunk_list(vectors, batch_size=100):
    client.batch_insert(database_id="my_db", vectors=batch)
```

**Future Implementation**: Planned for CLI version 2.0. Will include:
- `jade-db import <db> --file <path>` - Import from JSON/CSV
- `jade-db export <db> --file <path>` - Export to JSON/CSV
- `jade-db import <db> --format <csv|json|parquet>` - Multiple format support
- Progress indicators for large imports
- Resume capability for interrupted imports

#### 3. Multiple Output Formats (UI-016)
**Requirement**: CLI MUST support multiple output formats (JSON, YAML, table)

**Status**: Partially implemented
- **Distributed CLI**: ✅ Supports JSON, table, and compact formats
- **Python CLI**: ❌ JSON only
- **Shell CLI**: ❌ JSON only
- **JavaScript CLI**: ❌ JSON only

**Impact**: Limited integration with automation tools that require YAML or tabular output

**Workaround**:
```bash
# Convert JSON to YAML using yq
jade-db list-dbs --format json | yq eval -P

# Convert JSON to table using jq
jade-db list-dbs --format json | jq -r '.[] | [.id, .name, .dimension] | @tsv'
```

**Future Implementation**: Planned for all CLIs. Will add:
- `--format json` (default)
- `--format yaml`
- `--format table`
- `--format csv` (for data operations)

### Implemented Features Beyond Specifications

The CLI implementations include several features not explicitly required by the original specification:

#### 1. cURL Command Generation (Python & Shell CLIs)
**Feature**: `--curl-only` flag generates equivalent cURL commands

**Benefits**:
- API learning and documentation
- Debugging API calls
- Creating standalone scripts
- Understanding REST API structure

**Example**:
```bash
# Generate cURL command instead of executing
jade-db --curl-only create-db --name test --dimension 768

# Output:
# curl -X POST http://localhost:8080/api/v1/databases \
#   -H "Content-Type: application/json" \
#   -d '{"name":"test","dimension":768,"index_type":"HNSW"}'
```

#### 2. Distributed Cluster Management CLI
**Feature**: Specialized CLI for cluster operations

**Capabilities**:
- Real-time cluster monitoring
- Node health diagnostics
- Shard migration management
- Multiple output formats
- Performance metrics

**Note**: This was implemented to meet the distributed architecture requirements (spec FR-004, FR-005) but exceeds the original CLI specification scope.

#### 3. Environment Variable Configuration
**Feature**: All CLIs support environment-based configuration

**Variables**:
- `JADEVECTORDB_URL` - Default API URL
- `JADEVECTORDB_API_KEY` - Default API key
- `JADEVECTORDB_DATABASE_ID` - Default database (Shell CLI only)

**Benefits**:
- Simplified command-line usage
- Secure credential storage
- CI/CD integration
- Multi-environment workflows

### Compliance Summary

**Overall Compliance**: ~75% of CLI specification requirements implemented

**Strengths**:
- ✅ All core database operations (create, list, get, delete)
- ✅ All core vector operations (store, retrieve, delete, search)
- ✅ System health and monitoring
- ✅ Distributed cluster management
- ✅ Multiple CLI language implementations
- ✅ Environment variable configuration
- ✅ cURL generation for API learning

**Gaps**:
- ❌ User management commands
- ❌ Bulk import/export functionality
- ❌ Multiple output formats (except Distributed CLI)
- ❌ YAML output support

**Recommended Actions**:
1. **Priority 1**: Implement bulk import/export commands for production data workflows
2. **Priority 2**: Add multiple output format support to Python/Shell/JS CLIs
3. **Priority 3**: Implement user management commands for administrative operations
4. **Long-term**: Consider consolidating CLI implementations or providing consistent feature parity

### Design Rationale

The current CLI implementation prioritizes:

1. **Practical Usability**: Focus on most common operations (database and vector management)
2. **Multiple Audiences**: Different CLI implementations for different user personas
3. **Production Readiness**: Distributed CLI for operational management
4. **Developer Experience**: cURL generation and environment variable support

Administrative features like user management were deprioritized because:
- Initial deployments use default users (admin, dev, test) per spec FR-029
- User management is typically performed during setup, not ongoing operations
- API and Web UI provide alternative access methods for user management

Bulk import was deprioritized because:
- Python client library provides robust batch operations
- Initial use cases focused on real-time vector insertion
- Complex import scenarios often require custom data transformation logic

## Getting Help

For help with any CLI implementation:
- Use the `--help` flag for command-specific help
- Check the examples in the `/examples/cli/` directory
- Try the tutorial scripts in `/tutorials/cli/examples/` directory
- Review the tutorials in `/tutorials/cli/` directory (basics.md, advanced.md)
- Refer to the API documentation for endpoint details

## Troubleshooting

Common issues and solutions:

### Connection Issues
- Verify JadeVectorDB server is running
- Check network connectivity to the server
- Confirm the correct port is being used

### Authentication Issues
- Verify API key is correct
- Check that API key has required permissions
- Ensure authentication is configured correctly on the server

### Data Format Issues
- Validate JSON format for vectors and metadata
- Confirm vector dimensions match database configuration
- Check that IDs follow naming requirements