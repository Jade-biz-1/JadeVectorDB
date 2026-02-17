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

**Database Management:**
- `create-db` - Create a new database
- `list-dbs` - List all databases
- `get-db` - Get database information
- `delete-db` - Delete a database
- `update-db` - Update a database's name, description, dimension, or index type

**Vector Operations:**
- `store` - Store a vector
- `retrieve` - Retrieve a vector
- `delete` - Delete a vector
- `list-vectors` - List/paginate vectors in a database
- `update-vector` - Update vector values and/or metadata
- `batch-get` - Batch retrieve vectors by IDs
- `search` - Perform similarity search
- `advanced-search` - Advanced filtered search with metadata/value inclusion options
- `hybrid-search` - Hybrid search combining vector and keyword search

**Search & Reranking:**
- `rerank-search` - Search with reranking
- `rerank` - Standalone document reranking

**Index Management:**
- `create-index` - Create an index on a database
- `list-indexes` - List all indexes for a database
- `delete-index` - Delete an index

**Embeddings:**
- `generate-embedding` - Generate vector embeddings from text

**API Key Management:**
- `create-api-key` - Create a new API key
- `list-api-keys` - List API keys (optionally filtered by user)
- `revoke-api-key` - Revoke an API key

**User Management:**
- `user-add <username> --role <role>` - Add a new user
- `user-list` - List all users
- `user-show <user_id>` - Show user details
- `user-update <user_id>` - Update user information
- `user-delete <user_id>` - Delete a user
- `user-activate <user_id>` - Activate a user
- `user-deactivate <user_id>` - Deactivate a user
- `change-password` - Change a user's password

**Security & Audit:**
- `audit-log` - View audit log entries

**Analytics:**
- `analytics-stats` - View analytics statistics for a database

**Bulk Operations:**
- `import` - Import vectors from file (JSON/CSV)
- `export` - Export vectors to file (JSON/CSV)

**Hybrid Search (BM25):**
- `hybrid-build` - Build BM25 index for hybrid search
- `hybrid-status` - Get BM25 index build status
- `hybrid-rebuild` - Rebuild BM25 index from scratch

**System Operations:**
- `health` - Get system health
- `status` - Get system status

**Output Formats:**
All list and query commands support `--format` flag:
- `--format json` (default)
- `--format yaml` (requires PyYAML)
- `--format table` (requires tabulate)
- `--format csv`

#### Advanced Features

**Analytics:**
```bash
# View analytics stats for a database
jade-db analytics-stats --database-id my-db --granularity daily

# View analytics with time range
jade-db analytics-stats --database-id my-db --start-time 2026-01-01T00:00:00Z --end-time 2026-02-01T00:00:00Z
```

**API Key Management:**
```bash
# Create an API key
jade-db create-api-key --user-id user-123 --description "Production key" --validity-days 90

# List API keys for a user
jade-db list-api-keys --user-id user-123

# Revoke an API key
jade-db revoke-api-key --key-id key-456
```

**Index Management:**
```bash
# Create an HNSW index
jade-db create-index --database-id my-db --index-type HNSW --name my-index --parameters '{"M": 16}'

# List indexes
jade-db list-indexes --database-id my-db

# Delete an index
jade-db delete-index --database-id my-db --index-id idx-123
```

**Reranking:**
```bash
# Search with reranking
jade-db rerank-search --database-id my-db --query-text "vector databases" --top-k 5

# Standalone reranking
jade-db rerank --query "best databases" --documents '[{"id":"1","text":"Vector DB guide"},{"id":"2","text":"SQL tutorial"}]'
```

**Embeddings:**
```bash
# Generate embeddings from text
jade-db generate-embedding --text "What is a vector database?"
```

**Audit & Security:**
```bash
# View audit log
jade-db audit-log --limit 50

# View audit log filtered by user
jade-db audit-log --user-id user-123 --event-type login
```

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

**Database Management:**
- `create-db NAME [DESCRIPTION] [DIMENSION] [INDEX_TYPE]` - Create a new database
- `list-dbs` - List all databases
- `get-db ID` - Get database details

**Vector Operations:**
- `store ID VALUES [METADATA]` - Store a vector
- `retrieve ID` - Retrieve a vector
- `delete ID` - Delete a vector
- `search QUERY_VECTOR [TOP_K] [THRESHOLD]` - Search for similar vectors

**User Management:**
- `user-add <email> <role> [password]` - Add a new user
- `user-list [--role <role>] [--status <status>]` - List all users
- `user-show <email>` - Show user details
- `user-update <email> [--role <role>] [--status <status>]` - Update user
- `user-delete <email>` - Delete a user
- `user-activate <email>` - Activate a user
- `user-deactivate <email>` - Deactivate a user

**Bulk Operations:**
- `import <file> <database-id>` - Import vectors from JSON file
- `export <database-id> <output-file>` - Export vectors to JSON file

**System Operations:**
- `health` - Get system health
- `status` - Get system status

**Output Formats:**
All list and query commands support `--format` flag:
- `--format json` (default)
- `--format yaml` (requires yq)
- `--format table` (uses column command)
- `--format csv` (uses jq @csv)

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

**Database Management:**
- `database create` - Create a new database
- `database list` - List all databases
- `database get <id>` - Get database details

**Vector Operations:**
- `vector store` - Store a vector
- `vector retrieve` - Retrieve a vector
- `vector delete` - Delete a vector
- `search` - Perform similarity search

**User Management:**
- `user add <email> <role>` - Add a new user
- `user list` - List all users
- `user show <email>` - Show user details
- `user update <email>` - Update user information
- `user delete <email>` - Delete a user
- `user activate <email>` - Activate a user
- `user deactivate <email>` - Deactivate a user

**System Operations:**
- `health` - Get system health
- `status` - Get system status

**Output Formats:**
All list and query commands support `--format` flag:
- `--format json` (default)
- `--format yaml` (requires js-yaml)
- `--format table` (uses cli-table3)
- `--format csv`

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
- `PUT /v1/databases/{id}` - Update database
- `DELETE /v1/databases/{id}` - Delete database
- `POST /v1/databases/{id}/vectors` - Store vector
- `GET /v1/databases/{id}/vectors` - List vectors
- `GET /v1/databases/{id}/vectors/{vectorId}` - Retrieve vector
- `PUT /v1/databases/{id}/vectors/{vectorId}` - Update vector
- `DELETE /v1/databases/{id}/vectors/{vectorId}` - Delete vector
- `POST /v1/databases/{id}/vectors/batch` - Batch store vectors
- `POST /v1/databases/{id}/vectors/batch-get` - Batch get vectors
- `POST /v1/databases/{id}/search` - Search vectors
- `POST /v1/databases/{id}/search/advanced` - Advanced search
- `POST /v1/databases/{id}/search/hybrid` - Hybrid search
- `POST /v1/databases/{id}/search/rerank` - Rerank search
- `POST /v1/rerank` - Standalone reranking
- `GET /v1/databases/{id}/reranking/config` - Get reranking config
- `PUT /v1/databases/{id}/reranking/config` - Update reranking config
- `POST /v1/databases/{id}/indexes` - Create index
- `GET /v1/databases/{id}/indexes` - List indexes
- `PUT /v1/databases/{id}/indexes/{indexId}` - Update index
- `DELETE /v1/databases/{id}/indexes/{indexId}` - Delete index
- `POST /v1/embeddings/generate` - Generate embeddings
- `POST /v1/api-keys` - Create API key
- `GET /v1/api-keys` - List API keys
- `DELETE /v1/api-keys/{keyId}` - Revoke API key
- `GET /v1/security/audit-log` - Get audit log
- `GET /v1/security/sessions` - Get sessions
- `GET /v1/security/audit-stats` - Get audit stats
- `GET /v1/databases/{id}/analytics/stats` - Analytics stats
- `GET /v1/databases/{id}/analytics/queries` - Analytics queries
- `GET /v1/databases/{id}/analytics/patterns` - Analytics patterns
- `GET /v1/databases/{id}/analytics/insights` - Analytics insights
- `GET /v1/databases/{id}/analytics/trending` - Analytics trending
- `POST /v1/databases/{id}/analytics/feedback` - Submit feedback
- `GET /v1/databases/{id}/analytics/export` - Export analytics
- `PUT /v1/users/{id}/password` - Change password
- `PUT /v1/admin/users/{id}/reset-password` - Admin reset password
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
| `user add <email> --role <role>` | ✅ | ✅ | ✅ | ❌ | ✅ **Implemented** |
| `user list/show/update/delete` | ✅ | ✅ | ✅ | ❌ | ✅ **Implemented** |
| **UI-015: Data Operations** |  |  |  |  |  |
| `search <db> --vector "[...]"` | ✅ | ✅ | ✅ | ❌ | Implemented |
| `import <db> --file <path>` | ✅ | ✅ | ❌ | ❌ | ✅ **Implemented** |
| `export <db> --file <path>` | ✅ | ✅ | ❌ | ❌ | ✅ **Implemented** |
| **UI-016: Output Formats** |  |  |  |  |  |
| JSON output | ✅ | ✅ | ✅ | ✅ | Implemented |
| YAML output | ✅ | ✅ | ✅ | ❌ | ✅ **Implemented** |
| Table output | ✅ | ✅ | ✅ | ✅ | ✅ **Implemented** |
| CSV output | ✅ | ✅ | ✅ | ❌ | ✅ **Implemented** |
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

### Recently Implemented Features (Phase 16 - December 2025)

The following specification requirements were recently implemented to achieve 95%+ CLI specification compliance:

#### 1. User Management CLI (UI-014) ✅

**Status**: ✅ **Fully Implemented** (Python, Shell, JavaScript CLIs)

**Commands Implemented**:
- `user-add <email> <role>` / `user add <email> <role>` - Create new user with role
- `user-list` / `user list` - List all users with optional filtering
- `user-show <email>` / `user show <email>` - Display user details
- `user-update <email>` / `user update <email>` - Update user role/status
- `user-delete <email>` / `user delete <email>` - Delete a user
- `user-activate <email>` / `user activate <email>` - Activate user account
- `user-deactivate <email>` / `user deactivate <email>` - Deactivate user account

**Examples**:
```bash
# Python CLI
jade-db user-add admin@example.com admin --password secret123
jade-db user-list --role developer --format table

# Shell CLI
./jade-db.sh user-add developer@example.com developer
./jade-db.sh user-list --format yaml

# JavaScript CLI
node jade-db.js user add viewer@example.com viewer
node jade-db.js user list --status active
```

**API Integration**: All commands integrate with `/api/v1/users` endpoints with proper authentication and error handling.

#### 2. Bulk Import/Export (UI-015) ✅

**Status**: ✅ **Implemented** (Python CLI - Full, Shell CLI - Basic)

**Python CLI Features**:
- Import from JSON and CSV files with configurable batch sizes
- Export to JSON and CSV files
- Real-time progress tracking with progress bars
- Automatic error handling and retry logic
- Batch processing for memory efficiency
- Detailed import/export statistics

**Shell CLI Features**:
- Import from JSON files using jq parser
- Export to JSON files
- Simple progress indicators
- Basic error counting

**Examples**:
```bash
# Python CLI - Import with progress tracking
jade-db import my-database vectors.json --batch-size 100

# Python CLI - Export to CSV
jade-db export my-database output.csv --format csv

# Shell CLI - Import JSON file
./jade-db.sh import vectors.json my-database

# Shell CLI - Export to file
./jade-db.sh export my-database output.json
```

**Performance**: Can efficiently handle 10,000+ vectors with configurable batch sizes and progress feedback.

#### 3. Multiple Output Formats (UI-016) ✅

**Status**: ✅ **Fully Implemented** (All CLIs)

**Supported Formats**:
- **JSON** (default) - Machine-readable, standard format
- **YAML** - Human-readable, configuration-friendly
- **Table** - Terminal-friendly tabular display
- **CSV** - Data export and spreadsheet integration

**Implementation Details**:

**Python CLI**:
- JSON: Native JSON formatting
- YAML: PyYAML library (optional dependency, graceful fallback)
- Table: tabulate library (optional dependency, graceful fallback)
- CSV: Python csv module (built-in, no dependencies)

**Shell CLI**:
- JSON: jq formatting
- YAML: yq tool (optional, falls back to JSON with warning)
- Table: column command (standard shell tool)
- CSV: jq @csv formatter

**JavaScript CLI**:
- JSON: Native JSON.stringify
- YAML: js-yaml library (installed via npm)
- Table: cli-table3 library (installed via npm)
- CSV: Custom formatter with proper escaping

**Examples**:
```bash
# List databases in different formats
jade-db list-dbs --format json    # Default
jade-db list-dbs --format yaml    # YAML output
jade-db list-dbs --format table   # Pretty table
jade-db list-dbs --format csv     # CSV for spreadsheets

# User management with table format
jade-db user-list --format table

# Health check in YAML
jade-db health --format yaml
```

**Graceful Degradation**: All formatters include fallback mechanisms when optional dependencies are not installed, with helpful installation messages.

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

**Overall Compliance**: ✅ **95%+** of CLI specification requirements implemented

**Previous Compliance**: ~75% (before Phase 16 - December 2025)
**Current Compliance**: 95%+ (after Phase 16 implementation)

**Fully Implemented**:
- ✅ All core database operations (create, list, get, delete)
- ✅ All core vector operations (store, retrieve, delete, search)
- ✅ System health and monitoring
- ✅ Distributed cluster management
- ✅ Multiple CLI language implementations
- ✅ Environment variable configuration
- ✅ cURL generation for API learning
- ✅ **User management commands** (UI-014) - *NEW*
- ✅ **Bulk import/export functionality** (UI-015) - *NEW*
- ✅ **Multiple output formats** (UI-016) - JSON, YAML, Table, CSV - *NEW*

**Remaining Gaps**:
- JavaScript CLI bulk import/export (low priority - Python and Shell CLIs provide full coverage)
- Advanced import features (resume capability, Parquet format support)

**Phase 16 Achievements** (December 2025):
1. ✅ Implemented complete user management across all three CLIs (Python, Shell, JavaScript)
2. ✅ Added bulk import/export with progress tracking (Python CLI) and basic support (Shell CLI)
3. ✅ Implemented multiple output formats (JSON, YAML, Table, CSV) across all CLIs
4. ✅ Achieved 95%+ specification compliance, up from 75%

### Design Rationale

The current CLI implementation prioritizes:

1. **Practical Usability**: Focus on most common operations (database and vector management)
2. **Multiple Audiences**: Different CLI implementations for different user personas
3. **Production Readiness**: Distributed CLI for operational management, now with user management
4. **Developer Experience**: cURL generation and environment variable support
5. **Data Operations**: Bulk import/export for production data workflows
6. **Flexibility**: Multiple output formats for integration with various tools

**Evolution from Initial Implementation**:

The Phase 16 enhancements (December 2025) addressed previously deprioritized features based on user feedback and production requirements:

- **User Management**: Now implemented across all CLIs to support administrative workflows without requiring direct API access
- **Bulk Import/Export**: Implemented with progress tracking and batch processing to support production data migration scenarios
- **Multiple Output Formats**: Added YAML, Table, and CSV formats to improve integration with automation tools and data pipelines

These additions bring CLI specification compliance from 75% to 95%+, covering all major use cases for vector database management.

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