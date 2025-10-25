# JadeVectorDB CLI Documentation

This document provides comprehensive documentation for all three CLI implementations of JadeVectorDB.

## Overview

JadeVectorDB offers three command-line interface implementations to suit different environments and user preferences:

- **Python CLI**: Full-featured Python-based CLI for data science environments
- **Shell CLI**: Lightweight bash-based CLI for system administration
- **JavaScript CLI**: Node.js-based CLI for web development environments

All implementations provide the same core functionality through a consistent interface.

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

## Common Parameters

All implementations accept these common parameters:

- `--url <url>`: JadeVectorDB API URL (default: http://localhost:8080)
- `--api-key <key>`: API key for authentication (if required)

## Configuration

### Environment Variables

The JavaScript CLI supports these environment variables:
- `JADE_DB_URL` - Sets the default API URL
- `JADE_DB_API_KEY` - Sets the default API key

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

## Getting Help

For help with any CLI implementation:
- Use the `--help` flag for command-specific help
- Check the examples in the `/examples/cli/` directory
- Look at the tutorials in the `/tutorials/cli/` directory
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