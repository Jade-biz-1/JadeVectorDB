#!/bin/bash

# Generate API Documentation from OpenAPI Specification
# This script generates HTML documentation from the OpenAPI JSON specification

set -e

PROJECT_ROOT="/home/deepak/Public/JadeVectorDB"
API_SPEC="${PROJECT_ROOT}/docs/api/openapi.json"
HTML_OUTPUT="${PROJECT_ROOT}/docs/api/index.html"
DOCS_DIR="${PROJECT_ROOT}/docs/api"

echo "Generating API documentation from OpenAPI specification..."

# Create docs directory if it doesn't exist
mkdir -p "${DOCS_DIR}"

# Check if required tools are available
if ! command -v redoc-cli &> /dev/null; then
    echo "Installing Redoc CLI for documentation generation..."
    npm install -g redoc-cli
fi

# Generate HTML documentation using Redoc
echo "Generating HTML documentation..."
redoc-cli bundle "${API_SPEC}" -o "${HTML_OUTPUT}"

echo "API documentation generated successfully!"
echo "Documentation available at: ${HTML_OUTPUT}"

# Also generate a simple markdown version for quick reference
echo "Generating markdown reference..."

# Extract key endpoints and their descriptions
cat > "${DOCS_DIR}/api_reference.md" << EOF
# JadeVectorDB API Reference

**Version**: 1.0.0
**Base URL**: https://api.jadevectordb.com/v1

## Authentication

All API requests require authentication using an API key in the \`X-API-Key\` header.

\`\`\`
X-API-Key: YOUR_API_KEY
\`\`\`

## Core Endpoints

### Database Management

- \`GET /databases\` - List all databases
- \`POST /databases\` - Create a new database
- \`GET /databases/{databaseId}\` - Get database details
- \`PUT /databases/{databaseId}\` - Update database configuration
- \`DELETE /databases/{databaseId}\` - Delete a database
- \`GET /databases/{databaseId}/status\` - Get database status

### Vector Operations

- \`POST /databases/{databaseId}/vectors\` - Store a vector
- \`POST /databases/{databaseId}/vectors/batch\` - Store multiple vectors
- \`GET /databases/{databaseId}/vectors/{vectorId}\` - Retrieve a vector
- \`PUT /databases/{databaseId}/vectors/{vectorId}\` - Update a vector
- \`DELETE /databases/{databaseId}/vectors/{vectorId}\` - Delete a vector

### Search Operations

- \`POST /databases/{databaseId}/search\` - Perform similarity search
- \`POST /databases/{databaseId}/search/advanced\` - Perform advanced search with filters

### Index Management

- \`GET /databases/{databaseId}/indexes\` - List all indexes
- \`POST /databases/{databaseId}/indexes\` - Create a new index
- \`GET /databases/{databaseId}/indexes/{indexId}\` - Get index details
- \`PUT /databases/{databaseId}/indexes/{indexId}\` - Update an index
- \`DELETE /databases/{databaseId}/indexes/{indexId}\` - Delete an index

### Embedding Generation

- \`POST /embeddings/generate\` - Generate vector embedding from input data

### Monitoring

- \`GET /health\` - System health check
- \`GET /status\` - Detailed system status

## Data Models

### Vector

\`\`\`json
{
  "id": "string",
  "values": [0.1, 0.2, 0.3],
  "metadata": {
    "key": "value"
  }
}
\`\`\`

### Database

\`\`\`json
{
  "databaseId": "string",
  "name": "string",
  "description": "string",
  "vectorDimension": 0,
  "indexType": "HNSW",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
\`\`\`

## Error Responses

All error responses follow this format:

\`\`\`json
{
  "error": "ERROR_CODE",
  "message": "Human readable error message",
  "details": {
    "additional": "information"
  }
}
\`\`\`

Common error codes:
- \`INVALID_ARGUMENT\` - Invalid request parameters
- \`NOT_FOUND\` - Resource not found
- \`UNAUTHORIZED\` - Missing or invalid API key
- \`FORBIDDEN\` - Insufficient permissions
- \`CONFLICT\` - Resource already exists
- \`INTERNAL_ERROR\` - Internal server error

## Rate Limits

The API implements rate limiting to prevent abuse. Exceeding rate limits will result in a 429 (Too Many Requests) response.

## Response Codes

- \`200\` - Success
- \`201\` - Created
- \`204\` - No Content
- \`400\` - Bad Request
- \`401\` - Unauthorized
- \`403\` - Forbidden
- \`404\` - Not Found
- \`409\` - Conflict
- \`429\` - Too Many Requests
- \`500\` - Internal Server Error

EOF

echo "Markdown reference generated at: ${DOCS_DIR}/api_reference.md"
echo "API documentation enhancement completed!"