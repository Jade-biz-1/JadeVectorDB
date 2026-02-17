M# JadeVectorDB User Guide

## Quick Start

### Logging In

JadeVectorDB provides default test users for development and testing environments. Use these credentials to get started immediately:

#### Default User Credentials

| Username | Password | User ID | Access Level | Use Case |
|----------|----------|---------|--------------|----------|
| `admin` | `admin123` | user_admin_default | Administrator | Full system access, user management, database administration |
| `dev` | `dev123` | user_dev_default | Developer | API development, database operations, vector management |
| `test` | `test123` | user_test_default | Tester | Testing workflows, limited permissions |

**Note**: These users are automatically created only in development, test, and local environments. They are NOT created in production for security reasons.

### Logging In via API

```bash
# Login as admin
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": "user_admin_default",
  "username": "admin",
  "roles": ["admin", "developer", "user"]
}
```

**Save your token** for subsequent API calls:
```bash
export TOKEN="<your-token-value>"
```

### Logging In via Web UI

1. Navigate to **http://localhost:3000** in your web browser
2. Click **"Login"**
3. Enter credentials:
   - Username: `admin`
   - Password: `admin123`
4. Click **"Sign In"**

You'll be redirected to the dashboard with full access to:
- Database Management
- Vector Operations
- Search Interface
- User Management (admin only)
- API Key Management

## Working with Databases

### Creating a Database

**Via API**:
```bash
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "product_embeddings",
    "dimension": 384,
    "metric": "cosine",
    "description": "Product recommendation embeddings"
  }'
```

**Via Web UI**:
1. Click "Databases" in the navigation menu
2. Click "Create New Database"
3. Fill in the form:
   - Name: `product_embeddings`
   - Dimension: `384`
   - Metric: `cosine`
4. Click "Create"

### Listing Databases

**Via API**:
```bash
curl -X GET http://localhost:8080/v1/databases \
  -H "Authorization: Bearer $TOKEN"
```

**Via Web UI**:
- Navigate to "Databases" page
- All your databases will be displayed with their configurations

## Working with Vectors

### Storing Vectors

**Single Vector**:
```bash
curl -X POST http://localhost:8080/v1/databases/product_embeddings/vectors \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "id": "product_123",
    "values": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "product_name": "Blue Widget",
      "category": "widgets",
      "price": 29.99
    }
  }'
```

**Batch Upload**:
```bash
curl -X POST http://localhost:8080/v1/databases/product_embeddings/vectors/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "vectors": [
      {
        "id": "product_123",
        "values": [0.1, 0.2, 0.3, ...],
        "metadata": {"product_name": "Blue Widget"}
      },
      {
        "id": "product_124",
        "values": [0.4, 0.5, 0.6, ...],
        "metadata": {"product_name": "Red Widget"}
      }
    ]
  }'
```

### Searching Vectors

**Similarity Search**:
```bash
curl -X POST http://localhost:8080/v1/databases/product_embeddings/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": [0.15, 0.25, 0.35, ...],
    "top_k": 10,
    "include_vector_data": false,
    "include_metadata": true
  }'
```

**Search with Metadata Filtering**:
```bash
curl -X POST http://localhost:8080/v1/databases/product_embeddings/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": [0.15, 0.25, 0.35, ...],
    "top_k": 10,
    "filter_category": "widgets",
    "include_metadata": true
  }'
```

## Advanced Search Features

### Hybrid Search

Hybrid search combines vector similarity with keyword-based BM25 search for improved accuracy, especially for exact matches like product codes or model numbers.

**Building BM25 Index**:
```bash
# First, build the BM25 index for your database
curl -X POST http://localhost:8080/v1/databases/product_embeddings/search/bm25/build \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "text_field": "product_name",
    "incremental": false
  }'
```

**Hybrid Search Query**:
```bash
curl -X POST http://localhost:8080/v1/databases/product_embeddings/search/hybrid \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query_text": "blue widget model BW-2024",
    "query_vector": [0.15, 0.25, 0.35, ...],
    "top_k": 10,
    "fusion_method": "rrf",
    "alpha": 0.7
  }'
```

**Response**:
```json
{
  "results": [
    {
      "id": "product_123",
      "vector_score": 0.92,
      "bm25_score": 0.85,
      "hybrid_score": 0.89,
      "metadata": {
        "product_name": "Blue Widget Model BW-2024",
        "category": "widgets"
      }
    }
  ]
}
```

**Fusion Methods**:
- **`rrf`** (Reciprocal Rank Fusion): Rank-based fusion, balanced results
- **`linear`**: Weighted combination, use `alpha` parameter (0-1) to control vector vs. keyword weight

**Use Cases**:
- Product search with exact model numbers
- Technical documentation search
- E-commerce search with product codes
- RAG systems requiring keyword precision

### Re-ranking with Cross-Encoders

Re-ranking improves search precision by re-scoring top candidates using cross-encoder models. This typically improves precision@5 by 15-25% over hybrid search alone.

#### How Re-ranking Works

Re-ranking uses a **two-stage retrieval pattern**:

1. **Stage 1: Fast Retrieval** - Retrieve many candidates (e.g., top 100) using hybrid search
2. **Stage 2: Precise Re-ranking** - Re-score candidates using a cross-encoder model
3. **Stage 3: Return Top-K** - Return the most relevant results (e.g., top 10)

This approach balances speed and accuracy: fast retrieval for candidate generation, precise but slower cross-encoder for final ranking.

#### Basic Re-ranking Example

**Hybrid Search with Re-ranking**:
```bash
curl -X POST http://localhost:8080/v1/databases/documents/search/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "queryText": "How do I implement user authentication?",
    "queryVector": [0.15, 0.25, 0.35, ...],
    "topK": 10,
    "enableReranking": true,
    "rerankTopN": 100,
    "fusionMethod": "rrf",
    "alpha": 0.7
  }'
```

**Response**:
```json
{
  "results": [
    {
      "docId": "doc_045",
      "vectorScore": 0.78,
      "bm25Score": 0.65,
      "hybridScore": 0.72,
      "rerankScore": 0.94,
      "combinedScore": 0.87,
      "metadata": {
        "title": "User Authentication Guide",
        "source": "A comprehensive guide to implementing secure user authentication..."
      }
    }
  ],
  "timings": {
    "retrievalMs": 15,
    "rerankingMs": 185,
    "totalMs": 200
  }
}
```

**Parameters Explained**:
- **`queryText`**: The text query for BM25 keyword search
- **`queryVector`**: The embedding vector for semantic search
- **`topK`**: Number of final results to return (e.g., 10)
- **`enableReranking`**: Set to `true` to enable cross-encoder re-ranking
- **`rerankTopN`**: Number of candidates to retrieve and re-rank (e.g., 100)
- **`fusionMethod`**: How to combine vector and BM25 scores (`rrf` or `linear`)

#### Standalone Re-ranking

Re-rank any list of documents without needing a database:

```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": "What is vector search?",
    "documents": [
      {"id": "doc1", "text": "Vector search uses embeddings to find similar items"},
      {"id": "doc2", "text": "Traditional keyword search matches exact terms"},
      {"id": "doc3", "text": "Vector databases store and index embeddings"}
    ],
    "topK": 3
  }'
```

**Response**:
```json
{
  "results": [
    {"id": "doc1", "score": 0.95, "rank": 1},
    {"id": "doc3", "score": 0.82, "rank": 2},
    {"id": "doc2", "score": 0.41, "rank": 3}
  ],
  "latencyMs": 145
}
```

**Use Cases for Standalone Re-ranking**:
- Re-ranking results from external search systems
- A/B testing different re-ranking models
- Post-processing search results from multiple databases
- Research and experimentation

#### Configuration Management

**Get Current Configuration**:
```bash
curl -X GET http://localhost:8080/v1/databases/documents/reranking/config \
  -H "Authorization: Bearer $TOKEN"
```

**Response**:
```json
{
  "modelName": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "batchSize": 32,
  "scoreThreshold": 0.0,
  "combineScores": true,
  "rerankWeight": 0.7,
  "statistics": {
    "totalRequests": 1523,
    "failedRequests": 3,
    "avgLatencyMs": 175.4,
    "totalDocumentsReranked": 152300
  }
}
```

**Update Configuration**:
```bash
curl -X PUT http://localhost:8080/v1/databases/documents/reranking/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "modelName": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "batchSize": 16,
    "rerankWeight": 0.8
  }'
```

**Configuration Parameters**:
- **`modelName`**: Cross-encoder model to use
- **`batchSize`**: Batch size for inference (lower for less memory, higher for throughput)
- **`scoreThreshold`**: Minimum score to include in results (0.0 = no filtering)
- **`combineScores`**: Whether to combine rerank score with original hybrid score
- **`rerankWeight`**: Weight for rerank score vs. original (0.7 = 70% rerank, 30% hybrid)

#### Available Re-ranking Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Good | Default, production use |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | Better | Higher precision needs |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Very Fast | Basic | High-throughput applications |
| Custom fine-tuned models | Varies | Domain-specific | Specialized domains |

**Model Selection Guidance**:
- Start with **L-6-v2** (default) for balanced speed/quality
- Upgrade to **L-12-v2** if you need higher precision and can tolerate 2x latency
- Use **TinyBERT-L-2-v2** for high-throughput scenarios (>100 requests/sec)
- Fine-tune custom models for specialized domains (legal, medical, finance)

#### Complete Workflow Example: RAG System

Here's a complete example showing hybrid search with re-ranking for a RAG (Retrieval-Augmented Generation) system:

```bash
# Step 1: Build BM25 index (one-time setup)
curl -X POST http://localhost:8080/v1/databases/knowledge_base/search/bm25/build \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "textField": "content",
    "incremental": false
  }'

# Step 2: Configure re-ranking (one-time setup)
curl -X PUT http://localhost:8080/v1/databases/knowledge_base/reranking/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "modelName": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "batchSize": 32,
    "rerankWeight": 0.75
  }'

# Step 3: Perform search with re-ranking
USER_QUERY="How do neural networks learn from data?"
QUERY_EMBEDDING=$(generate_embedding "$USER_QUERY")  # Your embedding service

curl -X POST http://localhost:8080/v1/databases/knowledge_base/search/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"queryText\": \"$USER_QUERY\",
    \"queryVector\": $QUERY_EMBEDDING,
    \"topK\": 5,
    \"enableReranking\": true,
    \"rerankTopN\": 50,
    \"fusionMethod\": \"rrf\"
  }"

# Step 4: Use top results as context for LLM
# The top 5 results can now be used as context for your LLM prompt
```

#### When to Use Re-ranking

**Ideal Use Cases**:
- ✅ RAG (Retrieval-Augmented Generation) systems
- ✅ Question answering over documents
- ✅ High-precision semantic search
- ✅ Document retrieval for legal/medical applications
- ✅ Content recommendation with quality requirements

**Not Recommended**:
- ❌ Real-time search with <50ms latency requirements
- ❌ Simple keyword matching (use BM25 alone)
- ❌ Exact match queries (use SQL/NoSQL)
- ❌ Very large result sets (>1000 documents to re-rank)

#### Performance Considerations

**Latency Breakdown**:
- Hybrid search (retrieve 100 docs): ~15-30ms
- Re-ranking 100 docs (CPU): ~150-250ms
- Re-ranking 100 docs (GPU): ~50-100ms
- **Total**: ~200-300ms on CPU, ~100-150ms on GPU

**Optimization Tips**:
1. **Adjust `rerankTopN`**: Start with 50-100 candidates, adjust based on quality metrics
2. **Use GPU**: Deploy with CUDA support for 2-3x speedup
3. **Batch efficiently**: Set `batchSize` to 32-64 for optimal throughput
4. **Monitor statistics**: Check `/reranking/config` endpoint for performance metrics
5. **Consider caching**: Cache re-ranking results for repeated queries

**Scalability**:
- Single instance: ~10-20 requests/sec (CPU), ~30-50 requests/sec (GPU)
- For higher throughput: Use dedicated re-ranking service (see Architecture docs)
- Kubernetes deployment: Scale re-ranking pods independently

#### Best Practices

1. **Two-Stage Retrieval**: Always retrieve more candidates than you need
   - Example: Retrieve 100, return top 10
   - Rule of thumb: `rerankTopN` = 5-10x `topK`

2. **Combine with Hybrid Search**: Re-ranking works best with diverse candidate sets
   - Use RRF fusion to get both semantic and keyword matches
   - Re-ranking will elevate the truly relevant results

3. **Monitor and Iterate**:
   - Track re-ranking latency via `/config` endpoint
   - Measure precision improvements (target: +15-25% precision@5)
   - A/B test different models and weights

4. **Handle Documents Without Text**:
   - Ensure your vectors have text in metadata for re-ranking
   - For documents without text, re-ranking will use document IDs (suboptimal)

5. **Production Deployment**:
   - Pre-download models during deployment (avoid first-request delays)
   - Set appropriate timeouts (recommend 5-10 seconds)
   - Monitor Python subprocess health via logs
   - Consider dedicated re-ranking service for production scale

## User Management (Admin Only)

### ✅ Enhanced Access Control (Implemented)

JadeVectorDB includes a comprehensive **Role-Based Access Control (RBAC)** system:

- **Groups**: Organize users into teams/departments
- **Roles**: Predefined roles (Admin, User, ReadOnly, DataScientist)
- **Permissions**: Granular control (read, write, delete, admin, create)
- **Database-level permissions**: Control access per database
- **API Keys**: Long-lived tokens for service authentication
- **Audit logging**: Track all security events

See `docs/FRONTEND_RBAC_IMPLEMENTATION.md` and `TasksTracking/11-documentation-updates-summary.md` for implementation details and examples.

### Current User Management

### Creating New Users

**Via API**:
```bash
curl -X POST http://localhost:8080/v1/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "username": "john_doe",
    "password": "SecurePassword123!",
    "email": "john@example.com",
    "roles": ["user"]
  }'
```

**Via Web UI**:
1. Navigate to "Users" (admin access required)
2. Click "Add User"
3. Fill in user details:
   - Username
   - Password
   - Email
   - Roles (comma-separated: user, developer, admin)
4. Click "Create User"

### Managing Users

**List All Users**:
```bash
curl -X GET http://localhost:8080/v1/users \
  -H "Authorization: Bearer $TOKEN"
```

**Update User**:
```bash
curl -X PUT http://localhost:8080/v1/users/user_john_doe \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "email": "john.doe@newdomain.com",
    "roles": ["user", "developer"]
  }'
```

**Delete User**:
```bash
curl -X DELETE http://localhost:8080/v1/users/user_john_doe \
  -H "Authorization: Bearer $TOKEN"
```

## API Key Management

API keys are persisted in the database and survive server restarts. Each key is stored as a SHA-256 hash — the raw key value is only shown once at creation time.

### Creating API Keys

**Via API**:
```bash
curl -X POST http://localhost:8080/v1/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "user_id": "user_admin_default",
    "description": "Production API Key",
    "permissions": ["read", "write"],
    "validity_days": 90
  }'
```

Response:
```json
{
  "api_key": "jadevdb_a1b2c3d4e5f6g7h8",
  "user_id": "user_admin_default",
  "description": "Production API Key",
  "message": "API key created successfully",
  "created_at": "2026-02-16T12:00:00Z"
}
```

**IMPORTANT**: Save the `api_key` value immediately. You won't be able to retrieve it again! Subsequent list calls only return the key prefix (`jadevdb_a1b2`) for identification.

**Via Web UI**:
1. Navigate to "API Keys"
2. Click "Create New API Key"
3. Fill in the form:
   - Description: e.g., "Production API Key"
   - Permissions: e.g., "read, write"
   - Validity Period: e.g., 90 days
4. Click "Create"
5. **Copy the generated key immediately** - it won't be shown again

### Listing API Keys

```bash
# List all keys
curl http://localhost:8080/v1/api-keys \
  -H "Authorization: Bearer $TOKEN"

# List keys for a specific user
curl "http://localhost:8080/v1/api-keys?user_id=user_admin_default" \
  -H "Authorization: Bearer $TOKEN"
```

Response:
```json
{
  "api_keys": [
    {
      "key_id": "abc123",
      "key_prefix": "jadevdb_a1b2",
      "description": "Production API Key",
      "user_id": "user_admin_default",
      "is_active": true,
      "created_at": 1739700000,
      "expires_at": 1747476000,
      "last_used_at": 0,
      "usage_count": 0,
      "permissions": ["read", "write"]
    }
  ],
  "count": 1
}
```

### Using API Keys

Instead of a JWT token, you can use an API key for authentication:

```bash
curl -X GET http://localhost:8080/v1/databases \
  -H "X-API-Key: jadevdb_a1b2c3d4e5f6g7h8"
```

### Revoking API Keys

Use the `key_id` (database ID) from the list response, not the raw key value:

**Via API**:
```bash
curl -X DELETE http://localhost:8080/v1/api-keys/abc123 \
  -H "Authorization: Bearer $TOKEN"
```

Response:
```json
{
  "key_id": "abc123",
  "message": "API key revoked successfully"
}
```

Revoked keys remain in list results with `is_active: false` but can no longer be used for authentication.

**Via Web UI**:
1. Navigate to "API Keys"
2. Find the key you want to revoke
3. Click "Revoke"
4. Confirm the action

## Environment Modes

### Development Mode (Default)

export JADEVECTORDB_ENV=development
export JADEVECTORDB_ENV=development
./jadevectordb
```

export JADEVECTORDB_ENV=test
- ✅ Default users automatically created (admin, dev, test)
- ✅ Verbose logging
- ✅ Debug information included
- ✅ Relaxed security for testing
export JADEVECTORDB_ENV=production
### Test Mode

```bash
export JADEVECTORDB_ENV=test
echo $JADEVECTORDB_ENV  # Should be 'development', 'dev', 'test', or not set
```
export JADEVECTORDB_ENV=development
**Features**:
- ✅ Default users automatically created
- ✅ Isolated test databases
- ✅ Mock external services
- ✅ Simplified authentication for testing

### Production Mode

```bash
export JADEVECTORDB_ENV=production
./jadevectordb
```

**Features**:
- ❌ **NO default users created** (security)
- ✅ Production logging (errors and warnings only)
- ✅ Strict security enforcement
- ✅ Performance optimizations enabled
- ✅ Rate limiting enabled

**Production Security Checklist**:
- [ ] Set `JADEVECTORDB_ENV=production`
- [ ] Create admin users with strong passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerting
- [ ] Enable audit logging
- [ ] Configure backup strategies

## Common Workflows

### Workflow 1: Product Recommendation System

```bash
# 1. Create database
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name": "products", "dimension": 384, "metric": "cosine"}'

# 2. Upload product embeddings
curl -X POST http://localhost:8080/v1/databases/products/vectors/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d @product_embeddings.json

# 3. Search for similar products
curl -X POST http://localhost:8080/v1/databases/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": [/* user preference embedding */],
    "top_k": 10,
    "filter_category": "electronics"
  }'
```

### Workflow 2: Semantic Search

```bash
# 1. Create database for documents
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name": "documents", "dimension": 768, "metric": "cosine"}'

# 2. Index documents
curl -X POST http://localhost:8080/v1/databases/documents/vectors \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "id": "doc_001",
    "values": [/* document embedding */],
    "metadata": {
      "title": "Introduction to Vector Databases",
      "author": "John Doe",
      "date": "2025-01-15"
    }
  }'

# 3. Search by query
curl -X POST http://localhost:8080/v1/databases/documents/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": [/* query embedding */],
    "top_k": 5,
    "include_metadata": true
  }'
```

## Troubleshooting

### Can't Login with Default Credentials

**Issue**: Login fails with "Invalid credentials"

**Solutions**:
1. Verify you're running in development/test mode:
   ```bash
   echo $JADEVECTORDB_ENV  # Should be empty, "development", or "test"
   ```

2. Check server logs for user creation messages:
   ```
   [INFO] Created default user: admin with roles: [admin, developer, user]
   ```

3. If running in production, default users are not created. Create a user manually:
   ```bash
   curl -X POST http://localhost:8080/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "username": "myuser",
       "password": "SecurePass123!",
       "email": "user@example.com"
     }'
   ```

### Token Expired

**Issue**: API returns "Token expired" error

**Solution**: Login again to get a new token:
```bash
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### Permission Denied

**Issue**: API returns "Permission denied" or "403 Forbidden"

**Solution**: Check your user roles and permissions:
```bash
# Login and check your token payload
# Admin users have full access
# Regular users have limited access

# If you need admin access, ask an administrator to update your roles
```

## Best Practices

### Security

1. **Change Default Passwords**: In shared development environments, change default user passwords
2. **Use API Keys for Applications**: Create dedicated API keys for each application
3. **Rotate Keys Regularly**: Revoke and recreate API keys every 90 days
4. **Never Commit Credentials**: Don't commit passwords or API keys to version control
5. **Use Environment Variables**: Store credentials in environment variables, not code

### Performance

1. **Batch Operations**: Use batch endpoints for bulk uploads
2. **Limit Results**: Use `top_k` to limit search results
3. **Filter Early**: Apply metadata filters to reduce search space
4. **Use Appropriate Metrics**: Choose cosine/euclidean/dot-product based on your data

### Data Management

1. **Meaningful IDs**: Use descriptive vector IDs for easy reference
2. **Rich Metadata**: Include relevant metadata for filtering
3. **Consistent Dimensions**: Ensure all vectors in a database have the same dimension
4. **Regular Backups**: Back up your databases regularly

## Additional Resources

- **API Documentation**: `/docs/api_documentation.md`
- **Architecture Overview**: `/docs/architecture.md`
- **Installation Guide**: `/docs/INSTALLATION_GUIDE.md`
- **CLI Documentation**: `/docs/cli-documentation.md`
- **GitHub Repository**: [JadeVectorDB](https://github.com/Jade-biz-1/JadeVectorDB)

## Support

- **Issues**: [GitHub Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Jade-biz-1/JadeVectorDB/discussions)
- **Documentation**: `/docs/`

---

Happy vector searching with JadeVectorDB! 🚀
