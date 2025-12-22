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

### Creating API Keys

**Via API**:
```bash
curl -X POST http://localhost:8080/v1/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "description": "Production API Key",
    "permissions": ["read", "write"],
    "validity_days": 90
  }'
```

Response:
```json
{
  "api_key": "jvdb_a1b2c3d4e5f6g7h8i9j0",
  "key_id": "key_123456",
  "expires_at": "2025-02-28T12:00:00Z"
}
```

**IMPORTANT**: Save the `api_key` value immediately. You won't be able to retrieve it again!

**Via Web UI**:
1. Navigate to "API Keys"
2. Click "Create New API Key"
3. Fill in the form:
   - Description: e.g., "Production API Key"
   - Permissions: e.g., "read, write"
   - Validity Period: e.g., 90 days
4. Click "Create"
5. **Copy the generated key immediately** - it won't be shown again

### Using API Keys

Instead of a JWT token, you can use an API key for authentication:

```bash
curl -X GET http://localhost:8080/v1/databases \
  -H "X-API-Key: jvdb_a1b2c3d4e5f6g7h8i9j0"
```

### Revoking API Keys

**Via API**:
```bash
curl -X DELETE http://localhost:8080/v1/api-keys/key_123456 \
  -H "Authorization: Bearer $TOKEN"
```

**Via Web UI**:
1. Navigate to "API Keys"
2. Find the key you want to revoke
3. Click "Revoke"
4. Confirm the action

## Environment Modes

### Development Mode (Default)

export JADEVECTORDB_ENV=development
export JADE_ENV=development
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
export JADE_ENV=test
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
export JADE_ENV=production
./jadevectordb
```

**Features**:
- ❌ **NO default users created** (security)
- ✅ Production logging (errors and warnings only)
- ✅ Strict security enforcement
- ✅ Performance optimizations enabled
- ✅ Rate limiting enabled

**Production Security Checklist**:
- [ ] Set `JADE_ENV=production`
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
   echo $JADE_ENV  # Should be empty, "development", or "test"
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
