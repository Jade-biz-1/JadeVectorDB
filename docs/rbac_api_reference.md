# RBAC API Reference

**Last Updated**: December 17, 2025  
**Version**: 1.0  
**Status**: Production Ready

---

## Overview

JadeVectorDB implements a comprehensive Role-Based Access Control (RBAC) system that provides fine-grained permission management for users, groups, and database resources. The RBAC system is built on SQLite persistence with sub-millisecond performance.

**Key Features**:
- User and group management
- Role-based permissions (admin, user, readonly)
- Database-level access control
- API key authentication
- Session management
- Comprehensive audit logging

---

## Authentication Endpoints

### POST /v1/auth/register

Register a new user account.

**Request Body**:
```json
{
  "username": "alice",
  "password": "SecurePassword123!",
  "email": "alice@example.com"
}
```

**Response** (201 Created):
```json
{
  "user_id": "user_abc123",
  "username": "alice",
  "email": "alice@example.com",
  "roles": ["role_user"],
  "created_at": 1702857600
}
```

**Errors**:
- `400` - Invalid request (weak password, missing fields)
- `409` - Username or email already exists

---

### POST /v1/auth/login

Authenticate user and obtain access token.

**Request Body**:
```json
{
  "username": "alice",
  "password": "SecurePassword123!"
}
```

**Response** (200 OK):
```json
{
  "token": "eyJhbGc...",
  "user_id": "user_abc123",
  "username": "alice",
  "roles": ["role_user"],
  "expires_at": 1702861200
}
```

**Errors**:
- `401` - Invalid credentials
- `403` - Account locked (too many failed attempts)

---

### POST /v1/auth/logout

Revoke current authentication token.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "message": "Logged out successfully"
}
```

---

### GET /v1/auth/me

Get current authenticated user information.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "user_id": "user_abc123",
  "username": "alice",
  "email": "alice@example.com",
  "roles": ["role_user"],
  "is_active": true,
  "created_at": 1702857600,
  "last_login": 1702857650
}
```

---

## API Key Management

### POST /v1/auth/api-keys

Create a new API key for the authenticated user.

**Headers**:
```
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "name": "Production API Key",
  "scopes": ["read", "write"],
  "expires_in_days": 90
}
```

**Response** (201 Created):
```json
{
  "api_key_id": "key_xyz789",
  "api_key": "jvdb_abc123...",
  "name": "Production API Key",
  "key_prefix": "jvdb_abc",
  "scopes": ["read", "write"],
  "created_at": 1702857600,
  "expires_at": 1710720000
}
```

**Note**: The full `api_key` value is only shown once. Store it securely.

---

### GET /v1/auth/api-keys

List all API keys for the authenticated user.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "api_keys": [
    {
      "api_key_id": "key_xyz789",
      "name": "Production API Key",
      "key_prefix": "jvdb_abc",
      "scopes": ["read", "write"],
      "is_active": true,
      "created_at": 1702857600,
      "expires_at": 1710720000,
      "last_used_at": 1702858000,
      "usage_count": 142
    }
  ]
}
```

---

### DELETE /v1/auth/api-keys/:key_id

Revoke an API key.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "message": "API key revoked successfully"
}
```

---

## User Management (Admin Only)

### GET /v1/users

List all users in the system.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Query Parameters**:
- `limit` (optional): Maximum number of users to return (default: 100)
- `offset` (optional): Number of users to skip (default: 0)

**Response** (200 OK):
```json
{
  "users": [
    {
      "user_id": "user_abc123",
      "username": "alice",
      "email": "alice@example.com",
      "roles": ["role_user"],
      "is_active": true,
      "created_at": 1702857600
    }
  ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

**Errors**:
- `403` - Forbidden (requires admin role)

---

### GET /v1/users/:user_id

Get detailed information about a specific user.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "user_id": "user_abc123",
  "username": "alice",
  "email": "alice@example.com",
  "roles": ["role_user"],
  "groups": ["group_developers"],
  "is_active": true,
  "created_at": 1702857600,
  "last_login": 1702857650,
  "failed_login_attempts": 0
}
```

---

### PUT /v1/users/:user_id/roles

Assign or update roles for a user.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "roles": ["role_admin", "role_user"]
}
```

**Response** (200 OK):
```json
{
  "user_id": "user_abc123",
  "roles": ["role_admin", "role_user"],
  "updated_at": 1702857700
}
```

---

### PUT /v1/users/:user_id/status

Activate or deactivate a user account.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "is_active": false
}
```

**Response** (200 OK):
```json
{
  "user_id": "user_abc123",
  "is_active": false,
  "updated_at": 1702857700
}
```

---

## Database Permissions

### POST /v1/databases/:db_id/permissions

Grant permission to a user or group for a database.

**Headers**:
```
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "principal_type": "user",
  "principal_id": "user_abc123",
  "permission": "database:read"
}
```

**Principal Types**:
- `user` - Grant to specific user
- `group` - Grant to all members of a group

**Available Permissions**:
- `database:read` - Read database and query vectors
- `database:write` - Insert, update, delete vectors
- `database:delete` - Delete the database
- `database:admin` - Full administrative access

**Response** (200 OK):
```json
{
  "database_id": "db_xyz789",
  "principal_type": "user",
  "principal_id": "user_abc123",
  "permission": "database:read",
  "granted_at": 1702857700,
  "granted_by": "admin_user"
}
```

---

### GET /v1/databases/:db_id/permissions

List all permissions for a database.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "database_id": "db_xyz789",
  "permissions": [
    {
      "principal_type": "user",
      "principal_id": "user_abc123",
      "permission": "database:read",
      "granted_at": 1702857700,
      "granted_by": "admin_user"
    },
    {
      "principal_type": "group",
      "principal_id": "group_developers",
      "permission": "database:write",
      "granted_at": 1702857600,
      "granted_by": "admin_user"
    }
  ]
}
```

---

### DELETE /v1/databases/:db_id/permissions

Revoke a permission from a user or group.

**Headers**:
```
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "principal_type": "user",
  "principal_id": "user_abc123",
  "permission": "database:read"
}
```

**Response** (200 OK):
```json
{
  "message": "Permission revoked successfully"
}
```

---

## Group Management

### POST /v1/groups

Create a new group.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "name": "developers",
  "description": "Development team members"
}
```

**Response** (201 Created):
```json
{
  "group_id": "group_xyz789",
  "name": "developers",
  "description": "Development team members",
  "created_at": 1702857700
}
```

---

### POST /v1/groups/:group_id/members

Add a user to a group.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "user_id": "user_abc123"
}
```

**Response** (200 OK):
```json
{
  "group_id": "group_xyz789",
  "user_id": "user_abc123",
  "added_at": 1702857700
}
```

---

### GET /v1/groups/:group_id/members

List all members of a group.

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "group_id": "group_xyz789",
  "members": [
    {
      "user_id": "user_abc123",
      "username": "alice",
      "joined_at": 1702857700
    }
  ]
}
```

---

## Audit Logs

### GET /v1/audit/logs

Retrieve audit logs for security events.

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Query Parameters**:
- `limit` (optional): Maximum logs to return (default: 100)
- `offset` (optional): Number of logs to skip (default: 0)
- `user_id` (optional): Filter by specific user
- `action` (optional): Filter by action type
- `start_time` (optional): Filter by start timestamp
- `end_time` (optional): Filter by end timestamp

**Response** (200 OK):
```json
{
  "logs": [
    {
      "log_id": "log_abc123",
      "user_id": "user_abc123",
      "action": "login",
      "resource_type": "authentication",
      "resource_id": "session_xyz",
      "ip_address": "192.168.1.100",
      "success": true,
      "details": "User logged in successfully",
      "timestamp": 1702857700
    }
  ],
  "total": 1523,
  "limit": 100,
  "offset": 0
}
```

**Logged Events**:
- User registration, login, logout
- Role assignments/revocations
- Permission grants/revokes
- API key creation/revocation
- Database creation/deletion
- Configuration changes
- Failed authentication attempts

---

## Permission Model

### Default Roles

JadeVectorDB provides three built-in roles:

**1. `role_admin`**
- Full system access
- User and group management
- Database creation and deletion
- Permission management
- Audit log access

**2. `role_user`**
- Create and manage own databases
- Query and modify own databases
- Create API keys
- View own audit logs

**3. `role_readonly`**
- Read-only access to assigned databases
- Cannot create or modify databases
- Cannot manage users or permissions

### Permission Hierarchy

Permissions follow a hierarchical structure:

```
database:admin
  ├── database:delete
  ├── database:write
  │     └── database:read
  └── database:read
```

Granting a higher-level permission automatically includes lower levels.

### Permission Inheritance

Users inherit permissions from:
1. **Direct user permissions** - Explicitly granted to user
2. **Group permissions** - Inherited from group membership
3. **Role permissions** - Default permissions for assigned roles

**Permission Resolution Order**:
1. Check direct user permissions
2. Check group permissions (if user is in groups)
3. Check role permissions
4. Deny by default if no permission found

---

## Performance Characteristics

Based on comprehensive benchmarking (Sprint 1.5):

| Operation | Performance | Target |
|-----------|-------------|--------|
| User lookup | 0.01ms | <10ms |
| Permission check | 0.01ms | <5ms |
| User creation | 0.51ms | <10ms |
| Role assignment | 0.44ms | <15ms |
| Concurrent operations | 1000 in 232ms | 1000+ |

**All targets exceeded by 20-500x** ⚡

---

## Best Practices

### Authentication

1. **Use strong passwords**: Minimum 10 characters, mix of upper/lower/numbers/symbols
2. **Rotate API keys**: Set expiration dates, rotate every 90 days
3. **Monitor failed logins**: Account locked after 5 failed attempts
4. **Use tokens for web**: Tokens for web apps, API keys for programmatic access

### Authorization

1. **Principle of least privilege**: Grant minimum permissions needed
2. **Use groups**: Manage permissions via groups rather than individual users
3. **Regular audits**: Review permissions quarterly
4. **Separate environments**: Different API keys for dev/staging/production

### Security

1. **Enable HTTPS**: Always use HTTPS in production
2. **Audit logs**: Monitor audit logs for suspicious activity
3. **Token expiration**: Configure appropriate token lifetimes
4. **IP restrictions**: Consider IP whitelisting for sensitive operations

---

## Code Examples

### Python Client

```python
import requests

# Login
response = requests.post(
    "https://api.example.com/v1/auth/login",
    json={"username": "alice", "password": "SecurePassword123!"}
)
token = response.json()["token"]

# Create API Key
response = requests.post(
    "https://api.example.com/v1/auth/api-keys",
    headers={"Authorization": f"Bearer {token}"},
    json={"name": "My API Key", "scopes": ["read", "write"]}
)
api_key = response.json()["api_key"]

# Use API Key for requests
response = requests.get(
    "https://api.example.com/v1/databases",
    headers={"Authorization": f"Bearer {api_key}"}
)
```

### cURL

```bash
# Login
TOKEN=$(curl -X POST https://api.example.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"SecurePassword123!"}' \
  | jq -r '.token')

# Grant permission
curl -X POST https://api.example.com/v1/databases/db_123/permissions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "principal_type": "user",
    "principal_id": "user_abc123",
    "permission": "database:read"
  }'
```

---

## Troubleshooting

### 401 Unauthorized

**Cause**: Invalid or expired token  
**Solution**: Re-authenticate to obtain new token

### 403 Forbidden

**Cause**: Insufficient permissions  
**Solution**: Contact admin to request required role/permission

### 429 Too Many Requests

**Cause**: Rate limit exceeded  
**Solution**: Implement exponential backoff, reduce request rate

### Account Locked

**Cause**: Too many failed login attempts  
**Solution**: Wait 15 minutes or contact admin to unlock

---

## See Also

- [RBAC Administration Guide](rbac_admin_guide.md)
- [Permission Model Documentation](rbac_permission_model.md)
- [Security Best Practices](security_best_practices.md)
- [API Authentication Guide](api_authentication.md)
