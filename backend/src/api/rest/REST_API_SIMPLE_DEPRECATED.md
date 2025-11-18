# DEPRECATION NOTICE: rest_api_simple.cpp

**Status**: DEPRECATED as of 2025-11-18
**Replacement**: Use `rest_api.cpp` instead

## Reason for Deprecation

The `rest_api_simple.cpp` file has been deprecated in favor of the comprehensive `rest_api.cpp` implementation due to the following reasons:

### Missing Critical Features

`rest_api_simple.cpp` lacks the following production-ready features that are present in `rest_api.cpp`:

1. **Authentication System** ❌
   - No JWT token management
   - No user registration/login/logout
   - No password reset functionality

2. **User Management** ❌
   - No user CRUD operations
   - No role-based access control (RBAC)
   - No user status management

3. **API Key Management** ❌
   - No API key generation
   - No API key revocation
   - No permission-based keys

4. **Security Features** ❌
   - No security audit logging
   - No session management
   - No IP tracking for authentication events

5. **Monitoring & Operations** ❌
   - No audit log endpoints
   - No alert management
   - No cluster management
   - No performance metrics

6. **Advanced Features** ❌
   - No lifecycle management
   - No replication routes
   - No backup/restore endpoints

### What rest_api_simple.cpp Has

`rest_api_simple.cpp` only provides basic functionality:
- Health check (`/health`)
- System status (`/status`)
- Database CRUD operations
- Vector CRUD operations
- Basic and advanced search

## Migration Guide

### For Developers

If you're currently using `rest_api_simple.cpp`:

1. **Switch to `rest_api.cpp`** - It has all the same endpoints plus much more
2. **Update build configuration** - Ensure you're compiling `rest_api.cpp` instead
3. **Add authentication** - Implement user authentication using the new endpoints
4. **Update API clients** - Add authentication headers to requests

### Endpoint Mapping

All endpoints in `rest_api_simple.cpp` exist in `rest_api.cpp` with identical functionality:

| Endpoint | rest_api_simple.cpp | rest_api.cpp | Notes |
|----------|-------------------|--------------|-------|
| GET /health | ✅ | ✅ | Identical |
| GET /status | ✅ | ✅ | Enhanced in rest_api.cpp |
| POST /v1/databases | ✅ | ✅ | Identical |
| GET /v1/databases | ✅ | ✅ | Enhanced with pagination |
| GET /v1/databases/{id} | ✅ | ✅ | Identical |
| PUT /v1/databases/{id} | ✅ | ✅ | Identical |
| DELETE /v1/databases/{id} | ✅ | ✅ | Identical |
| POST /v1/databases/{id}/vectors | ✅ | ✅ | Identical |
| GET /v1/databases/{id}/vectors/{vectorId} | ✅ | ✅ | Identical |
| PUT /v1/databases/{id}/vectors/{vectorId} | ✅ | ✅ | Identical |
| DELETE /v1/databases/{id}/vectors/{vectorId} | ✅ | ✅ | Identical |
| POST /v1/databases/{id}/vectors/batch | ✅ | ✅ | Identical |
| POST /v1/databases/{id}/search | ✅ | ✅ | Identical |
| POST /v1/databases/{id}/search/advanced | ✅ | ✅ | Enhanced with filters |

### Additional Endpoints in rest_api.cpp

When you migrate, you'll gain access to:

**Authentication** (5 endpoints):
- `POST /v1/auth/register`
- `POST /v1/auth/login`
- `POST /v1/auth/logout`
- `POST /v1/auth/forgot-password`
- `POST /v1/auth/reset-password`

**User Management** (5 endpoints):
- `GET /v1/users`
- `POST /v1/users`
- `PUT /v1/users/{userId}`
- `DELETE /v1/users/{userId}`
- `GET /v1/users/{userId}/status`

**API Key Management** (3 endpoints):
- `GET /v1/apikeys`
- `POST /v1/apikeys`
- `DELETE /v1/apikeys/{keyId}`

**Security** (1 endpoint):
- `GET /v1/security/audit-logs`

**Alerts** (3 endpoints):
- `GET /v1/alerts`
- `POST /v1/alerts`
- `POST /v1/alerts/{id}/acknowledge`

**Cluster** (2 endpoints):
- `GET /v1/cluster/nodes`
- `GET /v1/cluster/nodes/{nodeId}`

**Performance** (1 endpoint):
- `GET /v1/performance/metrics`

## Timeline

- **2025-11-18**: Deprecated, will remain in codebase for reference
- **Future**: May be removed in a future major version

## Questions?

If you have questions about this deprecation or need help migrating, please:
1. Review the `rest_api.cpp` implementation
2. Check the `AUTHENTICATION_IMPLEMENTATION_STATUS.md` document
3. Refer to `docs/api_documentation.md` for complete API reference

---

**Recommendation**: Use `rest_api.cpp` for all new development and production deployments.
