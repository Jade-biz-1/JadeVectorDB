# JadeVectorDB Admin Endpoints Reference

## Overview

This document describes the administrative endpoints available in JadeVectorDB. These endpoints provide server management capabilities and require elevated privileges (admin role) to access.

**Last Updated**: December 26, 2025
**Version**: 1.0

---

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Shutdown Endpoint](#shutdown-endpoint)
3. [Security Considerations](#security-considerations)
4. [Frontend Integration](#frontend-integration)
5. [Troubleshooting](#troubleshooting)

---

## Authentication & Authorization

All admin endpoints require:
- Valid JWT authentication token
- User must have the `admin` role assigned

### Getting an Admin Token

```bash
# Login as admin user
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'

# Response contains the JWT token
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": "b29b0d325d092122",
  "username": "admin"
}
```

### Using the Token

Include the token in the `Authorization` header:

```bash
Authorization: Bearer <your-jwt-token>
```

---

## Shutdown Endpoint

### POST /admin/shutdown

Initiates a graceful shutdown of the JadeVectorDB server.

**Endpoint**: `POST /admin/shutdown`
**Authentication**: Required (JWT)
**Authorization**: Requires `admin` role
**Content-Type**: `application/json`

#### Request

**Headers**:
```
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json
```

**Body**: None required

#### Response

**Success (200 OK)**:
```json
{
  "status": "shutting_down",
  "message": "Server shutdown initiated"
}
```

**Unauthorized (401)**:
```json
{
  "error": "Unauthorized: admin privileges required"
}
```

**Server Error (500)**:
```json
{
  "error": "Shutdown mechanism not configured"
}
```

#### Example Usage

**Using cURL**:
```bash
# Step 1: Login and get token
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | \
  jq -r '.token')

# Step 2: Call shutdown endpoint
curl -X POST http://localhost:8080/admin/shutdown \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

**Using Python**:
```python
import requests

# Login
login_response = requests.post(
    'http://localhost:8080/v1/auth/login',
    json={'username': 'admin', 'password': 'admin123'}
)
token = login_response.json()['token']

# Shutdown server
shutdown_response = requests.post(
    'http://localhost:8080/admin/shutdown',
    headers={'Authorization': f'Bearer {token}'}
)

print(shutdown_response.json())
```

**Using JavaScript (Node.js)**:
```javascript
const fetch = require('node-fetch');

async function shutdownServer() {
  // Login
  const loginResponse = await fetch('http://localhost:8080/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'admin', password: 'admin123' })
  });
  const { token } = await loginResponse.json();

  // Shutdown
  const shutdownResponse = await fetch('http://localhost:8080/admin/shutdown', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` }
  });

  const result = await shutdownResponse.json();
  console.log(result);
}

shutdownServer();
```

#### Behavior

When the shutdown endpoint is called:

1. **Authentication Check**: Validates the JWT token
2. **Authorization Check**: Verifies user has `admin` role
3. **Response Sent**: Returns success response to client
4. **Graceful Shutdown**: After 500ms delay (allows response to be sent):
   - Stops accepting new connections
   - Completes in-flight requests
   - Closes database connections
   - Shuts down HTTP server
   - Exits the process

**Timing**: The server will shut down approximately 500ms after the endpoint returns. This delay ensures the HTTP response is successfully sent to the client.

---

## Security Considerations

### Role-Based Access Control

- Only users with the `admin` role can access admin endpoints
- Default admin user is created in development mode (username: `admin`, password: `admin123`)
- In production, ensure strong passwords are used and default credentials are changed

### JWT Token Security

- Tokens are issued upon successful login
- Tokens contain user_id and are signed with JWT_SECRET
- Tokens should be kept secure and not shared
- Tokens have an expiration time (configurable)

### Audit Logging

All admin endpoint access is logged in the audit log:
- Timestamp of request
- User who made the request
- Action performed
- Result (success/failure)

Example audit log entry:
```
[2025-12-26 20:55:41] [INFO] Shutdown authorized by user: b29b0d325d092122
[2025-12-26 20:55:41] [INFO] Shutdown initiated successfully
[2025-12-26 20:55:41] [INFO] Executing shutdown callback...
```

### Network Security

- Consider restricting admin endpoints to internal networks only
- Use HTTPS in production to encrypt JWT tokens in transit
- Implement rate limiting to prevent brute force attacks
- Use firewall rules to restrict access to admin ports

---

## Frontend Integration

### Dashboard Button (Admin Only)

The JadeVectorDB frontend includes a shutdown button on the dashboard that is only visible to users with admin role.

**Implementation Details**:

1. **Role Check**: Frontend fetches user details on dashboard load
2. **Conditional Rendering**: Shutdown button only renders if user has `admin` role
3. **Confirmation Dialog**: User must confirm before shutdown is initiated
4. **API Call**: Calls `POST /admin/shutdown` with user's JWT token
5. **Feedback**: Shows success/error message to user

**Code Location**:
- Frontend API: `/frontend/src/lib/api.js` - `adminApi.shutdownServer()`
- Dashboard UI: `/frontend/src/pages/dashboard.js` - shutdown button component

**User Experience**:
```
1. Admin user logs into dashboard
2. Sees "Shutdown Server" button (red, next to Refresh button)
3. Clicks button
4. Confirmation dialog: "Are you sure you want to shut down the server?"
5. On confirmation, shutdown request sent
6. Success message displayed
7. After 2 seconds, redirected to home page
8. Server shuts down gracefully
```

---

## Troubleshooting

### Common Issues

#### 401 Unauthorized Error

**Symptom**: Getting "Unauthorized: admin privileges required" even with valid token

**Possible Causes**:
1. User doesn't have `admin` role
2. JWT token is expired
3. JWT token is invalid or malformed
4. Token not included in Authorization header

**Solution**:
```bash
# Verify user has admin role
curl -X GET http://localhost:8080/v1/users/<user-id> \
  -H "Authorization: Bearer $TOKEN"

# Check roles array in response should include "admin"
```

#### 405 Method Not Allowed

**Symptom**: Getting 405 error when calling shutdown endpoint

**Possible Cause**: Using GET instead of POST

**Solution**: Ensure you're using POST method:
```bash
# Correct
curl -X POST http://localhost:8080/admin/shutdown ...

# Incorrect
curl -X GET http://localhost:8080/admin/shutdown ...
```

#### 500 Shutdown Mechanism Not Configured

**Symptom**: Server returns "Shutdown mechanism not configured"

**Possible Cause**: Server was started without registering the shutdown callback

**Solution**: This is a server-side configuration issue. Check that:
- Server code properly registers shutdown callback in `main.cpp`
- `rest_api_service_->set_shutdown_callback()` is called during initialization

#### Frontend Button Not Visible

**Symptom**: Shutdown button doesn't appear on dashboard even when logged in as admin

**Possible Causes**:
1. User doesn't actually have admin role
2. Frontend failed to fetch user details
3. JavaScript error preventing rendering

**Solution**:
```javascript
// Check browser console for errors
// Verify user roles in localStorage
console.log(localStorage.getItem('jadevectordb_username'));

// Manually check user details
fetch('http://localhost:8080/v1/users/<user-id>', {
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('jadevectordb_auth_token')}`
  }
})
.then(r => r.json())
.then(data => console.log('User roles:', data.roles));
```

### Debug Mode

Enable debug logging to troubleshoot shutdown issues:

**Backend**:
```cpp
// Logs are automatically generated in DEBUG builds
// Check logs at /tmp/jadedb.log or console output
```

**Frontend**:
```javascript
// Open browser DevTools Console tab
// Check for API errors and network requests
```

---

## Implementation Details

### Backend Files

**REST API Handler** (`/backend/src/api/rest/rest_api.cpp`):
- `handle_shutdown_request()`: Main endpoint handler (lines 2642-2684)
- `authorize_api_key()`: Authorization helper (lines 2547-2625)
- `extract_api_key()`: Token extraction (lines 2528-2545)
- Route registration (lines 381-388)

**Main Application** (`/backend/src/main.cpp`):
- Shutdown callback registration (lines 243-247)
- `request_shutdown()`: Graceful shutdown implementation

### Frontend Files

**API Client** (`/frontend/src/lib/api.js`):
- `adminApi.shutdownServer()`: Shutdown API method (lines 406-416)

**Dashboard** (`/frontend/src/pages/dashboard.js`):
- Role checking logic (lines 22-35)
- Shutdown handler (lines 59-75)
- UI button (lines 258-265)

---

## Future Enhancements

Potential improvements to admin endpoints:

1. **Restart Endpoint**: Add `/admin/restart` to restart server without full shutdown
2. **Health Status**: Add `/admin/health/detailed` for comprehensive health checks
3. **Configuration Reload**: Add `/admin/config/reload` to reload configuration without restart
4. **Metrics Export**: Add `/admin/metrics/export` to download performance metrics
5. **Log Level Control**: Add `/admin/loglevel` to dynamically adjust logging verbosity
6. **Backup Trigger**: Add `/admin/backup/trigger` to manually trigger database backup

---

## Related Documentation

- [RBAC Admin Guide](rbac_admin_guide.md) - Role-based access control
- [API Documentation](api_documentation.md) - Complete API reference
- [Operations Runbook](operations_runbook.md) - Operational procedures
- [Security Policy](security_policy.md) - Security best practices
- [User Guide](UserGuide.md) - End-user documentation
