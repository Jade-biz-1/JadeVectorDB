# Known Issues in JadeVectorDB

This document details known issues in the JadeVectorDB codebase that developers should be aware of.

## 1. Duplicate API Route Handlers

### Issue Description
The backend REST API has a critical issue where API routes are being registered twice in `rest_api.cpp`, causing the application to crash on startup with the error:
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  handler already exists for /v1/databases
```

### Root Cause
Routes are registered using two different mechanisms in the `register_routes()` method:
1. `app_->route_dynamic("/v1/databases")` - which handles multiple HTTP methods in a single route
2. `CROW_ROUTE((*app_), "/v1/databases")` - which creates individual routes for each method

This double registration pattern was likely introduced during distributed system integration work.

### Affected Routes
- `/v1/databases` (POST and GET methods)
- `/v1/databases/<string>` (GET, PUT, DELETE methods)
- `/v1/databases/<string>/vectors` (POST method)
- `/v1/databases/<string>/vectors/batch` (POST method)
- `/v1/databases/<string>/vectors/batch-get` (POST method)
- `/v1/databases/<string>/vectors/<string>` (GET, PUT, DELETE methods)
- `/v1/databases/<string>/search` (POST method)
- `/v1/databases/<string>/search/advanced` (POST method)
- `/v1/databases/<string>/indexes` (POST, GET methods)
- `/v1/databases/<string>/indexes/<string>` (PUT, DELETE methods)

### Impact
- The application executable builds successfully
- The application crashes during startup when attempting to register duplicate routes
- The REST API endpoints are not available for use

### Status
- **Severity**: Critical
- **Status**: Unresolved
- **Location**: `/backend/src/api/rest/rest_api.cpp` lines 170-360, 400-610

### Workaround
There is no current workaround. The issue must be fixed by removing the duplicate route registration pattern in the code.

## 2. Test Compilation Failures

### Issue Description
When building with tests enabled (default behavior), multiple compilation errors may occur in test files due to API mismatches and access to private members.

### Root Cause
- Test files try to access private members that they shouldn't
- Incorrect API usage in test files (e.g., `metadata["key"]` syntax doesn't match the expected API)
- Type mismatches in test expectations (using `Database` instead of `DatabaseCreationParams`)

### Impact
- Tests fail to compile
- Continuous integration builds may fail if not properly configured

### Workaround
Use the build script with tests disabled:
```bash
cd backend && ./build.sh --no-tests --no-benchmarks
```

## 3. Frontend API Connection Issues

### Issue Description
The frontend cannot connect to backend services in local deployment due to backend crash issue.

### Root Cause
- The backend crashes on startup due to duplicate route handlers
- No backend services available for frontend to connect to

### Impact
- Local development environment does not run properly
- Frontend application cannot access backend API endpoints
- Authentication and other services are unavailable

### Status
- **Severity**: Critical (for local development)
- **Status**: Unresolved

### Workaround
The issue is directly tied to the backend crash issue. Both must be resolved together.

## 4. Batch Get Vectors Endpoint

### Issue Description
The batch get vectors endpoint is not yet implemented and returns a 501 Not Implemented error.

### Location
`/backend/src/api/rest/rest_api.cpp` - `handle_batch_get_vectors_request` method

### Impact
- Batch retrieval of vectors is not available
- Applications requiring bulk vector retrieval cannot use this feature

### Status
- **Severity**: Medium
- **Status**: Unresolved