# Local Deployment Guide for JadeVectorDB

This guide details how to deploy JadeVectorDB locally, including current limitations and workarounds.

## Overview

JadeVectorDB can be deployed locally using either Docker Compose or direct installation. However, there are critical known issues that currently prevent a fully functional local deployment.

## Current Status

**NOTE**: The startup crash caused by duplicate API route handlers was **fixed on 2025-12-12**. Local deployment should now work when using the latest code from `run-and-fix` or newer branches. If you see the `handler already exists for /v1/databases` error, pull the latest changes and rebuild.

## Deployment Options

### Option 1: Docker Compose (Recommended for Testing)

#### Prerequisites
- Docker version 20.10 or higher
- Docker Compose plugin or docker-compose v2.0+

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Jade-biz-1/JadeVectorDB.git
   cd JadeVectorDB
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

#### Expected Behavior
- **Backend service should start normally** when using the latest code (pull `run-and-fix` or newer and rebuild)
- Frontend service should be able to connect to the backend once started
- Monitoring services (Prometheus, Grafana) should work independently

#### Workaround for Testing
You can still run the services separately to test individual components:
```bash
# Start only monitoring services
docker-compose up prometheus grafana

# Or start backend and frontend separately to test their connectivity
docker-compose up jadevectordb
```

### Option 2: Direct Installation

#### Prerequisites
- C++20 compatible compiler (GCC 11+ or Clang 14+)
- CMake 3.20 or higher
- Node.js 16+ (for frontend)
- Docker (optional, for containerized deployment)

#### Backend Build & Run
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Build the application (excluding tests to avoid compilation issues):
   ```bash
   ./build.sh --no-tests --no-benchmarks
   ```

3. Run the application:
   ```bash
   cd build
   ./jadevectordb
   ```

#### Expected Behavior
- The executable builds successfully
- Application crashes during startup with the error: `handler already exists for /v1/databases`
- No API endpoints are available

## Known Issues and Limitations

### 1. Duplicate Route Handler Issue (Resolved)

**Problem (historical)**: The REST API previously had duplicate route registrations that caused a startup crash.

**Impact**: 
- Backend would not start and frontend could not connect

**Status**: **Resolved (2025-12-12)** — ensure you are on the latest branch and rebuild if you encounter similar behavior.

### 2. Frontend Connection Issues

**Problem**: Even if the backend were to start, the frontend cannot connect due to the backend crash.

**Impact**: 
- Web UI at `http://localhost:3000` cannot access backend API
- Authentication and all data operations unavailable
- Monitoring dashboards cannot retrieve real-time data

### 3. Test Compilation Issues

**Problem**: Building with tests enabled results in compilation errors.

**Workaround**: Use `--no-tests --no-benchmarks` flag during build.

## Workarounds and Testing Alternatives

### 1. Building Only for Code Testing
If you want to build the code without running:
```bash
cd backend
./build.sh --no-tests --no-benchmarks
```

### 2. Frontend Development
You can still run the frontend separately for UI development:
```bash
cd frontend
npm install
npm run dev
```
Note: The frontend won't be able to connect to the backend API.

### 3. Documentation and Architecture Review
- Browse the API documentation (`docs/api_documentation.md`)
- Review architecture documentation (`docs/architecture.md`)
- Study the system design without running the service

## Monitoring and Logging

Even though the main service cannot start, you can still access some monitoring components:

### Prometheus
- URL: `http://localhost:9090`
- Provides monitoring metrics for the services that do start

### Grafana
- URL: `http://localhost:3001`
- Username: `admin`
- Password: `admin`
- Note: Will have limited data since the main backend service isn't running

## Troubleshooting Current Issues

### Check Backend Build Status
```bash
cd backend/build
ls -la jadevectordb
file jadevectordb
```

### Check Docker Logs
```bash
docker-compose logs jadevectordb
```

### Common Error Messages
- `"handler already exists for /v1/databases"` - Duplicate route registration
- `"Connection refused"` - Backend service not running
- `"Address already in use"` - Port conflict (different issue from the main crash)

## Next Steps for Full Deployment

Before the local deployment can function properly, the following issues need to be resolved:

1. **Fix duplicate API route handlers** in `backend/src/api/rest/rest_api.cpp`
2. **Verify authentication endpoints** work correctly after route fixes
3. **Complete batch get vectors endpoint** implementation
4. **Resolve any test compilation issues**

## Support and Contributions

The development team is aware of these issues and working on fixes. If you have expertise in C++ REST API development, you can help by:

1. Examining `backend/src/api/rest/rest_api.cpp` to identify and fix the duplicate route registration
2. Reviewing the distributed system changes that may have introduced this issue
3. Testing API functionality after fixes are implemented

## Additional Resources

- [Known Issues](KNOWN_ISSUES.md) - Complete list of current problems
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - How to diagnose problems
- [API Documentation](api_documentation.md) - API endpoints and usage
- [Architecture Documentation](architecture.md) - System design overview

---

**Note**: This local deployment guide will be updated once the critical issues preventing startup are resolved.