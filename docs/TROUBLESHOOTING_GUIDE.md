# JadeVectorDB Troubleshooting Guide

This guide helps diagnose and resolve common issues with building, running, and using JadeVectorDB.

## Table of Contents
1. [Build Issues](#build-issues)
2. [Runtime Issues](#runtime-issues)
3. [API and Network Issues](#api-and-network-issues)
4. [Authentication Issues](#authentication-issues)
5. [Performance Issues](#performance-issues)
6. [Distributed System Issues](#distributed-system-issues)

## Build Issues

### 1. Test Compilation Failures

**Problem**: When building with tests enabled, compilation errors occur in test files.

**Error**: Multiple compilation errors related to:
- Accessing private members that tests shouldn't access
- Incorrect API usage in test files (e.g., `metadata["key"]` syntax)
- Type mismatches (using `Database` instead of `DatabaseCreationParams`)

**Solution**: 
```bash
# Build without tests to bypass compilation errors
./build.sh --no-tests --no-benchmarks

# Or use explicit CMake options
cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF ..
make -j$(nproc)
```

### 2. "Out of Memory" During Build

**Problem**: Build fails with memory exhaustion errors, particularly during Arrow build.

**Solution**:
```bash
# Reduce parallel jobs to lower memory usage
./build.sh --jobs 2

# Or use fewer cores during compilation
cmake --build backend/build -j1
```

### 3. Git Fetch Failures

**Problem**: CMake FetchContent fails to download dependencies.

**Error**: Network-related errors during dependency fetching.

**Solution**:
- Check internet connection
- Verify git is installed: `git --version`
- Check firewall settings
- Retry the build command: `./build.sh --clean`

### 4. Missing Compiler or Dependencies

**Problem**: Build fails with compiler or dependency detection errors.

**Error**: "C++20 compiler not found" or dependency missing errors.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake git

# Check versions
cmake --version  # Should be >= 3.20
g++ --version    # Should be >= GCC 11.0 or Clang 14.0
```

### 5. Slow First Build

**Problem**: Initial build takes much longer than expected (30+ minutes).

**Explanation**: First build includes downloading and compiling all dependencies (Eigen, Arrow, etc.).

**Solution**:
- This is normal for first builds
- Subsequent builds will be much faster (incremental compilation)
- Use `--no-tests --no-benchmarks` to speed up builds during development

## Runtime Issues

### 1. Duplicate Route Handler Crash (Resolved)

**Problem (historical)**: Application crashed on startup with:
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  handler already exists for /v1/databases
```

**Root Cause**: Duplicate route registration in `rest_api.cpp` (both `app_->route_dynamic()` and `CROW_ROUTE()` used for same endpoints).

**Status**: **Resolved (2025-12-12)**. If you still see this error, update to the latest `run-and-fix` branch and rebuild the backend. See `docs/LOCAL_DEPLOYMENT.md` for troubleshooting steps.

### 2. Port Already in Use

**Problem**: Server fails to start with "Address already in use" error.

**Solution**:
```bash
# Check what's using port 8080
lsof -i :8080

# Kill the process or use a different port
export JDB_PORT=8081
./jadevectordb
```

### 3. High Memory Usage

**Problem**: Application uses more memory than expected.

**Solution**:
- Check system memory: `free -h`
- Reduce index cache size in configuration
- Use memory-mapped files for large datasets
- Limit number of active databases/indexes

### 4. API Response Timeouts

**Problem**: API requests timeout or take longer than expected.

**Troubleshooting**:
```bash
# Check system resources
top -p $(pgrep jadevectordb)

# Check for resource constraints
free -h
df -h
```

**Solutions**:
- Increase timeout values in client configuration
- Check system resources (CPU, memory, disk I/O)
- Optimize index parameters for query patterns

## API and Network Issues

### 1. Connection Refused

**Problem**: API requests return "Connection refused" errors.

**Solution**:
```bash
# Check if server is running
ps aux | grep jadevectordb

# Check if listening on expected port
netstat -tlnp | grep 8080

# Verify the correct address is being used
curl http://localhost:8080/health
```

### 2. Authentication Required

**Problem**: API returns 401 Unauthorized errors.

**Solution**:
- Verify API key or JWT token is correct
- Check Authorization header format:
  ```
  Authorization: Bearer {api-key}
  ```
- Ensure token hasn't expired (for JWT tokens)
- Check that default users were created in development mode

### 3. Rate Limiting

**Problem**: API returns 429 Too Many Requests errors.

**Default Limits**:
- Database operations: 100 requests per minute
- Vector operations: 1000 requests per minute
- Search operations: 500 requests per minute
- Embedding generation: 10 requests per minute

**Solution**:
- Implement retry logic with exponential backoff
- Batch requests where possible
- Adjust rate limit configuration if appropriate

## Authentication Issues

### 1. Default Users Not Created

**Problem**: Cannot log in with default credentials.

**Check Environment**:
```bash
echo $JADEVECTORDB_ENV  # Should be 'development', 'dev', 'test', or not set

# If set to production, change to development
export JADEVECTORDB_ENV=development
./jadevectordb
```

### 2. Default User Seeding Messages

**Check server logs** for one of:
- Development/Test mode: "Seeding default users for development environment"
- Production mode: "Skipping default user seeding in production environment"
- Subsequent starts: "Default user 'admin' already exists, skipping"

### 3. Invalid Credentials

**Problem**: Login fails with valid-looking credentials.

**Solution**:
- Verify server is running in development mode
- Check server logs for user seeding confirmation
- Try registering a new user manually

## Performance Issues

### 1. Slow Query Response Times

**Check Index Types**:
- FLAT: Best for small datasets (< 10K vectors)
- HNSW: Good for most use cases
- IVF: Good for very large datasets
- LSH: Good for high-dimensional data

**Solutions**:
- Ensure appropriate index is built for the database
- Check system resources
- Optimize index parameters based on query patterns
- Consider using different index types for different workloads

### 2. High Query Latency

**Monitoring**:
```bash
# Check performance metrics
curl -X GET http://localhost:8080/status
```

**Solutions**:
- Verify index is properly configured
- Check system resources (CPU, memory, disk I/O)
- Consider increasing cache sizes
- Rebuild index if necessary

### 3. Memory Issues

**Problem**: Application runs out of memory.

**Solutions**:
- Reduce index cache size
- Limit number of active databases
- Use memory-mapped files for large datasets
- Monitor memory usage patterns

## Distributed System Issues

### 1. Cluster Communication Failures

**Problem**: Distributed components cannot communicate properly.

**Solution**:
- Verify network connectivity between nodes
- Check that firewall rules allow required ports
- Verify cluster configuration parameters
- Check node status with cluster monitoring endpoints

### 2. Sharding Configuration Issues

**Problem**: Data is not distributed properly across shards.

**Solution**:
- Verify sharding strategy configuration
- Check that all worker nodes are operational
- Review load balancing settings
- Monitor shard distribution statistics

### 3. Replication Failures

**Problem**: Data is not replicating across nodes as expected.

**Solution**:
- Check network connectivity between nodes
- Verify replication configuration
- Monitor replication lag statistics
- Check disk space on replica nodes

## General Troubleshooting Commands

```bash
# Check if JadeVectorDB is running
ps aux | grep jadevectordb

# Check network connections
netstat -tlnp | grep 8080

# Check application logs
tail -f logs/jadevectordb.log

# Test basic connectivity
curl -v http://localhost:8080/health

# Check system resources
htop
free -h
df -h

# Monitor performance
curl http://localhost:8080/status
```

## Getting Help

If you encounter issues not covered in this guide:

1. Check the logs in the `logs/` directory
2. Review the [Known Issues](KNOWN_ISSUES.md) document
3. Search existing GitHub issues
4. Open a new issue with:
   - CMake version (`cmake --version`)
   - Compiler version (`g++ --version` or `clang++ --version`)
   - Operating system details
   - Build command used
   - Complete error output
   - Steps to reproduce