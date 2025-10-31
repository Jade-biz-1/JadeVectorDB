# JadeVectorDB Docker Deployment Guide

## 📋 Overview

This guide covers local Docker-based deployment of JadeVectorDB. The current setup successfully builds the **core library** (`libjadevectordb_core.a`), which contains all the vector database functionality.

## 🎯 Current Status

### ✅ Working
- **Core Library**: Fully built and functional (33 MB)
- **All Core Features**:
  - Vector storage and retrieval
  - Similarity search (cosine, euclidean, dot product)
  - Multiple index types (HNSW, IVF, LSH, PQ, OPQ, SQ, Flat, Composite)
  - Metadata filtering
  - Database management
  - Authentication and authorization

### ⚠️ In Progress
- **REST API Server**: Has compilation issues in `rest_api.cpp` (lines 2787-2898)
  - LifecycleConfig struct definition missing
  - AuthManager::get_instance() method mismatch
  - Policy field access errors

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4 GB RAM
- 10 GB free disk space

### Option 1: Development Setup (Recommended for now)

Build and run the core library development environment:

```bash
# Build the development container
docker-compose -f docker-compose.dev.yml build

# Start the core library container
docker-compose -f docker-compose.dev.yml up -d jadevectordb-core

# Check the build status
docker logs jadevectordb-core-dev

# Access the development shell
docker exec -it jadevectordb-core-dev /bin/bash

# Inside the container, verify the library
cd /app/backend/build
ls -lh libjadevectordb_core.a
```

### Option 2: Interactive Development Shell

For active development work:

```bash
# Start the interactive shell
docker-compose -f docker-compose.dev.yml run --rm jadevectordb-shell

# Inside the shell, you can rebuild as needed
cd /app/backend/build
make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc) jadevectordb_core
```

### Option 3: With Monitoring Stack

Enable Prometheus and Grafana for monitoring:

```bash
# Start with monitoring profile
docker-compose -f docker-compose.dev.yml --profile monitoring up -d

# Access Grafana at http://localhost:3001
# Username: admin, Password: admin

# Access Prometheus at http://localhost:9090
```

## 📁 Available Docker Files

| File | Purpose | Status |
|------|---------|--------|
| `Dockerfile` | Original production Dockerfile | ⚠️ Needs REST API fixes |
| `Dockerfile.dev` | Development build (core library) | ✅ Working |
| `Dockerfile.local` | Local development variant | ⚠️ Needs REST API fixes |
| `Dockerfile.fixed` | Attempted fixes | ⚠️ Incomplete |
| `docker-compose.yml` | Full production stack | ⚠️ Needs REST API fixes |
| `docker-compose.dev.yml` | Development environment | ✅ Working |
| `docker-compose.fixed.yml` | Attempted fixes | ⚠️ Incomplete |
| `docker-compose.distributed.yml` | Distributed deployment | 📝 For future use |

## 🛠️ Development Workflow

### 1. Build the Development Image

```bash
docker-compose -f docker-compose.dev.yml build jadevectordb-core
```

### 2. Start the Container

```bash
docker-compose -f docker-compose.dev.yml up -d jadevectordb-core
```

### 3. Make Changes

Edit files in `./backend/` directory. The container mounts this as a read-only volume.

### 4. Rebuild Inside Container

```bash
docker exec -it jadevectordb-core-dev /bin/bash
cd /app/backend/build
make -j$(nproc) jadevectordb_core
```

### 5. Test Your Changes

```bash
# Run tests
make jadevectordb_tests
./jadevectordb_tests
```

## 🔧 Fixing the REST API Server

To complete the full server deployment, fix these issues in `backend/src/api/rest/rest_api.cpp`:

### Issue 1: LifecycleConfig (lines 2787-2800)
```cpp
// Missing struct definition - add to appropriate header
struct LifecycleConfig {
    std::string database_id;
    RetentionPolicy retention_policy;
    // ... other fields
};
```

### Issue 2: AuthManager (line 2856)
```cpp
// Current: auto auth_manager = AuthManager::get_instance();
// Fix: Either add get_instance() method or use constructor
auto auth_manager = std::make_shared<AuthManager>();
```

### Issue 3: Policy Fields (lines 2894-2898)
```cpp
// The policy variable needs proper type definition
// Review LifecycleService::get_lifecycle_status() return type
```

Once fixed, you can use the original Dockerfile:

```bash
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d
```

## 📊 Container Structure

### Development Container (`jadevectordb-core`)
```
/app/backend/                     # Source code
/app/backend/build/      # Build directory
  ├── libjadevectordb_core.a      # Core library (33 MB)
  ├── CMakeCache.txt              # CMake configuration
  └── ...                         # Other build artifacts
/data/                            # Data volume
/config/                          # Configuration volume
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JADE_DB_DATA_DIR` | `/data` | Data storage directory |
| `JADE_DB_CONFIG_DIR` | `/config` | Configuration directory |
| `JADE_DB_LOG_LEVEL` | `INFO` | Logging level |
| `JADE_DB_PORT` | `8080` | Server port (when REST API is fixed) |

## 🗂️ Volumes

| Volume | Purpose | Persistence |
|--------|---------|-------------|
| `jadevectordb_data` | Vector data storage | Persistent |
| `jadevectordb_config` | Configuration files | Persistent |
| `jadevectordb_build` | Build artifacts | Persistent (dev only) |
| `prometheus_data` | Metrics storage | Persistent |
| `grafana_data` | Dashboard config | Persistent |

## 🌐 Network

The development setup creates a bridge network `jadevectordb_dev_network` for inter-container communication.

## 🧪 Testing the Build

### Verify Core Library

```bash
# Check library exists and size
docker exec jadevectordb-core-dev ls -lh /app/backend/build/libjadevectordb_core.a

# Check library symbols
docker exec jadevectordb-core-dev nm /app/backend/build/libjadevectordb_core.a | grep -i "search\|vector\|database" | head -20
```

### Run Unit Tests (if available)

```bash
docker exec jadevectordb-core-dev /bin/bash -c "cd /app/backend/build && make jadevectordb_tests && ./jadevectordb_tests"
```

## 📝 Common Commands

```bash
# View logs
docker-compose -f docker-compose.dev.yml logs -f jadevectordb-core

# Stop all services
docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes (⚠️ deletes data)
docker-compose -f docker-compose.dev.yml down -v

# Rebuild from scratch
docker-compose -f docker-compose.dev.yml build --no-cache

# Check running containers
docker ps | grep jadevectordb

# Inspect container
docker inspect jadevectordb-core-dev

# Check resource usage
docker stats jadevectordb-core-dev
```

## 🐛 Troubleshooting

### Build Fails

1. **Check Docker resources**: Ensure Docker has at least 4 GB RAM
2. **Clean build**: `docker-compose -f docker-compose.dev.yml build --no-cache`
3. **Check disk space**: `df -h`

### Container Won't Start

1. **Check logs**: `docker logs jadevectordb-core-dev`
2. **Verify volumes**: `docker volume ls | grep jadevectordb`
3. **Check ports**: `netstat -tuln | grep 8080`

### Permission Issues

If you encounter permission issues with volumes:

```bash
# Fix volume permissions
docker exec -u root jadevectordb-core-dev chown -R jadevector:jadevector /data /config
```

## 🔜 Next Steps

1. **Fix REST API Issues**: Complete the server implementation
2. **Add Frontend**: Integrate the React UI from `frontend/` directory
3. **Production Build**: Create optimized production Dockerfile
4. **CI/CD Pipeline**: Set up automated builds and tests
5. **Kubernetes**: Create K8s deployment manifests
6. **Distributed Mode**: Use `docker-compose.distributed.yml` for multi-node setup

## 📚 Additional Resources

- [Main README](./README.md)
- [Backend Build Guide](./backend/README.md)
- [API Documentation](./docs/api/)
- [Architecture Docs](./docs/architecture/)

## 🤝 Contributing

When contributing Docker-related changes:

1. Test with `docker-compose.dev.yml` first
2. Update this documentation
3. Ensure backward compatibility
4. Add comments to Dockerfiles
5. Update .dockerignore if needed

## 📄 License

See [LICENSE](./LICENSE) file for details.

---

**Last Updated**: 2025-10-26
**Version**: 1.0.0-dev
**Status**: Core Library Functional, REST API In Progress
