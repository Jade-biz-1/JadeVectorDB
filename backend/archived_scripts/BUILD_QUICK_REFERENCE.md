# JadeVectorDB Build System - Quick Reference

## 🚀 Quick Start

```bash
# Local build
cd backend && ./build.sh

# Docker build
docker build -f Dockerfile -t jadevectordb .

# Run
./build/jadevectordb  # or: docker run -p 8080:8080 jadevectordb
```

## 📋 Common Commands

### Local Builds

| Task | Command |
|------|---------|
| **Standard build** | `./build.sh` |
| **Clean build** | `./build.sh --clean` |
| **Debug build** | `./build.sh --type Debug --clean` |
| **Production** | `./build.sh --no-tests --no-benchmarks` |
| **Fast build** | `./build.sh --jobs 8` |
| **With tests** | `./build.sh` (tests ON by default) |
| **Coverage** | `./build.sh --type Debug --coverage --clean` |

### Docker Builds

| Task | Command |
|------|---------|
| **Standard** | `docker build -f Dockerfile -t jadevectordb .` |
| **Production** | `docker build -f Dockerfile --build-arg BUILD_TESTS=OFF -t jadevectordb:prod .` |
| **Development** | `docker build -f Dockerfile --build-arg BUILD_TYPE=Debug -t jadevectordb:dev .` |

## ⚙️ Build Options

```bash
./build.sh [OPTIONS]

--clean                 # Remove build directory first
--type TYPE             # Debug | Release | RelWithDebInfo
--dir DIR               # Custom build directory
--no-tests              # Skip tests
--no-benchmarks         # Skip benchmarks
--with-grpc             # Enable gRPC (slow!)
--coverage              # Code coverage
--jobs N                # Parallel jobs
--verbose               # Verbose output
```

## 🎯 Use Cases

### For Development
```bash
./build.sh --type Debug --clean
```

### For Production Deployment
```bash
./build.sh --no-tests --no-benchmarks
```

### For CI/CD
```bash
./build.sh --type Release --jobs 2
```

### For Testing Coverage
```bash
./build.sh --type Debug --coverage --clean
cd build && cmake --build . --target coverage
```

### For Performance Testing
```bash
./build.sh --type Release
cd build && ./search_benchmarks
```

## 🐳 Docker Commands

### Build
```bash
# Production image (recommended)
docker build -f Dockerfile -t jadevectordb:latest .

# Development image
docker build -f Dockerfile \
  --build-arg BUILD_TYPE=Debug \
  --build-arg BUILD_TESTS=ON \
  -t jadevectordb:dev .
```

### Run
```bash
# Simple run
docker run -p 8080:8080 jadevectordb:latest

# With volume mounts
docker run -d \
  --name jadevectordb \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  jadevectordb:latest

# Interactive (for debugging)
docker run -it --rm jadevectordb:latest /bin/bash
```

### Manage
```bash
# Stop
docker stop jadevectordb

# Remove
docker rm jadevectordb

# Logs
docker logs -f jadevectordb

# Shell access
docker exec -it jadevectordb /bin/bash
```

## 📊 Build Artifacts

After build, find these in `build/`:

- `jadevectordb` - Main executable
- `libjadevectordb_core.a` - Core library
- `jadevectordb_tests` - Test suite (if enabled)
- `search_benchmarks` - Benchmarks (if enabled)

## 🔧 Environment Variables

```bash
export BUILD_TYPE=Release
export BUILD_TESTS=OFF
export BUILD_BENCHMARKS=OFF
export PARALLEL_JOBS=4

./build.sh
```

## ⏱️ Build Times

| Configuration | First Build | Incremental |
|--------------|-------------|-------------|
| Standard (no gRPC) | ~12 min | ~1 min |
| With gRPC | ~40 min | ~1 min |
| Minimal | ~8 min | ~30 sec |

*Times on 4-core CPU with good internet*

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `./build.sh --jobs 2` |
| Slow build | `./build.sh --no-tests --no-benchmarks` |
| Clean needed | `./build.sh --clean` |
| CMake issues | `rm -rf build && ./build.sh` |

## 📚 More Information

- Full documentation: [BUILD.md](BUILD.md)
- Architecture: [../docs/architecture.md](../docs/architecture.md)
- API docs: [../docs/api_documentation.md](../docs/api_documentation.md)
- Build system overview: [../docs/COMPLETE_BUILD_SYSTEM_SETUP.md](../docs/COMPLETE_BUILD_SYSTEM_SETUP.md)

## ✅ Self-Contained Build

**No package installation needed!**
- All dependencies fetched from source
- Works in Docker without apt-get
- Reproducible builds everywhere
- Same build system for local, Docker, and CI/CD
