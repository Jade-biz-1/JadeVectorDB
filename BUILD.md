# JadeVectorDB Build Guide

**Quick Start:** `cd backend && ./build.sh`

---

## Quick Build Commands

```bash
# Standard build (recommended)
cd backend && ./build.sh

# Production build (optimized, no tests/benchmarks)
./build.sh --no-tests --no-benchmarks

# Debug build for development
./build.sh --type Debug --clean

# Fast rebuild after changes
./build.sh
```

**Build Output:**
- `build/jadevectordb` - Main executable (3.1 MB)
- `build/libjadevectordb_core.a` - Static library (5.9 MB)

---

## System Requirements

**Required Tools:**
- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+)
- Git (for dependency fetching)
- 4+ GB RAM
- 2+ GB disk space

**Installation (Ubuntu/Debian):**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git ninja-build pkg-config
```

---

## Build Options

### Command Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--clean` | Clean build (removes build directory) | Off |
| `--type TYPE` | Build type: Debug, Release, RelWithDebInfo | Release |
| `--no-tests` | Skip building tests | Off |
| `--no-benchmarks` | Skip building benchmarks | Off |
| `--with-grpc` | Enable full gRPC support (slow!) | Off |
| `--jobs N` | Number of parallel jobs | CPU count |
| `--verbose` | Show detailed build output | Off |
| `--help` | Show all options | - |

### Environment Variables

Set these before running `build.sh`:

```bash
export BUILD_TYPE=Debug          # Build type
export BUILD_TESTS=OFF           # Skip tests
export PARALLEL_JOBS=4           # Parallel jobs
export CLEAN_BUILD=true          # Clean build
```

---

## Build System Architecture

### Self-Contained Dependencies

All dependencies are fetched from source via CMake `FetchContent`:

| Dependency | Version | Purpose | Build Time |
|-----------|---------|---------|------------|
| Eigen | 3.4.0 | Linear algebra | ~30s |
| FlatBuffers | v23.5.26 | Serialization | ~45s |
| Crow | v1.0+5 | REST API | ~20s |
| Google Test | v1.14.0 | Testing (optional) | ~30s |
| Google Benchmark | v1.8.3 | Benchmarks (optional) | ~25s |
| Apache Arrow | 14.0.0 | In-memory format | ~3min |
| gRPC | v1.60.0 | RPC (optional) | ~30min |

**Benefits:**
- ✅ No system package installation required
- ✅ Reproducible builds across all platforms
- ✅ Same versions on local, Docker, and CI/CD
- ✅ No dependency conflicts

### Build Process

```
1. CMake configures project
   ↓
2. FetchContent downloads dependencies (first build only)
   ↓
3. Dependencies built from source (cached)
   ↓
4. JadeVectorDB core library compiled
   ↓
5. Main executable linked
```

---

## Build Times

| Configuration | First Build | Incremental | Notes |
|---------------|-------------|-------------|-------|
| Standard (Release) | ~8-12 min | ~30 sec | Recommended |
| Debug | ~6-10 min | ~30 sec | Development |
| Minimal (no tests) | ~5-8 min | ~20 sec | Production |
| With gRPC | ~35-45 min | ~30 sec | Rarely needed |

*Times based on 4-core CPU with SSD*

---

## Docker Build

### Build Image

```bash
# Standard build
docker build -f Dockerfile -t jadevectordb:latest .

# Production build
docker build -f Dockerfile \
  --build-arg BUILD_TESTS=OFF \
  --build-arg BUILD_BENCHMARKS=OFF \
  -t jadevectordb:prod .
```

### Run Container

```bash
docker run -d \
  --name jadevectordb \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  jadevectordb:latest
```

### Docker Image Sizes

- Builder stage: ~2 GB (includes all build artifacts)
- Runtime stage: ~100 MB (minimal, production-ready)

---

## Common Issues & Solutions

### Issue: Build takes too long

**Solution:** Skip optional components
```bash
./build.sh --no-tests --no-benchmarks --jobs 2
```

### Issue: Out of memory during build

**Solution:** Reduce parallel jobs
```bash
./build.sh --jobs 2
```

### Issue: CMake configuration errors

**Solution:** Clean build
```bash
./build.sh --clean
```

### Issue: Linking errors with gRPC

**Note:** gRPC is optional and disabled by default. The project uses stubs to build without gRPC. Only enable with `--with-grpc` if you need full gRPC support.

### Issue: Build fails after git pull

**Solution:** Clean rebuild
```bash
rm -rf build && mkdir build && cd build
cmake .. && cmake --build . -j$(nproc)
```

---

## Development Workflow

### Standard Workflow

```bash
# 1. Clone repository
git clone <repo-url>
cd JadeVectorDB/backend

# 2. First build (fetches dependencies)
./build.sh --type Debug

# 3. Make changes to code
vim src/main.cpp

# 4. Rebuild (incremental, fast)
./build.sh

# 5. Run executable
./build/jadevectordb

# 6. Run tests (if built)
cd build && ctest --output-on-failure
```

### Testing Workflow

```bash
# Build with tests
./build.sh --type Debug

# Run all tests
cd build && ctest

# Run specific test
./jadevectordb_tests --gtest_filter="DatabaseServiceTest.*"

# Run with verbose output
ctest --output-on-failure --verbose
```

---

## Code Patterns

### Error Handling

This codebase uses `tl::expected` for error handling:

```cpp
#include "lib/error_handling.h"

Result<Vector> get_vector(const std::string& id) {
    if (id.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "ID cannot be empty");
    }
    Vector vec = /* ... */;
    return vec;  // Success
}

// Usage
auto result = get_vector("vec123");
if (result.has_value()) {
    Vector vec = result.value();
    // Use vec
} else {
    LOG_ERROR(logger_, ErrorHandler::format_error(result.error()));
}
```

### Logging

```cpp
#include "lib/logging.h"

class MyService {
private:
    std::shared_ptr<logging::Logger> logger_;

public:
    MyService() {
        logger_ = logging::LoggerManager::get_logger("MyService");
    }

    void some_method() {
        LOG_DEBUG(logger_, "Debug message");
        LOG_INFO(logger_, "Info message");
        LOG_WARN(logger_, "Warning");
        LOG_ERROR(logger_, "Error occurred");
    }
};
```

### Singleton Pattern

```cpp
// AuthManager is a singleton
auto* auth = AuthManager::get_instance();  // Don't use make_unique!
auth->some_method();
```

---

## Adding New Services

When creating a new service:

1. ✅ Create header in `src/services/`
2. ✅ Create implementation in `src/services/`
3. ✅ Add .cpp to `CORE_SOURCES` in `CMakeLists.txt`
4. ✅ Include `lib/logging.h` and `lib/error_handling.h`
5. ✅ Use `Result<T>` for error handling
6. ✅ Initialize logger in constructor
7. ✅ Mark mutexes `mutable` if used in const methods

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake git ninja-build

      - name: Build
        run: |
          cd backend
          ./build.sh --type Release --jobs 2

      - name: Run Tests
        run: |
          cd backend/build
          ctest --output-on-failure

      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: jadevectordb
          path: backend/build/jadevectordb
```

---

## Performance Tips

### Faster Builds

```bash
# Use Ninja instead of Make
cmake -G Ninja ..
ninja -j$(nproc)

# Use ccache for faster rebuilds
export CC="ccache gcc"
export CXX="ccache g++"
cmake ..
```

### Runtime Performance

The codebase includes:
- SIMD operations (`src/lib/simd_ops.cpp`)
- GPU acceleration (`src/lib/gpu_acceleration.cpp`)
- Thread pool (`src/lib/thread_pool.cpp`)
- Memory-mapped files (`src/lib/mmap_utils.cpp`)

---

## Project Structure

```
JadeVectorDB/
├── backend/
│   ├── build.sh                 # Build script
│   ├── CMakeLists.txt           # Build configuration
│   ├── src/
│   │   ├── main.cpp             # Application entry point
│   │   ├── models/              # Data models
│   │   ├── services/            # Core services
│   │   ├── api/                 # REST & gRPC APIs
│   │   └── lib/                 # Utility libraries
│   └── tests/                   # Test files
├── Dockerfile                   # Docker build
├── docker-compose*.yml          # Docker Compose configs
└── docs/                        # Documentation
```

---

## Getting Help

**Build Issues:**
1. Try `./build.sh --clean`
2. Check logs in `build/CMakeFiles/CMakeError.log`
3. Search this document for your error message
4. Open GitHub issue with full build logs

**Runtime Issues:**
1. Check application logs
2. Verify configuration files
3. See documentation in `docs/`

---

## Summary

**Quick Start:**
```bash
cd backend && ./build.sh
./build/jadevectordb
```

**Key Points:**
- ✅ All dependencies built from source (no apt-get needed)
- ✅ Same build process for local, Docker, and CI/CD
- ✅ Fast incremental builds (~30 seconds)
- ✅ Production-ready binaries (3.1 MB executable + 5.9 MB library)
- ✅ Comprehensive testing and benchmarking support

**Build Time:** 8-12 minutes (first build), ~30 seconds (incremental)

**For More Details:** See documentation in `docs/` directory

---

*Last Updated: November 2, 2025*
*Build System Version: 2.0*
