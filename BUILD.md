# JadeVectorDB Build System

## Overview

JadeVectorDB uses a unified, self-contained build system that works consistently across all environments:
- Local development
- Docker containers
- CI/CD pipelines
- Production deployments

**Key Feature:** All dependencies are fetched from source via CMake FetchContent. No external package installation required!

## Quick Start

### Local Build

```bash
cd backend
./build.sh
```

The executable will be in `backend/build/jadevectordb`.

**Note:** The build script must be run from the `backend/` directory.

### Docker Build

```bash
# From project root
docker build -t jadevectordb:latest .
```

### Run

```bash
# Local
cd backend/build
./jadevectordb

# Docker
docker run -p 8080:8080 jadevectordb:latest
```

## Build Options

### Using the Build Script

```bash
./build.sh [OPTIONS]

Options:
  --help                  Show help message
  --clean                 Perform clean build (removes build directory)
  --type TYPE             Build type: Debug, Release, RelWithDebInfo (default: Release)
  --dir DIR               Build directory (default: build)
  --no-tests              Disable building tests
  --no-benchmarks         Disable building benchmarks
  --with-grpc             Enable full gRPC support (increases build time significantly)
  --coverage              Enable code coverage instrumentation
  --jobs N                Number of parallel build jobs (default: nproc)
  --verbose               Enable verbose build output
```

### Examples

```bash
# Standard release build
./build.sh

# Debug build with clean
./build.sh --type Debug --clean

# Production build (no tests/benchmarks, optimized)
./build.sh --no-tests --no-benchmarks

# Development build with full features
./build.sh --type Debug --with-grpc --clean

# Coverage build for testing
./build.sh --type Debug --coverage --clean

# Fast build with limited parallelism (for CI)
./build.sh --jobs 2
```

### Using CMake Directly

```bash
# Navigate to backend directory
cd backend

# Configure
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_WITH_GRPC=OFF

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Dependencies

All dependencies are automatically fetched from source during the CMake configure step:

### Core Dependencies (Always Built)
- **Eigen 3.4.0** - Linear algebra library
- **FlatBuffers v23.5.26** - Serialization
- **Crow v1.0+5** - Web framework for REST API
- **Google Test v1.14.0** - Testing framework (if BUILD_TESTS=ON)
- **Google Benchmark v1.8.3** - Performance testing (if BUILD_BENCHMARKS=ON)
- **Apache Arrow 14.0.0** - In-memory columnar format

### Optional Dependencies
- **gRPC v1.60.0** - RPC framework (only if BUILD_WITH_GRPC=ON)
  - Note: This is VERY large and significantly increases build time
  - Default: OFF (uses stub implementation)

## Build System Features

### Self-Contained
- No external package manager required (apt, yum, brew, etc.)
- All dependencies fetched from source
- Reproducible builds across all platforms

### Flexible Configuration
- CMake options for all features
- Environment variable support
- Command-line argument support

### Optimized Builds
- Minimal Arrow build (only required components)
- Optional gRPC (uses stubs by default)
- Parallel compilation support
- Incremental builds supported

### Docker Integration
- Multi-stage builds for small images
- Uses same build system as local
- No dependency installation in container
- Optimized for CI/CD

## Build Configurations

### Development Build
Best for local development and debugging:
```bash
./build.sh --type Debug --clean
```
- Debug symbols enabled
- Optimizations disabled
- Tests and benchmarks included

### Release Build (Default)
Optimized for production:
```bash
./build.sh --type Release
```
- Full optimizations (-O3)
- Debug symbols stripped
- Tests and benchmarks included

### Production Build
Minimal size, maximum performance:
```bash
./build.sh --type Release --no-tests --no-benchmarks
```
- Full optimizations
- No tests or benchmarks
- Smallest binary size

### Coverage Build
For test coverage analysis:
```bash
./build.sh --type Debug --coverage --clean
cd backend/build
cmake --build . --target coverage
```
- Coverage instrumentation
- Generates HTML coverage report

## Troubleshooting

### Build Fails with "Out of Memory"

Arrow build can be memory-intensive. Solutions:
```bash
# Reduce parallel jobs
./build.sh --jobs 2

# Or build incrementally
cmake --build backend/build -j1
```

### Git Fetch Fails

Ensure you have internet access and Git is installed:
```bash
git --version
# Should output: git version 2.x.x
```

### Missing Compiler

Install GCC 11+ or Clang 14+:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y build-essential cmake git

# macOS
brew install cmake git
```

### Slow Build

First build fetches and compiles all dependencies (can take 10-30 minutes).
Subsequent builds are much faster (incremental compilation).

To speed up:
```bash
# Skip tests and benchmarks
./build.sh --no-tests --no-benchmarks

# Use more CPU cores
./build.sh --jobs 8

# Don't build gRPC (uses stubs instead)
# This is the default, but explicitly:
cd backend && cmake -B build -DBUILD_WITH_GRPC=OFF
```

### Known Build Issues

#### Test Compilation Notes
Most test files now compile and the current automated test suite reports **26/26 passing**. If you run into test compilation issues in specific test files (rare), you can skip tests for a fast build:

```bash
./build.sh --no-tests --no-benchmarks
```
See `TasksTracking/SPRINT_2_3_TEST_RESULTS.md` and `CleanupReport.md` for details on test coverage and results.

#### Runtime Startup Issues (Address Already in Use)
If the server fails to start due to port conflicts:
```bash
# Check what's using port 8080
lsof -i :8080

# Use a different port
export JDB_PORT=8081
./jadevectordb
```

#### Runtime Crash: Duplicate Route Handlers (Resolved)
An earlier issue caused the application to crash on startup with the message `handler already exists for /v1/databases` due to duplicate route registrations in `rest_api.cpp`.

**Cause**: Routes were registered both via `app_->route_dynamic()` and `CROW_ROUTE()` for the same endpoints, introduced during integration.

**Status**: **Resolved (fixed 2025-12-12)** — the duplicate registration was removed and the startup crash has been addressed. If you still encounter this error, ensure you are on the latest `run-and-fix` branch and rebuild the project.

**Note**: The application should start normally after building; if problems persist, consult `docs/LOCAL_DEPLOYMENT.md` and `docs/TROUBLESHOOTING_GUIDE.md` for troubleshooting steps.

## Environment Variables

You can control the build using environment variables:

```bash
# Build configuration
export BUILD_TYPE=Release           # Debug, Release, RelWithDebInfo
export BUILD_DIR=build              # Build directory name
export BUILD_TESTS=ON               # ON or OFF
export BUILD_BENCHMARKS=ON          # ON or OFF
export BUILD_COVERAGE=OFF           # ON or OFF
export BUILD_WITH_GRPC=OFF          # ON or OFF
export CLEAN_BUILD=false            # true or false
export PARALLEL_JOBS=4              # Number of parallel jobs
export VERBOSE=false                # true or false

# Run build
cd backend && ./build.sh
```

## Docker Build Options

### Development Image (with tests)
```bash
docker build \
  --build-arg BUILD_TESTS=ON \
  --build-arg BUILD_BENCHMARKS=ON \
  -t jadevectordb:dev .
```

### Production Image (minimal)
```bash
docker build \
  --build-arg BUILD_TESTS=OFF \
  --build-arg BUILD_BENCHMARKS=OFF \
  -t jadevectordb:prod .
```

### With gRPC Support
```bash
docker build \
  --build-arg BUILD_WITH_GRPC=ON \
  -t jadevectordb:grpc .
```

## Build Artifacts

After a successful build, you'll find:

```
backend/build/
├── jadevectordb              # Main executable
├── libjadevectordb_core.a    # Core static library
├── jadevectordb_tests        # Test suite (if BUILD_TESTS=ON)
├── search_benchmarks         # Benchmarks (if BUILD_BENCHMARKS=ON)
└── filtered_search_benchmarks
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Build JadeVectorDB
  run: |
    cd backend
    ./build.sh --type Release --no-benchmarks --jobs 2
```

### GitLab CI
```yaml
build:
  script:
    - cd backend
    - ./build.sh --type Release --no-benchmarks
```

### Jenkins
```groovy
sh 'cd backend && ./build.sh --type Release'
```

## Performance

### Build Times (approximate)

| Configuration | First Build | Incremental |
|--------------|-------------|-------------|
| Release (no gRPC) | 10-15 min | 1-2 min |
| Debug (no gRPC) | 8-12 min | 1-2 min |
| With gRPC | 30-45 min | 1-2 min |
| Production (minimal) | 8-10 min | 30 sec |

*Times vary based on CPU cores and internet speed*

### Optimizations

The build system includes several optimizations:
- **GIT_SHALLOW TRUE** - Fetches only latest commits
- **Minimal Arrow** - Builds only required Arrow components
- **Static linking** - Reduces runtime dependencies
- **Parallel compilation** - Uses all available CPU cores
- **Incremental builds** - Only recompiles changed files

## Support

For build issues:
1. Check this documentation and the quick reference guide in `backend/BUILD_QUICK_REFERENCE.md`
2. Review build logs
3. Try a clean build: `cd backend && ./build.sh --clean`
4. Open an issue on GitHub with:
   - CMake version
   - Compiler version
   - Build command used
   - Complete error output

## Additional Documentation

- **Quick Reference:** `backend/BUILD_QUICK_REFERENCE.md` - Common commands and examples
- **Getting Started:** `backend/README_BUILD.md` - Simple introduction to the build system
- **Detailed Backend Guide:** `backend/BUILD.md` - Backend-specific build documentation
- **Build System Overview:** `docs/COMPLETE_BUILD_SYSTEM_SETUP.md` - Complete build system setup and features