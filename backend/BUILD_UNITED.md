# JadeVectorDB Build System - United Documentation

## Overview

JadeVectorDB uses a unified, self-contained build system that works consistently across all environments:
- Local development
- Docker containers
- CI/CD pipelines
- Production deployments

**Key Feature:** All dependencies are fetched from source via CMake FetchContent. No external package installation required!

## 🎯 Quick Start

```bash
# One command to build everything
./build.sh
```

That's it! All dependencies are automatically fetched from source.

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

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--type TYPE` | Debug, Release, RelWithDebInfo | Release |
| `--clean` | Remove build directory first | false |
| `--no-tests` | Skip building tests | tests enabled |
| `--no-benchmarks` | Skip building benchmarks | benchmarks enabled |
| `--with-grpc` | Enable full gRPC (adds 30min!) | OFF (uses stubs) |
| `--coverage` | Code coverage instrumentation | OFF |
| `--jobs N` | Parallel build jobs | all CPUs |
| `--verbose` | Verbose output | quiet |

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
cd build
cmake --build . --target coverage
```
- Coverage instrumentation
- Generates HTML coverage report

## 📦 What Gets Built

After successful build, find in `build/`:
- `jadevectordb` - Main executable
- `libjadevectordb_core.a` - Core library
- `jadevectordb_tests` - Test suite (if enabled)
- `search_benchmarks` - Performance tests (if enabled)

## 🔧 Using CMake Directly

```bash
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

## 🔧 Environment Variables

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
./build.sh
```

## ⚡ Performance

| Build Type | First Build | Incremental |
|-----------|-------------|-------------|
| Standard | ~12 minutes | ~1 minute |
| Minimal | ~8 minutes | ~30 seconds |
| With gRPC | ~40 minutes | ~1 minute |

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

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails | `./build.sh --clean` |
| Out of memory | `./build.sh --jobs 2` |
| Slow build | `./build.sh --no-tests --no-benchmarks` |
| Clean needed | `./build.sh --clean` |
| CMake issues | `rm -rf build && ./build.sh` |

### Build Fails with "Out of Memory"

Arrow build can be memory-intensive. Solutions:
```bash
# Reduce parallel jobs
./build.sh --jobs 2

# Or build incrementally
cmake --build build -j1
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
cmake -B build -DBUILD_WITH_GRPC=OFF
```

## ✨ Key Features

- ✅ **Self-Contained** - All dependencies built from source
- ✅ **No Installation** - No apt-get, yum, or brew needed
- ✅ **Consistent** - Same build everywhere (local, Docker, CI/CD)
- ✅ **Fast** - Incremental builds in ~1 minute
- ✅ **Flexible** - Many configuration options

## 📊 Build Artifacts

After build, find these in `build/`:

- `jadevectordb` - Main executable
- `libjadevectordb_core.a` - Core library
- `jadevectordb_tests` - Test suite (if enabled)
- `search_benchmarks` - Benchmarks (if enabled)

## 🚦 CI/CD Integration

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

## 📚 Additional Documentation

- Full documentation: [BUILD.md](BUILD.md)
- Architecture: [../docs/architecture.md](../docs/architecture.md)
- API docs: [../docs/api_documentation.md](../docs/api_documentation.md)
- Build system enhancement: [../docs/BUILD_SYSTEM_ENHANCEMENT_SUMMARY.md](../docs/BUILD_SYSTEM_ENHANCEMENT_SUMMARY.md)
- Quick Reference: [BUILD_QUICK_REFERENCE.md](BUILD_QUICK_REFERENCE.md) - Common commands and examples
- Complete Guide: [BUILD.md](BUILD.md) - Full documentation with troubleshooting
- Enhancement Summary: [../docs/BUILD_SYSTEM_ENHANCEMENT_SUMMARY.md](../docs/BUILD_SYSTEM_ENHANCEMENT_SUMMARY.md) - Technical details

## 📖 More Help

```bash
./build.sh --help
```

For detailed information, see [BUILD.md](BUILD.md).

## Support

For build issues:
1. Check this documentation and the quick reference guide in `BUILD_QUICK_REFERENCE.md`
2. Review build logs
3. Try a clean build: `./build.sh --clean`
4. Open an issue on GitHub with:
   - CMake version
   - Compiler version
   - Build command used
   - Complete error output

## ✅ Self-Contained Build

**No package installation needed!**
- All dependencies fetched from source
- Works in Docker without apt-get
- Reproducible builds everywhere
- Same build system for local, Docker, and CI/CD