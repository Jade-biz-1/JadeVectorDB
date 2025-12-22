# JadeVectorDB Backend - Build System

## 🎯 Quick Start

```bash
# One command to build everything
./build.sh
```

That's it! All dependencies are automatically fetched from source.

**Run the server:**
```bash
cd build
./jadevectordb
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

## ✨ Key Features

- ✅ **Self-Contained** - All dependencies built from source via CMake FetchContent
- ✅ **No Installation** - No apt-get, yum, or brew needed
- ✅ **Consistent** - Same build everywhere (local, Docker, CI/CD)
- ✅ **Fast** - Incremental builds in ~1 minute
- ✅ **Flexible** - Many configuration options

## 📦 What Gets Built

After successful build, find in `build/`:
- `jadevectordb` - Main executable (server)
- `libjadevectordb_core.a` - Core library
- `jadevectordb_tests` - Test suite (if enabled)
- `search_benchmarks` - Performance tests (if enabled)

## ⚡ Performance

| Build Type | First Build | Incremental |
|-----------|-------------|-------------|
| Standard | ~12 minutes | ~1 minute |
| Minimal (no tests) | ~8 minutes | ~30 seconds |
| With gRPC | ~40 minutes | ~1 minute |

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails | Try `./build.sh --clean` |
| Out of memory | Limit jobs: `./build.sh --jobs 2` |
| Slow build | Skip extras: `./build.sh --no-tests --no-benchmarks` |
| Missing dependencies | None needed! Build fetches everything from source |

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

# With persistent data
docker run -p 8080:8080 -v $(pwd)/data:/app/data jadevectordb:latest

# With environment variables
docker run -p 8080:8080 \
  -e SERVER_PORT=8080 \
  -e LOG_LEVEL=INFO \
  jadevectordb:latest
```

## 🔧 Using CMake Directly

If you prefer to use CMake directly instead of the build script:

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

## 📝 Environment Variables

The build script respects these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BUILD_TYPE` | Build type | Release |
| `BUILD_DIR` | Build directory | build |
| `BUILD_TESTS` | Enable tests | ON |
| `BUILD_BENCHMARKS` | Enable benchmarks | ON |
| `BUILD_WITH_GRPC` | Enable gRPC | OFF |
| `PARALLEL_JOBS` | Build jobs | nproc |
| `CLEAN_BUILD` | Clean first | false |

Example:
```bash
BUILD_TYPE=Debug CLEAN_BUILD=true ./build.sh
```

## 🧪 Testing

After building with tests enabled:

```bash
cd build

# Run all tests
ctest --output-on-failure

# Run specific test
./jadevectordb_tests --gtest_filter=DatabaseTest.*

# Run with verbose output
ctest -V
```

## 📖 Additional Documentation

- **Project README**: [../README.md](../README.md) - Project overview
- **Complete Setup Guide**: [../docs/COMPLETE_BUILD_SYSTEM_SETUP.md](../docs/COMPLETE_BUILD_SYSTEM_SETUP.md) - Full build system details
- **Installation Guide**: [../docs/INSTALLATION_GUIDE.md](../docs/INSTALLATION_GUIDE.md) - Installation instructions

## 💡 Tips

1. **First Build**: Be patient, first build takes ~12 minutes as it fetches and compiles all dependencies
2. **Incremental Builds**: After first build, rebuilds are typically ~1 minute
3. **Clean Builds**: Use `--clean` only when needed (dependency changes, build issues)
4. **CI/CD**: Limit jobs with `--jobs 2` to avoid memory issues
5. **Development**: Use Debug builds for better error messages and debugging

## 🚀 Quick Examples

```bash
# Standard development workflow
./build.sh --type Debug --clean
cd build && ./jadevectordb

# Production deployment
./build.sh --no-tests --no-benchmarks
cd build && ./jadevectordb

# Test coverage report
./build.sh --type Debug --coverage --clean
cd build && cmake --build . --target coverage

# Fast iteration (no tests/benchmarks)
./build.sh --no-tests --no-benchmarks --jobs 4
```

## ❓ Getting Help

```bash
# Show all build script options
./build.sh --help

# Check CMake configuration
cd build && cmake -L ..

# View build variables
cd build && cmake -LAH ..
```

For more help, see the main project [README.md](../README.md) or open an issue on GitHub.
