# JadeVectorDB Backend - Build System

## 🎯 Quick Start

```bash
# One command to build everything
./build.sh
```

That's it! All dependencies are automatically fetched from source.

## 📚 Documentation

- **Quick Reference**: [BUILD_QUICK_REFERENCE.md](BUILD_QUICK_REFERENCE.md) - Common commands and examples
- **Complete Guide**: [BUILD.md](BUILD.md) - Full documentation with troubleshooting
- **Build System Overview**: [../docs/COMPLETE_BUILD_SYSTEM_SETUP.md](../docs/COMPLETE_BUILD_SYSTEM_SETUP.md) - Complete setup guide

## 🐳 Docker

```bash
# From project root
docker build -f Dockerfile -t jadevectordb .
docker run -p 8080:8080 jadevectordb
```

## ✨ Key Features

- ✅ **Self-Contained** - All dependencies built from source
- ✅ **No Installation** - No apt-get, yum, or brew needed
- ✅ **Consistent** - Same build everywhere (local, Docker, CI/CD)
- ✅ **Fast** - Incremental builds in ~1 minute
- ✅ **Flexible** - Many configuration options

## 🚀 Common Commands

```bash
# Development build
./build.sh --type Debug --clean

# Production build
./build.sh --no-tests --no-benchmarks

# Fast build (fewer CPU cores)
./build.sh --jobs 2

# With full features
./build.sh --with-grpc
```

## 📦 What Gets Built

After successful build, find in `build/`:
- `jadevectordb` - Main executable
- `libjadevectordb_core.a` - Core library
- `jadevectordb_tests` - Test suite (if enabled)
- `search_benchmarks` - Performance tests (if enabled)

## 🔧 Configuration Options

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

## ⚡ Performance

| Build Type | First Build | Incremental |
|-----------|-------------|-------------|
| Standard | ~12 minutes | ~1 minute |
| Minimal | ~8 minutes | ~30 seconds |
| With gRPC | ~40 minutes | ~1 minute |

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails | `./build.sh --clean` |
| Out of memory | `./build.sh --jobs 2` |
| Slow build | `./build.sh --no-tests --no-benchmarks` |

## 📖 More Help

```bash
./build.sh --help
```

For detailed information, see [BUILD.md](BUILD.md).
