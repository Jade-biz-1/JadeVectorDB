# Cross-Platform Build Support

## Overview

JadeVectorDB is designed to build and run on multiple platforms with proper dependency management and platform-specific code handling.

## Supported Platforms

### ✅ macOS (Primary Development Platform)
- **Minimum Version**: macOS 10.15 (Catalina)
- **Compiler**: Apple Clang 12.0+ or GCC 11+
- **Architecture**: x86_64, ARM64 (Apple Silicon)

### ✅ Linux
- **Distributions**: Ubuntu 20.04+, Debian 11+, RHEL 8+, Fedora 34+
- **Compiler**: GCC 11+ or Clang 14+
- **Architecture**: x86_64, ARM64

### ✅ Windows
- **Minimum Version**: Windows 10 (1909+)
- **Compiler**: MSVC 2019+ or MinGW-w64 with GCC 11+
- **Architecture**: x86_64

## Platform-Specific Dependencies

### macOS
```bash
# Install via Homebrew (all required)
brew install cmake openssl boost sqlite3 asio

# Optional: For full gRPC support
brew install grpc protobuf
```

### Linux (Ubuntu/Debian)
```bash
# Essential dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libboost-all-dev \
    libasio-dev

# Optional: For full gRPC support
sudo apt-get install -y \
    libgrpc++-dev \
    protobuf-compiler-grpc
```

### Linux (RHEL/Fedora)
```bash
# Essential dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    git \
    openssl-devel \
    zlib-devel \
    sqlite-devel \
    boost-devel \
    asio-devel

# Optional: For full gRPC support
sudo dnf install -y \
    grpc-devel \
    grpc-plugins \
    protobuf-compiler
```

### Windows
```powershell
# Using vcpkg (recommended)
vcpkg install openssl:x64-windows boost:x64-windows sqlite3:x64-windows asio:x64-windows

# Or using Chocolatey for build tools
choco install cmake git

# Build with vcpkg toolchain
cmake -B build -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
```

## Platform-Specific Code Handling

### Compiler Detection
The build system automatically detects the platform using CMake:

```cmake
# Platform detection
if(APPLE)
    # macOS-specific settings
endif()

if(UNIX AND NOT APPLE)
    # Linux-specific settings
endif()

if(WIN32)
    # Windows-specific settings
endif()
```

### C++ Preprocessor Macros
Use these macros for platform-specific code:

```cpp
#ifdef __linux__
    // Linux-specific code
    #include <sys/sysinfo.h>
#elif __APPLE__
    // macOS-specific code
    #include <sys/sysctl.h>
    #include <mach/mach.h>
#elif _WIN32
    // Windows-specific code
    #include <windows.h>
#endif
```

### Example: Memory Information
```cpp
size_t get_available_memory() {
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
#elif __APPLE__
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        vm_size_t page_size;
        host_page_size(mach_host_self(), &page_size);
        return vm_stats.free_count * page_size;
    }
#elif _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullAvailPhys;
    }
#endif
    return 0;
}
```

## Dependency Management

### Fetched from Source (via CMake FetchContent)
These dependencies are automatically downloaded and built:
- **Eigen 3.4.0** - Linear algebra
- **FlatBuffers v23.5.26** - Serialization
- **nlohmann/json v3.11.2** - JSON parsing
- **TartanLlama/expected v1.1.0** - std::expected backport
- **Crow v1.2.1.2** - Web framework
- **Google Test v1.14.0** - Testing framework
- **Google Benchmark v1.8.3** - Performance testing

### System Dependencies (must be installed)
- **OpenSSL 1.1.1+** - Cryptography
- **Boost 1.64+** - Required by Crow for Boost.System/Date_Time
- **ASIO (standalone)** - Async I/O (required by Crow)
- **SQLite3 3.26+** - Database
- **zlib** - Compression
- **Threads** - POSIX threads (pthreads on Linux/macOS, Win32 threads on Windows)

## C++20 Standard Library Features

### Replaced Features
| Deprecated | Current Replacement | Platform |
|------------|---------------------|----------|
| `std::result_of` | `std::invoke_result` | All |
| `boost::thread::shared_mutex` | `std::shared_mutex` | All |
| `boost::optional` | `std::optional` | All (via Crow v1.2+) |

## Build Configuration

### Cross-Platform Build Script
The `build.sh` script automatically detects CPU count:

```bash
# Detect number of CPU cores (cross-platform)
if command -v nproc >/dev/null 2>&1; then
    PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"        # Linux
elif command -v sysctl >/dev/null 2>&1; then
    PARALLEL_JOBS="${PARALLEL_JOBS:-$(sysctl -n hw.ncpu)}"  # macOS
else
    PARALLEL_JOBS="${PARALLEL_JOBS:-4}"               # Fallback
fi
```

### CMake Options
```bash
# Basic options (all platforms)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      ..

# Platform-specific toolchain (example)
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake ..
```

## Testing Cross-Platform Builds

### macOS
```bash
cd backend
./build.sh --clean
./build/jadevectordb
```

### Linux
```bash
cd backend
./build.sh --clean
./build/jadevectordb
```

### Windows (via CMake)
```powershell
cd backend
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
.\Release\jadevectordb.exe
```

## Known Platform Issues

### macOS
- ✅ Apple Silicon (M1/M2) fully supported
- ✅ Rosetta 2 compatibility for x86_64 binaries
- ⚠️ Older Xcode versions may need updated Command Line Tools

### Linux
- ✅ Most distributions work out of the box
- ⚠️ Alpine Linux requires musl-compatible builds
- ⚠️ Older glibc versions may need compatibility flags

### Windows
- ⚠️ MSVC requires `/std:c++20` flag
- ⚠️ MinGW requires careful Boost configuration
- ⚠️ Path separators must use `\\` or `/` consistently

## Troubleshooting

### Issue: Boost not found
**Solution**: Install Boost or set `BOOST_ROOT` environment variable
```bash
export BOOST_ROOT=/path/to/boost
```

### Issue: OpenSSL version mismatch
**Solution**: Set OpenSSL path explicitly
```bash
cmake -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl@3 ..
```

### Issue: Compiler too old
**Solution**: Update compiler or use newer toolchain
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install gcc-11 g++-11
export CXX=g++-11
```

## CI/CD Platform Matrix

### GitHub Actions Example
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    include:
      - os: ubuntu-latest
        compiler: gcc-11
      - os: macos-latest
        compiler: clang
      - os: windows-latest
        compiler: msvc
```

## Future Platform Support

### Planned
- ☐ FreeBSD
- ☐ ARM Linux (Raspberry Pi)
- ☐ Windows ARM64

### Experimental
- ☐ WebAssembly (WASM)
- ☐ Android NDK
- ☐ iOS (limited functionality)

## Contributing

When adding platform-specific code:
1. Use preprocessor macros for platform detection
2. Test on all three major platforms
3. Document platform-specific dependencies
4. Add fallback implementations when possible
5. Update this document with new requirements

## Resources

- [CMake Platform Detection](https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html)
- [C++ Preprocessor Macros](https://en.cppreference.com/w/cpp/preprocessor)
- [Boost Platform Support](https://www.boost.org/doc/libs/1_88_0/more/getting_started/)
