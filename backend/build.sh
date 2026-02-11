#!/bin/bash

#############################################################################
# JadeVectorDB Unified Build Script
#############################################################################
# This script provides a consistent build process for:
# - Local development builds
# - Docker container builds
# - CI/CD pipeline builds
# - Service builds
#
# All dependencies are fetched from source via CMake FetchContent
# No external package installation required!
#############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default values
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
BUILD_BENCHMARKS="${BUILD_BENCHMARKS:-ON}"
BUILD_COVERAGE="${BUILD_COVERAGE:-OFF}"
BUILD_WITH_GRPC="${BUILD_WITH_GRPC:-OFF}"
CLEAN_BUILD="${CLEAN_BUILD:-false}"
# Detect number of CPU cores (cross-platform)
if command -v nproc >/dev/null 2>&1; then
    PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"
elif command -v sysctl >/dev/null 2>&1; then
    PARALLEL_JOBS="${PARALLEL_JOBS:-$(sysctl -n hw.ncpu)}"
else
    PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
fi
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         JadeVectorDB Unified Build System                 ║"
    echo "║         High-Performance Vector Database                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

print_config() {
    echo "Build Configuration:"
    echo "  Build Type:       ${BUILD_TYPE}"
    echo "  Build Directory:  ${BUILD_DIR}"
    echo "  Build Tests:      ${BUILD_TESTS}"
    echo "  Build Benchmarks: ${BUILD_BENCHMARKS}"
    echo "  Build Coverage:   ${BUILD_COVERAGE}"
    echo "  Build with gRPC:  ${BUILD_WITH_GRPC}"
    echo "  Parallel Jobs:    ${PARALLEL_JOBS}"
    echo "  Clean Build:      ${CLEAN_BUILD}"
    echo ""
}

detect_platform() {
    PLATFORM="$(uname -s)"
    case "${PLATFORM}" in
        Linux*)     PLATFORM_NAME="Linux";;
        Darwin*)    PLATFORM_NAME="macOS";;
        CYGWIN*|MINGW*|MSYS*) PLATFORM_NAME="Windows";;
        *)          PLATFORM_NAME="Unknown";;
    esac
    log_info "Detected platform: ${PLATFORM_NAME} ($(uname -m))"
}

setup_macos_deps() {
    if [ "${PLATFORM_NAME}" != "macOS" ]; then
        return
    fi

    log_info "Setting up macOS dependency paths..."

    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        log_error "Homebrew is not installed. Please install it from https://brew.sh"
        exit 1
    fi

    HOMEBREW_PREFIX=$(brew --prefix)
    log_success "Found Homebrew at ${HOMEBREW_PREFIX}"

    # Check and hint for required dependencies
    local missing_deps=()

    if ! brew list openssl &> /dev/null; then
        missing_deps+=("openssl")
    fi
    if ! brew list boost &> /dev/null; then
        missing_deps+=("boost")
    fi
    if ! brew list sqlite3 &> /dev/null; then
        missing_deps+=("sqlite3")
    fi
    if ! brew list asio &> /dev/null; then
        missing_deps+=("asio")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing Homebrew dependencies: ${missing_deps[*]}"
        log_info "Install them with: brew install ${missing_deps[*]}"
        exit 1
    fi

    # Set environment hints for CMake
    export OPENSSL_ROOT_DIR="${HOMEBREW_PREFIX}/opt/openssl@3"
    log_success "OpenSSL: ${OPENSSL_ROOT_DIR}"
    log_success "Boost: ${HOMEBREW_PREFIX}/opt/boost"
    log_success "SQLite3: ${HOMEBREW_PREFIX}/opt/sqlite"
}

check_requirements() {
    log_info "Checking build requirements..."

    # Detect platform first
    detect_platform

    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake is not installed. Please install CMake 3.20 or later."
        if [ "${PLATFORM_NAME}" = "macOS" ]; then
            log_info "Install via: brew install cmake"
        elif [ "${PLATFORM_NAME}" = "Linux" ]; then
            log_info "Install via: sudo apt-get install cmake"
        fi
        exit 1
    fi

    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    log_success "Found CMake ${CMAKE_VERSION}"

    # Check for C++ compiler
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        log_error "No C++ compiler found. Please install GCC 11+ or Clang 14+."
        if [ "${PLATFORM_NAME}" = "macOS" ]; then
            log_info "Install via: xcode-select --install"
        fi
        exit 1
    fi

    if command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -n1)
        log_success "Found ${CLANG_VERSION}"
    elif command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1)
        log_success "Found ${GCC_VERSION}"
    fi

    # Check for Git (needed for FetchContent)
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Git is required to fetch dependencies."
        exit 1
    fi
    log_success "Found Git $(git --version | cut -d' ' -f3)"

    # Setup platform-specific dependencies
    setup_macos_deps

    echo ""
}

clean_build_directory() {
    if [ "${CLEAN_BUILD}" = "true" ]; then
        log_info "Performing clean build..."
        if [ -d "${BUILD_DIR}" ]; then
            log_warning "Removing existing build directory: ${BUILD_DIR}"
            rm -rf "${BUILD_DIR}"
        fi
    fi
}

configure_project() {
    log_info "Configuring CMake project..."

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
        -DBUILD_TESTS="${BUILD_TESTS}"
        -DBUILD_BENCHMARKS="${BUILD_BENCHMARKS}"
        -DBUILD_COVERAGE="${BUILD_COVERAGE}"
        -DBUILD_WITH_GRPC="${BUILD_WITH_GRPC}"
    )

    if [ "${VERBOSE}" = "true" ]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi

    cmake "${CMAKE_ARGS[@]}" ..

    log_success "CMake configuration complete"
    echo ""
}

build_project() {
    log_info "Building JadeVectorDB..."
    log_info "Using ${PARALLEL_JOBS} parallel jobs"

    BUILD_ARGS=(--build . --config "${BUILD_TYPE}" -j "${PARALLEL_JOBS}")

    if [ "${VERBOSE}" = "true" ]; then
        BUILD_ARGS+=(--verbose)
    fi

    cmake "${BUILD_ARGS[@]}"

    echo ""
    log_success "Build completed successfully!"
}

run_tests() {
    if [ "${BUILD_TESTS}" = "ON" ]; then
        log_info "Running tests..."
        ctest --output-on-failure -C "${BUILD_TYPE}" || {
            log_warning "Some tests failed. Check output above."
            return 1
        }
        log_success "All tests passed!"
        echo ""
    fi
}

print_artifacts() {
    log_info "Build artifacts:"

    if [ -f "jadevectordb" ]; then
        EXECUTABLE_SIZE=$(du -h jadevectordb | cut -f1)
        echo "  Main executable:     jadevectordb (${EXECUTABLE_SIZE})"
    fi

    if [ -f "libjadevectordb_core.a" ]; then
        LIB_SIZE=$(du -h libjadevectordb_core.a | cut -f1)
        echo "  Core library:        libjadevectordb_core.a (${LIB_SIZE})"
    fi

    if [ "${BUILD_TESTS}" = "ON" ] && [ -f "jadevectordb_tests" ]; then
        TEST_SIZE=$(du -h jadevectordb_tests | cut -f1)
        echo "  Test suite:          jadevectordb_tests (${TEST_SIZE})"
    fi

    if [ "${BUILD_BENCHMARKS}" = "ON" ] && [ -f "search_benchmarks" ]; then
        BENCH_SIZE=$(du -h search_benchmarks | cut -f1)
        echo "  Benchmarks:          search_benchmarks (${BENCH_SIZE})"
    fi

    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --help                  Show this help message
    --clean                 Perform a clean build (removes build directory)
    --type TYPE             Build type: Debug, Release, RelWithDebInfo (default: Release)
    --dir DIR               Build directory (default: build)
    --no-tests              Disable building tests
    --no-benchmarks         Disable building benchmarks
    --with-grpc             Enable full gRPC support (increases build time)
    --coverage              Enable code coverage instrumentation
    --jobs N                Number of parallel build jobs (default: nproc)
    --verbose               Enable verbose build output

Environment Variables:
    BUILD_TYPE              Build type (overridden by --type)
    BUILD_DIR               Build directory (overridden by --dir)
    BUILD_TESTS             Build tests (ON/OFF)
    BUILD_BENCHMARKS        Build benchmarks (ON/OFF)
    BUILD_COVERAGE          Build with coverage (ON/OFF)
    BUILD_WITH_GRPC         Build with gRPC (ON/OFF)
    CLEAN_BUILD             Clean build (true/false)
    PARALLEL_JOBS           Number of parallel jobs
    VERBOSE                 Verbose output (true/false)

Examples:
    # Standard release build
    ./build.sh

    # Debug build with tests
    ./build.sh --type Debug --clean

    # Production build without tests/benchmarks
    ./build.sh --no-tests --no-benchmarks

    # Build with full gRPC support
    ./build.sh --with-grpc

    # Coverage build
    ./build.sh --type Debug --coverage --clean

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            print_usage
            exit 0
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS=OFF
            shift
            ;;
        --with-grpc)
            BUILD_WITH_GRPC=ON
            shift
            ;;
        --coverage)
            BUILD_COVERAGE=ON
            shift
            ;;
        --jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

#############################################################################
# Main Build Process
#############################################################################

main() {
    local start_time=$(date +%s)

    print_banner
    print_config
    check_requirements
    clean_build_directory
    configure_project
    build_project
    run_tests
    print_artifacts

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "╔════════════════════════════════════════════════════════════╗"
    log_success "║  Build completed successfully in ${duration} seconds!            ║"
    log_success "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "To run the application:"
    echo "  cd ${BUILD_DIR} && ./jadevectordb"
    echo ""
}

# Run main function
main
