#!/bin/bash
# Standalone build script for Sprint 2.2 integration tests (Service Layer)

set -e

echo "Building Sprint 2.2 Service Integration Tests..."

cd "$(dirname "$0")"

# Ensure main library is built
if [ ! -f "build/libjadevectordb_core.a" ]; then
    echo "Error: Main library not built. Run ./build.sh --no-tests --no-benchmarks first."
    exit 1
fi

# Create test build directory
mkdir -p build/sprint22_tests
cd build/sprint22_tests

# Compile service-based test files
echo "Compiling test_compaction_service.cpp..."
g++ -std=c++20 -I../../src \
    -I../_deps/eigen-src \
    -I../_deps/nlohmann_json-src/include \
    -I../_deps/crow-src/include \
    -I../_deps/googletest-src/googletest/include \
    -I../_deps/googletest-src/googlemock/include \
    -c ../../tests/integration/test_compaction_service.cpp -o test_compaction_service.o

echo "Compiling test_backup_service.cpp..."
g++ -std=c++20 -I../../src \
    -I../_deps/eigen-src \
    -I../_deps/nlohmann_json-src/include \
    -I../_deps/crow-src/include \
    -I../_deps/googletest-src/googletest/include \
    -I../_deps/googletest-src/googlemock/include \
    -c ../../tests/integration/test_backup_service.cpp -o test_backup_service.o

# Link test executable (include distributed_master_client object from main build)
echo "Linking sprint22_service_tests executable..."
g++ -o sprint22_service_tests \
    test_compaction_service.o \
    test_backup_service.o \
    ../CMakeFiles/jadevectordb.dir/src/api/grpc/distributed_master_client.cpp.o \
    ../libjadevectordb_core.a \
    ../lib/libgtest.a \
    ../lib/libgtest_main.a \
    -lpthread -lssl -lcrypto -lsqlite3 -lz -ldl

echo "Sprint 2.2 service tests built successfully!"
echo "Run with: cd build/sprint22_tests && ./sprint22_service_tests"
