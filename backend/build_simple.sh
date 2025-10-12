#!/bin/bash

# Simple build script for JadeVectorDB core components

set -e  # Exit on any error

echo "Building JadeVectorDB core components..."

# Create build directory
mkdir -p build
cd build

# Compile core components with basic dependencies
echo "Compiling core components..."

# Compile all .cpp files in the src directory
g++ -std=c++20 -Wall -Wextra -O3 \
    -I../src \
    -I../src/lib \
    -I../src/models \
    -I../src/services \
    -I../src/api \
    $(find ../src -name "*.cpp" | grep -v test | grep -v benchmark) \
    -pthread \
    -o jadevectordb \
    2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable created: build/jadevectordb"
    echo ""
    echo "To run the server:"
    echo "  cd build && ./jadevectordb"
else
    echo "Build failed. Check build.log for details."
    exit 1
fi