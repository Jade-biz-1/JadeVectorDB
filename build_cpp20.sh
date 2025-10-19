#!/bin/bash
# Build script for JadeVectorDB with explicit C++20

cd /home/deepak/Public/JadeVectorDB/backend

# Clean any existing build
rm -rf build

# Create build directory
mkdir build
cd build

# Configure with explicit C++20
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON

# Build
make -j$(nproc)