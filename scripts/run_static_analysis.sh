#!/bin/bash

# Static analysis script for JadeVectorDB
set -e  # Exit on any error

echo "Running static analysis on JadeVectorDB..."

# Check if required tools are available
if ! command -v clang-tidy &> /dev/null; then
    echo "clang-tidy could not be found. Please install clang-tools."
    exit 1
fi

if ! command -v cppcheck &> /dev/null; then
    echo "cppcheck could not be found. Please install cppcheck."
    exit 1
fi

# Define source directories
SOURCE_DIRS="backend/src backend/tests"

# Create build directory if it doesn't exist and generate compile_commands.json
if [ ! -f "backend/build/compile_commands.json" ]; then
    echo "Generating compile_commands.json..."
    mkdir -p backend/build
    cd backend/build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. > /dev/null
    cd ../..
fi

echo "Running clang-tidy..."
clang-tidy $SOURCE_DIRS/*.cpp $SOURCE_DIRS/**/*.cpp --warnings-as-errors=* --extra-arg=-Ibackend/src --extra-arg=-Ibackend/build/_deps/eigen-src --quiet

echo "Running cppcheck..."
cppcheck --enable=all --std=c++20 --verbose --quiet $SOURCE_DIRS

echo "Static analysis completed successfully!"