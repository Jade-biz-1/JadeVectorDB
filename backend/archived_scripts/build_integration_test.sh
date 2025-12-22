#!/bin/bash

# Integration test build and run script

echo "Building integration test..."

# Navigate to the project root
cd /home/deepak/Public/JadeVectorDB/backend

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build the integration test
make integration_test

if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Run the integration test
    echo "Running integration test..."
    ./integration_test
    
    if [ $? -eq 0 ]; then
        echo "Integration test passed!"
        
        # Show the generated security audit log
        echo "Checking generated security audit log..."
        if [ -f "./test_security_audit.log" ]; then
            echo "Security audit log contents:"
            cat ./test_security_audit.log
        else
            echo "No security audit log found"
        fi
        
    else
        echo "Integration test failed!"
        exit 1
    fi
else
    echo "Build failed!"
    exit 1
fi