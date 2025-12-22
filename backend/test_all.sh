#!/bin/bash
#############################################################################
# JadeVectorDB - Test Everything Script
#############################################################################
# Single command to run all tests (backend unit tests + CLI tests)
# Usage: ./test_all.sh
#############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}JadeVectorDB - Test Everything${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build/ directory not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

# Check if backend binary exists
if [ ! -f "build/jadevectordb" ]; then
    echo -e "${RED}Error: build/jadevectordb not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

# Test counters
BACKEND_TESTS_PASSED=0
BACKEND_TESTS_FAILED=0
CLI_TESTS_PASSED=0
CLI_TESTS_FAILED=0

# ===================================
# 1. Run Backend Unit Tests
# ===================================
echo -e "${YELLOW}[1/3] Running Backend Unit Tests...${NC}"
echo ""

cd build

# Check if tests were built
if [ ! -f "CTestTestfile.cmake" ] && [ ! -f "jadevectordb_tests" ]; then
    echo -e "${YELLOW}⊘ Backend unit tests not found (build was run with --no-tests)${NC}"
    echo -e "${YELLOW}  To run backend tests, rebuild with: ./build.sh${NC}"
    cd ..
else
    if command -v ctest &> /dev/null && [ -f "CTestTestfile.cmake" ]; then
        # Use ctest if available
        if ctest --output-on-failure 2>&1 | grep -q "No tests were found"; then
            echo -e "${YELLOW}⊘ No backend tests configured${NC}"
        elif ctest --output-on-failure; then
            BACKEND_TESTS_PASSED=1
            echo -e "${GREEN}✓ Backend unit tests PASSED${NC}"
        else
            BACKEND_TESTS_FAILED=1
            echo -e "${RED}✗ Backend unit tests FAILED${NC}"
        fi
    elif [ -f "jadevectordb_tests" ]; then
        # Fallback: run test binary directly
        if ./jadevectordb_tests; then
            BACKEND_TESTS_PASSED=1
            echo -e "${GREEN}✓ Backend unit tests PASSED${NC}"
        else
            BACKEND_TESTS_FAILED=1
            echo -e "${RED}✗ Backend unit tests FAILED${NC}"
        fi
    else
        echo -e "${YELLOW}⊘ Backend unit tests not available${NC}"
    fi
    cd ..
fi
echo ""

# ===================================
# 2. Start Backend Server
# ===================================
echo -e "${YELLOW}[2/3] Starting Backend Server...${NC}"
echo ""

# Check if server is already running
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Server already running on port 8080${NC}"
    SERVER_WAS_RUNNING=1
else
    echo "Starting server in background..."
    cd build
    ./jadevectordb > server.log 2>&1 &
    SERVER_PID=$!
    cd ..
    
    echo "Waiting for server to start (PID: $SERVER_PID)..."
    sleep 3
    
    # Check if server is running
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ Server failed to start${NC}"
        echo "Server log:"
        cat build/server.log
        exit 1
    fi
    
    # Check if server responds
    if curl -s http://localhost:8080/health > /dev/null; then
        echo -e "${GREEN}✓ Server started successfully${NC}"
        SERVER_WAS_RUNNING=0
    else
        echo -e "${RED}✗ Server not responding to health check${NC}"
        kill $SERVER_PID 2>/dev/null || true
        cat build/server.log
        exit 1
    fi
fi

echo ""

# ===================================
# 3. Run CLI Tests
# ===================================
echo -e "${YELLOW}[3/3] Running CLI Tests...${NC}"
echo ""

cd ../tests

if python3 run_cli_tests.py; then
    CLI_TESTS_PASSED=1
    echo -e "${GREEN}✓ CLI tests completed${NC}"
else
    CLI_TESTS_FAILED=1
    echo -e "${RED}✗ CLI tests had failures${NC}"
fi

cd ../backend
echo ""

# ===================================
# Cleanup
# ===================================
if [ "$SERVER_WAS_RUNNING" = "0" ]; then
    echo "Stopping test server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    sleep 1
fi

# ===================================
# Summary
# ===================================
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}======================================${NC}"

if [ $BACKEND_TESTS_PASSED -eq 1 ]; then
    echo -e "${GREEN}✓ Backend Unit Tests: PASSED${NC}"
elif [ $BACKEND_TESTS_FAILED -eq 1 ]; then
    echo -e "${RED}✗ Backend Unit Tests: FAILED${NC}"
else
    echo -e "${YELLOW}⊘ Backend Unit Tests: SKIPPED${NC}"
fi

if [ $CLI_TESTS_PASSED -eq 1 ]; then
    echo -e "${GREEN}✓ CLI Tests: PASSED${NC}"
else
    echo -e "${RED}✗ CLI Tests: FAILED${NC}"
fi

echo ""

# Exit code
if [ $BACKEND_TESTS_FAILED -eq 1 ] || [ $CLI_TESTS_FAILED -eq 1 ]; then
    echo -e "${RED}Some tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
