#!/bin/bash
# JadeVectorDB - Master CLI Test Runner
# Runs all CLI tests (Python CLI tests + Shell tests)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "JadeVectorDB - Master CLI Test Runner"
echo "================================================================================"
echo ""

# Check if server is running
echo -e "${BLUE}Checking if server is running...${NC}"
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${RED}✗ Server is not running!${NC}"
    echo ""
    echo "Please start the server first:"
    echo "  cd backend/build"
    echo "  ./jadevectordb"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Run the main test suite (includes all Phase 16 tests)
echo -e "${YELLOW}=== Running Complete CLI Test Suite ===${NC}"
echo ""

if python3 run_cli_tests.py; then
    echo ""
    echo -e "${GREEN}✓ All tests completed${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
