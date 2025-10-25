#!/usr/bin/env bash
# Test script for verifying cURL command generation in both Python CLI and Shell Script CLI

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test variables
TEST_URL="http://localhost:8080"
TEST_API_KEY="test-api-key"

echo "Testing JadeVectorDB cURL Command Generation"
echo "=========================================="

# Test 1: Python CLI cURL generation
echo -e "\n${YELLOW}Test 1: Python CLI cURL Generation${NC}"
echo "----------------------------------"

echo "Testing create-db command..."
python3 -m jadevectordb.cli --curl-only --url $TEST_URL --api-key $TEST_API_KEY create-db --name test-db --description "Test database" --dimension 128 --index-type HNSW > /tmp/python_create_db_curl.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python CLI create-db cURL generation: PASSED${NC}"
else
    echo -e "${RED}✗ Python CLI create-db cURL generation: FAILED${NC}"
    cat /tmp/python_create_db_curl.txt
fi

echo "Testing list-dbs command..."
python3 -m jadevectordb.cli --curl-only --url $TEST_URL --api-key $TEST_API_KEY list-dbs > /tmp/python_list_dbs_curl.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python CLI list-dbs cURL generation: PASSED${NC}"
else
    echo -e "${RED}✗ Python CLI list-dbs cURL generation: FAILED${NC}"
    cat /tmp/python_list_dbs_curl.txt
fi

# Test 2: Shell Script CLI cURL generation
echo -e "\n${YELLOW}Test 2: Shell Script CLI cURL Generation${NC}"
echo "----------------------------------------"

echo "Testing create-db command..."
./jade-db.sh --curl-only --url $TEST_URL --api-key $TEST_API_KEY create-db test-db "Test database" 128 HNSW > /tmp/shell_create_db_curl.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Shell CLI create-db cURL generation: PASSED${NC}"
else
    echo -e "${RED}✗ Shell CLI create-db cURL generation: FAILED${NC}"
    cat /tmp/shell_create_db_curl.txt
fi

echo "Testing list-dbs command..."
./jade-db.sh --curl-only --url $TEST_URL --api-key $TEST_API_KEY list-dbs > /tmp/shell_list_dbs_curl.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Shell CLI list-dbs cURL generation: PASSED${NC}"
else
    echo -e "${RED}✗ Shell CLI list-dbs cURL generation: FAILED${NC}"
    cat /tmp/shell_list_dbs_curl.txt
fi

# Test 3: Verify cURL commands are valid
echo -e "\n${YELLOW}Test 3: cURL Command Validation${NC}"
echo "-------------------------------"

# Check if generated cURL commands contain expected elements
if grep -q "curl.*POST.*databases" /tmp/python_create_db_curl.txt && grep -q "Content-Type.*application/json" /tmp/python_create_db_curl.txt; then
    echo -e "${GREEN}✓ Python CLI generated valid cURL command for create-db${NC}"
else
    echo -e "${RED}✗ Python CLI did not generate valid cURL command for create-db${NC}"
fi

if grep -q "curl.*GET.*databases" /tmp/python_list_dbs_curl.txt && grep -q "Content-Type.*application/json" /tmp/python_list_dbs_curl.txt; then
    echo -e "${GREEN}✓ Python CLI generated valid cURL command for list-dbs${NC}"
else
    echo -e "${RED}✗ Python CLI did not generate valid cURL command for list-dbs${NC}"
fi

if grep -q "curl.*POST.*databases" /tmp/shell_create_db_curl.txt && grep -q "Content-Type.*application/json" /tmp/shell_create_db_curl.txt; then
    echo -e "${GREEN}✓ Shell CLI generated valid cURL command for create-db${NC}"
else
    echo -e "${RED}✗ Shell CLI did not generate valid cURL command for create-db${NC}"
fi

if grep -q "curl.*GET.*databases" /tmp/shell_list_dbs_curl.txt && grep -q "Content-Type.*application/json" /tmp/shell_list_dbs_curl.txt; then
    echo -e "${GREEN}✓ Shell CLI generated valid cURL command for list-dbs${NC}"
else
    echo -e "${RED}✗ Shell CLI did not generate valid cURL command for list-dbs${NC}"
fi

# Clean up temporary files
rm -f /tmp/python_create_db_curl.txt /tmp/python_list_dbs_curl.txt /tmp/shell_create_db_curl.txt /tmp/shell_list_dbs_curl.txt

echo -e "\n${YELLOW}Summary:${NC}"
echo "--------"
echo "Both Python CLI and Shell Script CLI now support cURL command generation!"
echo "Users can use the --curl-only flag to generate cURL commands instead of executing operations directly."
echo "This feature is useful for learning the JadeVectorDB API and debugging API calls."

echo -e "\n${GREEN}All tests completed!${NC}"