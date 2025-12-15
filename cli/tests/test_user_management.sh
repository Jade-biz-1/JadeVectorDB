#!/usr/bin/env bash
"""
User Management CLI Tests - Shell Script (T264)

Comprehensive test suite for user management commands in Shell CLI.
Tests all user management operations without requiring a running backend.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test configuration
SHELL_CLI="../shell/scripts/jade-db.sh"
TEST_URL="http://localhost:8080"
TEST_API_KEY="test-api-key-12345"

# Helper function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}Test $TESTS_RUN: $test_name${NC}"

    # Run command and capture output
    output=$(eval "$command" 2>&1)
    exit_code=$?

    # Check if output matches expected pattern or command succeeded
    if [[ "$output" =~ $expected_pattern ]] || [[ $exit_code -eq 0 && "$expected_pattern" == "success" ]]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Expected pattern: $expected_pattern"
        echo "  Got output: $output"
        echo "  Exit code: $exit_code"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Helper function to test cURL command generation
test_curl_generation() {
    local test_name="$1"
    local command="$2"
    local expected_endpoint="$3"
    local expected_method="$4"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}Test $TESTS_RUN: $test_name${NC}"

    # Run command with --curl-only flag
    output=$(eval "$command --curl-only" 2>&1)

    # Check if output contains expected cURL elements
    if echo "$output" | grep -q "curl" && \
       echo "$output" | grep -q "$expected_endpoint" && \
       echo "$output" | grep -q "$expected_method"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        echo "  Generated cURL command correctly"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Expected: curl ... $expected_method ... $expected_endpoint"
        echo "  Got: $output"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

echo "========================================"
echo "User Management CLI Tests - Shell Script"
echo "========================================"
echo ""

# Check if shell CLI exists
if [ ! -f "$SHELL_CLI" ]; then
    echo -e "${RED}ERROR: Shell CLI not found at $SHELL_CLI${NC}"
    exit 1
fi

echo -e "${YELLOW}Part 1: User Management Command Availability${NC}"
echo "-------------------------------------------"

# Test 1: Check if user-add command exists in help
run_test \
    "user-add command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-add'" \
    "user-add"

# Test 2: Check if user-list command exists in help
run_test \
    "user-list command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-list'" \
    "user-list"

# Test 3: Check if user-show command exists in help
run_test \
    "user-show command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-show'" \
    "user-show"

# Test 4: Check if user-update command exists in help
run_test \
    "user-update command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-update'" \
    "user-update"

# Test 5: Check if user-delete command exists in help
run_test \
    "user-delete command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-delete'" \
    "user-delete"

# Test 6: Check if user-activate command exists in help
run_test \
    "user-activate command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-activate'" \
    "user-activate"

# Test 7: Check if user-deactivate command exists in help
run_test \
    "user-deactivate command in help" \
    "bash $SHELL_CLI --help | grep -i 'user-deactivate'" \
    "user-deactivate"

echo ""
echo -e "${YELLOW}Part 2: cURL Command Generation Tests${NC}"
echo "-------------------------------------"

# Test 8: user-add cURL generation
test_curl_generation \
    "user-add cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-add test@example.com developer password123" \
    "/api/v1/users" \
    "POST"

# Test 9: user-list cURL generation
test_curl_generation \
    "user-list cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-list" \
    "/api/v1/users" \
    "GET"

# Test 10: user-show cURL generation
test_curl_generation \
    "user-show cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-show test@example.com" \
    "/api/v1/users" \
    "GET"

# Test 11: user-update cURL generation
test_curl_generation \
    "user-update cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-update test@example.com --role admin" \
    "/api/v1/users" \
    "PUT"

# Test 12: user-delete cURL generation
test_curl_generation \
    "user-delete cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-delete test@example.com" \
    "/api/v1/users" \
    "DELETE"

# Test 13: user-activate cURL generation
test_curl_generation \
    "user-activate cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-activate test@example.com" \
    "/api/v1/users" \
    "PUT"

# Test 14: user-deactivate cURL generation
test_curl_generation \
    "user-deactivate cURL generation" \
    "bash $SHELL_CLI --url $TEST_URL --api-key $TEST_API_KEY user-deactivate test@example.com" \
    "/api/v1/users" \
    "PUT"

echo ""
echo -e "${YELLOW}Part 3: Parameter Validation Tests${NC}"
echo "----------------------------------"

# Test 15: user-add without required email parameter
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-add missing email parameter${NC}"
output=$(bash $SHELL_CLI --curl-only user-add 2>&1)
if echo "$output" | grep -iq "error\|required\|usage"; then
    echo -e "${GREEN}✓ PASSED - Correctly reports missing parameter${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Should report missing parameter${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 16: user-list with role filter
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-list with role filter${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-list --role admin 2>&1)
if echo "$output" | grep -q "role.*admin"; then
    echo -e "${GREEN}✓ PASSED - Role filter included in request${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Role filter not included${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 17: user-list with status filter
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-list with status filter${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-list --status active 2>&1)
if echo "$output" | grep -q "status.*active"; then
    echo -e "${GREEN}✓ PASSED - Status filter included in request${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Status filter not included${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo -e "${YELLOW}Part 4: JSON Payload Validation${NC}"
echo "-------------------------------"

# Test 18: user-add JSON payload structure
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-add JSON payload contains email, role, password${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-add test@example.com developer password123 2>&1)
if echo "$output" | grep -q "email.*test@example.com" && \
   echo "$output" | grep -q "role.*developer" && \
   echo "$output" | grep -q "password.*password123"; then
    echo -e "${GREEN}✓ PASSED - JSON payload correctly structured${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - JSON payload missing required fields${NC}"
    echo "  Output: $output"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 19: user-update JSON payload structure
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-update JSON payload contains role${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-update test@example.com --role admin 2>&1)
if echo "$output" | grep -q "role.*admin"; then
    echo -e "${GREEN}✓ PASSED - Update payload correctly structured${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Update payload missing role${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo -e "${YELLOW}Part 5: Output Format Tests${NC}"
echo "-------------------------"

# Test 20: user-list with JSON format
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-list supports --format json${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-list --format json 2>&1)
# Should not error with --format json
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ PASSED - JSON format accepted${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - JSON format not accepted${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 21: user-list with YAML format
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-list supports --format yaml${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-list --format yaml 2>&1)
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ PASSED - YAML format accepted${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - YAML format not accepted${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 22: user-list with Table format
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-list supports --format table${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-list --format table 2>&1)
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ PASSED - Table format accepted${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Table format not accepted${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 23: user-list with CSV format
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: user-list supports --format csv${NC}"
output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-list --format csv 2>&1)
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ PASSED - CSV format accepted${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - CSV format not accepted${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Total tests run:    ${BLUE}$TESTS_RUN${NC}"
echo -e "Tests passed:       ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed:       ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Please review the output above.${NC}"
    exit 1
fi
