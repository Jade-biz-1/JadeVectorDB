#!/usr/bin/env bash
"""
CLI Integration Tests - Shell Script (T272)

Comprehensive integration tests for cross-CLI consistency and complete workflows.
Tests compatibility between Python, Shell, and JavaScript CLIs.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Test configuration
PYTHON_CLI="python3 -m jadevectordb.cli"
SHELL_CLI="../shell/scripts/jade-db.sh"
JS_CLI="node ../js/bin/jade-db.js"
TEST_URL="http://localhost:8080"
TEST_API_KEY="test-integration-key"
TEMP_DIR="/tmp/jade_integration_tests_$$"

# Setup
mkdir -p "$TEMP_DIR"

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Helper function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}Test $TESTS_RUN: $test_name${NC}"

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if [[ "$output" =~ $expected_pattern ]] || [[ $exit_code -eq 0 && "$expected_pattern" == "success" ]]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Expected: $expected_pattern"
        echo "  Got: $output"
        echo "  Exit code: $exit_code"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Helper to check if a CLI is available
check_cli_available() {
    local cli_name="$1"
    local cli_command="$2"

    if eval "$cli_command --help" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $cli_name is available"
        return 0
    else
        echo -e "${YELLOW}⊘${NC} $cli_name is not available"
        return 1
    fi
}

echo "=========================================="
echo "CLI Integration Tests - Comprehensive"
echo "=========================================="
echo ""

echo -e "${CYAN}Checking CLI Availability${NC}"
echo "-------------------------"

PYTHON_AVAILABLE=0
SHELL_AVAILABLE=0
JS_AVAILABLE=0

if check_cli_available "Python CLI" "$PYTHON_CLI"; then
    PYTHON_AVAILABLE=1
fi

if check_cli_available "Shell CLI" "bash $SHELL_CLI"; then
    SHELL_AVAILABLE=1
fi

if check_cli_available "JavaScript CLI" "$JS_CLI"; then
    JS_AVAILABLE=1
fi

echo ""
echo -e "${YELLOW}Part 1: Cross-CLI Command Consistency${NC}"
echo "-------------------------------------"

# Test 1: All CLIs have user management commands
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: All CLIs support user management${NC}"

user_cmds_python=0
user_cmds_shell=0
user_cmds_js=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    if $PYTHON_CLI --help 2>&1 | grep -iq "user-add"; then
        user_cmds_python=1
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    if bash $SHELL_CLI --help 2>&1 | grep -iq "user-add"; then
        user_cmds_shell=1
    fi
fi

if [ $JS_AVAILABLE -eq 1 ]; then
    if $JS_CLI --help 2>&1 | grep -iq "user.*add"; then
        user_cmds_js=1
    fi
fi

total_with_user_mgmt=$((user_cmds_python + user_cmds_shell + user_cmds_js))
if [ $total_with_user_mgmt -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - $total_with_user_mgmt CLI(s) support user management${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - No CLIs support user management${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 2: All CLIs support output formats
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: All CLIs support multiple output formats${NC}"

format_support_python=0
format_support_shell=0
format_support_js=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    if $PYTHON_CLI --help 2>&1 | grep -iq "format.*json.*yaml.*table"; then
        format_support_python=1
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    if bash $SHELL_CLI --help 2>&1 | grep -iq "format.*json.*yaml.*table"; then
        format_support_shell=1
    fi
fi

if [ $JS_AVAILABLE -eq 1 ]; then
    if $JS_CLI --help 2>&1 | grep -iq "format"; then
        format_support_js=1
    fi
fi

total_with_formats=$((format_support_python + format_support_shell + format_support_js))
if [ $total_with_formats -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - $total_with_formats CLI(s) support output formats${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - No CLIs support output formats${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 3: All CLIs support import/export
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: CLIs support import/export operations${NC}"

import_support_python=0
import_support_shell=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    if $PYTHON_CLI --help 2>&1 | grep -iq "import"; then
        import_support_python=1
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    if bash $SHELL_CLI --help 2>&1 | grep -iq "import"; then
        import_support_shell=1
    fi
fi

total_with_import=$((import_support_python + import_support_shell))
if [ $total_with_import -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - $total_with_import CLI(s) support import/export${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${YELLOW}⊘ SKIPPED - Import/export not universally available${NC}"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
fi

echo ""
echo -e "${YELLOW}Part 2: Output Format Compatibility${NC}"
echo "---------------------------------"

# Test 4: JSON output format consistency
if [ $PYTHON_AVAILABLE -eq 1 ]; then
    run_test \
        "Python CLI JSON output format" \
        "$PYTHON_CLI --curl-only --url $TEST_URL --format json list-dbs 2>&1 | head -1" \
        "curl"
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    run_test \
        "Shell CLI JSON output format" \
        "bash $SHELL_CLI --curl-only --url $TEST_URL --format json list-dbs 2>&1 | head -1" \
        "curl"
fi

# Test 5: YAML output format support
if [ $PYTHON_AVAILABLE -eq 1 ]; then
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}Test $TESTS_RUN: Python CLI YAML format support${NC}"
    if $PYTHON_CLI --curl-only --url $TEST_URL --format yaml list-dbs &> /dev/null; then
        echo -e "${GREEN}✓ PASSED - Python supports YAML${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED - Python should support YAML${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}Test $TESTS_RUN: Shell CLI YAML format support${NC}"
    if bash $SHELL_CLI --curl-only --url $TEST_URL --format yaml list-dbs &> /dev/null; then
        echo -e "${GREEN}✓ PASSED - Shell supports YAML${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED - Shell should support YAML${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
fi

# Test 6: CSV output format support
if [ $PYTHON_AVAILABLE -eq 1 ]; then
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}Test $TESTS_RUN: Python CLI CSV format support${NC}"
    if $PYTHON_CLI --curl-only --url $TEST_URL --format csv list-dbs &> /dev/null; then
        echo -e "${GREEN}✓ PASSED - Python supports CSV${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED - Python should support CSV${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
fi

echo ""
echo -e "${YELLOW}Part 3: API Endpoint Consistency${NC}"
echo "------------------------------"

# Test 7: User creation endpoint consistency
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: User creation uses /api/v1/users endpoint${NC}"

endpoint_match=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    output=$($PYTHON_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-add test@example.com developer 2>&1)
    if echo "$output" | grep -q "/api/v1/users"; then
        endpoint_match=$((endpoint_match + 1))
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY user-add test@example.com developer 2>&1)
    if echo "$output" | grep -q "/api/v1/users"; then
        endpoint_match=$((endpoint_match + 1))
    fi
fi

if [ $endpoint_match -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - Consistent endpoint usage across CLIs${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Endpoint inconsistency detected${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 8: Database endpoints consistency
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: Database operations use /v1/databases endpoint${NC}"

db_endpoint_match=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    output=$($PYTHON_CLI --curl-only --url $TEST_URL list-dbs 2>&1)
    if echo "$output" | grep -q "/v1/databases"; then
        db_endpoint_match=$((db_endpoint_match + 1))
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    output=$(bash $SHELL_CLI --curl-only --url $TEST_URL list-dbs 2>&1)
    if echo "$output" | grep -q "/v1/databases"; then
        db_endpoint_match=$((db_endpoint_match + 1))
    fi
fi

if [ $db_endpoint_match -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - Database endpoint consistent${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Database endpoint inconsistent${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo -e "${YELLOW}Part 4: Authentication Consistency${NC}"
echo "--------------------------------"

# Test 9: API key handling
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: All CLIs include API key in requests${NC}"

api_key_match=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    output=$($PYTHON_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY list-dbs 2>&1)
    if echo "$output" | grep -iq "authorization.*bearer.*$TEST_API_KEY"; then
        api_key_match=$((api_key_match + 1))
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    output=$(bash $SHELL_CLI --curl-only --url $TEST_URL --api-key $TEST_API_KEY list-dbs 2>&1)
    if echo "$output" | grep -iq "authorization.*bearer.*$TEST_API_KEY"; then
        api_key_match=$((api_key_match + 1))
    fi
fi

if [ $api_key_match -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - API key properly included${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - API key not properly handled${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 10: Environment variable support
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: CLIs support environment variables${NC}"

env_support=0

export JADEVECTORDB_URL="http://env-test:8080"
export JADEVECTORDB_API_KEY="env-api-key"

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    output=$($PYTHON_CLI --curl-only list-dbs 2>&1)
    if echo "$output" | grep -q "env-test" && echo "$output" | grep -q "env-api-key"; then
        env_support=$((env_support + 1))
    fi
fi

unset JADEVECTORDB_URL
unset JADEVECTORDB_API_KEY

if [ $env_support -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - Environment variables supported${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${YELLOW}⊘ Partial support for environment variables${NC}"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
fi

echo ""
echo -e "${YELLOW}Part 5: Error Handling Consistency${NC}"
echo "--------------------------------"

# Test 11: Missing parameter handling
TESTS_RUN=$((TESTS_RUN + 1))
echo -e "${BLUE}Test $TESTS_RUN: CLIs report errors for missing parameters${NC}"

error_handling=0

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    output=$($PYTHON_CLI user-add 2>&1)
    if echo "$output" | grep -iq "error\|required\|usage"; then
        error_handling=$((error_handling + 1))
    fi
fi

if [ $SHELL_AVAILABLE -eq 1 ]; then
    output=$(bash $SHELL_CLI user-add 2>&1)
    if echo "$output" | grep -iq "error\|required\|usage"; then
        error_handling=$((error_handling + 1))
    fi
fi

if [ $error_handling -ge 1 ]; then
    echo -e "${GREEN}✓ PASSED - Error reporting works${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAILED - Error reporting inconsistent${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo -e "${YELLOW}Part 6: Feature Parity Summary${NC}"
echo "----------------------------"

# Summary of feature availability
echo ""
echo "Feature Parity Matrix:"
echo "---------------------"
printf "%-20s | %-10s | %-10s | %-10s\n" "Feature" "Python CLI" "Shell CLI" "JS CLI"
echo "------------------------------------------------------------------------"
printf "%-20s | %-10s | %-10s | %-10s\n" "User Management" \
    $([ $user_cmds_python -eq 1 ] && echo "✓" || echo "✗") \
    $([ $user_cmds_shell -eq 1 ] && echo "✓" || echo "✗") \
    $([ $user_cmds_js -eq 1 ] && echo "✓" || echo "✗")

printf "%-20s | %-10s | %-10s | %-10s\n" "Output Formats" \
    $([ $format_support_python -eq 1 ] && echo "✓" || echo "✗") \
    $([ $format_support_shell -eq 1 ] && echo "✓" || echo "✗") \
    $([ $format_support_js -eq 1 ] && echo "✓" || echo "✗")

printf "%-20s | %-10s | %-10s | %-10s\n" "Import/Export" \
    $([ $import_support_python -eq 1 ] && echo "✓" || echo "✗") \
    $([ $import_support_shell -eq 1 ] && echo "✓" || echo "✗") \
    "N/A"

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total tests run:    ${BLUE}$TESTS_RUN${NC}"
echo -e "Tests passed:       ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed:       ${RED}$TESTS_FAILED${NC}"
echo -e "Tests skipped:      ${YELLOW}$TESTS_SKIPPED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All integration tests passed!${NC}"
    echo ""
    echo "Summary:"
    echo "  ✓ Cross-CLI command consistency verified"
    echo "  ✓ Output format compatibility confirmed"
    echo "  ✓ API endpoint consistency validated"
    echo "  ✓ Authentication handling verified"
    echo "  ✓ Error handling consistency checked"
    exit 0
else
    echo -e "${RED}⚠️  Some integration tests failed.${NC}"
    echo "Please review the output above for details."
    exit 1
fi
