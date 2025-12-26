#!/bin/bash
# =============================================================================
# JadeVectorDB Smoke Tests
# =============================================================================
# Quick sanity tests for authentication and search endpoints
# Usage: ./smoke_tests.sh [base_url]
# =============================================================================

set -e

BASE_URL="${1:-http://localhost:8080}"
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASS_COUNT++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAIL_COUNT++))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# =============================================================================
# Health Check
# =============================================================================

echo "=============================================="
echo "JadeVectorDB Smoke Tests"
echo "Base URL: $BASE_URL"
echo "=============================================="
echo ""

log_info "Testing server health..."

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")
if [ "$HEALTH_RESPONSE" = "200" ]; then
    log_pass "Health endpoint returned 200"
else
    log_fail "Health endpoint returned $HEALTH_RESPONSE (expected 200)"
    echo "Server may not be running. Exiting."
    exit 1
fi

# =============================================================================
# Authentication Tests
# =============================================================================

echo ""
echo "--- Authentication Tests ---"

# Test 1: Login with valid credentials (development mode)
log_info "Testing login with default admin user..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"admin123"}' 2>/dev/null)

if echo "$LOGIN_RESPONSE" | grep -q '"success":true\|"token"'; then
    log_pass "Login successful with default admin user"
    # Extract token for subsequent tests
    TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)
else
    log_fail "Login failed: $LOGIN_RESPONSE"
    TOKEN=""
fi

# Test 2: Login with invalid credentials
log_info "Testing login with invalid credentials..."
INVALID_LOGIN=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"wrongpassword"}' 2>/dev/null)

if [ "$INVALID_LOGIN" = "401" ] || [ "$INVALID_LOGIN" = "403" ]; then
    log_pass "Invalid login correctly rejected with $INVALID_LOGIN"
else
    log_fail "Invalid login returned $INVALID_LOGIN (expected 401 or 403)"
fi

# Test 3: Access protected endpoint without token
log_info "Testing protected endpoint without authentication..."
UNAUTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/users" 2>/dev/null)

if [ "$UNAUTH_RESPONSE" = "401" ] || [ "$UNAUTH_RESPONSE" = "403" ]; then
    log_pass "Protected endpoint correctly rejected unauthenticated request"
else
    log_fail "Protected endpoint returned $UNAUTH_RESPONSE (expected 401 or 403)"
fi

# Test 4: Access protected endpoint with token
if [ -n "$TOKEN" ]; then
    log_info "Testing protected endpoint with valid token..."
    AUTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/users" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    
    if [ "$AUTH_RESPONSE" = "200" ]; then
        log_pass "Protected endpoint accessible with valid token"
    else
        log_fail "Protected endpoint returned $AUTH_RESPONSE with token (expected 200)"
    fi
fi

# =============================================================================
# Database Operations Tests
# =============================================================================

echo ""
echo "--- Database Operations Tests ---"

# Test 5: List databases
log_info "Testing database listing..."
if [ -n "$TOKEN" ]; then
    DB_LIST_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/databases" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    
    if [ "$DB_LIST_RESPONSE" = "200" ]; then
        log_pass "Database listing returned 200"
    else
        log_fail "Database listing returned $DB_LIST_RESPONSE (expected 200)"
    fi
else
    log_info "Skipping - no auth token available"
fi

# Test 6: Create a test database
TEST_DB_ID="smoke_test_db_$(date +%s)"
log_info "Creating test database: $TEST_DB_ID..."
if [ -n "$TOKEN" ]; then
    CREATE_DB_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/databases" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$TEST_DB_ID\",\"dimension\":128}" 2>/dev/null)
    
    if [ "$CREATE_DB_RESPONSE" = "201" ] || [ "$CREATE_DB_RESPONSE" = "200" ]; then
        log_pass "Database created successfully"
    else
        log_fail "Database creation returned $CREATE_DB_RESPONSE (expected 201)"
    fi
else
    log_info "Skipping - no auth token available"
fi

# =============================================================================
# Search Tests
# =============================================================================

echo ""
echo "--- Search Tests ---"

# Test 7: Search endpoint (may return empty if no vectors stored)
log_info "Testing search endpoint..."
if [ -n "$TOKEN" ]; then
    # Generate a simple 128-dimensional query vector
    QUERY_VECTOR=$(python3 -c "import json; print(json.dumps([0.1]*128))" 2>/dev/null || echo "[0.1]")
    
    SEARCH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/databases/$TEST_DB_ID/search" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"queryVector\":$QUERY_VECTOR,\"topK\":10}" 2>/dev/null)
    
    if [ "$SEARCH_RESPONSE" = "200" ] || [ "$SEARCH_RESPONSE" = "404" ]; then
        log_pass "Search endpoint returned $SEARCH_RESPONSE"
    else
        log_fail "Search endpoint returned $SEARCH_RESPONSE (expected 200 or 404)"
    fi
else
    log_info "Skipping - no auth token available"
fi

# Test 8: Search with includeVectorData parameter
log_info "Testing search with includeVectorData parameter..."
if [ -n "$TOKEN" ]; then
    SEARCH_WITH_VECTOR=$(curl -s -X POST "$BASE_URL/v1/databases/$TEST_DB_ID/search" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"queryVector\":$QUERY_VECTOR,\"topK\":10,\"includeVectorData\":true,\"includeMetadata\":true}" 2>/dev/null)
    
    SEARCH_STATUS=$(echo "$SEARCH_WITH_VECTOR" | head -c 1)
    if echo "$SEARCH_WITH_VECTOR" | grep -q '"results"\|"count"'; then
        log_pass "Search with parameters returned valid response"
    else
        # Check if it's a valid error response
        if echo "$SEARCH_WITH_VECTOR" | grep -q '"error"\|"message"'; then
            log_pass "Search returned expected error response"
        else
            log_fail "Search with parameters returned unexpected response"
        fi
    fi
else
    log_info "Skipping - no auth token available"
fi

# =============================================================================
# API Key Tests
# =============================================================================

echo ""
echo "--- API Key Tests ---"

# Test 9: Generate API key
log_info "Testing API key generation..."
if [ -n "$TOKEN" ]; then
    API_KEY_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/api-keys" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"name":"smoke_test_key","description":"Smoke test API key"}' 2>/dev/null)
    
    if echo "$API_KEY_RESPONSE" | grep -q '"key"\|"apiKey"\|"success":true'; then
        log_pass "API key generation successful"
        API_KEY=$(echo "$API_KEY_RESPONSE" | grep -o '"key":"[^"]*"\|"apiKey":"[^"]*"' | cut -d'"' -f4)
    else
        log_fail "API key generation failed: $API_KEY_RESPONSE"
    fi
else
    log_info "Skipping - no auth token available"
fi

# Test 10: List API keys
log_info "Testing API key listing..."
if [ -n "$TOKEN" ]; then
    LIST_KEYS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/api-keys" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    
    if [ "$LIST_KEYS_RESPONSE" = "200" ]; then
        log_pass "API key listing returned 200"
    else
        log_fail "API key listing returned $LIST_KEYS_RESPONSE (expected 200)"
    fi
else
    log_info "Skipping - no auth token available"
fi

# =============================================================================
# Logout Test
# =============================================================================

echo ""
echo "--- Logout Test ---"

# Test 11: Logout
if [ -n "$TOKEN" ]; then
    log_info "Testing logout..."
    LOGOUT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/auth/logout" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    
    if [ "$LOGOUT_RESPONSE" = "200" ]; then
        log_pass "Logout successful"
    else
        log_fail "Logout returned $LOGOUT_RESPONSE (expected 200)"
    fi
    
    # Test 12: Verify token is invalid after logout
    log_info "Testing token invalidation after logout..."
    POST_LOGOUT=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/users" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    
    if [ "$POST_LOGOUT" = "401" ] || [ "$POST_LOGOUT" = "403" ]; then
        log_pass "Token correctly invalidated after logout"
    else
        log_fail "Token still valid after logout (returned $POST_LOGOUT)"
    fi
else
    log_info "Skipping - no auth token available"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Smoke Test Summary"
echo "=============================================="
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some smoke tests failed.${NC}"
    exit 1
fi
