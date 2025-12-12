#!/bin/bash
# =============================================================================
# JadeVectorDB Performance Tests
# =============================================================================
# Performance benchmarks for authentication and search endpoints
# Usage: ./performance_tests.sh [base_url] [iterations]
# =============================================================================

set -e

BASE_URL="${1:-http://localhost:8080}"
ITERATIONS="${2:-100}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_result() {
    echo -e "${GREEN}[RESULT]${NC} $1"
}

log_metric() {
    echo -e "${CYAN}[METRIC]${NC} $1"
}

# =============================================================================
# Setup
# =============================================================================

echo "=============================================="
echo "JadeVectorDB Performance Tests"
echo "Base URL: $BASE_URL"
echo "Iterations: $ITERATIONS"
echo "=============================================="
echo ""

# Login to get token
log_info "Authenticating..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"Admin@123456"}' 2>/dev/null)

TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    echo "Failed to get authentication token. Ensure server is running with JADE_ENV=development"
    exit 1
fi

log_info "Authentication successful"

# =============================================================================
# Performance Test: Login Endpoint
# =============================================================================

echo ""
echo "--- Login Endpoint Performance ---"

TOTAL_TIME=0
for i in $(seq 1 $ITERATIONS); do
    TIME=$(curl -s -o /dev/null -w "%{time_total}" -X POST "$BASE_URL/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username":"admin","password":"Admin@123456"}' 2>/dev/null)
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
done

AVG_LOGIN_TIME=$(echo "scale=4; $TOTAL_TIME / $ITERATIONS" | bc)
RPS_LOGIN=$(echo "scale=2; 1 / $AVG_LOGIN_TIME" | bc)

log_result "Login endpoint:"
log_metric "  Total time: ${TOTAL_TIME}s for $ITERATIONS requests"
log_metric "  Average latency: ${AVG_LOGIN_TIME}s"
log_metric "  Requests/second: ${RPS_LOGIN}"

# =============================================================================
# Performance Test: User List Endpoint (Authenticated)
# =============================================================================

echo ""
echo "--- User List Endpoint Performance ---"

TOTAL_TIME=0
for i in $(seq 1 $ITERATIONS); do
    TIME=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/v1/users" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
done

AVG_USERS_TIME=$(echo "scale=4; $TOTAL_TIME / $ITERATIONS" | bc)
RPS_USERS=$(echo "scale=2; 1 / $AVG_USERS_TIME" | bc)

log_result "User list endpoint:"
log_metric "  Total time: ${TOTAL_TIME}s for $ITERATIONS requests"
log_metric "  Average latency: ${AVG_USERS_TIME}s"
log_metric "  Requests/second: ${RPS_USERS}"

# =============================================================================
# Performance Test: Database List Endpoint
# =============================================================================

echo ""
echo "--- Database List Endpoint Performance ---"

TOTAL_TIME=0
for i in $(seq 1 $ITERATIONS); do
    TIME=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/v1/databases" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
done

AVG_DB_TIME=$(echo "scale=4; $TOTAL_TIME / $ITERATIONS" | bc)
RPS_DB=$(echo "scale=2; 1 / $AVG_DB_TIME" | bc)

log_result "Database list endpoint:"
log_metric "  Total time: ${TOTAL_TIME}s for $ITERATIONS requests"
log_metric "  Average latency: ${AVG_DB_TIME}s"
log_metric "  Requests/second: ${RPS_DB}"

# =============================================================================
# Performance Test: Token Validation
# =============================================================================

echo ""
echo "--- Token Validation Performance ---"

TOTAL_TIME=0
for i in $(seq 1 $ITERATIONS); do
    TIME=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/v1/auth/validate" \
        -H "Authorization: Bearer $TOKEN" 2>/dev/null)
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
done

AVG_VALIDATE_TIME=$(echo "scale=4; $TOTAL_TIME / $ITERATIONS" | bc)
RPS_VALIDATE=$(echo "scale=2; 1 / $AVG_VALIDATE_TIME" | bc)

log_result "Token validation endpoint:"
log_metric "  Total time: ${TOTAL_TIME}s for $ITERATIONS requests"
log_metric "  Average latency: ${AVG_VALIDATE_TIME}s"
log_metric "  Requests/second: ${RPS_VALIDATE}"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Performance Test Summary"
echo "=============================================="
echo ""
echo "| Endpoint        | Avg Latency | Req/sec |"
echo "|-----------------|-------------|---------|"
printf "| Login           | %9ss  | %7s |\n" "$AVG_LOGIN_TIME" "$RPS_LOGIN"
printf "| User List       | %9ss  | %7s |\n" "$AVG_USERS_TIME" "$RPS_USERS"
printf "| Database List   | %9ss  | %7s |\n" "$AVG_DB_TIME" "$RPS_DB"
printf "| Token Validate  | %9ss  | %7s |\n" "$AVG_VALIDATE_TIME" "$RPS_VALIDATE"
echo ""

# Performance thresholds
THRESHOLD_MS=50  # 50ms target

check_threshold() {
    local name=$1
    local time_s=$2
    local time_ms=$(echo "scale=2; $time_s * 1000" | bc)
    
    if (( $(echo "$time_ms < $THRESHOLD_MS" | bc -l) )); then
        echo -e "${GREEN}✓${NC} $name: ${time_ms}ms (under ${THRESHOLD_MS}ms target)"
    else
        echo -e "${YELLOW}⚠${NC} $name: ${time_ms}ms (above ${THRESHOLD_MS}ms target)"
    fi
}

echo "Threshold Check (target: <${THRESHOLD_MS}ms):"
check_threshold "Login" "$AVG_LOGIN_TIME"
check_threshold "User List" "$AVG_USERS_TIME"
check_threshold "Database List" "$AVG_DB_TIME"
check_threshold "Token Validate" "$AVG_VALIDATE_TIME"

echo ""
log_info "Performance tests complete"
