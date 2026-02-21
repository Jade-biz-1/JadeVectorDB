#!/usr/bin/env bash
# ============================================================
# JadeVectorDB API Examples — Runner Script
# ============================================================
# This script:
#   1. Creates a Python virtual environment (if not already present)
#   2. Installs the jadevectordb client library + dependencies
#   3. Loads configuration from .env
#   4. Optionally auto-registers a demo user and obtains an auth token
#   5. Runs every example in sequence, reporting pass/fail
#
# Usage:
#   chmod +x APIExamples/run_all_examples.sh
#   ./APIExamples/run_all_examples.sh
#
# Or run a single example:
#   ./APIExamples/run_all_examples.sh 04
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ENV_FILE="$SCRIPT_DIR/.env"
CLI_DIR="$PROJECT_ROOT/cli/python"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[PASS]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $1"; }

# ---------------------------------------------------------------------------
# 1. Load .env configuration
# ---------------------------------------------------------------------------
banner "JadeVectorDB API Examples Runner"

if [ -f "$ENV_FILE" ]; then
    info "Loading configuration from $ENV_FILE"
    # Source .env but only export known variables (skip comments/blanks)
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        export "$key=$value"
    done < "$ENV_FILE"
else
    warn ".env file not found at $ENV_FILE — using defaults"
fi

# Defaults
export JADEVECTORDB_URL="${JADEVECTORDB_URL:-http://localhost:8080}"
export JADEVECTORDB_API_KEY="${JADEVECTORDB_API_KEY:-}"
export JADEVECTORDB_USER_ID="${JADEVECTORDB_USER_ID:-}"

info "Server URL: $JADEVECTORDB_URL"

# ---------------------------------------------------------------------------
# 2. Create Python virtual environment
# ---------------------------------------------------------------------------
banner "Setting Up Python Environment"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    fail "Python 3 not found. Please install Python 3.8+ and retry."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Found $PYTHON_CMD ($PYTHON_VERSION)"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created"
else
    info "Virtual environment already exists at $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtual environment"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
banner "Installing Dependencies"

# Upgrade pip quietly
pip install --upgrade pip --quiet 2>/dev/null

# Install the jadevectordb client library in editable mode
info "Installing jadevectordb client library from $CLI_DIR"
pip install -e "$CLI_DIR" --quiet 2>/dev/null
success "jadevectordb installed"

# Verify the import works
python -c "from jadevectordb import JadeVectorDB; print('  Import check: OK')"

# ---------------------------------------------------------------------------
# 4. Check server connectivity
# ---------------------------------------------------------------------------
banner "Checking Server Connectivity"

SERVER_OK=false
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$JADEVECTORDB_URL/health" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    success "Server is reachable at $JADEVECTORDB_URL (HTTP $HTTP_CODE)"
    SERVER_OK=true
else
    fail "Server not reachable at $JADEVECTORDB_URL (HTTP $HTTP_CODE)"
    echo ""
    echo "  Please start the JadeVectorDB backend:"
    echo "    cd backend/build && ./jadevectordb"
    echo ""
    echo "  Then re-run this script."
    deactivate 2>/dev/null || true
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Auto-register demo user and obtain auth token (if no API key set)
# ---------------------------------------------------------------------------
if [ -z "$JADEVECTORDB_API_KEY" ]; then
    banner "Auto-Registering Demo User"

    DEMO_USER="api_examples_runner_$(date +%s)"
    DEMO_PASS="RunnerPass123!"
    DEMO_EMAIL="${DEMO_USER}@examples.jadevectordb.local"

    info "Registering user: $DEMO_USER"

    # Register (may fail if user already exists — that's fine)
    curl -s -X POST "$JADEVECTORDB_URL/v1/auth/register" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$DEMO_USER\",\"password\":\"$DEMO_PASS\",\"email\":\"$DEMO_EMAIL\"}" \
        > /dev/null 2>&1 || true

    # Login to get token
    LOGIN_RESPONSE=$(curl -s -X POST "$JADEVECTORDB_URL/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$DEMO_USER\",\"password\":\"$DEMO_PASS\"}" 2>/dev/null)

    TOKEN=$(echo "$LOGIN_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
    USER_ID=$(echo "$LOGIN_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('user_id',''))" 2>/dev/null || echo "")

    if [ -n "$TOKEN" ]; then
        export JADEVECTORDB_API_KEY="$TOKEN"
        success "Obtained auth token"
    else
        warn "Could not obtain auth token — some examples may fail"
        warn "Login response: $LOGIN_RESPONSE"
    fi

    if [ -n "$USER_ID" ]; then
        export JADEVECTORDB_USER_ID="$USER_ID"
        info "User ID: $USER_ID"
    fi
else
    info "Using API key from .env"
fi

# ---------------------------------------------------------------------------
# 6. Run examples
# ---------------------------------------------------------------------------
banner "Running API Examples"

# Collect all example scripts in order
EXAMPLES=(
    "01_getting_started.py"
    "02_database_management.py"
    "03_vector_operations.py"
    "04_similarity_search.py"
    "05_hybrid_search.py"
    "06_reranking.py"
    "07_index_management.py"
    "08_embeddings.py"
    "09_user_management.py"
    "10_api_key_management.py"
    "11_security_audit.py"
    "12_analytics.py"
    "13_password_management.py"
    "14_import_export.py"
    "15_error_handling.py"
)

# If a specific example number was passed (e.g., ./run_all_examples.sh 04),
# run only that one.
FILTER="${1:-}"

TOTAL=0
PASSED=0
FAILED=0
FAILED_LIST=()

for example in "${EXAMPLES[@]}"; do
    # Apply filter if provided
    if [ -n "$FILTER" ] && [[ ! "$example" == "${FILTER}"* ]]; then
        continue
    fi

    EXAMPLE_PATH="$SCRIPT_DIR/$example"

    if [ ! -f "$EXAMPLE_PATH" ]; then
        warn "File not found: $example — skipping"
        continue
    fi

    TOTAL=$((TOTAL + 1))
    EXAMPLE_NAME="${example%.py}"

    echo ""
    echo -e "${BOLD}--- [$TOTAL] $EXAMPLE_NAME ---${NC}"

    # Run with a timeout of 60 seconds (portable: works on macOS and Linux)
    set +e
    if command -v gtimeout &>/dev/null; then
        # macOS with coreutils installed via Homebrew
        OUTPUT=$(gtimeout 60 python "$EXAMPLE_PATH" 2>&1)
        EXIT_CODE=$?
    elif command -v timeout &>/dev/null; then
        # Linux / GNU coreutils
        OUTPUT=$(timeout 60 python "$EXAMPLE_PATH" 2>&1)
        EXIT_CODE=$?
    else
        # Fallback: run without timeout, use a background process with kill
        python "$EXAMPLE_PATH" > /tmp/jade_example_out.txt 2>&1 &
        CHILD_PID=$!
        SECONDS_WAITED=0
        while kill -0 "$CHILD_PID" 2>/dev/null; do
            sleep 1
            SECONDS_WAITED=$((SECONDS_WAITED + 1))
            if [ $SECONDS_WAITED -ge 60 ]; then
                kill "$CHILD_PID" 2>/dev/null
                wait "$CHILD_PID" 2>/dev/null
                EXIT_CODE=124
                break
            fi
        done
        if [ $SECONDS_WAITED -lt 60 ]; then
            wait "$CHILD_PID"
            EXIT_CODE=$?
        fi
        OUTPUT=$(cat /tmp/jade_example_out.txt 2>/dev/null || echo "")
        rm -f /tmp/jade_example_out.txt
    fi
    set -e

    # Print output (indented)
    echo "$OUTPUT" | sed 's/^/  /'

    if [ $EXIT_CODE -eq 0 ]; then
        success "$EXAMPLE_NAME"
        PASSED=$((PASSED + 1))
    elif [ $EXIT_CODE -eq 124 ]; then
        fail "$EXAMPLE_NAME (timed out after 60s)"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$EXAMPLE_NAME (timeout)")
    else
        fail "$EXAMPLE_NAME (exit code $EXIT_CODE)"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$EXAMPLE_NAME (exit $EXIT_CODE)")
    fi
done

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
banner "Results Summary"

echo ""
echo -e "  Total   : ${BOLD}$TOTAL${NC}"
echo -e "  Passed  : ${GREEN}$PASSED${NC}"
echo -e "  Failed  : ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed examples:${NC}"
    for f in "${FAILED_LIST[@]}"; do
        echo -e "  - $f"
    done
    echo ""
fi

if [ $FAILED -eq 0 ] && [ $TOTAL -gt 0 ]; then
    echo -e "${GREEN}${BOLD}All $TOTAL examples passed!${NC}"
else
    echo -e "${YELLOW}$PASSED/$TOTAL examples passed.${NC}"
fi

echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true

# Exit with appropriate code
[ $FAILED -eq 0 ]
