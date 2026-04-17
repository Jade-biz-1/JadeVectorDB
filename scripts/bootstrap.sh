#!/bin/bash
# =============================================================================
# JadeVectorDB — First-Run Bootstrap Script
# =============================================================================
# Run this ONCE on a fresh deployment before starting the full stack.
# It will:
#   1. Create .env from the template if one doesn't exist
#   2. Skip setup if JADEVECTORDB_API_KEY is already set in .env
#   3. Start JadeVectorDB alone and wait until healthy
#   4. Login as admin, generate a service API key
#   5. Write the key into .env
#   6. Start the full stack
#
# Usage:
#   bash scripts/bootstrap.sh
#
# Requirements:
#   - Docker + docker compose installed
#   - python3 (for JSON parsing)
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[bootstrap]${NC} $*"; }
success() { echo -e "${GREEN}[bootstrap]${NC} ✅ $*"; }
warn()    { echo -e "${YELLOW}[bootstrap]${NC} ⚠️  $*"; }
error()   { echo -e "${RED}[bootstrap]${NC} ❌ $*"; exit 1; }

# ── Locate repo root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
ENV_TEMPLATE="$REPO_ROOT/EnterpriseRAG/.env.docker"

cd "$REPO_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        JadeVectorDB — First-Run Bootstrap                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Create .env from template if missing ──────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
  info "No .env found — copying template from EnterpriseRAG/.env.docker"
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  success ".env created. Review and edit it before continuing if needed."
  echo ""
  warn "Important fields to review in .env:"
  warn "  ADMIN_DEFAULT_PASSWORD — change this from the default"
  warn "  JWT_SECRET_KEY         — use a long random string in production"
  warn "  RAG_API_KEY            — set to protect EnterpriseRAG endpoints"
  echo ""
  read -p "Press ENTER when ready to continue, or Ctrl-C to exit and edit .env first: "
fi

# ── Step 2: Check if key already exists ──────────────────────────────────────
EXISTING_KEY=$(grep -E "^JADEVECTORDB_API_KEY=.+" "$ENV_FILE" | cut -d'=' -f2 || true)
if [ -n "$EXISTING_KEY" ]; then
  success "JADEVECTORDB_API_KEY already set in .env — skipping key generation."
  info "Starting full stack..."
  docker compose up -d
  success "All services started. Run 'docker compose ps' to check status."
  exit 0
fi

# ── Step 3: Read admin credentials from .env ──────────────────────────────────
ADMIN_USER=$(grep -E "^ADMIN_USERNAME=" "$ENV_FILE" | cut -d'=' -f2)
ADMIN_PASS=$(grep -E "^ADMIN_DEFAULT_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
JADE_PORT=$(grep -E "^JADEVECTORDB_PORT=" "$ENV_FILE" | cut -d'=' -f2)

# Fall back to docker-compose defaults if not set in .env
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASS="${ADMIN_PASS:-Admin@1234}"
JADE_PORT="${JADE_PORT:-8081}"
JADE_URL="http://localhost:${JADE_PORT}"

info "Admin user  : ${ADMIN_USER}"
info "JadeVectorDB: ${JADE_URL}"
echo ""

# ── Step 4: Start JadeVectorDB only ──────────────────────────────────────────
info "Starting JadeVectorDB (phase 1 of 2)..."
docker compose up -d jadevectordb

# ── Step 5: Wait for JadeVectorDB to be healthy ───────────────────────────────
info "Waiting for JadeVectorDB to become healthy..."
MAX_WAIT=120
ELAPSED=0
while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${JADE_URL}/health" 2>/dev/null || echo "000")
  if [ "$STATUS" = "200" ]; then
    success "JadeVectorDB is healthy."
    break
  fi
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    error "JadeVectorDB did not become healthy within ${MAX_WAIT}s. Check: docker logs jadevectordb"
  fi
  echo -n "."
  sleep 3
  ELAPSED=$((ELAPSED + 3))
done
echo ""

# ── Step 6: Login and obtain JWT token ────────────────────────────────────────
info "Logging in as '${ADMIN_USER}'..."
LOGIN_RESPONSE=$(curl -s -X POST "${JADE_URL}/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${ADMIN_USER}\",\"password\":\"${ADMIN_PASS}\"}")

TOKEN=$(echo "$LOGIN_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('token',''))" 2>/dev/null)
USER_ID=$(echo "$LOGIN_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('user_id',''))" 2>/dev/null)

if [ -z "$TOKEN" ] || [ -z "$USER_ID" ]; then
  echo ""
  error "Login failed. Response was: ${LOGIN_RESPONSE}
  Check ADMIN_USERNAME and ADMIN_DEFAULT_PASSWORD in .env."
fi
success "Logged in. User ID: ${USER_ID}"

# ── Step 7: Create API key ────────────────────────────────────────────────────
info "Creating service API key..."
KEY_RESPONSE=$(curl -s -X POST "${JADE_URL}/v1/api-keys" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"${USER_ID}\",\"description\":\"EnterpriseRAG service key (bootstrap)\"}")

API_KEY=$(echo "$KEY_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('api_key',''))" 2>/dev/null)

if [ -z "$API_KEY" ]; then
  error "API key creation failed. Response was: ${KEY_RESPONSE}"
fi
success "API key created: ${API_KEY:0:16}... (truncated for display)"

# ── Step 8: Write key into .env ───────────────────────────────────────────────
info "Writing API key to .env..."
# Replace the empty or missing JADEVECTORDB_API_KEY line
if grep -qE "^JADEVECTORDB_API_KEY=" "$ENV_FILE"; then
  # Line exists but empty — replace it
  sed -i "" "s|^JADEVECTORDB_API_KEY=.*|JADEVECTORDB_API_KEY=${API_KEY}|" "$ENV_FILE"
else
  # Line missing entirely — append it
  echo "JADEVECTORDB_API_KEY=${API_KEY}" >> "$ENV_FILE"
fi
success "API key written to .env"

# ── Step 9: Start the full stack ──────────────────────────────────────────────
info "Starting full stack (phase 2 of 2)..."
docker compose up -d

# ── Step 10: Wait for rag-backend health check ────────────────────────────────
info "Waiting for rag-backend to become healthy..."
RAG_PORT=$(grep -E "^RAG_BACKEND_PORT=" "$ENV_FILE" | cut -d'=' -f2)
RAG_PORT="${RAG_PORT:-8000}"
ELAPSED=0
while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${RAG_PORT}/api/health" 2>/dev/null || echo "000")
  if [ "$STATUS" = "200" ]; then
    success "rag-backend is healthy."
    break
  fi
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    warn "rag-backend health check timed out. Check: docker logs rag-backend"
    break
  fi
  echo -n "."
  sleep 3
  ELAPSED=$((ELAPSED + 3))
done
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Bootstrap Complete! 🎉                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
success "JadeVectorDB UI    : http://localhost:3000"
success "EnterpriseRAG      : http://localhost:$(grep -E '^RAG_FRONTEND_PORT=' "$ENV_FILE" | cut -d'=' -f2 || echo 3002)"
success "Grafana            : http://localhost:3001  (admin / admin)"
success "Prometheus         : http://localhost:9090"
echo ""
info "The API key has been saved to .env — it will be loaded"
info "automatically on all future 'docker compose up' runs."
info ""
info "Next run (no bootstrap needed):"
info "  docker compose up -d"
echo ""
