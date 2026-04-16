# Exercise 7: API Key Management

## Learning Objectives

By the end of this exercise, you will be able to:
- Create API keys for users with specific permissions
- Set key expiry for time-limited access
- List and inspect existing API keys
- Use an API key to authenticate CLI and HTTP requests
- Revoke keys when they are no longer needed
- Implement a key rotation workflow for production deployments

## Prerequisites

- Completed Exercise 6: User Management
- JadeVectorDB running at `http://localhost:8080`
- Admin credentials and API key set in your environment
- A test user created (we'll use `alice` from Exercise 6, or create one below)

## Background: Why API Keys?

Passwords are for humans; API keys are for machines. Use API keys when:

| Scenario | Recommended approach |
|----------|---------------------|
| A developer logs into the UI | Password |
| A CI/CD pipeline loads vectors | API key with `database:write` |
| A read-only analytics service queries data | API key with `database:read` |
| A third-party integration | API key with expiry set |
| Short-term contractor access | API key with `--validity-days 30` |

API keys inherit the permissions of the user they belong to, but can be further restricted with an explicit `--permissions` list.

---

## Exercise Steps

### Step 1: Create a Test User

If you don't have a test user from Exercise 6, create one now:

```bash
jade-db user-add apidemo \
  --role user \
  --password "TempPass@2024" \
  --email apidemo@example.com
```

```bash
DEMO_USER_ID=<user-id-from-response>
```

---

### Step 2: Create a Permanent API Key

Create a key with no expiry — suitable for a long-running service:

```bash
jade-db create-api-key \
  --user-id $DEMO_USER_ID \
  --description "Local dev key"
```

Save the key value from the response immediately — it is shown only once:

```bash
DEMO_KEY=<key-value-from-response>
DEMO_KEY_ID=<key-id-from-response>
```

> ⚠️ **The key value is shown only at creation time.** Store it securely (e.g. a password manager or secrets vault). The CLI will only show the key ID afterwards.

---

### Step 3: Create a Scoped Key with Limited Permissions

Create a read-only key that can only query vectors, not modify them:

```bash
jade-db create-api-key \
  --user-id $DEMO_USER_ID \
  --description "Read-only analytics key" \
  --permissions '["database:read"]'
```

```bash
READONLY_KEY_ID=<key-id-from-response>
```

Create a write key for an ingestion pipeline:

```bash
jade-db create-api-key \
  --user-id $DEMO_USER_ID \
  --description "Ingestion pipeline key" \
  --permissions '["database:read", "database:write"]'
```

Available permission values:

| Permission | Grants |
|------------|--------|
| `database:read` | Query and read vectors |
| `database:write` | Insert, update, delete vectors |
| `database:admin` | All operations including permission management |

---

### Step 4: Create a Time-Limited Key

Create a key that expires in 30 days — useful for contractors or short-term integrations:

```bash
jade-db create-api-key \
  --user-id $DEMO_USER_ID \
  --description "Contractor access - 30 days" \
  --validity-days 30
```

```bash
EXPIRING_KEY_ID=<key-id-from-response>
```

---

### Step 5: List API Keys

List all keys in the system:

```bash
jade-db list-api-keys --format table
```

List keys for a specific user only:

```bash
jade-db list-api-keys --user-id $DEMO_USER_ID --format table
```

The output shows each key's ID, description, user, permissions, and expiry date.

✅ **Checkpoint:** You should see three keys for `$DEMO_USER_ID` — the permanent key, the read-only key, and the expiring key.

---

### Step 6: Authenticate Using an API Key

Test that the key works by making a CLI call with it:

```bash
# Override the default API key for a single command
jade-db list-dbs --api-key $DEMO_KEY
```

Or set it as the environment variable for your session:

```bash
export JADE_DB_API_KEY=$DEMO_KEY
jade-db list-dbs
jade-db health
```

Using a key in a direct HTTP request with `curl`:

```bash
curl -s http://localhost:8080/v1/databases \
  -H "Authorization: Bearer $DEMO_KEY" | jq '.'
```

Reset to your admin key when done:

```bash
export JADE_DB_API_KEY=<your-admin-key>
```

---

### Step 7: Key Rotation

Key rotation replaces an old key with a new one without service downtime. The pattern:

1. Create a new key
2. Update your service to use the new key
3. Revoke the old key

```bash
# Step 1: Create the replacement key
jade-db create-api-key \
  --user-id $DEMO_USER_ID \
  --description "Ingestion pipeline key (rotated $(date +%Y-%m-%d))" \
  --permissions '["database:read", "database:write"]'

NEW_KEY=<key-value-from-response>
NEW_KEY_ID=<key-id-from-response>

# Step 2: Update your service config to use $NEW_KEY
# (In a real deployment: update the secret in your CI/CD system or secrets manager)

# Step 3: Revoke the old key
jade-db revoke-api-key --key-id $DEMO_KEY_ID
```

Verify the old key is gone:
```bash
jade-db list-api-keys --user-id $DEMO_USER_ID --format table
```

---

### Step 8: Revoke All Keys for a User

When offboarding a user, revoke all their keys before deleting the account:

```bash
# List all key IDs for the user, then revoke each
jade-db list-api-keys --user-id $DEMO_USER_ID --format json | \
  jq -r '.[].id' | \
  while read key_id; do
    echo "Revoking $key_id..."
    jade-db revoke-api-key --key-id "$key_id"
  done
```

Then delete the user:
```bash
jade-db user-delete $DEMO_USER_ID
```

---

## Automation Example: Key Rotation Script

This script automates key rotation for a named service and updates a local `.env` file:

```bash
#!/bin/bash
# rotate_key.sh <user-id> <env-file>
# Usage: ./rotate_key.sh abc123 /etc/myservice/.env

USER_ID="$1"
ENV_FILE="$2"
TODAY=$(date +%Y-%m-%d)

if [[ -z "$USER_ID" || -z "$ENV_FILE" ]]; then
  echo "Usage: $0 <user-id> <env-file>"
  exit 1
fi

echo "Creating new API key for user $USER_ID..."
RESULT=$(jade-db create-api-key \
  --user-id "$USER_ID" \
  --description "Service key (rotated $TODAY)" \
  --permissions '["database:read", "database:write"]')

NEW_KEY=$(echo "$RESULT" | jq -r '.key')
NEW_KEY_ID=$(echo "$RESULT" | jq -r '.id')

if [[ -z "$NEW_KEY" || "$NEW_KEY" == "null" ]]; then
  echo "❌ Failed to create new key"
  exit 1
fi

# Update the env file
sed -i.bak "s/^JADE_DB_API_KEY=.*/JADE_DB_API_KEY=$NEW_KEY/" "$ENV_FILE"
echo "✅ Updated $ENV_FILE with new key"

# Revoke all old keys (keep only the new one)
jade-db list-api-keys --user-id "$USER_ID" --format json | \
  jq -r '.[].id' | \
  while read key_id; do
    [[ "$key_id" == "$NEW_KEY_ID" ]] && continue
    echo "Revoking old key $key_id..."
    jade-db revoke-api-key --key-id "$key_id"
  done

echo "✅ Key rotation complete. New key ID: $NEW_KEY_ID"
```

---

## Verification

Expected outcomes after completing all steps:
- ✅ Created a permanent key, a scoped read-only key, and an expiring 30-day key
- ✅ Authenticated CLI calls using the key value
- ✅ Successfully rotated a key (create new → revoke old)
- ✅ Revoked all keys before deleting the test user

---

## Challenges (Optional)

1. **Challenge 1:** Write a script that lists all keys expiring within the next 7 days and sends a warning
2. **Challenge 2:** Create a key for each of three environments (dev, staging, prod) with different descriptions and compare the `list-api-keys` output
3. **Challenge 3:** Extend the rotation script to post a Slack/webhook notification after successful rotation

---

## Best Practices

1. **One key per service** — never share a key between services; if one is compromised, you can revoke it without affecting others
2. **Use the minimum permissions needed** — read-only services should get `database:read` only, not `database:write`
3. **Set expiry for external access** — always use `--validity-days` for contractors, partners, or third-party integrations
4. **Rotate keys regularly** — use the rotation script on a schedule (e.g., monthly via cron)
5. **Never commit keys to source control** — use environment variables or a secrets manager
6. **Revoke before offboarding** — always revoke a user's keys before deleting their account

---

## Next Steps

- **Exercise 8:** Advanced Features — hybrid search, reranking, embeddings, and analytics

## See Also

- `docs/rbac_api_reference.md` — REST API reference for API key endpoints
- `docs/rbac_admin_guide.md` — production security guide including key management policies
- `cli/RBAC_COMMANDS_REFERENCE.md` — full CLI reference for all key management commands
