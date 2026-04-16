# Exercise 6: User Management

## Learning Objectives

By the end of this exercise, you will be able to:
- Create users with specific roles via the CLI
- List and inspect user accounts
- Update user roles and status
- Activate and deactivate users
- Delete users safely
- Change passwords
- View the audit log to track administrative activity

## Prerequisites

- Completed Exercise 1: CLI Basics
- JadeVectorDB running at `http://localhost:8080`
- Admin credentials (default: `admin` / `admin123` for local development)
- Logged in as an admin: `export JADE_DB_API_KEY=<your-admin-api-key>`

## Background: Roles

JadeVectorDB uses Role-Based Access Control (RBAC). The two built-in roles are:

| Role | Access |
|------|--------|
| `admin` | Full system access — create users, assign roles, manage all databases |
| `user` | Standard access — create databases, manage own resources |

---

## Exercise Steps

### Step 1: List Existing Users

Before creating any users, see who already exists:

```bash
jade-db user-list
```

For a more readable view:
```bash
jade-db user-list --format table
```

Filter by role to see only admins:
```bash
jade-db user-list --role admin --format table
```

Filter by status to see only active accounts:
```bash
jade-db user-list --status active --format table
```

✅ **Checkpoint:** Note how many users exist and what roles they have.

---

### Step 2: Create a Developer User

Create a standard user account for a developer on your team:

```bash
jade-db user-add alice \
  --role user \
  --password "SecurePass@2024" \
  --email alice@example.com
```

Save the user ID returned in the response — you'll use it throughout this exercise.

```bash
# Store the user ID for convenience
ALICE_ID=<user-id-from-response>
```

✅ **Checkpoint:** Run `jade-db user-list --format table` and confirm Alice appears.

---

### Step 3: Create an Admin User

Create a second user with admin privileges:

```bash
jade-db user-add bob \
  --role admin \
  --password "AdminPass@2024" \
  --email bob@example.com
```

```bash
BOB_ID=<user-id-from-response>
```

✅ **Checkpoint:** Run `jade-db user-list --role admin --format table` — both the original admin and Bob should appear.

---

### Step 4: Inspect a User

View the full details of a user account:

```bash
jade-db user-show $ALICE_ID
```

The response includes the user's ID, username, email, role, status, and creation timestamp.

For a cleaner view:
```bash
jade-db user-show $ALICE_ID --format yaml
```

---

### Step 5: Update a User's Role

Alice has been promoted — update her role to admin:

```bash
jade-db user-update $ALICE_ID --role admin
```

Verify the change:
```bash
jade-db user-show $ALICE_ID --format table
```

Now demote her back to a standard user:
```bash
jade-db user-update $ALICE_ID --role user
```

---

### Step 6: Deactivate a User

When a team member leaves or needs their access suspended, deactivate rather than delete:

```bash
jade-db user-deactivate $BOB_ID
```

Verify the status changed:
```bash
jade-db user-show $BOB_ID --format table
```

Check that Bob no longer appears in the active users list:
```bash
jade-db user-list --status active --format table
```

But still appears when listing all users:
```bash
jade-db user-list --format table
```

---

### Step 7: Reactivate a User

Reactivate Bob's account:

```bash
jade-db user-activate $BOB_ID
```

```bash
jade-db user-list --status active --format table
```

✅ **Checkpoint:** Bob should be back in the active list.

---

### Step 8: Change a Password

Change Alice's password:

```bash
jade-db change-password \
  --user-id $ALICE_ID \
  --old-password "SecurePass@2024" \
  --new-password "NewSecurePass@2025"
```

---

### Step 9: View the Audit Log

Every administrative action is recorded. Review what happened during this exercise:

```bash
jade-db audit-log --limit 20
```

Filter to see only events for Alice:
```bash
jade-db audit-log --user-id $ALICE_ID
```

Filter by event type to see only role changes:
```bash
jade-db audit-log --event-type role_change --limit 10
```

✅ **Checkpoint:** You should see entries for user creation, role update, deactivation, reactivation, and password change.

---

### Step 10: Delete a User

When you're done, clean up the test accounts:

```bash
jade-db user-delete $BOB_ID
```

```bash
jade-db user-delete $ALICE_ID
```

Verify they're gone:
```bash
jade-db user-list --format table
```

> ⚠️ **Warning:** Deletion is permanent. Always prefer `user-deactivate` over `user-delete` for real accounts — deactivated users can be restored, deleted users cannot.

---

## Automation Example: Onboard Multiple Users

Real deployments often need to create many users at once. Here's a script that reads a CSV file and creates accounts in bulk:

```bash
#!/bin/bash
# onboard_users.sh
# CSV format: username,role,email,password

INPUT_FILE="new_users.csv"
SUCCESS=0
FAILED=0

while IFS=',' read -r username role email password; do
  # Skip header line
  [[ "$username" == "username" ]] && continue

  echo "Creating user: $username ($role)..."
  if jade-db user-add "$username" \
      --role "$role" \
      --email "$email" \
      --password "$password" > /dev/null 2>&1; then
    ((SUCCESS++))
    echo "  ✅ $username created"
  else
    ((FAILED++))
    echo "  ❌ Failed to create $username"
  fi
done < "$INPUT_FILE"

echo ""
echo "Onboarding complete: $SUCCESS succeeded, $FAILED failed"
```

**Sample `new_users.csv`:**
```
username,role,email,password
carol,user,carol@example.com,TempPass@001
dave,user,dave@example.com,TempPass@002
eve,admin,eve@example.com,AdminTemp@001
```

---

## Verification

Expected outcomes after completing all steps:
- ✅ Created two users (alice, bob) with different roles
- ✅ Updated alice's role from `user` to `admin` and back
- ✅ Deactivated and reactivated bob
- ✅ Changed alice's password
- ✅ Audit log shows a history of all changes
- ✅ Both test users deleted cleanly

---

## Challenges (Optional)

1. **Challenge 1:** Write a script that exports all active users to a CSV file using `jade-db user-list --format csv`
2. **Challenge 2:** Write a script that deactivates all users who haven't been seen in the audit log for the past 30 days
3. **Challenge 3:** Create a role-rotation script that temporarily elevates a `user` to `admin`, performs an operation, then reverts the role

---

## Best Practices

1. **Deactivate, don't delete** — deactivated accounts retain audit history and can be restored
2. **Use strong passwords** — enforce a minimum of 12 characters with mixed case, numbers, and symbols
3. **Assign least-privilege roles** — give users the `user` role unless admin access is specifically required
4. **Review the audit log regularly** — use `jade-db audit-log` in your monitoring scripts to detect unexpected changes
5. **Automate onboarding** — use the bulk-create script pattern for new team deployments

---

## Next Steps

- **Exercise 7:** API Key Management — create and rotate keys for programmatic access
- **Exercise 8:** Advanced Features — hybrid search, reranking, embeddings, and analytics

## See Also

- `docs/rbac_permission_model.md` — deep dive into role permissions and access control logic
- `docs/rbac_admin_guide.md` — production admin guide
- `docs/rbac_api_reference.md` — REST API reference for all user management endpoints
- `cli/RBAC_COMMANDS_REFERENCE.md` — full CLI command reference for RBAC
