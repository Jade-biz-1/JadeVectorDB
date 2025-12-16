# CLI RBAC Commands Reference (Future)

**Status**: 📋 PLANNED - Not Yet Implemented  
**Related Epic**: T11-PERSISTENCE  
**Implementation**: Phase 1+2 (Weeks 1-3)

---

## Overview

This document specifies the new RBAC (Role-Based Access Control) commands that will be added to all three JadeVectorDB CLI tools as part of the persistent storage implementation.

---

## Python CLI Commands

### Group Management

```bash
# Create a new group
jade-db group create <group-name> \
  --description "Team description" \
  --owner <user-id>

# List all groups
jade-db group list

# Get group details
jade-db group get <group-id>

# Delete a group
jade-db group delete <group-id>

# Add user to group
jade-db group add-user <group-id> <user-id>

# Remove user from group
jade-db group remove-user <group-id> <user-id>

# List group members
jade-db group members <group-id>
```

### Role Management

```bash
# List all roles
jade-db role list

# Get role details (including permissions)
jade-db role get <role-id>

# Assign role to user
jade-db role assign <user-id> <role-id>

# Revoke role from user
jade-db role revoke <user-id> <role-id>

# List user's roles
jade-db user roles <user-id>
```

### Permission Management

```bash
# List all available permissions
jade-db permission list

# List user's effective permissions
jade-db permission list-user <user-id>

# Grant database-level permission to user
jade-db permission grant \
  --database <database-id> \
  --user <user-id> \
  --permission <permission-name>

# Grant database-level permission to group
jade-db permission grant \
  --database <database-id> \
  --group <group-id> \
  --permission <permission-name>

# Revoke database-level permission
jade-db permission revoke \
  --database <database-id> \
  --user <user-id> \
  --permission <permission-name>

# Check if user has permission on database
jade-db permission check \
  --database <database-id> \
  --user <user-id> \
  --permission <permission-name>
```

### API Key Management

```bash
# Create API key for current user
jade-db api-key create <key-name> \
  --scopes "databases.read,databases.write" \
  --expires-in-days 365

# List current user's API keys
jade-db api-key list

# Get API key details
jade-db api-key get <api-key-id>

# Revoke API key
jade-db api-key revoke <api-key-id>

# Test API key authentication
jade-db --api-key <key> health
```

### Enhanced User Management

```bash
# Create user with roles
jade-db user create <username> \
  --email <email> \
  --password <password> \
  --roles "user,developer"

# Update user roles
jade-db user update <user-id> --roles "admin"

# List user's groups
jade-db user groups <user-id>

# Get user's full profile (roles, groups, permissions)
jade-db user profile <user-id>
```

---

## Shell CLI Commands

### Group Management

```bash
# Create group
./jade-db.sh group-create <group-name> \
  --description "Team description" \
  --owner <user-id>

# List groups
./jade-db.sh group-list

# Add user to group
./jade-db.sh group-add-user <group-id> <user-id>

# Remove user from group
./jade-db.sh group-remove-user <group-id> <user-id>
```

### Role Management

```bash
# List roles
./jade-db.sh role-list

# Assign role
./jade-db.sh role-assign <user-id> <role-id>

# Revoke role
./jade-db.sh role-revoke <user-id> <role-id>

# List user roles
./jade-db.sh user-roles <user-id>
```

### Permission Management

```bash
# List permissions
./jade-db.sh permission-list

# Grant database permission
./jade-db.sh permission-grant \
  <database-id> <user-id> <permission-name>

# Revoke database permission
./jade-db.sh permission-revoke \
  <database-id> <user-id> <permission-name>

# Check permission
./jade-db.sh permission-check \
  <database-id> <user-id> <permission-name>
```

### API Key Management

```bash
# Create API key
./jade-db.sh api-key-create <key-name> \
  --scopes "databases.read,databases.write" \
  --expires-days 365

# List API keys
./jade-db.sh api-key-list

# Revoke API key
./jade-db.sh api-key-revoke <api-key-id>

# Use API key for authentication
./jade-db.sh --api-key <key> health
```

---

## JavaScript CLI Commands

### Group Management

```bash
# Create group
jade-db group create <group-name> \
  --description "Team description" \
  --owner <user-id>

# List groups
jade-db group list

# Add user to group
jade-db group add-member <group-id> <user-id>

# Remove user from group
jade-db group remove-member <group-id> <user-id>
```

### Role Management

```bash
# List roles
jade-db role list

# Assign role
jade-db role assign <user-id> <role-id>

# Revoke role
jade-db role revoke <user-id> <role-id>
```

### Permission Management

```bash
# Grant database permission
jade-db permission grant \
  --db <database-id> \
  --user <user-id> \
  --perm <permission-name>

# Revoke database permission
jade-db permission revoke \
  --db <database-id> \
  --user <user-id> \
  --perm <permission-name>
```

### API Key Management

```bash
# Create API key
jade-db apikey create <key-name> \
  --scopes databases.read,databases.write \
  --expires 365

# List API keys
jade-db apikey list

# Revoke API key
jade-db apikey revoke <api-key-id>
```

---

## Common Usage Patterns

### Setting Up a Team

```bash
# 1. Create group
jade-db group create data-science-team \
  --description "Data Science Team" \
  --owner admin_user_id

# 2. Add team members
jade-db group add-user ds-team-id user1_id
jade-db group add-user ds-team-id user2_id
jade-db group add-user ds-team-id user3_id

# 3. Grant team access to databases
jade-db permission grant \
  --database prod_vectors_db \
  --group ds-team-id \
  --permission database.read

jade-db permission grant \
  --database dev_vectors_db \
  --group ds-team-id \
  --permission database.write
```

### Setting Up a Service Account

```bash
# 1. Create service user
jade-db user create ml-pipeline-service \
  --email ml-service@example.com \
  --password <secure-password> \
  --roles "user"

# 2. Grant specific permissions
jade-db permission grant \
  --database embeddings_db \
  --user ml-service-id \
  --permission database.write

# 3. Create long-lived API key
jade-db api-key create "ML Pipeline Key" \
  --scopes "databases.write,databases.read" \
  --expires-in-days 730  # 2 years

# 4. Use API key in application
export JADE_API_KEY="jade_key_abc123..."
jade-db --api-key $JADE_API_KEY create-db ml_embeddings
```

### Granting Read-Only Access

```bash
# 1. Assign ReadOnly role to user
jade-db role assign analyst_user_id role_readonly

# 2. Grant read permission to specific database
jade-db permission grant \
  --database analytics_vectors \
  --user analyst_user_id \
  --permission database.read

# 3. Verify permissions
jade-db permission check \
  --database analytics_vectors \
  --user analyst_user_id \
  --permission database.read
# Output: ✓ Permission granted

jade-db permission check \
  --database analytics_vectors \
  --user analyst_user_id \
  --permission database.write
# Output: ✗ Permission denied
```

---

## Authentication Methods

### 1. Username/Password (Interactive)

```bash
# Login to get token
jade-db auth login
# Prompts for username and password
# Saves token for subsequent commands
```

### 2. API Key (Programmatic)

```bash
# Option A: Command-line flag
jade-db --api-key jade_key_abc123... create-db my_database

# Option B: Environment variable
export JADE_API_KEY="jade_key_abc123..."
jade-db create-db my_database
```

### 3. Token (Session-based)

```bash
# Option A: Command-line flag
jade-db --token eyJhbGc... create-db my_database

# Option B: Environment variable
export JADE_TOKEN="eyJhbGc..."
jade-db create-db my_database
```

---

## Permission Types

### Database Permissions

| Permission Name | Description | Grants Access To |
|----------------|-------------|------------------|
| `database.read` | Read access | List vectors, search, get database info |
| `database.write` | Write access | Insert vectors, update vectors |
| `database.delete` | Delete access | Delete vectors, delete indexes |
| `database.admin` | Full admin access | All operations + configuration changes |
| `database.create` | Create databases | Create new database instances |

### System Permissions

| Permission Name | Description | Grants Access To |
|----------------|-------------|------------------|
| `system.admin` | System administration | User management, system configuration |
| `system.monitor` | Monitoring access | Health checks, metrics, logs |

---

## Predefined Roles

### Admin Role
- **Permissions**: All system and database permissions
- **Use Case**: System administrators

### User Role
- **Permissions**: `database.create`, `database.read`, `database.write`
- **Use Case**: Standard application users

### ReadOnly Role
- **Permissions**: `database.read`
- **Use Case**: Analysts, reporting systems

### DataScientist Role
- **Permissions**: `database.create`, `database.read`, `database.write`, `database.admin`
- **Use Case**: ML engineers, data scientists

---

## Error Handling

### Common Error Messages

```bash
# Insufficient permissions
$ jade-db create-db restricted_db
Error: Permission denied
  Required permission: database.create
  User: john_doe (user_abc123)
  Hint: Contact your administrator to grant database.create permission

# Invalid API key
$ jade-db --api-key invalid_key health
Error: Authentication failed
  Reason: Invalid API key
  Hint: Check your API key or generate a new one with: jade-db api-key create

# Expired token
$ jade-db --token expired_token create-db my_db
Error: Authentication failed
  Reason: Token expired
  Hint: Login again with: jade-db auth login
```

---

## Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Authentication error |
| 3 | Permission denied |
| 4 | Invalid arguments |
| 5 | Resource not found |
| 6 | Conflict (resource already exists) |

---

## Testing

### Verify RBAC Setup

```bash
# Test script to verify RBAC functionality
#!/bin/bash

echo "Testing RBAC functionality..."

# 1. Create test group
GROUP_ID=$(jade-db group create test-group --owner admin_id | jq -r '.group_id')
echo "✓ Created group: $GROUP_ID"

# 2. Create test user
USER_ID=$(jade-db user create test-user --email test@example.com --password Test123! | jq -r '.user_id')
echo "✓ Created user: $USER_ID"

# 3. Add user to group
jade-db group add-user $GROUP_ID $USER_ID
echo "✓ Added user to group"

# 4. Assign role
jade-db role assign $USER_ID role_user
echo "✓ Assigned role"

# 5. Create database
DB_ID=$(jade-db create-db test-db --dimension 128 | jq -r '.database_id')
echo "✓ Created database: $DB_ID"

# 6. Grant permission
jade-db permission grant --database $DB_ID --user $USER_ID --permission database.read
echo "✓ Granted permission"

# 7. Verify permission
jade-db permission check --database $DB_ID --user $USER_ID --permission database.read
echo "✓ Permission verified"

echo "All RBAC tests passed!"
```

---

## Implementation Tasks

This command reference corresponds to the following implementation tasks:

- **T11.15.1**: Python CLI RBAC commands
- **T11.16.1**: Shell CLI RBAC commands
- **T11.17.1**: JavaScript CLI RBAC commands
- **T11.18.1**: Cross-CLI consistency verification

See `TasksTracking/11-persistent-storage-implementation.md` for details.

---

**Status**: 📋 PLANNED  
**Review Status**: Pending team review  
**Implementation**: Weeks 1-3 (Phase 1+2)  
**Last Updated**: December 16, 2025
