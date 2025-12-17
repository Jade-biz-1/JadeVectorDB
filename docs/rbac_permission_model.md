# JadeVectorDB Permission Model Deep Dive

**Last Updated**: December 17, 2025  
**Version**: 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Permission Types](#permission-types)
4. [Role Hierarchy](#role-hierarchy)
5. [Permission Inheritance](#permission-inheritance)
6. [Permission Resolution](#permission-resolution)
7. [Database Schema](#database-schema)
8. [Permission Checking Algorithm](#permission-checking-algorithm)
9. [Performance Characteristics](#performance-characteristics)
10. [Security Considerations](#security-considerations)

---

## Overview

JadeVectorDB implements a hybrid RBAC (Role-Based Access Control) and ABAC (Attribute-Based Access Control) system that provides:

- **Role-based permissions** for system-level access
- **Resource-level permissions** for database access
- **Group-based permissions** for team management
- **Hierarchical permission inheritance**
- **Efficient permission caching**

### Design Goals

1. **Security First**: Default-deny, explicit permissions required
2. **Performance**: Sub-millisecond permission checks (benchmarked at 0.01ms)
3. **Flexibility**: Support complex organizational structures
4. **Auditability**: All permission changes logged
5. **Scalability**: Support thousands of users and databases

---

## Core Concepts

### Principals

A **principal** is an entity that can be granted permissions:

- **User**: Individual user account
- **Group**: Collection of users
- **API Key**: Programmatic access credential (inherits user permissions)

### Resources

A **resource** is an entity that can be protected:

- **System**: JadeVectorDB instance itself
- **Database**: Individual vector database
- **Collection**: (Future) Subset of vectors within database
- **Index**: (Future) Search index

### Permissions

A **permission** is an action that can be performed on a resource:

- Format: `<resource_type>:<action>`
- Examples: `database:read`, `database:write`, `database:admin`

### Roles

A **role** is a named collection of permissions:

- System-level abstractions (admin, user, readonly)
- Assigned to users or groups
- Provide baseline capabilities

---

## Permission Types

### System-Level Permissions

Controlled by roles:

| Role | Permissions | Description |
|------|-------------|-------------|
| `role_admin` | Full system access | Create users, assign roles, manage all databases |
| `role_user` | Standard user | Create databases, manage own resources |
| `role_readonly` | Read-only access | View databases, query vectors (no modifications) |

### Database-Level Permissions

Granular control per database:

| Permission | Actions Allowed |
|------------|-----------------|
| `database:admin` | All operations, grant/revoke permissions |
| `database:delete` | Delete database (destructive) |
| `database:write` | Insert, update, delete vectors |
| `database:read` | Query and read vectors |

### Permission Hierarchy

Permissions follow a hierarchy:

```
database:admin
    ├── database:delete
    ├── database:write
    │   └── database:read
    └── database:read
```

**Inheritance Rule**: Higher permissions include lower permissions.
- `database:admin` includes `database:delete`, `database:write`, `database:read`
- `database:write` includes `database:read`
- `database:read` is the minimum permission

---

## Role Hierarchy

### System Role Hierarchy

```
role_admin
    ├── Full system access
    ├── User management
    ├── Role assignment
    ├── All database permissions
    └── Audit log access

role_user
    ├── Create databases
    ├── Manage own databases
    ├── API key management
    └── Read own audit logs

role_readonly
    └── Read-only access to permitted databases
```

### Role Capabilities Matrix

| Capability | Admin | User | Readonly |
|------------|-------|------|----------|
| Create users | ✅ | ❌ | ❌ |
| Assign roles | ✅ | ❌ | ❌ |
| Create databases | ✅ | ✅ | ❌ |
| Grant DB permissions | ✅ | Own DBs | ❌ |
| Write vectors | ✅ | With permission | ❌ |
| Read vectors | ✅ | With permission | With permission |
| View audit logs | All | Own | Own |
| Manage API keys | ✅ | Own | Own |

---

## Permission Inheritance

### Group-Based Inheritance

Users inherit permissions from all groups they belong to:

```
User: alice
├── Direct roles: role_user
├── Direct permissions: database:read on db_personal
└── Group: developers
    ├── Group roles: (none)
    └── Group permissions: database:write on db_shared

Effective permissions:
- database:read on db_personal (direct)
- database:write on db_shared (from group)
- database:read on db_shared (inherited from database:write)
```

### Database Owner Privileges

Database creators automatically receive `database:admin`:

```sql
-- When user creates database
INSERT INTO databases (id, name, owner_user_id) 
VALUES ('db_123', 'vectors_db', 'user_alice');

-- Automatically grants
INSERT INTO database_permissions (database_id, user_id, permission)
VALUES ('db_123', 'user_alice', 'database:admin');
```

### Permission Accumulation

Permissions are **additive** (union of all sources):

```
alice's permissions on db_shared:
├── From user alice: database:read (direct grant)
├── From group developers: database:write (includes database:read)
└── From group admins: database:admin (includes all)

Result: database:admin (highest permission wins)
```

---

## Permission Resolution

### Resolution Algorithm

When checking if user `U` has permission `P` on database `D`:

1. **Admin Override**: If user has `role_admin`, grant access
2. **Direct Permission**: Check user's direct database permissions
3. **Group Permissions**: Check all groups user belongs to
4. **Permission Hierarchy**: Check if user has higher permission
5. **Database Owner**: Check if user owns the database
6. **Default Deny**: If none match, deny access

### Pseudocode

```python
def has_permission(user_id, database_id, required_permission):
    # Step 1: Check if user is admin
    if user.has_role("role_admin"):
        return True
    
    # Step 2: Check direct user permissions
    user_perms = get_user_permissions(user_id, database_id)
    if required_permission in user_perms:
        return True
    
    # Step 3: Check higher permissions (hierarchy)
    for perm in user_perms:
        if permission_includes(perm, required_permission):
            return True
    
    # Step 4: Check group permissions
    user_groups = get_user_groups(user_id)
    for group in user_groups:
        group_perms = get_group_permissions(group.id, database_id)
        if required_permission in group_perms:
            return True
        # Check hierarchy for group perms
        for perm in group_perms:
            if permission_includes(perm, required_permission):
                return True
    
    # Step 5: Check database ownership
    database = get_database(database_id)
    if database.owner_user_id == user_id:
        return True
    
    # Step 6: Default deny
    return False

def permission_includes(higher_perm, lower_perm):
    """Check if higher permission includes lower permission"""
    hierarchy = {
        "database:admin": ["database:delete", "database:write", "database:read"],
        "database:write": ["database:read"],
        "database:delete": []
    }
    
    if higher_perm == lower_perm:
        return True
    
    return lower_perm in hierarchy.get(higher_perm, [])
```

### Example Resolution

**Scenario**: Check if `alice` can write to `db_shared`

```
User: alice (user_123)
Database: db_shared (db_456)
Required: database:write

Resolution steps:
1. Is alice admin? → Check roles table
   - alice has role_user → NOT ADMIN
   
2. Direct permissions? → Check database_permissions
   - alice has database:read on db_456 → INSUFFICIENT
   
3. Higher permissions? → Check permission hierarchy
   - database:read does NOT include database:write → NO
   
4. Group permissions? → Check group_memberships + database_permissions
   - alice in group 'developers' (group_789)
   - group_789 has database:write on db_456 → MATCH!
   
Result: GRANT (via group membership)
```

---

## Database Schema

### Core Tables

#### users
```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

#### roles
```sql
CREATE TABLE roles (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

-- Default roles
INSERT INTO roles (id, name, description) VALUES
    ('role_admin', 'Administrator', 'Full system access'),
    ('role_user', 'User', 'Standard user access'),
    ('role_readonly', 'Read-only', 'Read-only access');
```

#### user_roles
```sql
CREATE TABLE user_roles (
    user_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    granted_by TEXT,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
);

CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);
```

#### databases
```sql
CREATE TABLE databases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_databases_owner ON databases(owner_user_id);
```

#### database_permissions
```sql
CREATE TABLE database_permissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    database_id TEXT NOT NULL,
    user_id TEXT,
    group_id TEXT,
    permission TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    granted_by TEXT,
    FOREIGN KEY (database_id) REFERENCES databases(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
    CHECK ((user_id IS NOT NULL AND group_id IS NULL) OR 
           (user_id IS NULL AND group_id IS NOT NULL))
);

CREATE INDEX idx_db_perms_database ON database_permissions(database_id);
CREATE INDEX idx_db_perms_user ON database_permissions(user_id);
CREATE INDEX idx_db_perms_group ON database_permissions(group_id);
CREATE INDEX idx_db_perms_permission ON database_permissions(permission);
```

#### groups
```sql
CREATE TABLE groups (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);
```

#### group_memberships
```sql
CREATE TABLE group_memberships (
    user_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (user_id, group_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
);

CREATE INDEX idx_group_memberships_user ON group_memberships(user_id);
CREATE INDEX idx_group_memberships_group ON group_memberships(group_id);
```

### Indexes for Performance

Critical indexes for sub-millisecond queries:

1. **Permission Lookup**: `idx_db_perms_user`, `idx_db_perms_group`
2. **Role Check**: `idx_user_roles_user`
3. **Group Membership**: `idx_group_memberships_user`
4. **Database Owner**: `idx_databases_owner`

---

## Permission Checking Algorithm

### C++ Implementation

```cpp
bool SQLitePersistenceLayer::check_database_permission(
    const std::string& user_id,
    const std::string& database_id,
    const std::string& required_permission) const {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    // Step 1: Check admin role
    sqlite3_stmt* stmt;
    const char* sql = R"(
        SELECT 1 FROM user_roles 
        WHERE user_id = ? AND role_id = 'role_admin'
        LIMIT 1
    )";
    
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return true; // Admin has all permissions
        }
        sqlite3_finalize(stmt);
    }
    
    // Step 2 & 3: Check direct user permissions (including hierarchy)
    sql = R"(
        SELECT permission FROM database_permissions
        WHERE database_id = ? AND user_id = ?
    )";
    
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string perm = reinterpret_cast<const char*>(
                sqlite3_column_text(stmt, 0));
            
            if (perm == required_permission || 
                permission_includes(perm, required_permission)) {
                sqlite3_finalize(stmt);
                return true;
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Step 4: Check group permissions
    sql = R"(
        SELECT dp.permission 
        FROM database_permissions dp
        JOIN group_memberships gm ON dp.group_id = gm.group_id
        WHERE dp.database_id = ? AND gm.user_id = ?
    )";
    
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string perm = reinterpret_cast<const char*>(
                sqlite3_column_text(stmt, 0));
            
            if (perm == required_permission || 
                permission_includes(perm, required_permission)) {
                sqlite3_finalize(stmt);
                return true;
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Step 5: Check database ownership
    sql = "SELECT owner_user_id FROM databases WHERE id = ?";
    
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string owner = reinterpret_cast<const char*>(
                sqlite3_column_text(stmt, 0));
            sqlite3_finalize(stmt);
            if (owner == user_id) {
                return true; // Owner has all permissions
            }
        } else {
            sqlite3_finalize(stmt);
        }
    }
    
    // Step 6: Default deny
    return false;
}

bool SQLitePersistenceLayer::permission_includes(
    const std::string& higher, const std::string& lower) const {
    
    if (higher == lower) return true;
    
    if (higher == "database:admin") {
        return lower == "database:delete" || 
               lower == "database:write" || 
               lower == "database:read";
    }
    
    if (higher == "database:write") {
        return lower == "database:read";
    }
    
    return false;
}
```

### SQL Optimization

**Single-Query Permission Check** (fastest):

```sql
-- Check all permission sources in one query
SELECT 1 FROM (
    -- Check admin role
    SELECT 1 FROM user_roles 
    WHERE user_id = ? AND role_id = 'role_admin'
    
    UNION ALL
    
    -- Check direct user permissions
    SELECT 1 FROM database_permissions
    WHERE database_id = ? AND user_id = ?
      AND (permission = ? OR 
           permission IN (SELECT parent FROM permission_hierarchy WHERE child = ?))
    
    UNION ALL
    
    -- Check group permissions
    SELECT 1 FROM database_permissions dp
    JOIN group_memberships gm ON dp.group_id = gm.group_id
    WHERE dp.database_id = ? AND gm.user_id = ?
      AND (dp.permission = ? OR 
           dp.permission IN (SELECT parent FROM permission_hierarchy WHERE child = ?))
    
    UNION ALL
    
    -- Check database ownership
    SELECT 1 FROM databases
    WHERE id = ? AND owner_user_id = ?
) LIMIT 1;
```

---

## Performance Characteristics

### Benchmarked Performance

**From `test_performance_benchmark.cpp`**:

| Operation | Target | Actual | Factor |
|-----------|--------|--------|--------|
| Permission check | 500ms | 0.01ms | **500x faster** |
| User creation | 500ms | 0.51ms | **980x faster** |
| Role assignment | 100ms | 0.51ms | **196x faster** |
| Concurrent operations (1000) | 10s | 0.232s | **43x faster** |

### Scalability Characteristics

**Users**:
- Up to 100,000 users: O(1) permission checks (indexed)
- Up to 1,000,000 users: O(log n) with proper indexing

**Groups**:
- Up to 10,000 groups: O(1) per group lookup
- Avoid deeply nested groups (max 5 levels recommended)

**Databases**:
- Up to 1,000,000 databases: O(1) per database permission check

**Permissions**:
- Permission check: **0.01ms average**
- Permission grant: **0.5ms average**
- Audit log write: **0.1ms average**

### Caching Strategy

**In-Memory Cache** (optional, configurable):

```cpp
// Cache user permissions for fast repeated checks
struct PermissionCacheEntry {
    std::string user_id;
    std::string database_id;
    std::string permission;
    bool granted;
    std::chrono::steady_clock::time_point cached_at;
};

// Cache expires after 5 minutes
constexpr auto CACHE_TTL = std::chrono::minutes(5);

// Cache invalidated on:
// - Permission grant/revoke
// - Role assignment/revocation
// - Group membership change
// - User activation/deactivation
```

**Benefits**:
- Permission checks: 0.01ms → 0.001ms (10x faster)
- Reduced database load for repeated checks

**Trade-offs**:
- Slight delay in permission changes (max 5 minutes)
- Increased memory usage (~100 bytes per cached permission)

---

## Security Considerations

### Default Deny

The system uses **default-deny** approach:
- No implicit permissions
- All access must be explicitly granted
- Unknown resources are inaccessible

### Privilege Escalation Prevention

1. **Role Assignment**: Only admins can assign roles
2. **Permission Grants**: Only database admins can grant permissions
3. **Group Management**: Only admins can create groups
4. **Audit Logging**: All permission changes logged

### SQL Injection Prevention

All queries use parameterized statements:

```cpp
// SAFE: Parameterized
sqlite3_prepare_v2(db_, "SELECT * FROM users WHERE id = ?", -1, &stmt, nullptr);
sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);

// UNSAFE: String concatenation (NEVER DO THIS)
std::string sql = "SELECT * FROM users WHERE id = '" + user_id + "'";
```

### Time-of-Check to Time-of-Use (TOCTOU)

Permission checks are atomic with resource access:

```cpp
// SAFE: Check and access in same transaction
BEGIN TRANSACTION;
SELECT check_permission(user_id, database_id, 'database:write');
INSERT INTO vectors (database_id, data) VALUES (?, ?);
COMMIT;

// UNSAFE: Separate check and access (race condition)
if (check_permission(...)) {  // Check
    // Time passes... permission might be revoked here
    insert_vector(...);        // Use
}
```

### Audit Trail Integrity

Audit logs are append-only and tamper-evident:

```sql
-- Audit logs cannot be updated or deleted
CREATE TRIGGER prevent_audit_update
BEFORE UPDATE ON audit_logs
BEGIN
    SELECT RAISE(ABORT, 'Audit logs cannot be modified');
END;

CREATE TRIGGER prevent_audit_delete
BEFORE DELETE ON audit_logs
BEGIN
    SELECT RAISE(ABORT, 'Audit logs cannot be deleted');
END;
```

---

## Future Enhancements

### Planned Features

1. **Attribute-Based Access Control (ABAC)**:
   - Time-based permissions (9am-5pm only)
   - IP-based restrictions
   - Device-based restrictions

2. **Dynamic Roles**:
   - Custom roles with configurable permissions
   - Role templates for common patterns

3. **Permission Delegation**:
   - Users can delegate their permissions temporarily
   - "Share with" functionality

4. **Advanced Auditing**:
   - Real-time alerts on suspicious activity
   - ML-based anomaly detection
   - Compliance reporting (SOC 2, GDPR)

5. **Multi-Tenancy**:
   - Organization-level isolation
   - Cross-organization permissions (B2B)

---

## References

- **NIST RBAC Standard**: [NIST RBAC Model](https://csrc.nist.gov/projects/role-based-access-control)
- **OWASP Access Control**: [OWASP Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- **SQLite Security**: [SQLite Security](https://www.sqlite.org/security.html)

---

**Last Updated**: December 17, 2025  
**Version**: 1.0  
**Next Review**: March 17, 2026
