# Persistent Storage Implementation Plan

**Epic ID**: T11-PERSISTENCE
**Priority**: CRITICAL (P0)
**Status**: PLANNING
**Created**: 2024-12-16
**Est. Duration**: 6-7 weeks
**Total Tasks**: 169 (60 backend core + 42 testing + 15 CLI + 9 API docs + 43 frontend)

---

## 🎯 Executive Summary

### Problem Statement
JadeVectorDB currently uses in-memory storage (`InMemoryDatabasePersistence`), causing complete data loss on server restart. This violates specification FR-001 which mandates persistent storage and makes the system unsuitable for production use.

### Solution Overview
Implement hybrid persistent storage:
- **SQLite** for transactional data (users, groups, roles, permissions, database metadata)
- **Memory-mapped files** for vector data (optimized for SIMD operations)

### Decision Record

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Storage Architecture** | Hybrid (SQLite + mmap) | SQLite for ACID compliance on metadata, mmap for vector performance |
| **Access Control** | Full RBAC | Users + Groups + Roles + Permissions + API Keys |
| **Migration** | Clean Start | Fresh development, no production data to migrate |
| **Delivery** | Strategic Split | Phase 1+2 (SQLite), then Phase 3 (Vectors) |
| **File Layout** | Hybrid | Single system.db + per-database directories |
| **Durability** | WAL + Batching | Balance between safety and performance |

---

## 📋 Implementation Phases

### Phase 1+2: SQLite Persistence Layer (Weeks 1-3)
**Deliverable**: Users, groups, roles, permissions, and database metadata persist across restarts
**Parallel Track**: Frontend RBAC UI development (Groups, Roles, Permissions, Enhanced API Keys)

### Phase 3: Vector Data Persistence (Weeks 4-7)
**Deliverable**: Vector embeddings persist in memory-mapped files
**Parallel Track**: Frontend testing, E2E testing, documentation

---

## 🏗️ Architecture

### File System Layout

```
/var/lib/jadevectordb/
├── system.db                 # SQLite - All metadata
└── databases/
    ├── {db_uuid_001}/
    │   ├── vectors.mmap      # Memory-mapped vector data
    │   └── indexes/
    │       ├── hnsw.index
    │       └── ivf.index
    ├── {db_uuid_002}/
    │   ├── vectors.mmap
    │   └── indexes/
    └── {db_uuid_003}/
        ├── vectors.mmap
        └── indexes/
```

### Configuration
- **Default data directory**: `/var/lib/jadevectordb` (Linux/Mac), `C:\ProgramData\JadeVectorDB` (Windows)
- **Configurable via**: Environment variable `JADEVECTORDB_DATA_DIR` or config file
- **Permissions**: 0700 (owner only)

---

## 📊 Database Schema Design

### SQLite Schema (`system.db`)

#### 1. Users Table
```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    is_system_admin INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,  -- Unix timestamp
    updated_at INTEGER NOT NULL,
    last_login INTEGER,
    failed_login_attempts INTEGER DEFAULT 0,
    account_locked_until INTEGER,
    metadata TEXT  -- JSON blob for extensibility
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);
```

#### 2. Groups Table
```sql
CREATE TABLE groups (
    group_id TEXT PRIMARY KEY,
    group_name TEXT UNIQUE NOT NULL,
    description TEXT,
    owner_user_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT,
    FOREIGN KEY (owner_user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_groups_name ON groups(group_name);
CREATE INDEX idx_groups_owner ON groups(owner_user_id);
```

#### 3. User-Group Membership
```sql
CREATE TABLE user_groups (
    user_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    joined_at INTEGER NOT NULL,
    role_in_group TEXT DEFAULT 'member',  -- member, admin
    PRIMARY KEY (user_id, group_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES groups(group_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_groups_user ON user_groups(user_id);
CREATE INDEX idx_user_groups_group ON user_groups(group_id);
```

#### 4. Roles Table
```sql
CREATE TABLE roles (
    role_id TEXT PRIMARY KEY,
    role_name TEXT UNIQUE NOT NULL,
    description TEXT,
    is_system_role INTEGER DEFAULT 0,  -- Predefined roles
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Predefined system roles
INSERT INTO roles (role_id, role_name, description, is_system_role, created_at, updated_at)
VALUES 
    ('role_admin', 'Admin', 'Full system administration', 1, strftime('%s', 'now'), strftime('%s', 'now')),
    ('role_user', 'User', 'Standard user with database creation', 1, strftime('%s', 'now'), strftime('%s', 'now')),
    ('role_readonly', 'ReadOnly', 'Read-only access to databases', 1, strftime('%s', 'now'), strftime('%s', 'now')),
    ('role_data_scientist', 'DataScientist', 'Advanced analytics and query access', 1, strftime('%s', 'now'), strftime('%s', 'now'));
```

#### 5. User-Role Assignment
```sql
CREATE TABLE user_roles (
    user_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    assigned_at INTEGER NOT NULL,
    assigned_by TEXT,  -- user_id who assigned the role
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(role_id) ON DELETE CASCADE,
    FOREIGN KEY (assigned_by) REFERENCES users(user_id)
);

CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);
```

#### 6. Permissions Table
```sql
CREATE TABLE permissions (
    permission_id TEXT PRIMARY KEY,
    permission_name TEXT UNIQUE NOT NULL,
    resource_type TEXT NOT NULL,  -- 'database', 'vector', 'system'
    action TEXT NOT NULL,  -- 'read', 'write', 'delete', 'admin', 'create'
    description TEXT,
    created_at INTEGER NOT NULL
);

-- Predefined permissions
INSERT INTO permissions VALUES
    ('perm_db_read', 'database.read', 'database', 'read', 'Read database and vectors', strftime('%s', 'now')),
    ('perm_db_write', 'database.write', 'database', 'write', 'Insert/update vectors', strftime('%s', 'now')),
    ('perm_db_delete', 'database.delete', 'database', 'delete', 'Delete vectors and databases', strftime('%s', 'now')),
    ('perm_db_admin', 'database.admin', 'database', 'admin', 'Full database administration', strftime('%s', 'now')),
    ('perm_db_create', 'database.create', 'database', 'create', 'Create new databases', strftime('%s', 'now')),
    ('perm_sys_admin', 'system.admin', 'system', 'admin', 'System administration', strftime('%s', 'now'));
```

#### 7. Role-Permission Assignment
```sql
CREATE TABLE role_permissions (
    role_id TEXT NOT NULL,
    permission_id TEXT NOT NULL,
    granted_at INTEGER NOT NULL,
    PRIMARY KEY (role_id, permission_id),
    FOREIGN KEY (role_id) REFERENCES roles(role_id) ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(permission_id) ON DELETE CASCADE
);

-- Default role-permission mappings
INSERT INTO role_permissions VALUES
    ('role_admin', 'perm_sys_admin', strftime('%s', 'now')),
    ('role_admin', 'perm_db_admin', strftime('%s', 'now')),
    ('role_admin', 'perm_db_create', strftime('%s', 'now')),
    ('role_user', 'perm_db_create', strftime('%s', 'now')),
    ('role_user', 'perm_db_read', strftime('%s', 'now')),
    ('role_user', 'perm_db_write', strftime('%s', 'now')),
    ('role_readonly', 'perm_db_read', strftime('%s', 'now'));
```

#### 8. Database-Level Permissions
```sql
CREATE TABLE database_permissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    database_id TEXT NOT NULL,
    principal_type TEXT NOT NULL,  -- 'user' or 'group'
    principal_id TEXT NOT NULL,  -- user_id or group_id
    permission_id TEXT NOT NULL,
    granted_at INTEGER NOT NULL,
    granted_by TEXT,  -- user_id
    FOREIGN KEY (permission_id) REFERENCES permissions(permission_id),
    FOREIGN KEY (granted_by) REFERENCES users(user_id),
    UNIQUE (database_id, principal_type, principal_id, permission_id)
);

CREATE INDEX idx_db_perms_database ON database_permissions(database_id);
CREATE INDEX idx_db_perms_principal ON database_permissions(principal_type, principal_id);
```

#### 9. API Keys
```sql
CREATE TABLE api_keys (
    api_key_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,  -- bcrypt hash of the key
    key_name TEXT,  -- User-friendly name
    key_prefix TEXT NOT NULL,  -- First 8 chars for identification
    scopes TEXT,  -- JSON array of permitted scopes
    is_active INTEGER DEFAULT 1,
    expires_at INTEGER,  -- NULL for no expiration
    created_at INTEGER NOT NULL,
    last_used_at INTEGER,
    usage_count INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);
```

#### 10. Authentication Tokens (Session Tokens)
```sql
CREATE TABLE auth_tokens (
    token_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    issued_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    last_used_at INTEGER NOT NULL,
    is_valid INTEGER DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_tokens_user ON auth_tokens(user_id);
CREATE INDEX idx_tokens_expires ON auth_tokens(expires_at);
CREATE INDEX idx_tokens_valid ON auth_tokens(is_valid);
```

#### 11. Sessions
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_id TEXT,
    ip_address TEXT,
    created_at INTEGER NOT NULL,
    last_activity INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    is_active INTEGER DEFAULT 1,
    metadata TEXT,  -- JSON for additional session data
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (token_id) REFERENCES auth_tokens(token_id) ON DELETE SET NULL
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);
CREATE INDEX idx_sessions_active ON sessions(is_active);
```

#### 12. Databases Table
```sql
CREATE TABLE databases (
    database_id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    owner_user_id TEXT NOT NULL,
    owner_group_id TEXT,  -- Optional group ownership
    vector_dimension INTEGER NOT NULL,
    index_type TEXT NOT NULL,
    index_parameters TEXT,  -- JSON blob
    sharding_strategy TEXT,
    num_shards INTEGER DEFAULT 1,
    replication_factor INTEGER DEFAULT 1,
    replication_sync INTEGER DEFAULT 1,
    embedding_models TEXT,  -- JSON array
    metadata_schema TEXT,  -- JSON object
    retention_policy TEXT,  -- JSON object
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    is_active INTEGER DEFAULT 1,
    FOREIGN KEY (owner_user_id) REFERENCES users(user_id),
    FOREIGN KEY (owner_group_id) REFERENCES groups(group_id)
);

CREATE INDEX idx_databases_name ON databases(name);
CREATE INDEX idx_databases_owner_user ON databases(owner_user_id);
CREATE INDEX idx_databases_owner_group ON databases(owner_group_id);
CREATE INDEX idx_databases_active ON databases(is_active);
```

#### 13. Indexes Table
```sql
CREATE TABLE indexes (
    index_id TEXT PRIMARY KEY,
    database_id TEXT NOT NULL,
    index_name TEXT NOT NULL,
    index_type TEXT NOT NULL,  -- HNSW, IVF, LSH
    index_parameters TEXT,  -- JSON blob
    status TEXT DEFAULT 'building',  -- building, ready, failed
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    build_duration_ms INTEGER,
    vector_count INTEGER DEFAULT 0,
    FOREIGN KEY (database_id) REFERENCES databases(database_id) ON DELETE CASCADE,
    UNIQUE (database_id, index_name)
);

CREATE INDEX idx_indexes_database ON indexes(database_id);
CREATE INDEX idx_indexes_status ON indexes(status);
```

#### 14. Audit Log
```sql
CREATE TABLE audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    ip_address TEXT,
    user_agent TEXT,
    success INTEGER NOT NULL,
    error_message TEXT,
    metadata TEXT,  -- JSON blob
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_action ON audit_log(action);
CREATE INDEX idx_audit_resource ON audit_log(resource_type, resource_id);
```

---

## 🔧 C++ Implementation Details

### New Classes to Implement

#### 1. SQLitePersistenceLayer
```cpp
// backend/src/services/sqlite_persistence.h
namespace jadevectordb {

class SQLitePersistenceLayer {
public:
    explicit SQLitePersistenceLayer(const std::string& db_path);
    ~SQLitePersistenceLayer();
    
    // Lifecycle
    Result<void> initialize();
    Result<void> close();
    Result<void> checkpoint();  // WAL checkpoint
    
    // User operations
    Result<std::string> create_user(const UserCredentials& user);
    Result<UserCredentials> get_user(const std::string& user_id);
    Result<UserCredentials> get_user_by_username(const std::string& username);
    Result<std::vector<UserCredentials>> list_users();
    Result<void> update_user(const UserCredentials& user);
    Result<void> delete_user(const std::string& user_id);
    
    // Group operations
    Result<std::string> create_group(const Group& group);
    Result<Group> get_group(const std::string& group_id);
    Result<std::vector<Group>> list_groups();
    Result<void> add_user_to_group(const std::string& user_id, const std::string& group_id);
    Result<void> remove_user_from_group(const std::string& user_id, const std::string& group_id);
    Result<std::vector<Group>> get_user_groups(const std::string& user_id);
    
    // Role operations
    Result<void> assign_role(const std::string& user_id, const std::string& role_id);
    Result<void> revoke_role(const std::string& user_id, const std::string& role_id);
    Result<std::vector<Role>> get_user_roles(const std::string& user_id);
    Result<std::vector<Permission>> get_user_permissions(const std::string& user_id);
    
    // Database permission operations
    Result<void> grant_database_permission(const std::string& database_id, 
                                          const std::string& principal_type,
                                          const std::string& principal_id,
                                          const std::string& permission_id);
    Result<bool> check_database_permission(const std::string& user_id,
                                          const std::string& database_id,
                                          const std::string& permission_id);
    
    // API Key operations
    Result<APIKey> create_api_key(const APIKey& key);
    Result<APIKey> get_api_key_by_prefix(const std::string& prefix);
    Result<void> revoke_api_key(const std::string& api_key_id);
    Result<std::vector<APIKey>> list_user_api_keys(const std::string& user_id);
    
    // Token operations
    Result<AuthToken> create_token(const AuthToken& token);
    Result<AuthToken> get_token(const std::string& token_id);
    Result<void> invalidate_token(const std::string& token_id);
    Result<void> cleanup_expired_tokens();
    
    // Session operations
    Result<Session> create_session(const Session& session);
    Result<Session> get_session(const std::string& session_id);
    Result<void> update_session_activity(const std::string& session_id);
    Result<void> terminate_session(const std::string& session_id);
    
    // Database metadata operations
    Result<std::string> create_database_metadata(const Database& db);
    Result<Database> get_database_metadata(const std::string& database_id);
    Result<std::vector<Database>> list_database_metadata();
    Result<void> update_database_metadata(const Database& db);
    Result<void> delete_database_metadata(const std::string& database_id);
    
    // Index metadata operations
    Result<void> create_index_metadata(const Index& index);
    Result<Index> get_index_metadata(const std::string& index_id);
    Result<std::vector<Index>> list_index_metadata(const std::string& database_id);
    Result<void> update_index_metadata(const Index& index);
    Result<void> delete_index_metadata(const std::string& index_id);
    
    // Audit logging
    Result<void> log_audit_event(const AuditEvent& event);
    
private:
    sqlite3* db_;
    std::string db_path_;
    std::shared_ptr<logging::Logger> logger_;
    mutable std::shared_mutex db_mutex_;
    
    // Helper methods
    Result<void> execute_sql(const std::string& sql);
    Result<void> create_schema();
    Result<void> migrate_schema();
    Result<void> enable_wal_mode();
};

} // namespace jadevectordb
```

#### 2. HybridDatabasePersistence (replaces InMemoryDatabasePersistence)
```cpp
// backend/src/services/database_layer.h
namespace jadevectordb {

class HybridDatabasePersistence : public DatabasePersistenceInterface {
public:
    explicit HybridDatabasePersistence(
        const std::string& data_directory,
        std::shared_ptr<ShardingService> sharding_service = nullptr,
        std::shared_ptr<QueryRouter> query_router = nullptr,
        std::shared_ptr<ReplicationService> replication_service = nullptr
    );
    ~HybridDatabasePersistence() override;
    
    // Initialize/shutdown
    Result<void> initialize();
    Result<void> shutdown();
    
    // DatabasePersistenceInterface implementation
    Result<std::string> create_database(const Database& db) override;
    Result<Database> get_database(const std::string& database_id) override;
    Result<std::vector<Database>> list_databases() override;
    Result<void> update_database(const std::string& database_id, const Database& db) override;
    Result<void> delete_database(const std::string& database_id) override;
    
    Result<void> store_vector(const std::string& database_id, const Vector& vector) override;
    Result<Vector> retrieve_vector(const std::string& database_id, const std::string& vector_id) override;
    Result<std::vector<Vector>> retrieve_vectors(const std::string& database_id, 
                                                 const std::vector<std::string>& vector_ids) override;
    Result<void> update_vector(const std::string& database_id, const Vector& vector) override;
    Result<void> delete_vector(const std::string& database_id, const std::string& vector_id) override;
    
    Result<void> batch_store_vectors(const std::string& database_id, 
                                     const std::vector<Vector>& vectors) override;
    Result<void> batch_delete_vectors(const std::string& database_id,
                                     const std::vector<std::string>& vector_ids) override;
    
    Result<void> create_index(const std::string& database_id, const Index& index) override;
    Result<Index> get_index(const std::string& database_id, const std::string& index_id) override;
    Result<std::vector<Index>> list_indexes(const std::string& database_id) override;
    Result<void> update_index(const std::string& database_id, const std::string& index_id, const Index& index) override;
    Result<void> delete_index(const std::string& database_id, const std::string& index_id) override;
    
    Result<bool> database_exists(const std::string& database_id) const override;
    Result<bool> vector_exists(const std::string& database_id, const std::string& vector_id) const override;
    Result<bool> index_exists(const std::string& database_id, const std::string& index_id) const override;
    
    Result<size_t> get_vector_count(const std::string& database_id) const override;
    Result<std::vector<std::string>> get_all_vector_ids(const std::string& database_id) const override;
    
private:
    std::string data_directory_;
    std::unique_ptr<SQLitePersistenceLayer> sql_layer_;
    
    // Vector storage (Phase 3)
    std::unordered_map<std::string, std::unique_ptr<MemoryMappedVectorStore>> vector_stores_;
    mutable std::shared_mutex vector_stores_mutex_;
    
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<QueryRouter> query_router_;
    std::shared_ptr<ReplicationService> replication_service_;
    
    // Helper methods
    Result<std::string> get_database_directory(const std::string& database_id) const;
    Result<void> ensure_database_directory(const std::string& database_id);
    Result<MemoryMappedVectorStore*> get_or_create_vector_store(const std::string& database_id);
};

} // namespace jadevectordb
```

#### 3. MemoryMappedVectorStore (Phase 3)
```cpp
// backend/src/storage/memory_mapped_vector_store.h
namespace jadevectordb {

class MemoryMappedVectorStore {
public:
    explicit MemoryMappedVectorStore(const std::string& file_path, size_t dimension);
    ~MemoryMappedVectorStore();
    
    // Lifecycle
    Result<void> open();
    Result<void> close();
    Result<void> flush();
    
    // Vector operations
    Result<void> store_vector(const std::string& vector_id, const std::vector<float>& values);
    Result<std::vector<float>> retrieve_vector(const std::string& vector_id);
    Result<void> delete_vector(const std::string& vector_id);
    
    // Batch operations
    Result<void> batch_store(const std::vector<std::pair<std::string, std::vector<float>>>& vectors);
    Result<std::vector<std::vector<float>>> batch_retrieve(const std::vector<std::string>& vector_ids);
    
    // Stats
    size_t get_vector_count() const;
    size_t get_dimension() const;
    size_t get_file_size() const;
    
private:
    std::string file_path_;
    size_t dimension_;
    
    // Memory mapping
    void* mapped_memory_;
    size_t mapped_size_;
    int fd_;  // File descriptor
    
    // Vector index (vector_id -> file offset)
    std::unordered_map<std::string, size_t> vector_index_;
    std::vector<size_t> free_slots_;  // For deleted vectors
    
    mutable std::shared_mutex mmap_mutex_;
    std::shared_ptr<logging::Logger> logger_;
    
    // Helper methods
    Result<void> map_file();
    Result<void> unmap_file();
    Result<void> resize_file(size_t new_size);
    Result<size_t> allocate_slot();
    void deallocate_slot(size_t offset);
};

} // namespace jadevectordb
```

#### 4. Enhanced AuthenticationService
```cpp
// backend/src/services/authentication_service.h
// Modifications to existing AuthenticationService

class AuthenticationService {
public:
    explicit AuthenticationService(
        const AuthenticationConfig& config,
        std::shared_ptr<SQLitePersistenceLayer> persistence,  // NEW
        std::shared_ptr<SecurityAuditLogger> audit_logger = nullptr
    );
    
    // NEW: Persistence-backed methods replace in-memory versions
    Result<std::string> register_user(const std::string& username, 
                                      const std::string& password, 
                                      const std::string& email);
    
    Result<AuthToken> login(const std::string& username, 
                           const std::string& password,
                           const std::string& ip_address = "",
                           const std::string& user_agent = "");
    
    // NEW: API Key authentication
    Result<std::string> create_api_key(const std::string& user_id, 
                                       const std::string& key_name,
                                       const std::vector<std::string>& scopes,
                                       int expiry_days = 365);
    Result<std::string> authenticate_api_key(const std::string& api_key);
    Result<void> revoke_api_key(const std::string& api_key_id);
    
    // NEW: Permission checking
    Result<bool> check_permission(const std::string& user_id,
                                 const std::string& permission_name);
    Result<bool> check_database_permission(const std::string& user_id,
                                          const std::string& database_id,
                                          const std::string& permission_name);
    
    // Existing methods remain, but backed by SQLite
    // ...
    
private:
    std::shared_ptr<SQLitePersistenceLayer> persistence_;  // NEW
    // Remove in-memory maps (users_, tokens_, sessions_, api_keys_)
    // ...
};
```

### New Model Structs

```cpp
// backend/src/models/auth.h
namespace jadevectordb {

struct Group {
    std::string group_id;
    std::string group_name;
    std::string description;
    std::string owner_user_id;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    bool validate() const {
        return !group_id.empty() && !group_name.empty() && !owner_user_id.empty();
    }
};

struct Role {
    std::string role_id;
    std::string role_name;
    std::string description;
    bool is_system_role;
    std::vector<Permission> permissions;
    
    bool validate() const {
        return !role_id.empty() && !role_name.empty();
    }
};

struct Permission {
    std::string permission_id;
    std::string permission_name;
    std::string resource_type;  // 'database', 'vector', 'system'
    std::string action;  // 'read', 'write', 'delete', 'admin', 'create'
    std::string description;
    
    bool validate() const {
        return !permission_id.empty() && !permission_name.empty() &&
               !resource_type.empty() && !action.empty();
    }
};

struct APIKey {
    std::string api_key_id;
    std::string user_id;
    std::string key_value;  // Plain text (only on creation, not stored)
    std::string key_hash;
    std::string key_name;
    std::string key_prefix;  // First 8 chars for UI display
    std::vector<std::string> scopes;
    bool is_active;
    std::chrono::system_clock::time_point expires_at;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_used_at;
    int usage_count;
    
    bool validate() const {
        return !api_key_id.empty() && !user_id.empty() && !key_hash.empty();
    }
};

struct Session {
    std::string session_id;
    std::string user_id;
    std::string token_id;
    std::string ip_address;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_activity;
    std::chrono::system_clock::time_point expires_at;
    bool is_active;
    std::map<std::string, std::string> metadata;
    
    bool validate() const {
        return !session_id.empty() && !user_id.empty();
    }
};

struct AuditEvent {
    std::string user_id;
    std::string action;
    std::string resource_type;
    std::string resource_id;
    std::string ip_address;
    std::string user_agent;
    bool success;
    std::string error_message;
    std::map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point timestamp;
};

} // namespace jadevectordb
```

---

## 📝 Detailed Task Breakdown

### Phase 1+2: SQLite Persistence (Weeks 1-3)

#### Sprint 1.1: Foundation & Schema (Days 1-3)
- [ ] **T11.1.1**: Create data directory structure
  - Implement directory creation with proper permissions
  - Add configuration for `JADEVECTORDB_DATA_DIR`
  - Cross-platform path handling (Linux/Mac/Windows)
  - **Acceptance**: Server creates `/var/lib/jadevectordb` on first run

- [ ] **T11.1.2**: Implement SQLitePersistenceLayer class skeleton
  - Create `backend/src/services/sqlite_persistence.h/cpp`
  - Initialize SQLite connection with WAL mode
  - Add connection pooling (optional, for Phase 2)
  - **Acceptance**: Can open/close SQLite connection

- [ ] **T11.1.3**: Create complete SQL schema
  - Implement `create_schema()` method
  - Add all 14 tables from design
  - Add indexes for performance
  - **Acceptance**: Schema creation succeeds, all tables present

- [ ] **T11.1.4**: Add SQLite to build system
  - Update `backend/CMakeLists.txt` to link SQLite3
  - Add vcpkg dependency or system SQLite
  - Test compilation
  - **Acceptance**: Project builds with SQLite linked

#### Sprint 1.2: User & Auth Management (Days 4-7)
- [ ] **T11.2.1**: Implement User CRUD operations
  - `create_user()`, `get_user()`, `update_user()`, `delete_user()`
  - Password hashing (bcrypt)
  - Username/email uniqueness validation
  - **Acceptance**: Can create, retrieve, update users in SQLite

- [ ] **T11.2.2**: Implement Group operations
  - Create Group model struct
  - `create_group()`, `add_user_to_group()`, `get_user_groups()`
  - Group ownership validation
  - **Acceptance**: Users can create and join groups

- [ ] **T11.2.3**: Implement Role & Permission system
  - Create Role, Permission model structs
  - `assign_role()`, `revoke_role()`, `get_user_permissions()`
  - Load predefined system roles
  - **Acceptance**: Can assign roles and query permissions

- [ ] **T11.2.4**: Implement database-level permissions
  - `grant_database_permission()`, `check_database_permission()`
  - Support user and group principals
  - **Acceptance**: Can grant DB permissions to users/groups

- [ ] **T11.2.5**: Update AuthenticationService to use SQLite
  - Replace in-memory maps with SQLitePersistenceLayer calls
  - Maintain existing API compatibility
  - Update `register_user()` and `login()` methods
  - **Acceptance**: Registration/login persists across restarts

#### Sprint 1.3: Tokens & API Keys (Days 8-10)
- [ ] **T11.3.1**: Implement token persistence
  - `create_token()`, `get_token()`, `invalidate_token()`
  - Token expiry cleanup job
  - **Acceptance**: Tokens survive server restart

- [ ] **T11.3.2**: Implement session persistence
  - `create_session()`, `update_session_activity()`, `terminate_session()`
  - Session expiry management
  - **Acceptance**: Sessions persist and expire correctly

- [ ] **T11.3.3**: Implement API Key management
  - Create APIKey model struct
  - `create_api_key()`, `revoke_api_key()`, `authenticate_api_key()`
  - Key generation with secure random
  - Key prefix for UI display
  - **Acceptance**: Can create and authenticate with API keys

- [ ] **T11.3.4**: Add API key authentication to REST API
  - Support `Authorization: ApiKey <key>` header
  - Add API key endpoints (create, list, revoke)
  - **Acceptance**: Can use API key for authentication

#### Sprint 1.4: Database Metadata (Days 11-14)
- [ ] **T11.4.1**: Implement database metadata persistence
  - `create_database_metadata()`, `get_database_metadata()`
  - Store owner, permissions, configuration
  - **Acceptance**: Database metadata persists

- [ ] **T11.4.2**: Implement index metadata persistence
  - `create_index_metadata()`, `update_index_metadata()`
  - Track index build status
  - **Acceptance**: Index metadata persists

- [ ] **T11.4.3**: Create HybridDatabasePersistence class
  - Implement DatabasePersistenceInterface
  - Delegate metadata to SQLitePersistenceLayer
  - Keep vectors in-memory (Phase 3 will add persistence)
  - **Acceptance**: Database operations use SQLite for metadata

- [ ] **T11.4.4**: Replace InMemoryDatabasePersistence with HybridDatabasePersistence
  - Update REST API initialization
  - Update DatabaseService
  - Remove old InMemoryDatabasePersistence class
  - **Acceptance**: Server uses HybridDatabasePersistence

#### Sprint 1.5: Testing & Integration (Days 15-21)
- [ ] **T11.5.1**: Write unit tests for SQLitePersistenceLayer
  - Test all CRUD operations
  - Test transaction handling
  - Test constraint enforcement
  - **Acceptance**: 95%+ code coverage

- [ ] **T11.5.2**: Write integration tests for AuthenticationService
  - Test registration → login → permission check flow
  - Test API key creation and usage
  - Test role assignment and permission checking
  - **Acceptance**: All auth flows work with persistence

- [ ] **T11.5.3**: Update CLI tests for persistence
  - Verify users persist across restarts
  - Verify databases persist across restarts
  - Add group/role/permission tests
  - **Acceptance**: All CLI tests pass with persistence

- [ ] **T11.5.4**: Performance testing
  - Benchmark user operations (target: <10ms for lookup)
  - Benchmark permission checks (target: <5ms)
  - Test concurrent access (1000+ simultaneous users)
  - **Acceptance**: Meets performance targets

- [ ] **T11.5.5**: Add audit logging
  - Implement `log_audit_event()`
  - Log all authentication events
  - Log permission changes
  - **Acceptance**: All security events logged

- [ ] **T11.5.6**: Documentation
  - Update API documentation for RBAC endpoints
  - Document permission model
  - Create admin guide for user/group management
  - **Acceptance**: Complete documentation

---

### Phase 3: Vector Data Persistence (Weeks 4-5)

#### Sprint 3.1: Memory-Mapped File Infrastructure (Days 22-25)
- [ ] **T11.6.1**: Implement MemoryMappedVectorStore class
  - Create `backend/src/storage/memory_mapped_vector_store.h/cpp`
  - Implement file mapping (mmap on Unix, CreateFileMapping on Windows)
  - Add SIMD-aligned memory allocation
  - **Acceptance**: Can create and map vector files

- [ ] **T11.6.2**: Implement vector serialization format
  - Design binary layout: [header][vector_index][vector_data]
  - Header: version, dimension, vector_count
  - Vector index: map of vector_id → offset
  - **Acceptance**: Can serialize/deserialize vectors

- [ ] **T11.6.3**: Implement vector CRUD operations
  - `store_vector()`, `retrieve_vector()`, `delete_vector()`
  - Handle file growth and free space management
  - **Acceptance**: Can store and retrieve vectors from mmap file

- [ ] **T11.6.4**: Implement batch operations
  - `batch_store()`, `batch_retrieve()`
  - Optimize for bulk inserts (single mmap resize)
  - **Acceptance**: Batch operations 10x faster than individual

#### Sprint 3.2: Integration with HybridDatabasePersistence (Days 26-28)
- [ ] **T11.7.1**: Integrate MemoryMappedVectorStore into HybridDatabasePersistence
  - Create vector store per database
  - Update `store_vector()` to use mmap
  - Update `retrieve_vector()` to use mmap
  - **Acceptance**: Vector operations use persistent storage

- [ ] **T11.7.2**: Implement lazy loading of vector stores
  - Load mmap files on first access
  - Cache open file descriptors
  - **Acceptance**: Minimal memory footprint at startup

- [ ] **T11.7.3**: Add vector persistence flush/sync
  - Periodic flush (every 5 seconds)
  - Flush on shutdown
  - msync/FlushViewOfFile for durability
  - **Acceptance**: Vectors persist across ungraceful shutdown

- [ ] **T11.7.4**: Handle database deletion
  - Delete mmap files when database deleted
  - Clean up directory structure
  - **Acceptance**: Database deletion removes all files

#### Sprint 3.3: Testing & Optimization (Days 29-35)
- [ ] **T11.8.1**: Write unit tests for MemoryMappedVectorStore
  - Test large vector datasets (1M+ vectors)
  - Test concurrent access
  - Test file growth and compaction
  - **Acceptance**: 95%+ code coverage

- [ ] **T11.8.2**: Write integration tests for full persistence
  - Create database → insert vectors → restart → verify vectors
  - Test with various vector dimensions (128, 384, 768, 1536)
  - **Acceptance**: All vectors survive restart

- [ ] **T11.8.3**: Performance benchmarking
  - Measure insert throughput (target: 10K+ vectors/sec)
  - Measure search latency with persistent storage
  - Compare to in-memory baseline
  - **Acceptance**: <10% performance degradation vs in-memory

- [ ] **T11.8.4**: SIMD optimization
  - Verify vector data alignment (16-byte for SSE, 32-byte for AVX)
  - Benchmark SIMD operations on mmap data
  - **Acceptance**: SIMD operations work on mmap memory

- [ ] **T11.8.5**: Cross-platform testing
  - Test on Linux (development)
  - Test on macOS (if available)
  - Test on Windows (WSL or native)
  - **Acceptance**: Works on all target platforms

- [ ] **T11.8.6**: Update CLI tests for vector persistence
  - Verify vectors persist across restarts
  - Test large batch inserts
  - **Acceptance**: All CLI tests pass

- [ ] **T11.8.7**: Final documentation
  - Update architecture documentation
  - Document file format specification
  - Create performance tuning guide
  - **Acceptance**: Complete documentation

---

## 🧪 Comprehensive Testing Strategy

### Testing Tasks Breakdown

#### Unit Testing Tasks

- [ ] **T11.9.1**: SQLitePersistenceLayer Unit Tests
  - Test all CRUD operations for users, groups, roles
  - Test permission checking logic
  - Test API key generation and validation
  - Test token creation and expiry
  - Test session management
  - Test database metadata operations
  - **Target**: 95%+ code coverage
  - **Acceptance**: All SQL operations covered with edge cases

- [ ] **T11.9.2**: MemoryMappedVectorStore Unit Tests
  - Test file creation and memory mapping
  - Test vector storage and retrieval
  - Test SIMD alignment verification
  - Test file growth and resizing
  - Test free space management
  - Test concurrent access patterns
  - **Target**: 95%+ code coverage
  - **Acceptance**: All mmap operations tested

- [ ] **T11.9.3**: HybridDatabasePersistence Unit Tests
  - Test database creation with both SQLite and mmap
  - Test vector operations persistence
  - Test metadata-vector coordination
  - Test error handling and rollback
  - **Target**: 90%+ code coverage
  - **Acceptance**: Integration layer fully tested

- [ ] **T11.9.4**: AuthenticationService Persistence Unit Tests
  - Test user registration with SQLite
  - Test login flow with persistent tokens
  - Test API key authentication
  - Test permission caching
  - Test session expiry cleanup
  - **Target**: 95%+ code coverage
  - **Acceptance**: All auth flows work with persistence

#### Integration Testing Tasks

- [ ] **T11.10.1**: User Authentication Flow Integration Test
  - Register user → Login → API call → Logout
  - Restart server → Login with same credentials
  - Verify token persistence across restart
  - Test failed login attempts and lockout
  - **Acceptance**: Complete auth flow survives restart

- [ ] **T11.10.2**: RBAC Integration Test
  - Create user → Assign role → Grant database permission
  - Test permission check enforcement
  - Test group membership and inherited permissions
  - Restart server → Verify permissions intact
  - **Acceptance**: Full RBAC flow works end-to-end

- [ ] **T11.10.3**: Database Lifecycle Integration Test
  - Create database → Store vectors → Create index
  - Restart server → Verify database exists
  - Verify metadata intact → Verify vectors intact
  - Search vectors → Delete database
  - **Acceptance**: Database lifecycle fully persistent

- [ ] **T11.10.4**: API Key Integration Test
  - Create API key → Use for authentication
  - Restart server → Use same API key
  - Test API key scopes and restrictions
  - Revoke API key → Verify access denied
  - **Acceptance**: API keys work across restarts

- [ ] **T11.10.5**: Concurrent Access Integration Test
  - Spawn 100 threads creating users
  - Spawn 100 threads storing vectors
  - Verify no race conditions or data corruption
  - Check SQLite locking behavior
  - **Acceptance**: Handle 1000+ concurrent operations

#### Performance Testing Tasks

- [ ] **T11.11.1**: SQLite Performance Benchmark
  - User lookup: Target <10ms (95th percentile)
  - Permission check: Target <5ms (95th percentile)
  - Database metadata query: Target <15ms
  - Batch user creation: Target 100+ users/sec
  - Test with 10K+ users, 100+ databases
  - **Acceptance**: Meets all performance targets

- [ ] **T11.11.2**: Vector Storage Performance Benchmark
  - Vector insert: Target 10K+ vectors/sec
  - Vector retrieval: Target <1ms per vector
  - Batch insert: Target 50K+ vectors/sec
  - Search with persistence: Compare to in-memory baseline
  - Target: <10% performance degradation
  - **Acceptance**: Meets throughput and latency targets

- [ ] **T11.11.3**: Restart Performance Test
  - Measure server startup time with 100K vectors
  - Measure lazy-loading effectiveness
  - Test memory usage after restart
  - Target: <5 seconds startup for 100K vectors
  - **Acceptance**: Fast startup with large datasets

- [ ] **T11.11.4**: SIMD Performance Verification
  - Verify SIMD instructions work on mmap memory
  - Benchmark cosine similarity on mmap vs in-memory
  - Test with AVX, AVX2, AVX-512 if available
  - Target: Same performance as in-memory
  - **Acceptance**: No SIMD performance regression

#### Security Testing Tasks

- [ ] **T11.12.1**: SQL Injection Security Test
  - Test all user inputs for SQL injection
  - Test username, email, database names
  - Verify parameterized queries used everywhere
  - Test with OWASP SQLi payload list
  - **Acceptance**: Zero SQL injection vulnerabilities

- [ ] **T11.12.2**: Authentication Security Test
  - Test password hashing (bcrypt work factor)
  - Test token security (JWT signing, expiry)
  - Test API key security (secure random generation)
  - Test session hijacking protection
  - **Acceptance**: All auth mechanisms secure

- [ ] **T11.12.3**: Permission Bypass Security Test
  - Attempt to access databases without permission
  - Attempt privilege escalation
  - Test group membership validation
  - Test role assignment authorization
  - **Acceptance**: No permission bypass vulnerabilities

- [ ] **T11.12.4**: File System Security Test
  - Verify data directory permissions (0700)
  - Test SQLite database file permissions
  - Test vector file access control
  - Verify no information leakage in logs
  - **Acceptance**: Secure file system access

#### Reliability Testing Tasks

- [ ] **T11.13.1**: Crash Recovery Test
  - Kill server during vector write
  - Restart and verify SQLite recovery
  - Test WAL checkpoint recovery
  - Verify no data corruption
  - **Acceptance**: Graceful recovery from crashes

- [ ] **T11.13.2**: Disk Full Scenario Test
  - Fill disk to 100% during writes
  - Verify graceful error handling
  - Verify no database corruption
  - Test recovery after disk space freed
  - **Acceptance**: Handle disk full gracefully

- [ ] **T11.13.3**: Large Dataset Stress Test
  - Insert 10M vectors across 100 databases
  - Test with 100K users and 10K groups
  - Measure memory usage and file sizes
  - Verify search performance maintained
  - **Acceptance**: Scale to production workload

- [ ] **T11.13.4**: Long-Running Stability Test
  - Run continuous operations for 72 hours
  - Monitor memory leaks
  - Monitor file descriptor leaks
  - Check SQLite checkpoint frequency
  - **Acceptance**: Stable for extended operation

---

## 🎨 Frontend Implementation Tasks

### Frontend Analysis Summary

**Current State**: 
- Next.js 14 with JavaScript (not TypeScript)
- Existing pages: `users.js`, `api-keys.js`, `databases.js`, `login.js`, `register.js`, `auth.js`
- API client library in `frontend/src/lib/api.js` with auth support
- UI components: `button.js`, `card.js`, `input.js`, `select.js`, `alert.js`
- Current user management UI only supports basic user CRUD

**Required Changes**:
- Add Group management UI
- Add Role assignment UI
- Add Permission management UI
- Enhance API key UI (scopes, expiration)
- Add database permission indicators
- Update existing pages for RBAC compatibility
- Add persistence status indicators

### Phase 1+2 Frontend Tasks (Parallel with Backend)

#### Group Management UI

- [ ] **T11.21.1**: Create Groups Page
  - Create `frontend/src/pages/groups.js`
  - Group list with create/edit/delete
  - Show member count and owner
  - Filter by group name
  - **Acceptance**: Full group management UI

- [ ] **T11.21.2**: Create Group Detail Page
  - Create `frontend/src/pages/groups/[id].js`
  - Show group details and metadata
  - List group members with add/remove
  - List inherited roles and permissions
  - **Acceptance**: Complete group detail view

- [ ] **T11.21.3**: Add Group API Functions
  - Update `frontend/src/lib/api.js`:
    - `groupApi.createGroup(name, description, ownerId)`
    - `groupApi.listGroups(limit, offset)`
    - `groupApi.getGroup(groupId)`
    - `groupApi.updateGroup(groupId, data)`
    - `groupApi.deleteGroup(groupId)`
    - `groupApi.addMember(groupId, userId)`
    - `groupApi.removeMember(groupId, userId)`
  - **Acceptance**: All group operations callable from frontend

- [ ] **T11.21.4**: Add Group Components
  - Create `frontend/src/components/GroupCard.js`
  - Create `frontend/src/components/GroupMemberList.js`
  - Create `frontend/src/components/AddMemberModal.js`
  - **Acceptance**: Reusable group components

#### Role & Permission Management UI

- [ ] **T11.22.1**: Create Roles Page
  - Create `frontend/src/pages/roles.js`
  - List system and custom roles
  - Show permissions for each role
  - Display users assigned to each role
  - **Acceptance**: View all roles and assignments

- [ ] **T11.22.2**: Add Role Assignment UI to User Page
  - Update `frontend/src/pages/users.js`:
    - Add "Roles" column to user table
    - Add role assignment modal/dropdown
    - Show inherited roles from groups
    - Add role revocation button
  - **Acceptance**: Manage user roles from users page

- [ ] **T11.22.3**: Add Permission Matrix View
  - Create `frontend/src/pages/permissions.js`
  - Show permission matrix (users × databases)
  - Color-code permission levels (read/write/admin)
  - Filter by user, group, or database
  - **Acceptance**: Visual permission overview

- [ ] **T11.22.4**: Add Role API Functions
  - Update `frontend/src/lib/api.js`:
    - `roleApi.listRoles()`
    - `roleApi.getUserRoles(userId)`
    - `roleApi.assignRole(userId, roleId)`
    - `roleApi.revokeRole(userId, roleId)`
    - `roleApi.getGroupRoles(groupId)`
    - `roleApi.assignGroupRole(groupId, roleId)`
  - **Acceptance**: All role operations available

- [ ] **T11.22.5**: Add Permission API Functions
  - Update `frontend/src/lib/api.js`:
    - `permissionApi.listPermissions()`
    - `permissionApi.getUserPermissions(userId)`
    - `permissionApi.grantDatabasePermission(dbId, principalId, principalType, permissionId)`
    - `permissionApi.revokeDatabasePermission(dbId, principalId, principalType, permissionId)`
    - `permissionApi.getDatabasePermissions(databaseId)`
  - **Acceptance**: All permission operations available

#### Enhanced API Key Management

- [ ] **T11.23.1**: Enhance API Keys Page
  - Update `frontend/src/pages/api-keys.js`:
    - Add scopes selection (multi-select or checkboxes)
    - Add expiration date picker
    - Show key status (active/expired/revoked)
    - Add key usage statistics (last used, usage count)
    - Add key prefix display (first 8 chars)
  - **Acceptance**: Full-featured API key management

- [ ] **T11.23.2**: Add API Key Scopes Component
  - Create `frontend/src/components/ApiKeyScopes.js`
  - Multi-select for available scopes
  - Display scope descriptions
  - Visual indicator for selected scopes
  - **Acceptance**: User-friendly scope selection

- [ ] **T11.23.3**: Add API Key Status Indicators
  - Visual badges for active/expired/revoked status
  - Countdown timer for expiring keys
  - Warning for keys expiring within 7 days
  - **Acceptance**: Clear key status visibility

#### Database Management Enhancements

- [ ] **T11.24.1**: Add Permission Indicators to Database List
  - Update `frontend/src/pages/databases.js`:
    - Show permission badges (read/write/admin/owner)
    - Add "Permissions" column to database table
    - Filter databases by permission level
    - Show shared databases vs owned databases
  - **Acceptance**: User sees their access level per database

- [ ] **T11.24.2**: Add Database Permission Management Page
  - Create `frontend/src/pages/databases/[id]/permissions.js`
  - List users/groups with access to database
  - Add "Grant Permission" button with modal
  - Revoke permission button for each entry
  - Filter by user or group
  - **Acceptance**: Manage database-level permissions

- [ ] **T11.24.3**: Add Database Owner Transfer UI
  - Add "Transfer Ownership" button to database detail
  - Modal with user selection dropdown
  - Confirmation dialog with warning
  - **Acceptance**: Database ownership transferable

#### User Management Enhancements

- [ ] **T11.25.1**: Enhance User List Page
  - Update `frontend/src/pages/users.js`:
    - Add "Groups" column showing group membership
    - Add "Roles" column showing assigned roles
    - Add filter by group, role, or active status
    - Add bulk operations (assign role, add to group)
  - **Acceptance**: Rich user list with RBAC info

- [ ] **T11.25.2**: Create User Detail/Profile Page
  - Create `frontend/src/pages/users/[id].js`
  - Show user details and metadata
  - List group memberships
  - List assigned roles
  - List effective permissions (direct + inherited)
  - List owned databases
  - List API keys for user
  - **Acceptance**: Complete user profile view

- [ ] **T11.25.3**: Add User Activity Timeline
  - Show recent logins
  - Show database access history
  - Show permission changes audit log
  - **Acceptance**: User activity visible

#### Navigation & Layout Updates

- [ ] **T11.26.1**: Update Navigation Menu
  - Update `frontend/src/components/Layout.js`:
    - Add "Groups" navigation link
    - Add "Roles" navigation link
    - Add "Permissions" navigation link
    - Add "Audit Logs" navigation link (if admin)
    - Organize menu into sections (Users, Access Control, Databases)
  - **Acceptance**: Easy navigation to all RBAC pages

- [ ] **T11.26.2**: Add Permission-Based UI Rendering
  - Create `frontend/src/hooks/usePermissions.js`
  - Check user permissions before showing UI elements
  - Hide "Delete" button if user lacks permission
  - Show read-only mode for users with only read permission
  - **Acceptance**: UI adapts to user permissions

- [ ] **T11.26.3**: Add Admin-Only UI Elements
  - Protect admin pages (users, groups, roles)
  - Add admin badge to current user indicator
  - Show "Admin" section in navigation only for admins
  - **Acceptance**: Admin features protected

#### Persistence Impact & Indicators

- [ ] **T11.27.1**: Add Save Status Indicators
  - Show "Saving..." indicator during API calls
  - Show "Saved successfully" confirmation
  - Show "Failed to save" error with retry option
  - Add auto-save indicator for draft operations
  - **Acceptance**: User knows when data is persisted

- [ ] **T11.27.2**: Add Data Freshness Indicators
  - Show "Last updated" timestamp on lists
  - Add "Refresh" button to reload data
  - Auto-refresh data every N seconds (configurable)
  - Show stale data warning if offline
  - **Acceptance**: User knows data freshness

- [ ] **T11.27.3**: Update Error Handling for Persistence Failures
  - Catch SQLite locked errors and show user-friendly message
  - Handle disk full errors gracefully
  - Retry logic for transient failures
  - **Acceptance**: Graceful error handling

#### Frontend State Management

- [ ] **T11.28.1**: Add Authentication State Management
  - Create `frontend/src/context/AuthContext.js`
  - Store current user, roles, permissions in context
  - Refresh on login/logout
  - Provide `useAuth()` hook for components
  - **Acceptance**: Global auth state available

- [ ] **T11.28.2**: Add Permission Caching
  - Cache user permissions in localStorage
  - Refresh permissions on login
  - Invalidate cache on permission changes
  - **Acceptance**: Fast permission checks

- [ ] **T11.28.3**: Add Optimistic UI Updates
  - Immediately show changes in UI before API response
  - Rollback if API call fails
  - Show loading states for async operations
  - **Acceptance**: Responsive UI with optimistic updates

### Frontend Testing Tasks

#### Unit Tests

- [ ] **T11.29.1**: Test Group Management Components
  - Test `GroupCard` rendering
  - Test `GroupMemberList` functionality
  - Test `AddMemberModal` validation
  - Mock API calls
  - **Acceptance**: 95%+ coverage for group components

- [ ] **T11.29.2**: Test Role & Permission Components
  - Test role assignment modal
  - Test permission matrix rendering
  - Test permission API functions
  - **Acceptance**: 95%+ coverage for role components

- [ ] **T11.29.3**: Test Enhanced API Key Components
  - Test scopes selection component
  - Test expiration date picker
  - Test API key validation
  - **Acceptance**: 95%+ coverage for API key components

- [ ] **T11.29.4**: Test User Management Enhancements
  - Test enhanced user list rendering
  - Test user profile page
  - Test activity timeline
  - **Acceptance**: 95%+ coverage for user components

#### Integration Tests

- [ ] **T11.30.1**: Test Group Workflows
  - Create group → Add members → Delete group
  - Test group list pagination
  - Test group search/filter
  - **Acceptance**: Full group workflows work

- [ ] **T11.30.2**: Test Role Assignment Workflows
  - Assign role to user → Verify permissions updated
  - Assign role to group → Verify inherited by members
  - Revoke role → Verify permissions removed
  - **Acceptance**: Role assignments work correctly

- [ ] **T11.30.3**: Test Database Permission Workflows
  - Grant database permission → User can access
  - Revoke permission → User cannot access
  - Transfer ownership → New owner has full access
  - **Acceptance**: Database permissions work

- [ ] **T11.30.4**: Test API Key Workflows
  - Create API key with scopes → Authenticate with key
  - Test expired key → Should fail authentication
  - Revoke key → Should fail authentication
  - **Acceptance**: API key lifecycle works

#### End-to-End Tests

- [ ] **T11.31.1**: E2E Test: Complete RBAC Setup
  - Admin creates group
  - Admin assigns users to group
  - Admin creates database
  - Admin grants group permission to database
  - User logs in and accesses database
  - **Acceptance**: Full RBAC workflow end-to-end

- [ ] **T11.31.2**: E2E Test: Permission Enforcement
  - User without permission tries to access database → Denied
  - User tries to delete database they don't own → Denied
  - User with read permission tries to write → Denied
  - **Acceptance**: Permissions enforced in UI

- [ ] **T11.31.3**: E2E Test: API Key Usage
  - User creates API key via UI
  - User copies key
  - User authenticates with key via CLI
  - User performs operations with API key
  - **Acceptance**: API key works from creation to usage

- [ ] **T11.31.4**: E2E Test: Persistence Verification
  - User creates group and database
  - User logs out
  - Restart backend server
  - User logs back in
  - Verify group and database still exist
  - **Acceptance**: Data persists across restarts

#### Accessibility & UX Testing

- [ ] **T11.32.1**: Test Keyboard Navigation
  - Tab through all forms
  - Test Enter/Escape key handling
  - Test focus management in modals
  - **Acceptance**: Full keyboard accessibility

- [ ] **T11.32.2**: Test Screen Reader Compatibility
  - Add ARIA labels to all interactive elements
  - Test with screen reader
  - Ensure proper heading hierarchy
  - **Acceptance**: Accessible to screen readers

- [ ] **T11.32.3**: Test Mobile Responsiveness
  - Test on mobile viewports
  - Test touch interactions
  - Test navigation on small screens
  - **Acceptance**: Fully responsive UI

#### Performance Testing

- [ ] **T11.33.1**: Test Large Dataset Rendering
  - Test user list with 10,000+ users
  - Test group list with 1,000+ groups
  - Test database list with 500+ databases
  - Verify virtualization/pagination works
  - **Acceptance**: Smooth rendering of large lists

- [ ] **T11.33.2**: Test Permission Check Performance
  - Measure permission check latency
  - Test with 100+ permissions per user
  - Test with complex group hierarchies
  - **Acceptance**: Permission checks <50ms

### Frontend Documentation

- [ ] **T11.34.1**: Create Frontend Developer Guide
  - Component architecture overview
  - API integration patterns
  - State management guide
  - Permission checking guide
  - **Acceptance**: Complete frontend developer docs

- [ ] **T11.34.2**: Create UI/UX Style Guide
  - Component usage examples
  - Design patterns for RBAC UI
  - Accessibility guidelines
  - **Acceptance**: Consistent UI development guide

- [ ] **T11.34.3**: Update User Guide with Screenshots
  - Add screenshots of new pages
  - Document group management workflows
  - Document role assignment workflows
  - Document permission management
  - **Acceptance**: User guide with visual aids

---

## 🔧 CLI Testing & Enhancement Tasks

### CLI Testing Infrastructure

- [ ] **T11.14.1**: Update CLI Test Runner for Persistence
  - Modify `tests/run_cli_tests.py` for restart testing
  - Add persistence verification tests
  - Test user creation persistence
  - Test database persistence across restarts
  - **Acceptance**: CLI tests verify persistence

- [ ] **T11.14.2**: Add RBAC CLI Tests
  - Test group creation via CLI
  - Test role assignment via CLI
  - Test permission granting via CLI
  - Test API key creation via CLI
  - **Acceptance**: All RBAC operations testable

- [ ] **T11.14.3**: Add CLI Negative Tests
  - Test invalid credentials
  - Test insufficient permissions
  - Test expired tokens
  - Test revoked API keys
  - **Acceptance**: CLI handles errors gracefully

### Python CLI Enhancements

- [ ] **T11.15.1**: Add RBAC Commands to Python CLI
  - `jade-db group create <name>`
  - `jade-db group add-user <group> <user>`
  - `jade-db role assign <user> <role>`
  - `jade-db permission grant <database> <user> <permission>`
  - `jade-db api-key create <name> --scopes <scopes>`
  - **Acceptance**: All RBAC operations available in CLI

- [ ] **T11.15.2**: Update Python CLI Documentation
  - Document group management commands
  - Document role and permission commands
  - Document API key management
  - Add examples and use cases
  - **Acceptance**: Complete CLI documentation

- [ ] **T11.15.3**: Add Python CLI Tests for RBAC
  - Unit tests for RBAC commands
  - Integration tests with server
  - Test error handling and edge cases
  - **Acceptance**: 95%+ test coverage for CLI

### Shell CLI Enhancements

- [ ] **T11.16.1**: Add RBAC Commands to Shell CLI
  - `jade-db.sh group-create <name>`
  - `jade-db.sh group-add-user <group> <user>`
  - `jade-db.sh role-assign <user> <role>`
  - `jade-db.sh permission-grant <database> <user> <permission>`
  - `jade-db.sh api-key-create <name>`
  - **Acceptance**: All RBAC operations in shell script

- [ ] **T11.16.2**: Update Shell CLI Documentation
  - Add RBAC command examples to README
  - Document authentication with API keys
  - Add troubleshooting section
  - **Acceptance**: Complete shell CLI docs

- [ ] **T11.16.3**: Add Shell CLI Tests for RBAC
  - Test all new RBAC commands
  - Test API key authentication
  - Test error scenarios
  - **Acceptance**: Shell CLI tests pass with RBAC

### JavaScript CLI Enhancements

- [ ] **T11.17.1**: Add RBAC Commands to JS CLI
  - `jade-db group create <name>`
  - `jade-db role assign <user> <role>`
  - `jade-db permission grant <db> <user> <perm>`
  - `jade-db apikey create <name>`
  - **Acceptance**: RBAC commands in JS CLI

- [ ] **T11.17.2**: Update JS CLI Documentation
  - Document new RBAC features
  - Add code examples
  - Update README with new commands
  - **Acceptance**: Complete JS CLI documentation

### CLI Integration Testing

- [ ] **T11.18.1**: Cross-CLI Consistency Test
  - Verify same operations work across all CLIs
  - Test data created in Python CLI visible in Shell CLI
  - Test API keys created in one CLI work in another
  - **Acceptance**: Consistent behavior across CLIs

- [ ] **T11.18.2**: CLI Performance Test
  - Benchmark CLI command execution time
  - Test with large responses (1000+ databases)
  - Test batch operations via CLI
  - **Acceptance**: CLI performs efficiently

---

## 📖 API Documentation Tasks

### REST API Documentation

- [ ] **T11.19.1**: Document User Management Endpoints
  - `POST /v1/users` - Create user
  - `GET /v1/users` - List users
  - `GET /v1/users/{id}` - Get user
  - `PUT /v1/users/{id}` - Update user
  - `DELETE /v1/users/{id}` - Delete user
  - **Acceptance**: Complete endpoint documentation

- [ ] **T11.19.2**: Document Group Management Endpoints
  - `POST /v1/groups` - Create group
  - `GET /v1/groups` - List groups
  - `POST /v1/groups/{id}/members` - Add member
  - `DELETE /v1/groups/{id}/members/{user_id}` - Remove member
  - **Acceptance**: Full group API documented

- [ ] **T11.19.3**: Document Role & Permission Endpoints
  - `POST /v1/users/{id}/roles` - Assign role
  - `DELETE /v1/users/{id}/roles/{role_id}` - Revoke role
  - `GET /v1/users/{id}/permissions` - List permissions
  - `POST /v1/databases/{id}/permissions` - Grant DB permission
  - **Acceptance**: RBAC endpoints fully documented

- [ ] **T11.19.4**: Document API Key Endpoints
  - `POST /v1/api-keys` - Create API key
  - `GET /v1/api-keys` - List user's API keys
  - `DELETE /v1/api-keys/{id}` - Revoke API key
  - Document API key authentication header format
  - **Acceptance**: API key endpoints documented

- [ ] **T11.19.5**: Create OpenAPI/Swagger Specification
  - Generate OpenAPI 3.0 spec for all endpoints
  - Include request/response schemas
  - Add authentication requirements
  - Host interactive API documentation
  - **Acceptance**: Browsable API documentation

### SDK Documentation

- [ ] **T11.20.1**: Update Python Client Library
  - Add RBAC methods to client
  - Add API key authentication support
  - Update examples and tutorials
  - **Acceptance**: Python SDK fully updated

- [ ] **T11.20.2**: Create Administrator Guide
  - User and group management best practices
  - Role assignment strategies
  - Permission model explanation
  - Security recommendations
  - **Acceptance**: Complete admin guide

- [ ] **T11.20.3**: Create Developer Integration Guide
  - API key usage patterns
  - Authentication flow examples
  - Permission checking strategies
  - Code samples in multiple languages
  - **Acceptance**: Developer guide complete

---

## ✅ Updated Definition of Done

### Phase 1+2 Complete When:
- ✅ Users persist across server restarts
- ✅ Groups, roles, and permissions work correctly
- ✅ API keys can be created and used for authentication
- ✅ Database metadata persists
- ✅ All existing CLI tests pass
- ✅ No performance regression (<10% slower than in-memory)
- ✅ SQL injection tests pass
- ✅ Audit logging captures all security events

### Phase 3 Success Criteria
- ✅ Vectors persist across server restarts
- ✅ Large datasets (1M+ vectors) handled efficiently
- ✅ Search performance maintained (<10% regression)
- ✅ SIMD operations work on mmap memory
- ✅ Cross-platform compatibility (Linux, Mac, Windows)
- ✅ Graceful and ungraceful shutdown both preserve data
- ✅ File corruption recovery mechanism

---

## 🚀 Deployment Plan

### Configuration
```bash
# Environment variables
export JADEVECTORDB_DATA_DIR=/var/lib/jadevectordb
export JADEVECTORDB_LOG_LEVEL=info

# Or in config file (backend/config/jadevectordb.conf)
data_directory=/var/lib/jadevectordb
sqlite_journal_mode=WAL
sqlite_cache_size=10000
vector_flush_interval_sec=5
```

### Migration Steps
1. **Stop old server** (if running)
2. **Deploy new binary** with persistence
3. **Server creates** `/var/lib/jadevectordb/system.db` on first run
4. **Create initial admin user** via CLI or API
5. **Users re-register** (clean start, no migration needed)
6. **Verify persistence** by restarting server

### Rollback Plan
If issues arise:
1. Stop new server
2. Deploy old binary (in-memory version)
3. System returns to in-memory mode
4. No data loss (since it's a clean start)

---

## 📚 Documentation Deliverables

1. **API Documentation**
   - RBAC endpoints (users, groups, roles, permissions)
   - API key management endpoints
   - Database permission endpoints

2. **Administrator Guide**
   - User and group management
   - Role assignment
   - Permission model explanation
   - API key usage

3. **Developer Guide**
   - SQLite schema reference
   - File format specification (mmap)
   - Adding new permissions
   - Extending RBAC system

4. **Operations Guide**
   - Backup and restore procedures
   - Data directory management
   - Performance tuning
   - Troubleshooting

---

## 🔗 Dependencies

### External Libraries
- **SQLite3** (3.35+): Core database engine
- **bcrypt** or **libsodium**: Password hashing
- **Existing dependencies**: Crow, nlohmann/json, etc.

### Internal Dependencies
- Depends on: AuthenticationService, DatabaseService, VectorStorageService
- Depended on by: REST API, CLI tools

---

## 📈 Metrics & Monitoring

### Key Metrics
- **SQLite operations**: Query time, transaction time, checkpoint time
- **Vector I/O**: Read/write throughput, mmap file size, cache hit rate
- **Authentication**: Login success/failure rate, token expiry rate
- **Permissions**: Permission check latency, cache hit rate

### Alerts
- SQLite database locked (>1 second)
- Vector file corruption detected
- Disk space <10% free
- Authentication failure rate >5%

---

## 🎯 Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1+2 - Sprint 1.1** | Days 1-3 | SQLite foundation and schema |
| **Phase 1+2 - Sprint 1.2** | Days 4-7 | User & RBAC system |
| **Phase 1+2 - Sprint 1.3** | Days 8-10 | Tokens & API keys |
| **Phase 1+2 - Sprint 1.4** | Days 11-14 | Database metadata persistence |
| **Phase 1+2 - Sprint 1.5** | Days 15-18 | Backend integration testing |
| **Phase 1+2 - Sprint 1.6** | Days 1-18 (parallel) | Frontend RBAC UI implementation |
| **Phase 1+2 - Sprint 1.7** | Days 19-25 | Frontend integration & E2E testing |
| **Phase 3 - Sprint 3.1** | Days 26-29 | Memory-mapped files |
| **Phase 3 - Sprint 3.2** | Days 30-32 | Vector persistence integration |
| **Phase 3 - Sprint 3.3** | Days 33-42 | Testing & optimization |
| **Total** | **6-7 weeks** | **Full persistent storage + RBAC UI** |

---

## ✅ Definition of Done

### Phase 1+2 Complete When:
- [ ] All Sprint 1.x tasks completed
- [ ] Unit tests pass (95%+ coverage)
- [ ] Integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] CLI tests updated and passing
- [ ] Documentation complete
- [ ] Code reviewed and merged

### Phase 3 Complete When:
- [ ] All Sprint 3.x tasks completed
- [ ] Vector persistence tests pass
- [ ] Performance benchmarks meet targets (<10% regression)
- [ ] Cross-platform tests pass
- [ ] Documentation complete
- [ ] Production deployment successful

---

## 🏁 Next Steps

1. **Review this plan** with the team
2. **Create GitHub issues** for each task
3. **Set up project board** with sprints
4. **Assign initial tasks** (T11.1.1 - T11.1.4)
5. **Begin Sprint 1.1** (Foundation & Schema)

---

**Document Version**: 1.0
**Last Updated**: 2024-12-16
**Author**: GitHub Copilot
**Reviewers**: [To be assigned]
