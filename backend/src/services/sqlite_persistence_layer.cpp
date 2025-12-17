#include "sqlite_persistence_layer.h"
#include "models/auth.h"
#include <random>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

namespace jadevectordb {

SQLitePersistenceLayer::SQLitePersistenceLayer(const std::string& data_directory)
    : db_(nullptr)
    , data_directory_(data_directory)
    , db_file_path_(data_directory + "/system.db") {
    logger_ = logging::LoggerManager::get_logger("SQLitePersistenceLayer");
}

SQLitePersistenceLayer::~SQLitePersistenceLayer() {
    if (db_) {
        close();
    }
}

Result<void> SQLitePersistenceLayer::initialize() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    // Create data directory if it doesn't exist
    mkdir(data_directory_.c_str(), 0700);
    
    // Open SQLite database
    int rc = sqlite3_open(db_file_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string error = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        db_ = nullptr;
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to open SQLite database: " + error);
    }
    
    LOG_INFO(logger_, "SQLite database opened: " << db_file_path_);
    
    // Enable WAL mode for better concurrency
    auto wal_result = execute_sql("PRAGMA journal_mode=WAL");
    if (!wal_result.has_value()) {
        LOG_WARN(logger_, "Failed to enable WAL mode: " << wal_result.error().message);
    }
    
    // Set cache size (10MB)
    auto cache_result = execute_sql("PRAGMA cache_size=10000");
    (void)cache_result; // Intentionally unused
    
    // Enable foreign keys
    auto fk_result = execute_sql("PRAGMA foreign_keys=ON");
    if (!fk_result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to enable foreign keys");
    }
    
    // Create tables
    auto tables_result = create_tables();
    if (!tables_result.has_value()) {
        return tables_result;
    }
    
    // Create indexes
    auto indexes_result = create_indexes();
    if (!indexes_result.has_value()) {
        return indexes_result;
    }
    
    // Insert default roles and permissions
    auto defaults_result = insert_default_roles_and_permissions();
    if (!defaults_result.has_value()) {
        return defaults_result;
    }
    
    LOG_INFO(logger_, "SQLite persistence layer initialized successfully");
    return {};
}

Result<void> SQLitePersistenceLayer::close() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        LOG_INFO(logger_, "SQLite database closed");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::create_tables() {
    // Users table
    auto result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            is_system_admin INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            last_login INTEGER,
            failed_login_attempts INTEGER DEFAULT 0,
            account_locked_until INTEGER,
            metadata TEXT
        )
    )");
    if (!result.has_value()) return result;
    
    // Groups table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS groups (
            group_id TEXT PRIMARY KEY,
            group_name TEXT UNIQUE NOT NULL,
            description TEXT,
            owner_user_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            FOREIGN KEY (owner_user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    )");
    if (!result.has_value()) return result;
    
    // Group members (junction table)
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS group_members (
            group_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            joined_at INTEGER NOT NULL,
            PRIMARY KEY (group_id, user_id),
            FOREIGN KEY (group_id) REFERENCES groups(group_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    )");
    if (!result.has_value()) return result;
    
    // Roles table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS roles (
            role_id TEXT PRIMARY KEY,
            role_name TEXT UNIQUE NOT NULL,
            description TEXT,
            is_system_role INTEGER DEFAULT 0
        )
    )");
    if (!result.has_value()) return result;
    
    // Permissions table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS permissions (
            permission_id TEXT PRIMARY KEY,
            permission_name TEXT UNIQUE NOT NULL,
            resource_type TEXT NOT NULL,
            action TEXT NOT NULL,
            description TEXT
        )
    )");
    if (!result.has_value()) return result;
    
    // Role-Permission mapping
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id TEXT NOT NULL,
            permission_id TEXT NOT NULL,
            granted_at INTEGER NOT NULL,
            PRIMARY KEY (role_id, permission_id),
            FOREIGN KEY (role_id) REFERENCES roles(role_id) ON DELETE CASCADE,
            FOREIGN KEY (permission_id) REFERENCES permissions(permission_id) ON DELETE CASCADE
        )
    )");
    if (!result.has_value()) return result;
    
    // User-Role assignment
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            assigned_at INTEGER NOT NULL,
            assigned_by TEXT,
            PRIMARY KEY (user_id, role_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (role_id) REFERENCES roles(role_id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_by) REFERENCES users(user_id) ON DELETE SET NULL
        )
    )");
    if (!result.has_value()) return result;
    
    // Group-Role assignment
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS group_roles (
            group_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            assigned_at INTEGER NOT NULL,
            assigned_by TEXT,
            PRIMARY KEY (group_id, role_id),
            FOREIGN KEY (group_id) REFERENCES groups(group_id) ON DELETE CASCADE,
            FOREIGN KEY (role_id) REFERENCES roles(role_id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_by) REFERENCES users(user_id) ON DELETE SET NULL
        )
    )");
    if (!result.has_value()) return result;
    
    // Database-level permissions
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS database_permissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            database_id TEXT NOT NULL,
            principal_type TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            permission_id TEXT NOT NULL,
            granted_at INTEGER NOT NULL,
            granted_by TEXT,
            FOREIGN KEY (permission_id) REFERENCES permissions(permission_id),
            FOREIGN KEY (granted_by) REFERENCES users(user_id),
            UNIQUE (database_id, principal_type, principal_id, permission_id)
        )
    )");
    if (!result.has_value()) return result;
    
    // API Keys table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS api_keys (
            api_key_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            key_hash TEXT NOT NULL,
            key_name TEXT,
            key_prefix TEXT NOT NULL,
            scopes TEXT,
            is_active INTEGER DEFAULT 1,
            expires_at INTEGER,
            created_at INTEGER NOT NULL,
            last_used_at INTEGER,
            usage_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    )");
    if (!result.has_value()) return result;
    
    // Authentication tokens table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS auth_tokens (
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
        )
    )");
    if (!result.has_value()) return result;
    
    // Sessions table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token_id TEXT,
            ip_address TEXT,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            is_active INTEGER DEFAULT 1,
            metadata TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (token_id) REFERENCES auth_tokens(token_id) ON DELETE SET NULL
        )
    )");
    if (!result.has_value()) return result;
    
    // Databases metadata table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS databases (
            database_id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            owner_user_id TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            index_type TEXT,
            vector_count INTEGER DEFAULT 0,
            index_count INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            metadata TEXT,
            FOREIGN KEY (owner_user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    )");
    if (!result.has_value()) return result;
    
    // Audit logs table
    result = execute_sql(R"(
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            ip_address TEXT,
            success INTEGER NOT NULL,
            details TEXT,
            timestamp INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
        )
    )");
    if (!result.has_value()) return result;
    
    LOG_INFO(logger_, "All database tables created successfully");
    return {};
}

Result<void> SQLitePersistenceLayer::create_indexes() {
    // User indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)");
    
    // Group indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_groups_name ON groups(group_name)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_groups_owner ON groups(owner_user_id)");
    
    // Group members indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_group_members_user ON group_members(user_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_group_members_group ON group_members(group_id)");
    
    // User roles indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id)");
    
    // Group roles indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_group_roles_group ON group_roles(group_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_group_roles_role ON group_roles(role_id)");
    
    // Database permissions indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_db_perms_database ON database_permissions(database_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_db_perms_principal ON database_permissions(principal_type, principal_id)");
    
    // API keys indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)");
    
    // Auth tokens indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_tokens_expires ON auth_tokens(expires_at)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_tokens_valid ON auth_tokens(is_valid)");
    
    // Sessions indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)");
    
    // Database metadata indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_databases_name ON databases(name)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_databases_owner ON databases(owner_user_id)");
    
    // Audit logs indexes
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id)");
    (void)execute_sql("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)");
    
    LOG_INFO(logger_, "All database indexes created successfully");
    return {};
}

Result<void> SQLitePersistenceLayer::insert_default_roles_and_permissions() {
    // Check if roles already exist
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM roles", -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int count = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
            if (count > 0) {
                LOG_INFO(logger_, "Default roles and permissions already exist");
                return {};
            }
        } else {
            sqlite3_finalize(stmt);
        }
    }
    
    int64_t now = current_timestamp();
    
    // Insert default permissions
    const char* permissions_sql = R"(
        INSERT OR IGNORE INTO permissions (permission_id, permission_name, resource_type, action, description) VALUES
        ('perm_sys_admin', 'system:admin', 'system', 'admin', 'Full system administration'),
        ('perm_db_admin', 'database:admin', 'database', 'admin', 'Database administration'),
        ('perm_db_create', 'database:create', 'database', 'create', 'Create new databases'),
        ('perm_db_read', 'database:read', 'database', 'read', 'Read database content'),
        ('perm_db_write', 'database:write', 'database', 'write', 'Write to database'),
        ('perm_db_delete', 'database:delete', 'database', 'delete', 'Delete database'),
        ('perm_vec_read', 'vector:read', 'vector', 'read', 'Read vectors'),
        ('perm_vec_write', 'vector:write', 'vector', 'write', 'Write vectors'),
        ('perm_vec_delete', 'vector:delete', 'vector', 'delete', 'Delete vectors')
    )";
    
    auto result = execute_sql(permissions_sql);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to insert default permissions: " << result.error().message);
        return result;
    }
    
    // Insert default roles
    const char* roles_sql = R"(
        INSERT OR IGNORE INTO roles (role_id, role_name, description, is_system_role) VALUES
        ('role_admin', 'admin', 'System administrator with full access', 1),
        ('role_user', 'user', 'Regular user with standard permissions', 1),
        ('role_readonly', 'readonly', 'Read-only access to databases', 1)
    )";
    
    result = execute_sql(roles_sql);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to insert default roles: " << result.error().message);
        return result;
    }
    
    // Map permissions to roles
    std::stringstream role_perms_sql;
    role_perms_sql << "INSERT OR IGNORE INTO role_permissions (role_id, permission_id, granted_at) VALUES ";
    role_perms_sql << "('role_admin', 'perm_sys_admin', " << now << "),";
    role_perms_sql << "('role_admin', 'perm_db_admin', " << now << "),";
    role_perms_sql << "('role_admin', 'perm_db_create', " << now << "),";
    role_perms_sql << "('role_admin', 'perm_db_read', " << now << "),";
    role_perms_sql << "('role_admin', 'perm_db_write', " << now << "),";
    role_perms_sql << "('role_admin', 'perm_db_delete', " << now << "),";
    role_perms_sql << "('role_user', 'perm_db_create', " << now << "),";
    role_perms_sql << "('role_user', 'perm_db_read', " << now << "),";
    role_perms_sql << "('role_user', 'perm_db_write', " << now << "),";
    role_perms_sql << "('role_readonly', 'perm_db_read', " << now << "),";
    role_perms_sql << "('role_readonly', 'perm_vec_read', " << now << ")";
    
    result = execute_sql(role_perms_sql.str());
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to map role permissions: " << result.error().message);
        return result;
    }
    
    LOG_INFO(logger_, "Default roles and permissions inserted successfully");
    return {};
}

// === User Management Implementation ===

Result<std::string> SQLitePersistenceLayer::create_user(
    const std::string& username,
    const std::string& email,
    const std::string& password_hash,
    const std::string& salt) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string user_id = generate_id();
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO users (user_id, username, email, password_hash, salt, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, email.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, password_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, salt.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 6, now);
    sqlite3_bind_int64(stmt, 7, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        std::string error = sqlite3_errmsg(db_);
        if (error.find("UNIQUE constraint failed") != std::string::npos) {
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Username or email already exists");
        }
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create user: " + error);
    }
    
    LOG_INFO(logger_, "Created user: " << username << " (ID: " << user_id << ")");
    return user_id;
}

Result<User> SQLitePersistenceLayer::get_user(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT user_id, username, email, password_hash, salt, is_active, is_system_admin,
               created_at, updated_at, last_login, failed_login_attempts, account_locked_until, metadata
        FROM users WHERE user_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    User user;
    user.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    user.email = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    user.password_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    user.salt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    user.is_active = sqlite3_column_int(stmt, 5) != 0;
    user.is_system_admin = sqlite3_column_int(stmt, 6) != 0;
    user.created_at = sqlite3_column_int64(stmt, 7);
    user.updated_at = sqlite3_column_int64(stmt, 8);
    user.last_login = sqlite3_column_int64(stmt, 9);
    user.failed_login_attempts = sqlite3_column_int(stmt, 10);
    user.account_locked_until = sqlite3_column_int64(stmt, 11);
    if (sqlite3_column_text(stmt, 12)) {
        user.metadata = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12));
    }
    
    sqlite3_finalize(stmt);
    return user;
}

Result<User> SQLitePersistenceLayer::get_user_by_username(const std::string& username) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT user_id, username, email, password_hash, salt, is_active, is_system_admin,
               created_at, updated_at, last_login, failed_login_attempts, account_locked_until, metadata
        FROM users WHERE username = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + username);
    }
    
    User user;
    user.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    user.email = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    user.password_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    user.salt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    user.is_active = sqlite3_column_int(stmt, 5) != 0;
    user.is_system_admin = sqlite3_column_int(stmt, 6) != 0;
    user.created_at = sqlite3_column_int64(stmt, 7);
    user.updated_at = sqlite3_column_int64(stmt, 8);
    user.last_login = sqlite3_column_int64(stmt, 9);
    user.failed_login_attempts = sqlite3_column_int(stmt, 10);
    user.account_locked_until = sqlite3_column_int64(stmt, 11);
    if (sqlite3_column_text(stmt, 12)) {
        user.metadata = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12));
    }
    
    sqlite3_finalize(stmt);
    return user;
}

Result<std::vector<User>> SQLitePersistenceLayer::list_users(int limit, int offset) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT user_id, username, email, password_hash, salt, is_active, is_system_admin,
               created_at, updated_at, last_login, failed_login_attempts, account_locked_until, metadata
        FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_int(stmt, 1, limit);
    sqlite3_bind_int(stmt, 2, offset);
    
    std::vector<User> users;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        User user;
        user.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        user.email = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        user.password_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        user.salt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        user.is_active = sqlite3_column_int(stmt, 5) != 0;
        user.is_system_admin = sqlite3_column_int(stmt, 6) != 0;
        user.created_at = sqlite3_column_int64(stmt, 7);
        user.updated_at = sqlite3_column_int64(stmt, 8);
        user.last_login = sqlite3_column_int64(stmt, 9);
        user.failed_login_attempts = sqlite3_column_int(stmt, 10);
        user.account_locked_until = sqlite3_column_int64(stmt, 11);
        if (sqlite3_column_text(stmt, 12)) {
            user.metadata = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12));
        }
        users.push_back(user);
    }
    
    sqlite3_finalize(stmt);
    return users;
}

Result<void> SQLitePersistenceLayer::update_last_login(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE users SET last_login = ? WHERE user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_int64(stmt, 1, current_timestamp());
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update last login");
    }
    
    return {};
}

Result<User> SQLitePersistenceLayer::get_user_by_email(const std::string& email) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT user_id, username, email, password_hash, salt, is_active, is_system_admin,
               created_at, updated_at, last_login, failed_login_attempts, account_locked_until
        FROM users WHERE email = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, email.c_str(), -1, SQLITE_TRANSIENT);
    
    User user;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        user.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        user.username = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        user.email = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        user.password_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        user.salt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        user.is_active = sqlite3_column_int(stmt, 5) != 0;
        user.is_system_admin = sqlite3_column_int(stmt, 6) != 0;
        user.created_at = sqlite3_column_int64(stmt, 7);
        user.updated_at = sqlite3_column_int64(stmt, 8);
        user.last_login = sqlite3_column_int64(stmt, 9);
        user.failed_login_attempts = sqlite3_column_int(stmt, 10);
        user.account_locked_until = sqlite3_column_int64(stmt, 11);
        
        sqlite3_finalize(stmt);
        return user;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found");
}

Result<void> SQLitePersistenceLayer::update_user(const std::string& user_id, const User& user) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        UPDATE users SET 
            username = ?, 
            email = ?, 
            password_hash = ?, 
            salt = ?,
            is_active = ?,
            is_system_admin = ?,
            updated_at = ?,
            failed_login_attempts = ?,
            account_locked_until = ?
        WHERE user_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user.username.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user.email.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, user.password_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, user.salt.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 5, user.is_active ? 1 : 0);
    sqlite3_bind_int(stmt, 6, user.is_system_admin ? 1 : 0);
    sqlite3_bind_int64(stmt, 7, now);
    sqlite3_bind_int(stmt, 8, user.failed_login_attempts);
    sqlite3_bind_int64(stmt, 9, user.account_locked_until);
    sqlite3_bind_text(stmt, 10, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update user");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::delete_user(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "DELETE FROM users WHERE user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to delete user");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::increment_failed_login(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE users SET failed_login_attempts = failed_login_attempts + 1 WHERE user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to increment failed login attempts");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::reset_failed_login(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE users SET failed_login_attempts = 0, account_locked_until = 0 WHERE user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to reset failed login attempts");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::lock_account(const std::string& user_id, int64_t lock_until_timestamp) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE users SET account_locked_until = ? WHERE user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, lock_until_timestamp);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to lock account");
    }
    
    return {};
}

// === Group Management Implementation ===

Result<std::string> SQLitePersistenceLayer::create_group(
    const std::string& group_name,
    const std::string& description,
    const std::string& owner_user_id) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string group_id = generate_id();
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO groups (group_id, group_name, description, owner_user_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, group_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, owner_user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, now);
    sqlite3_bind_int64(stmt, 6, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create group: " + std::string(sqlite3_errmsg(db_)));
    }
    
    return group_id;
}

Result<Group> SQLitePersistenceLayer::get_group(const std::string& group_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT group_id, group_name, description, owner_user_id, created_at, updated_at
        FROM groups WHERE group_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    
    Group group;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        group.group_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        group.group_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        group.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        group.owner_user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        group.created_at = sqlite3_column_int64(stmt, 4);
        group.updated_at = sqlite3_column_int64(stmt, 5);
        
        sqlite3_finalize(stmt);
        return group;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Group not found");
}

Result<std::vector<Group>> SQLitePersistenceLayer::list_groups(int limit, int offset) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT group_id, group_name, description, owner_user_id, created_at, updated_at
        FROM groups
        ORDER BY group_name
        LIMIT ? OFFSET ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int(stmt, 1, limit);
    sqlite3_bind_int(stmt, 2, offset);
    
    std::vector<Group> groups;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Group group;
        group.group_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        group.group_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        group.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        group.owner_user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        group.created_at = sqlite3_column_int64(stmt, 4);
        group.updated_at = sqlite3_column_int64(stmt, 5);
        groups.push_back(group);
    }
    
    sqlite3_finalize(stmt);
    return groups;
}

Result<void> SQLitePersistenceLayer::update_group(const std::string& group_id, const Group& group) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        UPDATE groups SET 
            group_name = ?, 
            description = ?,
            owner_user_id = ?,
            updated_at = ?
        WHERE group_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group.group_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, group.description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, group.owner_user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 4, now);
    sqlite3_bind_text(stmt, 5, group_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update group");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::delete_group(const std::string& group_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "DELETE FROM groups WHERE group_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to delete group");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::add_user_to_group(const std::string& group_id, const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "INSERT INTO group_members (group_id, user_id, joined_at) VALUES (?, ?, ?)";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to add user to group");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::remove_user_from_group(const std::string& group_id, const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "DELETE FROM group_members WHERE group_id = ? AND user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to remove user from group");
    }
    
    return {};
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_group_members(const std::string& group_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT user_id FROM group_members
        WHERE group_id = ?
        ORDER BY joined_at
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> user_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        user_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return user_ids;
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_user_groups(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT group_id FROM group_members
        WHERE user_id = ?
        ORDER BY joined_at
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> group_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        group_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return group_ids;
}

// === Role Management Implementation ===

Result<void> SQLitePersistenceLayer::assign_role_to_user(
    const std::string& user_id,
    const std::string& role_id) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "INSERT INTO user_roles (user_id, role_id, assigned_at) VALUES (?, ?, ?)";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, role_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to assign role to user");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::revoke_role_from_user(const std::string& user_id, const std::string& role_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, role_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to revoke role from user");
    }
    
    return {};
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_user_roles(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT role_id FROM user_roles
        WHERE user_id = ?
        ORDER BY assigned_at
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> role_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        role_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return role_ids;
}

Result<void> SQLitePersistenceLayer::assign_role_to_group(
    const std::string& group_id,
    const std::string& role_id) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "INSERT INTO group_roles (group_id, role_id, assigned_at) VALUES (?, ?, ?)";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, role_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to assign role to group");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::revoke_role_from_group(const std::string& group_id, const std::string& role_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "DELETE FROM group_roles WHERE group_id = ? AND role_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, role_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to revoke role from group");
    }
    
    return {};
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_group_roles(const std::string& group_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT role_id FROM group_roles
        WHERE group_id = ?
        ORDER BY assigned_at
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, group_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> role_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        role_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return role_ids;
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_role_permissions(const std::string& role_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT permission_id FROM role_permissions
        WHERE role_id = ?
        ORDER BY granted_at
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, role_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> permission_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        permission_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return permission_ids;
}

// === Helper Methods ===

Result<void> SQLitePersistenceLayer::execute_sql(const std::string& sql) {
    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_msg);
    
    if (rc != SQLITE_OK) {
        std::string error = error_msg ? error_msg : "Unknown error";
        sqlite3_free(error_msg);
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "SQL execution failed: " + error);
    }
    
    return {};
}

std::string SQLitePersistenceLayer::generate_id() const {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << dis(gen);
    return ss.str();
}

int64_t SQLitePersistenceLayer::current_timestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

Result<bool> SQLitePersistenceLayer::user_exists(const std::string& username) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "SELECT COUNT(*) FROM users WHERE username = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_text(stmt, 1, username.c_str(), -1, SQLITE_TRANSIENT);
    
    bool exists = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        exists = sqlite3_column_int(stmt, 0) > 0;
    }
    
    sqlite3_finalize(stmt);
    return exists;
}

Result<bool> SQLitePersistenceLayer::email_exists(const std::string& email) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "SELECT COUNT(*) FROM users WHERE email = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_text(stmt, 1, email.c_str(), -1, SQLITE_TRANSIENT);
    
    bool exists = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        exists = sqlite3_column_int(stmt, 0) > 0;
    }
    
    sqlite3_finalize(stmt);
    return exists;
}

Result<bool> SQLitePersistenceLayer::group_exists(const std::string& group_name) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "SELECT COUNT(*) FROM groups WHERE group_name = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_text(stmt, 1, group_name.c_str(), -1, SQLITE_TRANSIENT);
    
    bool exists = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        exists = sqlite3_column_int(stmt, 0) > 0;
    }
    
    sqlite3_finalize(stmt);
    return exists;
}

Result<bool> SQLitePersistenceLayer::database_name_exists(const std::string& name) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "SELECT COUNT(*) FROM databases WHERE name = ?";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement");
    }
    
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    
    bool exists = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        exists = sqlite3_column_int(stmt, 0) > 0;
    }
    
    sqlite3_finalize(stmt);
    return exists;
}

// === Permission Management Implementation ===

Result<std::vector<Permission>> SQLitePersistenceLayer::list_permissions() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT permission_id, permission_name, resource_type, action, description
        FROM permissions
        ORDER BY permission_name
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    std::vector<Permission> permissions;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Permission perm;
        perm.permission_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        perm.permission_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        perm.resource_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        perm.action = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        perm.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        permissions.push_back(perm);
    }
    
    sqlite3_finalize(stmt);
    return permissions;
}

Result<Permission> SQLitePersistenceLayer::get_permission(const std::string& permission_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT permission_id, permission_name, resource_type, action, description
        FROM permissions WHERE permission_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, permission_id.c_str(), -1, SQLITE_TRANSIENT);
    
    Permission perm;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        perm.permission_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        perm.permission_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        perm.resource_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        perm.action = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        perm.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        
        sqlite3_finalize(stmt);
        return perm;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Permission not found");
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_user_permissions(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT DISTINCT p.permission_id
        FROM permissions p
        INNER JOIN role_permissions rp ON p.permission_id = rp.permission_id
        INNER JOIN user_roles ur ON rp.role_id = ur.role_id
        WHERE ur.user_id = ?
        UNION
        SELECT DISTINCT p.permission_id
        FROM permissions p
        INNER JOIN role_permissions rp ON p.permission_id = rp.permission_id
        INNER JOIN group_roles gr ON rp.role_id = gr.role_id
        INNER JOIN group_members gm ON gr.group_id = gm.group_id
        WHERE gm.user_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> permission_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        permission_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return permission_ids;
}

Result<void> SQLitePersistenceLayer::grant_database_permission(
    const std::string& database_id,
    const std::string& principal_type,
    const std::string& principal_id,
    const std::string& permission_id,
    const std::string& granted_by) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO database_permissions (database_id, principal_type, principal_id, permission_id, granted_at, granted_by)
        VALUES (?, ?, ?, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, principal_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, principal_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, permission_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, now);
    sqlite3_bind_text(stmt, 6, granted_by.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to grant database permission");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::revoke_database_permission(
    const std::string& database_id,
    const std::string& principal_type,
    const std::string& principal_id,
    const std::string& permission_id) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        DELETE FROM database_permissions 
        WHERE database_id = ? AND principal_type = ? AND principal_id = ? AND permission_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, principal_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, principal_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, permission_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to revoke database permission");
    }
    
    return {};
}

Result<std::vector<std::string>> SQLitePersistenceLayer::get_database_permissions(
    const std::string& database_id,
    const std::string& user_id) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT DISTINCT dp.permission_id
        FROM database_permissions dp
        WHERE dp.database_id = ? 
        AND (
            (dp.principal_type = 'user' AND dp.principal_id = ?)
            OR
            (dp.principal_type = 'group' AND dp.principal_id IN (
                SELECT group_id FROM group_members WHERE user_id = ?
            ))
        )
        ORDER BY dp.permission_id
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<std::string> permission_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        permission_ids.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }
    
    sqlite3_finalize(stmt);
    return permission_ids;
}

Result<bool> SQLitePersistenceLayer::check_database_permission(
    const std::string& database_id,
    const std::string& user_id,
    const std::string& permission_name) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT COUNT(*) FROM database_permissions dp
        INNER JOIN permissions p ON dp.permission_id = p.permission_id
        WHERE dp.database_id = ? 
        AND p.permission_name = ?
        AND (
            (dp.principal_type = 'user' AND dp.principal_id = ?)
            OR
            (dp.principal_type = 'group' AND dp.principal_id IN (
                SELECT group_id FROM group_members WHERE user_id = ?
            ))
        )
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, permission_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    bool has_permission = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        has_permission = sqlite3_column_int(stmt, 0) > 0;
    }
    
    sqlite3_finalize(stmt);
    return has_permission;
}

// === API Key Management Implementation ===

Result<std::string> SQLitePersistenceLayer::create_api_key(
    const std::string& user_id,
    const std::string& key_hash,
    const std::string& key_name,
    const std::string& key_prefix,
    const std::vector<std::string>& scopes,
    int64_t expires_at) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string api_key_id = generate_id();
    int64_t now = current_timestamp();
    
    // Convert scopes vector to JSON string
    std::stringstream scopes_json;
    scopes_json << "[";
    for (size_t i = 0; i < scopes.size(); ++i) {
        scopes_json << "\"" << scopes[i] << "\"";
        if (i < scopes.size() - 1) scopes_json << ",";
    }
    scopes_json << "]";
    
    const char* sql = R"(
        INSERT INTO api_keys (api_key_id, user_id, key_hash, key_name, key_prefix, scopes, 
                              is_active, created_at, expires_at, last_used_at, usage_count)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, 0, 0)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, api_key_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, key_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, key_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, key_prefix.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, scopes_json.str().c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 7, now);
    sqlite3_bind_int64(stmt, 8, expires_at);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create API key");
    }
    
    return api_key_id;
}

Result<APIKey> SQLitePersistenceLayer::get_api_key_by_id(const std::string& api_key_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT api_key_id, user_id, key_hash, key_name, key_prefix, scopes, is_active,
               created_at, expires_at, last_used_at, usage_count
        FROM api_keys WHERE api_key_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, api_key_id.c_str(), -1, SQLITE_TRANSIENT);
    
    APIKey api_key;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        api_key.api_key_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        api_key.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        api_key.key_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        api_key.key_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        api_key.key_prefix = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        // TODO: Parse JSON scopes from column 5 - skipped for now
        api_key.is_active = sqlite3_column_int(stmt, 6) != 0;
        api_key.created_at = sqlite3_column_int64(stmt, 7);
        api_key.expires_at = sqlite3_column_int64(stmt, 8);
        api_key.last_used_at = sqlite3_column_int64(stmt, 9);
        api_key.usage_count = sqlite3_column_int(stmt, 10);
        
        sqlite3_finalize(stmt);
        return api_key;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "API key not found");
}

Result<APIKey> SQLitePersistenceLayer::get_api_key_by_prefix(const std::string& key_prefix) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT api_key_id, user_id, key_hash, key_name, key_prefix, scopes, is_active,
               created_at, expires_at, last_used_at, usage_count
        FROM api_keys WHERE key_prefix = ? AND is_active = 1
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, key_prefix.c_str(), -1, SQLITE_TRANSIENT);
    
    APIKey api_key;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        api_key.api_key_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        api_key.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        api_key.key_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        api_key.key_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        api_key.key_prefix = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        // TODO: Parse JSON scopes from column 5 - skipped for now
        api_key.is_active = sqlite3_column_int(stmt, 6) != 0;
        api_key.created_at = sqlite3_column_int64(stmt, 7);
        api_key.expires_at = sqlite3_column_int64(stmt, 8);
        api_key.last_used_at = sqlite3_column_int64(stmt, 9);
        api_key.usage_count = sqlite3_column_int(stmt, 10);
        
        sqlite3_finalize(stmt);
        return api_key;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "API key not found");
}

Result<std::vector<APIKey>> SQLitePersistenceLayer::list_user_api_keys(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT api_key_id, user_id, key_hash, key_name, key_prefix, scopes, is_active,
               created_at, expires_at, last_used_at, usage_count
        FROM api_keys WHERE user_id = ?
        ORDER BY created_at DESC
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    std::vector<APIKey> api_keys;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        APIKey api_key;
        api_key.api_key_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        api_key.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        api_key.key_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        api_key.key_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        api_key.key_prefix = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        // TODO: Parse JSON scopes from column 5 - skipped for now
        api_key.is_active = sqlite3_column_int(stmt, 6) != 0;
        api_key.created_at = sqlite3_column_int64(stmt, 7);
        api_key.expires_at = sqlite3_column_int64(stmt, 8);
        api_key.last_used_at = sqlite3_column_int64(stmt, 9);
        api_key.usage_count = sqlite3_column_int(stmt, 10);
        api_keys.push_back(api_key);
    }
    
    sqlite3_finalize(stmt);
    return api_keys;
}

Result<void> SQLitePersistenceLayer::revoke_api_key(const std::string& api_key_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE api_keys SET is_active = 0 WHERE api_key_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, api_key_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to revoke API key");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::update_api_key_usage(const std::string& api_key_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "UPDATE api_keys SET last_used_at = ?, usage_count = usage_count + 1 WHERE api_key_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, now);
    sqlite3_bind_text(stmt, 2, api_key_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update API key usage");
    }
    
    return {};
}

// === Authentication Token Management Implementation ===

Result<std::string> SQLitePersistenceLayer::create_auth_token(
    const std::string& user_id,
    const std::string& token_hash,
    const std::string& ip_address,
    const std::string& user_agent,
    int64_t expires_at) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string token_id = generate_id();
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO auth_tokens (token_id, user_id, token_hash, ip_address, user_agent,
                                 issued_at, expires_at, last_used_at, is_valid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, token_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, token_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, ip_address.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, user_agent.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 6, now);
    sqlite3_bind_int64(stmt, 7, expires_at);
    sqlite3_bind_int64(stmt, 8, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create auth token");
    }
    
    return token_id;
}

Result<AuthToken> SQLitePersistenceLayer::get_auth_token(const std::string& token_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT token_id, user_id, token_hash, ip_address, user_agent,
               issued_at, expires_at, last_used_at, is_valid
        FROM auth_tokens WHERE token_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, token_id.c_str(), -1, SQLITE_TRANSIENT);
    
    AuthToken token;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        token.token_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        token.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        token.token_hash = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        token.ip_address = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        token.user_agent = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        token.issued_at = sqlite3_column_int64(stmt, 5);
        token.expires_at = sqlite3_column_int64(stmt, 6);
        token.last_used_at = sqlite3_column_int64(stmt, 7);
        token.is_valid = sqlite3_column_int(stmt, 8) != 0;
        
        sqlite3_finalize(stmt);
        return token;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Auth token not found");
}

Result<void> SQLitePersistenceLayer::invalidate_auth_token(const std::string& token_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE auth_tokens SET is_valid = 0 WHERE token_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, token_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to invalidate auth token");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::invalidate_user_tokens(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE auth_tokens SET is_valid = 0 WHERE user_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to invalidate user tokens");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::update_token_last_used(const std::string& token_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "UPDATE auth_tokens SET last_used_at = ? WHERE token_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, now);
    sqlite3_bind_text(stmt, 2, token_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update token last used");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::cleanup_expired_tokens() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "DELETE FROM auth_tokens WHERE expires_at > 0 AND expires_at < ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to cleanup expired tokens");
    }
    
    return {};
}

// === Session Management Implementation ===

Result<std::string> SQLitePersistenceLayer::create_session(
    const std::string& user_id,
    const std::string& token_id,
    const std::string& ip_address,
    int64_t expires_at) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string session_id = generate_id();
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO sessions (session_id, user_id, token_id, ip_address, 
                              created_at, last_activity, expires_at, is_active, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, '{}')
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, token_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, ip_address.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, now);
    sqlite3_bind_int64(stmt, 6, now);
    sqlite3_bind_int64(stmt, 7, expires_at);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create session");
    }
    
    return session_id;
}

Result<Session> SQLitePersistenceLayer::get_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT session_id, user_id, token_id, ip_address,
               created_at, last_activity, expires_at, is_active, metadata
        FROM sessions WHERE session_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
    
    Session session;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        session.session_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        session.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        session.token_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        session.ip_address = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        session.created_at = sqlite3_column_int64(stmt, 4);
        session.last_activity = sqlite3_column_int64(stmt, 5);
        session.expires_at = sqlite3_column_int64(stmt, 6);
        session.is_active = sqlite3_column_int(stmt, 7) != 0;
        session.metadata = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8));
        
        sqlite3_finalize(stmt);
        return session;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Session not found");
}

Result<void> SQLitePersistenceLayer::update_session_activity(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "UPDATE sessions SET last_activity = ? WHERE session_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, now);
    sqlite3_bind_text(stmt, 2, session_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update session activity");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::end_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "UPDATE sessions SET is_active = 0 WHERE session_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to end session");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::cleanup_expired_sessions() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = "DELETE FROM sessions WHERE expires_at > 0 AND expires_at < ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to cleanup expired sessions");
    }
    
    return {};
}

// === DATABASE METADATA MANAGEMENT ===

Result<std::string> SQLitePersistenceLayer::store_database_metadata(
    const std::string& name,
    const std::string& description,
    const std::string& owner_user_id,
    int vector_dimension,
    const std::string& index_type,
    const std::string& metadata_json) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string database_id = generate_id();
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO databases (database_id, name, description, owner_user_id,
                              vector_dimension, index_type, vector_count, index_count,
                              created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, owner_user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 5, vector_dimension);
    sqlite3_bind_text(stmt, 6, index_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 7, now);
    sqlite3_bind_int64(stmt, 8, now);
    sqlite3_bind_text(stmt, 9, metadata_json.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to store database metadata: " + std::string(sqlite3_errmsg(db_)));
    }
    
    return database_id;
}

Result<DatabaseMetadata> SQLitePersistenceLayer::get_database_metadata(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT database_id, name, description, owner_user_id,
               vector_dimension, index_type, vector_count, index_count,
               created_at, updated_at, metadata
        FROM databases WHERE database_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    
    if (rc == SQLITE_ROW) {
        DatabaseMetadata metadata;
        metadata.database_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        metadata.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        metadata.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        metadata.owner_user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        metadata.vector_dimension = sqlite3_column_int(stmt, 4);
        metadata.index_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        metadata.vector_count = sqlite3_column_int64(stmt, 6);
        metadata.index_count = sqlite3_column_int64(stmt, 7);
        metadata.created_at = sqlite3_column_int64(stmt, 8);
        metadata.updated_at = sqlite3_column_int64(stmt, 9);
        
        const char* metadata_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        metadata.metadata = metadata_text ? metadata_text : "";
        
        sqlite3_finalize(stmt);
        return metadata;
    }
    
    sqlite3_finalize(stmt);
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Database metadata not found");
}

Result<std::vector<DatabaseMetadata>> SQLitePersistenceLayer::list_database_metadata(int limit, int offset) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        SELECT database_id, name, description, owner_user_id,
               vector_dimension, index_type, vector_count, index_count,
               created_at, updated_at, metadata
        FROM databases
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int(stmt, 1, limit);
    sqlite3_bind_int(stmt, 2, offset);
    
    std::vector<DatabaseMetadata> databases_list;
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        DatabaseMetadata metadata;
        metadata.database_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        metadata.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        metadata.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        metadata.owner_user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        metadata.vector_dimension = sqlite3_column_int(stmt, 4);
        metadata.index_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        metadata.vector_count = sqlite3_column_int64(stmt, 6);
        metadata.index_count = sqlite3_column_int64(stmt, 7);
        metadata.created_at = sqlite3_column_int64(stmt, 8);
        metadata.updated_at = sqlite3_column_int64(stmt, 9);
        
        const char* metadata_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        metadata.metadata = metadata_text ? metadata_text : "";
        
        databases_list.push_back(metadata);
    }
    
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to list database metadata");
    }
    
    return databases_list;
}

Result<void> SQLitePersistenceLayer::update_database_metadata(
    const std::string& database_id,
    const DatabaseMetadata& metadata) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        UPDATE databases
        SET name = ?, description = ?, vector_dimension = ?, index_type = ?,
            updated_at = ?, metadata = ?
        WHERE database_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, metadata.name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, metadata.description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, metadata.vector_dimension);
    sqlite3_bind_text(stmt, 4, metadata.index_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, now);
    sqlite3_bind_text(stmt, 6, metadata.metadata.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, database_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update database metadata");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::delete_database_metadata(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "DELETE FROM databases WHERE database_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, database_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to delete database metadata");
    }
    
    return {};
}

Result<void> SQLitePersistenceLayer::update_database_stats(
    const std::string& database_id,
    int64_t vector_count,
    int64_t index_count) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        UPDATE databases
        SET vector_count = ?, index_count = ?, updated_at = ?
        WHERE database_id = ?
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_int64(stmt, 1, vector_count);
    sqlite3_bind_int64(stmt, 2, index_count);
    sqlite3_bind_int64(stmt, 3, now);
    sqlite3_bind_text(stmt, 4, database_id.c_str(), -1, SQLITE_TRANSIENT);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update database stats");
    }
    
    return {};
}

// === AUDIT LOGGING ===

Result<void> SQLitePersistenceLayer::log_audit_event(
    const std::string& user_id,
    const std::string& action,
    const std::string& resource_type,
    const std::string& resource_id,
    const std::string& ip_address,
    bool success,
    const std::string& details) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int64_t now = current_timestamp();
    
    const char* sql = R"(
        INSERT INTO audit_logs (user_id, action, resource_type, resource_id,
                               ip_address, success, details, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    sqlite3_bind_text(stmt, 1, user_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, action.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, resource_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, resource_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, ip_address.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 6, success ? 1 : 0);
    sqlite3_bind_text(stmt, 7, details.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 8, now);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log audit event: " + std::string(sqlite3_errmsg(db_)));
    }
    
    return {};
}

Result<std::vector<AuditLogEntry>> SQLitePersistenceLayer::get_audit_logs(
    int limit,
    int offset,
    const std::string& user_id,
    const std::string& action) {
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    // Build dynamic query based on filters
    std::string sql = R"(
        SELECT id, user_id, action, resource_type, resource_id,
               ip_address, success, details, timestamp
        FROM audit_logs
        WHERE 1=1
    )";
    
    if (!user_id.empty()) {
        sql += " AND user_id = ?";
    }
    if (!action.empty()) {
        sql += " AND action = ?";
    }
    
    sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }
    
    int param_idx = 1;
    if (!user_id.empty()) {
        sqlite3_bind_text(stmt, param_idx++, user_id.c_str(), -1, SQLITE_TRANSIENT);
    }
    if (!action.empty()) {
        sqlite3_bind_text(stmt, param_idx++, action.c_str(), -1, SQLITE_TRANSIENT);
    }
    sqlite3_bind_int(stmt, param_idx++, limit);
    sqlite3_bind_int(stmt, param_idx++, offset);
    
    std::vector<AuditLogEntry> logs;
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        AuditLogEntry entry;
        entry.id = sqlite3_column_int64(stmt, 0);
        entry.user_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        entry.action = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        entry.resource_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        entry.resource_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        entry.ip_address = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        entry.success = sqlite3_column_int(stmt, 6) == 1;
        
        const char* details_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        entry.details = details_text ? details_text : "";
        
        entry.timestamp = sqlite3_column_int64(stmt, 8);
        
        logs.push_back(entry);
    }
    
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get audit logs");
    }
    
    return logs;
}

// Transaction support
Result<void> SQLitePersistenceLayer::begin_transaction() {
    return execute_sql("BEGIN TRANSACTION");
}

Result<void> SQLitePersistenceLayer::commit_transaction() {
    return execute_sql("COMMIT");
}

Result<void> SQLitePersistenceLayer::rollback_transaction() {
    return execute_sql("ROLLBACK");
}

} // namespace jadevectordb
