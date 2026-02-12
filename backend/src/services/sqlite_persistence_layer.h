#ifndef JADEVECTORDB_SQLITE_PERSISTENCE_LAYER_H
#define JADEVECTORDB_SQLITE_PERSISTENCE_LAYER_H

#include <sqlite3.h>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "utils/circuit_breaker.h"

namespace jadevectordb {

// Forward declarations for RBAC models
struct User;
struct Group;
struct Role;
struct Permission;
struct APIKey;
struct AuthToken;
struct Session;
struct DatabaseMetadata;
struct AuditLogEntry;

// SQLite-based persistence layer for metadata
class SQLitePersistenceLayer {
public:
    explicit SQLitePersistenceLayer(const std::string& data_directory);
    ~SQLitePersistenceLayer();

    // Lifecycle
    Result<void> initialize();
    Result<void> close();
    
    // === User Management ===
    Result<std::string> create_user(const std::string& username,
                                     const std::string& email,
                                     const std::string& password_hash,
                                     const std::string& salt,
                                     bool must_change_password = false);
    Result<User> get_user(const std::string& user_id);
    Result<User> get_user_by_username(const std::string& username);
    Result<User> get_user_by_email(const std::string& email);
    Result<std::vector<User>> list_users(int limit = 100, int offset = 0);
    Result<void> update_user(const std::string& user_id, const User& user);
    Result<void> delete_user(const std::string& user_id);
    Result<void> update_last_login(const std::string& user_id);
    Result<void> increment_failed_login(const std::string& user_id);
    Result<void> reset_failed_login(const std::string& user_id);
    Result<void> lock_account(const std::string& user_id, int64_t until_timestamp);
    
    // === Group Management ===
    Result<std::string> create_group(const std::string& group_name,
                                      const std::string& description,
                                      const std::string& owner_user_id);
    Result<Group> get_group(const std::string& group_id);
    Result<std::vector<Group>> list_groups(int limit = 100, int offset = 0);
    Result<void> update_group(const std::string& group_id, const Group& group);
    Result<void> delete_group(const std::string& group_id);
    
    // Group membership
    Result<void> add_user_to_group(const std::string& group_id, const std::string& user_id);
    Result<void> remove_user_from_group(const std::string& group_id, const std::string& user_id);
    Result<std::vector<std::string>> get_group_members(const std::string& group_id);
    Result<std::vector<std::string>> get_user_groups(const std::string& user_id);
    
    // === Role Management ===
    Result<std::vector<Role>> list_roles();
    Result<Role> get_role(const std::string& role_id);
    Result<std::string> get_role_id_by_name(const std::string& role_name);
    Result<void> create_role(const std::string& role_id, const std::string& role_name, const std::string& description = "");
    
    // User role assignment
    Result<void> assign_role_to_user(const std::string& user_id, const std::string& role_id);
    Result<void> revoke_role_from_user(const std::string& user_id, const std::string& role_id);
    Result<std::vector<std::string>> get_user_roles(const std::string& user_id);
    
    // Group role assignment
    Result<void> assign_role_to_group(const std::string& group_id, const std::string& role_id);
    Result<void> revoke_role_from_group(const std::string& group_id, const std::string& role_id);
    Result<std::vector<std::string>> get_group_roles(const std::string& group_id);
    
    // === Permission Management ===
    Result<std::vector<Permission>> list_permissions();
    Result<Permission> get_permission(const std::string& permission_id);
    Result<std::vector<std::string>> get_user_permissions(const std::string& user_id);
    Result<std::vector<std::string>> get_role_permissions(const std::string& role_id);
    
    // Database-level permissions
    Result<void> grant_database_permission(const std::string& database_id,
                                           const std::string& principal_type, // 'user' or 'group'
                                           const std::string& principal_id,
                                           const std::string& permission_id,
                                           const std::string& granted_by);
    Result<void> revoke_database_permission(const std::string& database_id,
                                            const std::string& principal_type,
                                            const std::string& principal_id,
                                            const std::string& permission_id);
    Result<std::vector<std::string>> get_database_permissions(const std::string& database_id,
                                                              const std::string& user_id);
    Result<bool> check_database_permission(const std::string& database_id,
                                           const std::string& user_id,
                                           const std::string& permission_name);
    
    // === API Key Management ===
    Result<std::string> create_api_key(const std::string& user_id,
                                        const std::string& key_hash,
                                        const std::string& key_name,
                                        const std::string& key_prefix,
                                        const std::vector<std::string>& scopes,
                                        int64_t expires_at = 0);
    Result<APIKey> get_api_key_by_id(const std::string& api_key_id);
    Result<APIKey> get_api_key_by_prefix(const std::string& key_prefix);
    Result<std::vector<APIKey>> list_user_api_keys(const std::string& user_id);
    Result<void> revoke_api_key(const std::string& api_key_id);
    Result<void> update_api_key_usage(const std::string& api_key_id);
    
    // === Authentication Token Management ===
    Result<std::string> create_auth_token(const std::string& user_id,
                                           const std::string& token_hash,
                                           const std::string& ip_address,
                                           const std::string& user_agent,
                                           int64_t expires_at);
    Result<AuthToken> get_auth_token(const std::string& token_id);
    Result<void> invalidate_auth_token(const std::string& token_id);
    Result<void> invalidate_user_tokens(const std::string& user_id);
    Result<void> update_token_last_used(const std::string& token_id);
    Result<void> cleanup_expired_tokens();
    
    // === Session Management ===
    Result<std::string> create_session(const std::string& user_id,
                                        const std::string& token_id,
                                        const std::string& ip_address,
                                        int64_t expires_at);
    Result<Session> get_session(const std::string& session_id);
    Result<void> update_session_activity(const std::string& session_id);
    Result<void> end_session(const std::string& session_id);
    Result<void> cleanup_expired_sessions();
    
    // === Database Metadata ===
    Result<std::string> store_database_metadata(const std::string& name,
                                                 const std::string& description,
                                                 const std::string& owner_user_id,
                                                 int vector_dimension,
                                                 const std::string& index_type,
                                                 const std::string& metadata_json);
    Result<DatabaseMetadata> get_database_metadata(const std::string& database_id);
    Result<std::vector<DatabaseMetadata>> list_database_metadata(int limit = 100, int offset = 0);
    Result<void> update_database_metadata(const std::string& database_id, const DatabaseMetadata& metadata);
    Result<void> delete_database_metadata(const std::string& database_id);
    Result<void> update_database_stats(const std::string& database_id,
                                       int64_t vector_count,
                                       int64_t index_count);
    
    // === Audit Logging ===
    Result<void> log_audit_event(const std::string& user_id,
                                  const std::string& action,
                                  const std::string& resource_type,
                                  const std::string& resource_id,
                                  const std::string& ip_address,
                                  bool success,
                                  const std::string& details = "");
    Result<std::vector<AuditLogEntry>> get_audit_logs(int limit = 100,
                                                       int offset = 0,
                                                       const std::string& user_id = "",
                                                       const std::string& action = "");
    
    // === Utility Methods ===
    Result<bool> user_exists(const std::string& username);
    Result<bool> email_exists(const std::string& email);
    Result<bool> group_exists(const std::string& group_name);
    Result<bool> database_name_exists(const std::string& name);
    
    // Transaction support
    Result<void> begin_transaction();
    Result<void> commit_transaction();
    Result<void> rollback_transaction();
    
    // Health & Resilience
    Result<void> reconnect();
    bool is_healthy() const;
    bool is_connected() const;
    std::string get_health_status() const;
    
    // Circuit breaker access (for monitoring)
    const utils::CircuitBreaker& get_circuit_breaker() const { return *circuit_breaker_; }

private:
    sqlite3* db_;
    std::string data_directory_;
    std::string db_file_path_;
    std::mutex db_mutex_;
    std::shared_ptr<logging::Logger> logger_;
    std::unique_ptr<utils::CircuitBreaker> circuit_breaker_;
    std::atomic<bool> is_connected_;
    std::atomic<int> connection_retry_count_;
    static constexpr int MAX_RETRY_ATTEMPTS = 3;
    static constexpr int RETRY_DELAY_MS = 100;
    
    // Helper methods
    Result<void> create_tables();
    Result<void> create_indexes();
    Result<void> insert_default_roles_and_permissions();
    std::string generate_id() const;
    int64_t current_timestamp() const;
    
    // SQL execution helpers with retry logic
    Result<void> execute_sql(const std::string& sql);
    Result<void> execute_sql_with_retry(const std::string& sql);
    Result<sqlite3_stmt*> prepare_statement(const std::string& sql);
    Result<sqlite3_stmt*> prepare_statement_with_retry(const std::string& sql);
    void finalize_statement(sqlite3_stmt* stmt);
    void exponential_backoff(int attempt) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SQLITE_PERSISTENCE_LAYER_H
