#ifndef JADEVECTORDB_AUTHORIZATION_SERVICE_H
#define JADEVECTORDB_AUTHORIZATION_SERVICE_H

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "security_audit_logger.h"
#include "cache/permission_cache.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>

namespace jadevectordb {

// Permission definition
struct Permission {
    std::string permission_id;
    std::string resource_type;  // "database", "vector", "system", "cluster", etc.
    std::string action;         // "create", "read", "update", "delete", "search", "admin", etc.
    std::string scope;          // "global", "database:*", "database:specific_id", etc.
    std::string description;

    Permission() = default;
    Permission(const std::string& resource, const std::string& act, const std::string& scp = "global")
        : resource_type(resource), action(act), scope(scp),
          permission_id(resource + ":" + act + ":" + scp) {}
};

// Role definition
struct Role {
    std::string role_id;
    std::string role_name;
    std::string description;
    std::vector<Permission> permissions;
    bool is_system_role;  // System roles cannot be modified/deleted

    Role() : is_system_role(false) {}
};

// Policy definition for attribute-based access control (ABAC)
struct AccessPolicy {
    std::string policy_id;
    std::string policy_name;
    std::string description;
    std::vector<std::string> applicable_roles;
    std::unordered_map<std::string, std::string> conditions;  // attribute -> value
    bool is_active;

    AccessPolicy() : is_active(true) {}
};

// Access control list entry
struct ACLEntry {
    std::string resource_id;
    std::string resource_type;
    std::string principal_id;  // user_id or role_id
    std::string principal_type;  // "user" or "role"
    std::vector<std::string> permissions;
    bool allow;  // true for allow, false for deny

    ACLEntry() : allow(true) {}
};

// Authorization configuration
struct AuthorizationConfig {
    bool enabled = true;
    bool enable_rbac = true;  // Role-Based Access Control
    bool enable_abac = false;  // Attribute-Based Access Control
    bool enable_acl = true;    // Access Control Lists
    bool deny_by_default = true;  // If no explicit permission, deny
    bool log_authorization_decisions = true;
    std::string default_role = "user";  // Default role for new users
};

/**
 * @brief Authorization Service for Role-Based and Attribute-Based Access Control
 *
 * This service handles authorization decisions, role management, permission
 * management, and integrates with the security audit logger.
 */
class AuthorizationService {
private:
    std::shared_ptr<logging::Logger> logger_;
    AuthorizationConfig config_;
    std::shared_ptr<SecurityAuditLogger> audit_logger_;

    // Roles storage
    std::unordered_map<std::string, Role> roles_;

    // User-to-Role mappings
    std::unordered_map<std::string, std::vector<std::string>> user_roles_;  // user_id -> role_ids

    // Access policies
    std::unordered_map<std::string, AccessPolicy> policies_;

    // Access Control Lists
    std::unordered_map<std::string, std::vector<ACLEntry>> acls_;  // resource_id -> ACL entries

    // Permission cache for performance
    std::unique_ptr<cache::PermissionCache> permission_cache_;

    mutable std::mutex roles_mutex_;
    mutable std::mutex user_roles_mutex_;
    mutable std::mutex policies_mutex_;
    mutable std::mutex acls_mutex_;

public:
    explicit AuthorizationService();
    ~AuthorizationService() = default;

    // Initialize authorization service
    bool initialize(const AuthorizationConfig& config,
                   std::shared_ptr<SecurityAuditLogger> audit_logger = nullptr);

    // Authorization decision
    Result<bool> authorize(const std::string& user_id,
                          const std::string& resource_type,
                          const std::string& resource_id,
                          const std::string& action,
                          const std::unordered_map<std::string, std::string>& context = {});

    // Check if user has specific permission
    Result<bool> has_permission(const std::string& user_id,
                               const Permission& permission);

    // Role management
    Result<bool> create_role(const Role& role);
    Result<bool> update_role(const Role& role);
    Result<bool> delete_role(const std::string& role_id);
    Result<Role> get_role(const std::string& role_id) const;
    Result<std::vector<Role>> list_roles() const;

    // User-Role assignment
    Result<bool> assign_role_to_user(const std::string& user_id,
                                    const std::string& role_id);
    Result<bool> remove_role_from_user(const std::string& user_id,
                                       const std::string& role_id);
    Result<std::vector<std::string>> get_user_roles(const std::string& user_id) const;

    // Permission management
    Result<bool> add_permission_to_role(const std::string& role_id,
                                        const Permission& permission);
    Result<bool> remove_permission_from_role(const std::string& role_id,
                                             const Permission& permission);
    Result<std::vector<Permission>> get_role_permissions(const std::string& role_id) const;
    Result<std::vector<Permission>> get_user_permissions(const std::string& user_id) const;

    // Policy management (ABAC)
    Result<bool> create_policy(const AccessPolicy& policy);
    Result<bool> update_policy(const AccessPolicy& policy);
    Result<bool> delete_policy(const std::string& policy_id);
    Result<AccessPolicy> get_policy(const std::string& policy_id) const;

    // ACL management
    Result<bool> add_acl_entry(const ACLEntry& entry);
    Result<bool> remove_acl_entry(const std::string& resource_id,
                                  const std::string& principal_id);
    Result<std::vector<ACLEntry>> get_resource_acl(const std::string& resource_id) const;

    // Bulk operations
    Result<bool> assign_multiple_roles(const std::string& user_id,
                                       const std::vector<std::string>& role_ids);
    Result<bool> revoke_all_user_roles(const std::string& user_id);

    // Authorization statistics
    Result<std::unordered_map<std::string, std::string>> get_authz_stats() const;

    // Update configuration
    Result<bool> update_config(const AuthorizationConfig& new_config);
    AuthorizationConfig get_config() const;

    // Initialize default system roles
    void initialize_default_roles();

private:
    // Check RBAC authorization
    bool check_rbac_authorization(const std::string& user_id,
                                 const std::string& resource_type,
                                 const std::string& resource_id,
                                 const std::string& action);

    // Check ABAC authorization
    bool check_abac_authorization(const std::string& user_id,
                                 const std::string& resource_type,
                                 const std::string& resource_id,
                                 const std::string& action,
                                 const std::unordered_map<std::string, std::string>& context);

    // Check ACL authorization
    bool check_acl_authorization(const std::string& user_id,
                                const std::string& resource_id,
                                const std::string& action);

    // Check if permission matches scope
    bool matches_scope(const Permission& permission,
                      const std::string& resource_type,
                      const std::string& resource_id) const;

    // Evaluate policy conditions
    bool evaluate_policy_conditions(const AccessPolicy& policy,
                                   const std::unordered_map<std::string, std::string>& context);

    // Log authorization decision
    void log_authz_decision(const std::string& user_id,
                           const std::string& resource_type,
                           const std::string& resource_id,
                           const std::string& action,
                           bool granted,
                           const std::string& reason = "");

    // Validate configuration
    bool validate_config(const AuthorizationConfig& config) const;

    // Check if role is system role
    bool is_system_role(const std::string& role_id) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_AUTHORIZATION_SERVICE_H
