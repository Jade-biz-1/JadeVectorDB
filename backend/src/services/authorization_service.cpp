#include "authorization_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>

namespace jadevectordb {

AuthorizationService::AuthorizationService() {
    logger_ = logging::LoggerManager::get_logger("AuthorizationService");
    // Initialize permission cache: 100k entries, 5 minute TTL
    permission_cache_ = std::make_unique<cache::PermissionCache>(100000, 300);
}

bool AuthorizationService::initialize(const AuthorizationConfig& config,
                                     std::shared_ptr<SecurityAuditLogger> audit_logger) {
    try {
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid authorization configuration");
            return false;
        }

        config_ = config;
        audit_logger_ = audit_logger;

        // Initialize default system roles
        initialize_default_roles();

        LOG_INFO(logger_, "AuthorizationService initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize: " + std::string(e.what()));
        return false;
    }
}

void AuthorizationService::initialize_default_roles() {
    std::lock_guard<std::mutex> lock(roles_mutex_);

    // Admin role - full access
    Role admin_role;
    admin_role.role_id = "admin";
    admin_role.role_name = "Administrator";
    admin_role.description = "Full system access";
    admin_role.is_system_role = true;
    admin_role.permissions = {
        Permission("*", "*", "global"),  // Wildcard permission
        Permission("system", "admin", "global"),
        Permission("cluster", "manage", "global")
    };
    roles_[admin_role.role_id] = admin_role;

    // User role - standard access
    Role user_role;
    user_role.role_id = "user";
    user_role.role_name = "User";
    user_role.description = "Standard user access";
    user_role.is_system_role = true;
    user_role.permissions = {
        Permission("database", "create", "global"),
        Permission("database", "read", "global"),
        Permission("database", "update", "global"),
        Permission("vector", "create", "global"),
        Permission("vector", "read", "global"),
        Permission("vector", "update", "global"),
        Permission("vector", "search", "global")
    };
    roles_[user_role.role_id] = user_role;

    // Read-only role
    Role readonly_role;
    readonly_role.role_id = "readonly";
    readonly_role.role_name = "Read Only";
    readonly_role.description = "Read-only access";
    readonly_role.is_system_role = true;
    readonly_role.permissions = {
        Permission("database", "read", "global"),
        Permission("vector", "read", "global"),
        Permission("vector", "search", "global")
    };
    roles_[readonly_role.role_id] = readonly_role;

    LOG_INFO(logger_, "Default system roles initialized");
}

Result<bool> AuthorizationService::authorize(const std::string& user_id,
                                            const std::string& resource_type,
                                            const std::string& resource_id,
                                            const std::string& action,
                                            const std::unordered_map<std::string, std::string>& context) {
    try {
        if (!config_.enabled) {
            return true;  // Authorization disabled
        }

        // Check cache first
        std::string cache_key = resource_type + ":" + resource_id;
        auto cached = permission_cache_->get(user_id, cache_key, action);
        if (cached.has_value()) {
            // Cache hit - return cached result
            if (!cached.value()) {
                RETURN_ERROR(ErrorCode::PERMISSION_DENIED,
                            "Access denied (cached): " + user_id + " cannot " + action +
                            " " + resource_type + ":" + resource_id);
            }
            return true;
        }

        // Cache miss - perform full authorization check
        bool granted = false;
        std::string reason;

        // Check RBAC if enabled
        if (config_.enable_rbac) {
            granted = check_rbac_authorization(user_id, resource_type, resource_id, action);
            if (granted) {
                reason = "RBAC authorization";
            }
        }

        // Check ABAC if enabled and RBAC didn't grant
        if (!granted && config_.enable_abac) {
            granted = check_abac_authorization(user_id, resource_type, resource_id, action, context);
            if (granted) {
                reason = "ABAC authorization";
            }
        }

        // Check ACL if enabled and still not granted
        if (!granted && config_.enable_acl) {
            granted = check_acl_authorization(user_id, resource_id, action);
            if (granted) {
                reason = "ACL authorization";
            }
        }

        // Apply deny_by_default policy
        if (!granted && config_.deny_by_default) {
            granted = false;
            reason = "Deny by default";
        }

        // Log decision
        log_authz_decision(user_id, resource_type, resource_id, action, granted, reason);

        // Cache the result
        std::string cache_key = resource_type + ":" + resource_id;
        permission_cache_->put(user_id, cache_key, action, granted);

        if (!granted) {
            RETURN_ERROR(ErrorCode::PERMISSION_DENIED,
                        "Access denied: " + user_id + " cannot " + action +
                        " " + resource_type + ":" + resource_id);
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in authorize: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Authorization check failed: " + std::string(e.what()));
    }
}

bool AuthorizationService::check_rbac_authorization(const std::string& user_id,
                                                    const std::string& resource_type,
                                                    const std::string& resource_id,
                                                    const std::string& action) {
    std::lock_guard<std::mutex> user_roles_lock(user_roles_mutex_);
    std::lock_guard<std::mutex> roles_lock(roles_mutex_);

    // Get user's roles
    auto user_roles_it = user_roles_.find(user_id);
    if (user_roles_it == user_roles_.end()) {
        return false;  // User has no roles
    }

    // Check each role's permissions
    for (const auto& role_id : user_roles_it->second) {
        auto role_it = roles_.find(role_id);
        if (role_it == roles_.end()) {
            continue;
        }

        const Role& role = role_it->second;

        // Check if any permission matches
        for (const auto& permission : role.permissions) {
            // Check for wildcard permission
            if (permission.resource_type == "*" && permission.action == "*") {
                return true;  // Admin wildcard
            }

            // Check if resource type and action match
            if ((permission.resource_type == resource_type || permission.resource_type == "*") &&
                (permission.action == action || permission.action == "*")) {

                // Check if scope matches
                if (matches_scope(permission, resource_type, resource_id)) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool AuthorizationService::check_abac_authorization(const std::string& user_id,
                                                    const std::string& resource_type,
                                                    const std::string& resource_id,
                                                    const std::string& action,
                                                    const std::unordered_map<std::string, std::string>& context) {
    std::lock_guard<std::mutex> policies_lock(policies_mutex_);
    std::lock_guard<std::mutex> user_roles_lock(user_roles_mutex_);

    // Get user's roles
    auto user_roles_it = user_roles_.find(user_id);
    if (user_roles_it == user_roles_.end()) {
        return false;
    }

    // Check applicable policies
    for (const auto& [policy_id, policy] : policies_) {
        if (!policy.is_active) {
            continue;
        }

        // Check if policy applies to any of user's roles
        bool applies_to_user = false;
        for (const auto& role_id : user_roles_it->second) {
            if (std::find(policy.applicable_roles.begin(),
                         policy.applicable_roles.end(),
                         role_id) != policy.applicable_roles.end()) {
                applies_to_user = true;
                break;
            }
        }

        if (!applies_to_user) {
            continue;
        }

        // Evaluate policy conditions
        if (evaluate_policy_conditions(policy, context)) {
            return true;
        }
    }

    return false;
}

bool AuthorizationService::check_acl_authorization(const std::string& user_id,
                                                   const std::string& resource_id,
                                                   const std::string& action) {
    std::lock_guard<std::mutex> acls_lock(acls_mutex_);

    auto acl_it = acls_.find(resource_id);
    if (acl_it == acls_.end()) {
        return false;  // No ACL for this resource
    }

    // Check ACL entries
    for (const auto& entry : acl_it->second) {
        // Check if entry applies to this user
        if (entry.principal_type == "user" && entry.principal_id == user_id) {
            // Check if action is in permissions
            auto perm_it = std::find(entry.permissions.begin(),
                                    entry.permissions.end(),
                                    action);
            if (perm_it != entry.permissions.end()) {
                return entry.allow;  // Return allow/deny
            }
        }

        // Check if entry applies via user's roles
        if (entry.principal_type == "role") {
            std::lock_guard<std::mutex> user_roles_lock(user_roles_mutex_);
            auto user_roles_it = user_roles_.find(user_id);
            if (user_roles_it != user_roles_.end()) {
                auto role_it = std::find(user_roles_it->second.begin(),
                                        user_roles_it->second.end(),
                                        entry.principal_id);
                if (role_it != user_roles_it->second.end()) {
                    auto perm_it = std::find(entry.permissions.begin(),
                                            entry.permissions.end(),
                                            action);
                    if (perm_it != entry.permissions.end()) {
                        return entry.allow;
                    }
                }
            }
        }
    }

    return false;
}

bool AuthorizationService::matches_scope(const Permission& permission,
                                        const std::string& resource_type,
                                        const std::string& resource_id) const {
    if (permission.scope == "global") {
        return true;
    }

    if (permission.scope == resource_type + ":*") {
        return true;
    }

    if (permission.scope == resource_type + ":" + resource_id) {
        return true;
    }

    return false;
}

bool AuthorizationService::evaluate_policy_conditions(
    const AccessPolicy& policy,
    const std::unordered_map<std::string, std::string>& context) {

    // All conditions must match
    for (const auto& [attr, value] : policy.conditions) {
        auto context_it = context.find(attr);
        if (context_it == context.end() || context_it->second != value) {
            return false;
        }
    }

    return true;
}

Result<bool> AuthorizationService::create_role(const Role& role) {
    try {
        std::lock_guard<std::mutex> lock(roles_mutex_);

        if (roles_.find(role.role_id) != roles_.end()) {
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Role already exists");
        }

        roles_[role.role_id] = role;
        LOG_INFO(logger_, "Role created: " + role.role_id);

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_role: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Role creation failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::assign_role_to_user(const std::string& user_id,
                                                       const std::string& role_id) {
    try {
        // Check if role exists
        {
            std::lock_guard<std::mutex> roles_lock(roles_mutex_);
            if (roles_.find(role_id) == roles_.end()) {
                RETURN_ERROR(ErrorCode::NOT_FOUND, "Role not found");
            }
        }

        std::lock_guard<std::mutex> lock(user_roles_mutex_);

        // Add role to user
        auto& user_roles = user_roles_[user_id];
        if (std::find(user_roles.begin(), user_roles.end(), role_id) == user_roles.end()) {
            user_roles.push_back(role_id);
        }

        LOG_INFO(logger_, "Role " + role_id + " assigned to user " + user_id);
        
        // Invalidate user's permission cache
        invalidate_user_cache(user_id);
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in assign_role_to_user: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Role assignment failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::remove_role_from_user(const std::string& user_id,
                                                         const std::string& role_id) {
    try {
        std::lock_guard<std::mutex> lock(user_roles_mutex_);

        auto it = user_roles_.find(user_id);
        if (it == user_roles_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found");
        }

        auto& roles = it->second;
        auto role_it = std::find(roles.begin(), roles.end(), role_id);
        if (role_it == roles.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "User does not have this role");
        }

        roles.erase(role_it);
        
        // Invalidate user's permission cache
        invalidate_user_cache(user_id);
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_role_from_user: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Role removal failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::add_permission_to_role(const std::string& role_id,
                                                          const Permission& permission) {
    try {
        std::lock_guard<std::mutex> lock(roles_mutex_);

        auto it = roles_.find(role_id);
        if (it == roles_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Role not found");
        }

        it->second.permissions.push_back(permission);
        
        // Invalidate cache for all users with this role
        std::lock_guard<std::mutex> user_roles_lock(user_roles_mutex_);
        for (const auto& [user_id, user_roles] : user_roles_) {
            if (std::find(user_roles.begin(), user_roles.end(), role_id) != user_roles.end()) {
                invalidate_user_cache(user_id);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_permission_to_role: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Add permission to role failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::remove_permission_from_role(const std::string& role_id,
                                                               const Permission& permission) {
    try {
        std::lock_guard<std::mutex> lock(roles_mutex_);

        auto it = roles_.find(role_id);
        if (it == roles_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Role not found");
        }

        auto& permissions = it->second.permissions;
        auto perm_it = std::find_if(permissions.begin(), permissions.end(),
            [&permission](const Permission& p) {
                return p.resource == permission.resource && 
                       p.action == permission.action;
            });
        
        if (perm_it == permissions.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Permission not found in role");
        }

        permissions.erase(perm_it);
        
        // Invalidate cache for all users with this role
        std::lock_guard<std::mutex> user_roles_lock(user_roles_mutex_);
        for (const auto& [user_id, user_roles] : user_roles_) {
            if (std::find(user_roles.begin(), user_roles.end(), role_id) != user_roles.end()) {
                invalidate_user_cache(user_id);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_permission_from_role: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Remove permission from role failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::revoke_all_user_roles(const std::string& user_id) {
    try {
        std::lock_guard<std::mutex> lock(user_roles_mutex_);

        auto it = user_roles_.find(user_id);
        if (it == user_roles_.end()) {
            return true;  // Already no roles
        }

        it->second.clear();
        
        // Invalidate user's permission cache
        invalidate_user_cache(user_id);
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in revoke_all_user_roles: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Revoke all roles failed: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> AuthorizationService::get_user_roles(const std::string& user_id) const {
    try {
        std::lock_guard<std::mutex> lock(user_roles_mutex_);

        auto it = user_roles_.find(user_id);
        if (it != user_roles_.end()) {
            return it->second;
        }

        return std::vector<std::string>();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_user_roles: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Get user roles failed: " + std::string(e.what()));
    }
}

Result<std::vector<Permission>> AuthorizationService::get_user_permissions(const std::string& user_id) const {
    try {
        std::vector<Permission> all_permissions;

        std::lock_guard<std::mutex> user_roles_lock(user_roles_mutex_);
        std::lock_guard<std::mutex> roles_lock(roles_mutex_);

        auto user_roles_it = user_roles_.find(user_id);
        if (user_roles_it == user_roles_.end()) {
            return all_permissions;  // No roles, no permissions
        }

        // Collect permissions from all roles
        for (const auto& role_id : user_roles_it->second) {
            auto role_it = roles_.find(role_id);
            if (role_it != roles_.end()) {
                all_permissions.insert(all_permissions.end(),
                                      role_it->second.permissions.begin(),
                                      role_it->second.permissions.end());
            }
        }

        return all_permissions;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_user_permissions: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Get user permissions failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::add_acl_entry(const ACLEntry& entry) {
    try {
        std::lock_guard<std::mutex> lock(acls_mutex_);

        acls_[entry.resource_id].push_back(entry);
        LOG_INFO(logger_, "ACL entry added for resource: " + entry.resource_id);

        // Invalidate cache for the affected principal (user)
        invalidate_user_cache(entry.principal_id);

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_acl_entry: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Add ACL entry failed: " + std::string(e.what()));
    }
}

Result<bool> AuthorizationService::remove_acl_entry(const std::string& resource_id,
                                                    const std::string& principal_id) {
    try {
        std::lock_guard<std::mutex> lock(acls_mutex_);

        auto it = acls_.find(resource_id);
        if (it == acls_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Resource ACL not found");
        }

        auto& entries = it->second;
        auto entry_it = std::find_if(entries.begin(), entries.end(),
            [&principal_id](const ACLEntry& e) {
                return e.principal_id == principal_id;
            });
        
        if (entry_it == entries.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "ACL entry not found for principal");
        }

        entries.erase(entry_it);
        LOG_INFO(logger_, "ACL entry removed for resource: " + resource_id + 
                 ", principal: " + principal_id);

        // Invalidate cache for the affected principal (user)
        invalidate_user_cache(principal_id);

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_acl_entry: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR,
                    "Remove ACL entry failed: " + std::string(e.what()));
    }
}

void AuthorizationService::log_authz_decision(const std::string& user_id,
                                              const std::string& resource_type,
                                              const std::string& resource_id,
                                              const std::string& action,
                                              bool granted,
                                              const std::string& reason) {
    if (config_.log_authorization_decisions && audit_logger_) {
        SecurityEventType event_type = granted ?
            SecurityEventType::AUTHORIZATION_GRANTED :
            SecurityEventType::AUTHORIZATION_DENIED;

        SecurityEvent event(event_type, user_id, "", resource_id, action, granted);
        event.details = "Resource: " + resource_type + ", Reason: " + reason;
        audit_logger_->log_security_event(event);
    }

    if (granted) {
        LOG_DEBUG(logger_, "Authorization granted: " + user_id + " -> " +
                 action + " on " + resource_type + ":" + resource_id);
    } else {
        LOG_WARN(logger_, "Authorization denied: " + user_id + " -> " +
                action + " on " + resource_type + ":" + resource_id +
                " (" + reason + ")");
    }
}

bool AuthorizationService::validate_config(const AuthorizationConfig& config) const {
    // Basic validation
    if (!config.enable_rbac && !config.enable_abac && !config.enable_acl) {
        LOG_ERROR(logger_, "At least one authorization method must be enabled");
        return false;
    }

    return true;
}

bool AuthorizationService::is_system_role(const std::string& role_id) const {
    auto it = roles_.find(role_id);
    return it != roles_.end() && it->second.is_system_role;
}

// Cache management methods
cache::PermissionCache::CacheStats AuthorizationService::get_cache_stats() const {
    return permission_cache_->get_stats();
}

void AuthorizationService::clear_cache() {
    permission_cache_->clear();
    LOG_INFO(logger_, "Permission cache cleared");
}

void AuthorizationService::invalidate_user_cache(const std::string& user_id) {
    permission_cache_->invalidate_user(user_id);
    LOG_DEBUG(logger_, "Invalidated cache for user: " + user_id);
}

} // namespace jadevectordb
