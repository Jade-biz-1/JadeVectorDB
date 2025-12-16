#ifndef JADEVECTORDB_AUTH_MODELS_H
#define JADEVECTORDB_AUTH_MODELS_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace jadevectordb {

// User model
struct User {
    std::string user_id;
    std::string username;
    std::string email;
    std::string password_hash;
    std::string salt;
    bool is_active = true;
    bool is_system_admin = false;
    int64_t created_at = 0;
    int64_t updated_at = 0;
    int64_t last_login = 0;
    int failed_login_attempts = 0;
    int64_t account_locked_until = 0;
    std::string metadata; // JSON string for extensibility
    
    bool validate() const {
        return !user_id.empty() && !username.empty() && !email.empty();
    }
    
    bool is_locked() const {
        return account_locked_until > 0 && account_locked_until > std::time(nullptr);
    }
};

// Group model
struct Group {
    std::string group_id;
    std::string group_name;
    std::string description;
    std::string owner_user_id;
    int64_t created_at = 0;
    int64_t updated_at = 0;
    
    bool validate() const {
        return !group_id.empty() && !group_name.empty() && !owner_user_id.empty();
    }
};

// Role model
struct Role {
    std::string role_id;
    std::string role_name;
    std::string description;
    bool is_system_role = false;
    
    bool validate() const {
        return !role_id.empty() && !role_name.empty();
    }
};

// Permission model
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

// API Key model
struct APIKey {
    std::string api_key_id;
    std::string user_id;
    std::string key_hash;
    std::string key_name;
    std::string key_prefix;  // First 8 chars for display
    std::vector<std::string> scopes;
    bool is_active = true;
    int64_t expires_at = 0;  // 0 = no expiration
    int64_t created_at = 0;
    int64_t last_used_at = 0;
    int usage_count = 0;
    
    bool validate() const {
        return !api_key_id.empty() && !user_id.empty() && !key_hash.empty();
    }
    
    bool is_expired() const {
        return expires_at > 0 && expires_at < std::time(nullptr);
    }
};

// Authentication Token model
struct AuthToken {
    std::string token_id;
    std::string user_id;
    std::string token_hash;
    std::string ip_address;
    std::string user_agent;
    int64_t issued_at = 0;
    int64_t expires_at = 0;
    int64_t last_used_at = 0;
    bool is_valid = true;
    
    bool validate() const {
        return !token_id.empty() && !user_id.empty() && !token_hash.empty();
    }
    
    bool is_expired() const {
        return expires_at > 0 && expires_at < std::time(nullptr);
    }
};

// Session model
struct Session {
    std::string session_id;
    std::string user_id;
    std::string token_id;
    std::string ip_address;
    int64_t created_at = 0;
    int64_t last_activity = 0;
    int64_t expires_at = 0;
    bool is_active = true;
    std::string metadata;  // JSON for additional session data
    
    bool validate() const {
        return !session_id.empty() && !user_id.empty();
    }
    
    bool is_expired() const {
        return expires_at > 0 && expires_at < std::time(nullptr);
    }
};

// Database Metadata model
struct DatabaseMetadata {
    std::string database_id;
    std::string name;
    std::string description;
    std::string owner_user_id;
    int vector_dimension = 0;
    std::string index_type;
    int64_t vector_count = 0;
    int64_t index_count = 0;
    int64_t created_at = 0;
    int64_t updated_at = 0;
    std::string metadata;  // JSON string for additional metadata
    
    bool validate() const {
        return !database_id.empty() && !name.empty() && !owner_user_id.empty() &&
               vector_dimension > 0;
    }
};

// Audit Log Entry model
struct AuditLogEntry {
    int64_t id = 0;
    std::string user_id;
    std::string action;  // 'create', 'read', 'update', 'delete', 'login', 'logout', etc.
    std::string resource_type;  // 'user', 'group', 'database', 'vector', 'api_key', etc.
    std::string resource_id;
    std::string ip_address;
    bool success = false;
    std::string details;  // JSON or text for additional information
    int64_t timestamp = 0;
    
    bool validate() const {
        return !user_id.empty() && !action.empty() && !resource_type.empty();
    }
};

// Database Permission Entry
struct DatabasePermission {
    int64_t id = 0;
    std::string database_id;
    std::string principal_type;  // 'user' or 'group'
    std::string principal_id;
    std::string permission_id;
    int64_t granted_at = 0;
    std::string granted_by;  // user_id
    
    bool validate() const {
        return !database_id.empty() && !principal_type.empty() &&
               !principal_id.empty() && !permission_id.empty();
    }
};

} // namespace jadevectordb

#endif // JADEVECTORDB_AUTH_MODELS_H
