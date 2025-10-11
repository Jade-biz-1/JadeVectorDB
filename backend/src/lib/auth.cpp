#include "auth.h"
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>
#include <set>

// Include header for string hashing (we'll use a simple approach)
#include <functional>

namespace jadevectordb {

const std::string AuthManager::ADMIN_ROLE = "admin";
const std::string AuthManager::USER_ROLE = "user";
const std::string AuthManager::READER_ROLE = "reader";

AuthManager::AuthManager() {
    initialize_default_roles();
}

void AuthManager::initialize_default_roles() {
    // Create default admin role with all permissions
    Role admin_role("role_admin", "Administrator");
    admin_role.permissions = {
        "database:create", "database:read", "database:update", "database:delete",
        "vector:add", "vector:read", "vector:update", "vector:delete",
        "index:create", "index:read", "index:update", "index:delete",
        "search:execute", "user:manage", "api_key:manage", "config:manage"
    };
    admin_role.created_at = std::chrono::system_clock::now();
    admin_role.updated_at = std::chrono::system_clock::now();
    
    // Create default user role with standard permissions
    Role user_role("role_user", "Standard User");
    user_role.permissions = {
        "database:read", "vector:add", "vector:read", "vector:update",
        "search:execute", "index:read"
    };
    user_role.created_at = std::chrono::system_clock::now();
    user_role.updated_at = std::chrono::system_clock::now();
    
    // Create default reader role with read-only permissions
    Role reader_role("role_reader", "Reader");
    reader_role.permissions = {
        "database:read", "vector:read", "search:execute"
    };
    reader_role.created_at = std::chrono::system_clock::now();
    reader_role.updated_at = std::chrono::system_clock::now();
    
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    roles_[admin_role.role_id] = admin_role;
    roles_[user_role.role_id] = user_role;
    roles_[reader_role.role_id] = reader_role;
}

Result<std::string> AuthManager::create_user(const std::string& username,
                                           const std::string& email,
                                           const std::vector<std::string>& initial_roles) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    // Generate user ID
    std::string user_id = generate_id();
    
    // Check if user with this email already exists
    for (const auto& pair : users_) {
        if (pair.second.email == email) {
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "User with this email already exists");
        }
    }
    
    // Validate roles exist
    for (const auto& role : initial_roles) {
        if (roles_.find(role) == roles_.end()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid role: " + role);
        }
    }
    
    User user;
    user.user_id = user_id;
    user.username = username;
    user.email = email;
    user.roles = initial_roles;
    user.created_at = std::chrono::system_clock::now();
    user.is_active = true;
    
    users_[user_id] = user;
    
    return user_id;
}

Result<User> AuthManager::get_user(const std::string& user_id) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    return it->second;
}

Result<void> AuthManager::update_user(const std::string& user_id,
                                    const std::vector<std::string>& new_roles) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    // Validate roles exist
    for (const auto& role : new_roles) {
        if (roles_.find(role) == roles_.end()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid role: " + role);
        }
    }
    
    it->second.roles = new_roles;
    return {};
}

Result<void> AuthManager::deactivate_user(const std::string& user_id) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    it->second.is_active = false;
    return {};
}

Result<void> AuthManager::activate_user(const std::string& user_id) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    it->second.is_active = true;
    return {};
}

Result<std::string> AuthManager::create_role(const std::string& name,
                                           const std::vector<std::string>& permissions) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    std::string role_id = generate_id();
    
    // Check if role with this name already exists
    for (const auto& pair : roles_) {
        if (pair.second.name == name) {
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Role with this name already exists: " + name);
        }
    }
    
    Role role(role_id, name);
    role.permissions = permissions;
    role.created_at = std::chrono::system_clock::now();
    role.updated_at = std::chrono::system_clock::now();
    
    roles_[role_id] = role;
    
    return role_id;
}

Result<Role> AuthManager::get_role(const std::string& role_id) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = roles_.find(role_id);
    if (it == roles_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Role not found: " + role_id);
    }
    
    return it->second;
}

Result<void> AuthManager::update_role(const std::string& role_id,
                                    const std::vector<std::string>& new_permissions) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = roles_.find(role_id);
    if (it == roles_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Role not found: " + role_id);
    }
    
    it->second.permissions = new_permissions;
    it->second.updated_at = std::chrono::system_clock::now();
    
    return {};
}

Result<void> AuthManager::delete_role(const std::string& role_id) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = roles_.find(role_id);
    if (it == roles_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Role not found: " + role_id);
    }
    
    // Check if any users have this role
    for (auto& user_pair : users_) {
        auto& user_roles = user_pair.second.roles;
        user_roles.erase(
            std::remove(user_roles.begin(), user_roles.end(), role_id),
            user_roles.end()
        );
    }
    
    roles_.erase(it);
    return {};
}

Result<std::string> AuthManager::generate_api_key(const std::string& user_id,
                                                const std::vector<std::string>& permissions,
                                                const std::string& description,
                                                std::chrono::hours validity_duration) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto user_it = users_.find(user_id);
    if (user_it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    if (!user_it->second.is_active) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "User account is deactivated");
    }
    
    // Generate API key
    std::string raw_api_key = generate_random_api_key();
    std::string key_hash = hash_api_key(raw_api_key);
    
    // Generate key ID
    std::string key_id = generate_id();
    
    // Determine permissions based on user roles if not explicitly provided
    std::vector<std::string> final_permissions = permissions;
    if (final_permissions.empty()) {
        // Aggregate permissions from user's roles
        std::set<std::string> perm_set;
        for (const auto& role_id : user_it->second.roles) {
            auto role_it = roles_.find(role_id);
            if (role_it != roles_.end()) {
                for (const auto& perm : role_it->second.permissions) {
                    perm_set.insert(perm);
                }
            }
        }
        final_permissions.assign(perm_set.begin(), perm_set.end());
    } else {
        // Validate provided permissions are valid for the user's roles
        std::set<std::string> user_permissions;
        for (const auto& role_id : user_it->second.roles) {
            auto role_it = roles_.find(role_id);
            if (role_it != roles_.end()) {
                for (const auto& perm : role_it->second.permissions) {
                    user_permissions.insert(perm);
                }
            }
        }
        
        for (const auto& perm : permissions) {
            if (user_permissions.find(perm) == user_permissions.end()) {
                RETURN_ERROR(ErrorCode::PERMISSION_DENIED, 
                           "Permission not allowed for user's roles: " + perm);
            }
        }
    }
    
    ApiKey api_key;
    api_key.key_id = key_id;
    api_key.key_hash = key_hash;
    api_key.user_id = user_id;
    api_key.permissions = final_permissions;
    api_key.created_at = std::chrono::system_clock::now();
    api_key.expires_at = api_key.created_at + validity_duration;
    api_key.description = description;
    api_key.is_active = true;
    
    api_keys_[key_id] = api_key;
    
    // Return the raw API key (only time it's returned in full)
    return raw_api_key;
}

Result<bool> AuthManager::validate_api_key(const std::string& api_key) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    std::string key_hash = hash_api_key(api_key);
    
    // Find the API key with matching hash
    for (const auto& pair : api_keys_) {
        if (pair.second.key_hash == key_hash) {
            const auto& key = pair.second;
            
            // Check if key is active
            if (!key.is_active) {
                return false;
            }
            
            // Check if key has expired
            auto now = std::chrono::system_clock::now();
            if (now > key.expires_at) {
                return false;  // Key has expired
            }
            
            // Check if associated user is active
            auto user_it = users_.find(key.user_id);
            if (user_it == users_.end() || !user_it->second.is_active) {
                return false;  // User account is deactivated
            }
            
            return true;
        }
    }
    
    return false;  // API key not found
}

Result<std::string> AuthManager::get_user_from_api_key(const std::string& api_key) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    std::string key_hash = hash_api_key(api_key);
    
    // Find the API key with matching hash
    for (const auto& pair : api_keys_) {
        if (pair.second.key_hash == key_hash) {
            const auto& key = pair.second;
            
            // Check if key is active and not expired
            if (!key.is_active) {
                RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "API key is inactive");
            }
            
            auto now = std::chrono::system_clock::now();
            if (now > key.expires_at) {
                RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "API key has expired");
            }
            
            return key.user_id;
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Invalid API key");
}

Result<std::vector<std::string>> AuthManager::get_permissions_from_api_key(const std::string& api_key) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    std::string key_hash = hash_api_key(api_key);
    
    // Find the API key with matching hash
    for (const auto& pair : api_keys_) {
        if (pair.second.key_hash == key_hash) {
            const auto& key = pair.second;
            
            // Check if key is active and not expired
            if (!key.is_active) {
                RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "API key is inactive");
            }
            
            auto now = std::chrono::system_clock::now();
            if (now > key.expires_at) {
                RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "API key has expired");
            }
            
            return key.permissions;
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Invalid API key");
}

Result<void> AuthManager::revoke_api_key(const std::string& key_id) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = api_keys_.find(key_id);
    if (it == api_keys_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "API key not found: " + key_id);
    }
    
    it->second.is_active = false;  // Mark as inactive instead of deleting
    return {};
}

Result<bool> AuthManager::has_permission(const std::string& user_id, 
                                       const std::string& permission) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto user_it = users_.find(user_id);
    if (user_it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    if (!user_it->second.is_active) {
        RETURN_ERROR(ErrorCode::PERMISSION_DENIED, "User account is deactivated");
    }
    
    // Collect all permissions from user's roles
    std::set<std::string> user_permissions;
    for (const auto& role_id : user_it->second.roles) {
        auto role_it = roles_.find(role_id);
        if (role_it != roles_.end()) {
            for (const auto& perm : role_it->second.permissions) {
                user_permissions.insert(perm);
            }
        }
    }
    
    return user_permissions.count(permission) > 0;
}

Result<bool> AuthManager::has_permission_with_api_key(const std::string& api_key, 
                                                    const std::string& permission) const {
    auto user_id_result = get_user_from_api_key(api_key);
    if (!user_id_result.has_value()) {
        return false;  // Invalid API key
    }
    
    auto permissions_result = get_permissions_from_api_key(api_key);
    if (!permissions_result.has_value()) {
        return false;  // Should not happen if get_user_from_api_key succeeded
    }
    
    const auto& permissions = permissions_result.value();
    return std::find(permissions.begin(), permissions.end(), permission) != permissions.end();
}

std::string AuthManager::hash_api_key(const std::string& api_key) const {
    // This is a very simple hash implementation for demonstration purposes
    // In a real system, you should use a proper cryptographic hash function like SHA-256
    std::hash<std::string> hasher;
    size_t hash = hasher(api_key);
    
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << hash;
    return ss.str();
}

std::string AuthManager::generate_random_api_key() const {
    // Generate a random API key
    // This is a simple implementation - in production, use a cryptographically secure random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    std::string api_key = "jdb_";
    for (int i = 0; i < 32; ++i) {
        int val = dis(gen);
        std::stringstream ss;
        ss << std::hex << std::setw(2) << std::setfill('0') << val;
        api_key += ss.str();
    }
    
    return api_key;
}

std::string AuthManager::generate_id() const {
    // Generate a unique ID
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << std::hex << count;
    return ss.str();
}

} // namespace jadevectordb