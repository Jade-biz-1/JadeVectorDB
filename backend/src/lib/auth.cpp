#include "auth.h"
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>
#include <set>
#include <optional>
#include "zero_trust.h"

// Include header for string hashing (we'll use a simple approach)
#include <functional>

namespace jadevectordb {

const std::string AuthManager::ADMIN_ROLE = "admin";
const std::string AuthManager::USER_ROLE = "user";
const std::string AuthManager::READER_ROLE = "reader";

// Singleton instance and flag
std::unique_ptr<AuthManager> AuthManager::instance_ = nullptr;
std::once_flag AuthManager::once_flag_;

AuthManager::AuthManager() {
    initialize_default_roles();
    initialize_zero_trust();
}

AuthManager* AuthManager::get_instance() {
    std::call_once(once_flag_, []() {
        instance_ = std::unique_ptr<AuthManager>(new AuthManager());
    });
    return instance_.get();
}

void AuthManager::initialize_default_roles() {
    // Create default admin role with all permissions
    Role admin_role("role_admin", "Administrator");
    admin_role.permissions = {
        "database:create", "database:read", "database:update", "database:delete",
        "vector:add", "vector:read", "vector:update", "vector:delete",
        "index:create", "index:read", "index:update", "index:delete",
        "search:execute", "user:manage", "api_key:manage", "config:manage",
        "monitoring:read", "monitoring:write", "alert:read", "alert:ack", "audit:read"
    };
    admin_role.created_at = std::chrono::system_clock::now();
    admin_role.updated_at = std::chrono::system_clock::now();
    
    // Create default user role with standard permissions
    Role user_role("role_user", "Standard User");
    user_role.permissions = {
        "database:read", "vector:add", "vector:read", "vector:update",
        "search:execute", "index:read", "monitoring:read", "alert:read"
    };
    user_role.created_at = std::chrono::system_clock::now();
    user_role.updated_at = std::chrono::system_clock::now();
    
    // Create default reader role with read-only permissions
    Role reader_role("role_reader", "Reader");
    reader_role.permissions = {
        "database:read", "vector:read", "search:execute", "monitoring:read", "alert:read"
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

Result<void> AuthManager::create_user_with_id(const User& user) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);

    if (user.user_id.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "User id cannot be empty");
    }

    if (users_.find(user.user_id) != users_.end()) {
        RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "User already exists: " + user.user_id);
    }

    for (const auto& entry : users_) {
        if (!user.email.empty() && entry.second.email == user.email) {
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "User with this email already exists");
        }
        if (!user.username.empty() && entry.second.username == user.username) {
            RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "User with this username already exists");
        }
    }

    for (const auto& role : user.roles) {
        if (roles_.find(role) == roles_.end()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid role: " + role);
        }
    }

    User stored_user = user;
    if (stored_user.created_at.time_since_epoch().count() == 0) {
        stored_user.created_at = std::chrono::system_clock::now();
    }

    users_[stored_user.user_id] = stored_user;
    return {};
}

Result<User> AuthManager::get_user(const std::string& user_id) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }
    
    return it->second;
}

Result<User> AuthManager::get_user_by_username(const std::string& username) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);

    for (const auto& pair : users_) {
        if (pair.second.username == username) {
            return pair.second;
        }
    }

    RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + username);
}

Result<std::vector<User>> AuthManager::list_users() const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    std::vector<User> user_list;
    user_list.reserve(users_.size());
    
    for (const auto& pair : users_) {
        user_list.push_back(pair.second);
    }
    
    return user_list;
}

Result<void> AuthManager::update_user(const std::string& user_id,
                                    const std::vector<std::string>& new_roles) {
    return update_user_details(user_id, std::nullopt, std::nullopt, new_roles);
}

Result<void> AuthManager::update_user_details(const std::string& user_id,
                                            const std::optional<std::string>& new_username,
                                            const std::optional<std::string>& new_email,
                                            const std::vector<std::string>& new_roles) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);

    auto it = users_.find(user_id);
    if (it == users_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "User not found: " + user_id);
    }

    if (new_username.has_value()) {
        const std::string& username = new_username.value();
        if (username.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Username cannot be empty");
        }
        for (const auto& entry : users_) {
            if (entry.first != user_id && entry.second.username == username) {
                RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Username already in use: " + username);
            }
        }
        it->second.username = username;
    }

    if (new_email.has_value()) {
        const std::string& email = new_email.value();
        for (const auto& entry : users_) {
            if (entry.first != user_id && entry.second.email == email) {
                RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Email already in use: " + email);
            }
        }
        it->second.email = email;
    }

    if (!new_roles.empty()) {
        for (const auto& role : new_roles) {
            if (roles_.find(role) == roles_.end()) {
                RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid role: " + role);
            }
        }
        it->second.roles = new_roles;
    }

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

Result<std::vector<ApiKey>> AuthManager::list_api_keys() const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);

    std::vector<ApiKey> keys;
    keys.reserve(api_keys_.size());
    for (const auto& entry : api_keys_) {
        keys.push_back(entry.second);
    }
    return keys;
}

Result<std::vector<ApiKey>> AuthManager::list_api_keys_for_user(const std::string& user_id) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);

    std::vector<ApiKey> keys;
    for (const auto& entry : api_keys_) {
        if (entry.second.user_id == user_id) {
            keys.push_back(entry.second);
        }
    }
    return keys;
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

Result<std::string> AuthManager::create_secure_session(const std::string& user_id, 
                                                        const std::string& device_id,
                                                        const std::string& ip_address) {
    if (!zero_trust_orchestrator_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Zero-trust system not initialized");
    }
    
    // Check if user exists
    {
        std::shared_lock<std::shared_mutex> lock(auth_mutex_);
        if (users_.find(user_id) == users_.end()) {
            RETURN_ERROR(ErrorCode::AUTHENTICATION_ERROR, "User not found: " + user_id);
        }
    }
    
    // Generate session ID
    std::string session_id = "sess_" + generate_id();
    
    // Create session info
    zero_trust::SessionInfo session_info;
    session_info.session_id = session_id;
    session_info.user_id = user_id;
    session_info.device_id = device_id;
    session_info.ip_address = ip_address;
    session_info.created_at = std::chrono::system_clock::now();
    session_info.last_activity = session_info.created_at;
    session_info.expires_at = session_info.created_at + std::chrono::hours(24); // 24 hour default
    session_info.trust_level = zero_trust::TrustLevel::MEDIUM; // Initial trust level
    session_info.origin = "authentication";
    
    // Store session
    {
        std::unique_lock<std::shared_mutex> lock(auth_mutex_);
        active_sessions_[session_id] = session_info;
    }
    
    return session_id;
}

Result<bool> AuthManager::validate_session(const std::string& session_id) const {
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        return false;
    }
    
    auto now = std::chrono::system_clock::now();
    bool is_valid = now < it->second.expires_at;
    
    return is_valid;
}

Result<void> AuthManager::update_session_activity(const std::string& session_id) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        RETURN_ERROR(ErrorCode::AUTHENTICATION_ERROR, "Session not found: " + session_id);
    }
    
    it->second.last_activity = std::chrono::system_clock::now();
    
    return {};
}

Result<zero_trust::TrustLevel> AuthManager::evaluate_session_trust(const std::string& session_id) {
    if (!zero_trust_orchestrator_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Zero-trust system not initialized");
    }
    
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        RETURN_ERROR(ErrorCode::AUTHENTICATION_ERROR, "Session not found: " + session_id);
    }
    
    zero_trust::TrustLevel trust_level = zero_trust_orchestrator_->continuous_evaluation(session_id);
    
    // Update session trust level
    auto& session = const_cast<zero_trust::SessionInfo&>(it->second);
    session.trust_level = trust_level;
    
    return trust_level;
}

Result<zero_trust::AccessDecision> AuthManager::authorize_access(const std::string& session_id,
                                                                  const std::string& resource_id,
                                                                  zero_trust::AccessType access_type) {
    if (!zero_trust_orchestrator_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Zero-trust system not initialized");
    }
    
    std::shared_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        RETURN_ERROR(ErrorCode::AUTHENTICATION_ERROR, "Session not found: " + session_id);
    }
    
    const auto& session_info = it->second;
    
    // Create access request
    zero_trust::AccessRequest request;
    request.resource_id = resource_id;
    request.access_type = access_type;
    request.requester_id = session_info.user_id;
    request.device_id = session_info.device_id;
    request.ip_address = session_info.ip_address;
    request.justification = "Requested by user";
    request.requested_at = std::chrono::system_clock::now();
    
    // Create device identity (simplified for this example)
    zero_trust::DeviceIdentity device_identity;
    device_identity.device_id = session_info.device_id;
    device_identity.is_managed = true;
    device_identity.trust_level = session_info.trust_level;
    
    // Evaluate access request using zero-trust orchestrator
    zero_trust::AccessDecision decision = zero_trust_orchestrator_->evaluate_access_request(
        request, session_info, device_identity);
    
    return decision;
}

Result<void> AuthManager::terminate_session(const std::string& session_id) {
    std::unique_lock<std::shared_mutex> lock(auth_mutex_);
    
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        RETURN_ERROR(ErrorCode::AUTHENTICATION_ERROR, "Session not found: " + session_id);
    }
    
    active_sessions_.erase(it);
    
    return {};
}

Result<std::string> AuthManager::register_device(const zero_trust::DeviceIdentity& device_identity,
                                                  zero_trust::TrustLevel initial_trust_level) {
    if (!zero_trust_orchestrator_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Zero-trust system not initialized");
    }
    
    std::string device_id = zero_trust_orchestrator_->register_device(device_identity, initial_trust_level);
    
    return device_id;
}


Result<void> AuthManager::initialize_zero_trust() {
    // Create individual zero-trust components
    auto continuous_auth = std::make_unique<zero_trust::ContinuousAuthentication>();
    auto microseg = std::make_unique<zero_trust::MicroSegmentation>();
    auto jit_access = std::make_unique<zero_trust::JustInTimeAccess>();
    auto device_attestation = std::make_unique<zero_trust::DeviceAttestation>();

    // Create and initialize the orchestrator with these components
    zero_trust_orchestrator_ = std::make_unique<zero_trust::ZeroTrustOrchestrator>(
        std::move(continuous_auth),
        std::move(microseg),
        std::move(jit_access),
        std::move(device_attestation)
    );

    return {}; // Success
}

} // namespace jadevectordb
