#ifndef JADEVECTORDB_AUTH_H
#define JADEVECTORDB_AUTH_H

#include <string>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <set>

#include "error_handling.h"
#include "zero_trust.h"

namespace jadevectordb {

// Represents a user in the system
struct User {
    std::string user_id;
    std::string username;
    std::string email;
    std::vector<std::string> roles;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_login;
    bool is_active;
    
    User() : is_active(true) {}
};

// Represents a role in the system
struct Role {
    std::string role_id;
    std::string name;
    std::vector<std::string> permissions;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    Role() = default;
    Role(const std::string& id, const std::string& n) : role_id(id), name(n) {}
};

// Represents an API key
struct ApiKey {
    std::string key_id;
    std::string key_hash;  // Store hashed value of the key
    std::string user_id;
    std::vector<std::string> permissions;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point expires_at;
    std::string description;
    bool is_active;
    
    ApiKey() : is_active(true) {}
};

class AuthManager {
private:
    std::unordered_map<std::string, User> users_;
    std::unordered_map<std::string, Role> roles_;
    std::unordered_map<std::string, ApiKey> api_keys_;
    
    mutable std::shared_mutex auth_mutex_;
    
    // Zero-trust components
    std::unique_ptr<zero_trust::ZeroTrustOrchestrator> zero_trust_orchestrator_;
    std::unordered_map<std::string, zero_trust::SessionInfo> active_sessions_;
    
    // Default roles
    static const std::string ADMIN_ROLE;
    static const std::string USER_ROLE;
    static const std::string READER_ROLE;
    
public:
    AuthManager();
    ~AuthManager() = default;
    
    // User management
    Result<std::string> create_user(const std::string& username, 
                                  const std::string& email,
                                  const std::vector<std::string>& initial_roles = {});
    
    Result<User> get_user(const std::string& user_id) const;
    Result<void> update_user(const std::string& user_id, 
                           const std::vector<std::string>& new_roles);
    Result<void> deactivate_user(const std::string& user_id);
    Result<void> activate_user(const std::string& user_id);
    
    // Role management
    Result<std::string> create_role(const std::string& name, 
                                  const std::vector<std::string>& permissions);
    
    Result<Role> get_role(const std::string& role_id) const;
    Result<void> update_role(const std::string& role_id, 
                           const std::vector<std::string>& new_permissions);
    Result<void> delete_role(const std::string& role_id);
    
    // API Key management
    Result<std::string> generate_api_key(const std::string& user_id,
                                       const std::vector<std::string>& permissions = {},
                                       const std::string& description = "",
                                       std::chrono::hours validity_duration = std::chrono::hours(24 * 30)); // 30 days default
    
    Result<bool> validate_api_key(const std::string& api_key) const;
    Result<std::string> get_user_from_api_key(const std::string& api_key) const;
    Result<std::vector<std::string>> get_permissions_from_api_key(const std::string& api_key) const;
    Result<void> revoke_api_key(const std::string& key_id);
    
    // Permission checking
    Result<bool> has_permission(const std::string& user_id, const std::string& permission) const;
    Result<bool> has_permission_with_api_key(const std::string& api_key, 
                                           const std::string& permission) const;
    
    // Zero-trust related methods
    Result<void> initialize_zero_trust();
    Result<std::string> create_secure_session(const std::string& user_id, 
                                            const std::string& device_id,
                                            const std::string& ip_address);
    Result<bool> validate_session(const std::string& session_id) const;
    Result<void> update_session_activity(const std::string& session_id);
    Result<zero_trust::TrustLevel> evaluate_session_trust(const std::string& session_id);
    Result<zero_trust::AccessDecision> authorize_access(const std::string& session_id,
                                                       const std::string& resource_id,
                                                       zero_trust::AccessType access_type);
    Result<void> terminate_session(const std::string& session_id);
    Result<std::string> register_device(const zero_trust::DeviceIdentity& device_identity,
                                     zero_trust::TrustLevel initial_trust_level = zero_trust::TrustLevel::LOW);
    
    // Helper methods
    std::string hash_api_key(const std::string& api_key) const;
    std::string generate_random_api_key() const;
    
    // Initialize default roles
    void initialize_default_roles();
    
private:
    std::string generate_id() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_AUTH_H