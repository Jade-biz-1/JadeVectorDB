// Comprehensive test for Sprint 1.3 - Permissions, API Keys, Auth Tokens, Sessions
#include "services/sqlite_persistence_layer.h"
#include "models/auth.h"
#include "lib/logging.h"
#include <iostream>
#include <cassert>

using namespace jadevectordb;

#define TEST(name) std::cout << "\n" << name << "..." << std::endl
#define ASSERT(cond, msg) if (!(cond)) { std::cerr << "  ❌ FAILED: " << msg << std::endl; return 1; } else { std::cout << "  ✓ " << msg << std::endl; }

int main() {
    std::cout << "=== Sprint 1.3 Persistence Test ===" << std::endl;
    
    // Initialize logging
    logging::LoggerManager::initialize(logging::LogLevel::WARN);
    
    // Create test data directory
    std::string test_dir = "/tmp/jadevectordb_test_sprint13";
    system(("rm -rf " + test_dir).c_str());
    system(("mkdir -p " + test_dir).c_str());
    
    SQLitePersistenceLayer persistence(test_dir);
    
    TEST("1. Initialize database");
    auto init_result = persistence.initialize();
    ASSERT(init_result.has_value(), "Database initialized");
    
    // Create test users
    auto user1_result = persistence.create_user("alice", "alice@example.com", "hash1", "salt1");
    ASSERT(user1_result.has_value(), "Created user alice");
    std::string user1_id = user1_result.value();
    
    auto user2_result = persistence.create_user("bob", "bob@example.com", "hash2", "salt2");
    ASSERT(user2_result.has_value(), "Created user bob");
    std::string user2_id = user2_result.value();
    
    // === PERMISSION TESTS ===
    
    TEST("2. List all permissions");
    auto perms_result = persistence.list_permissions();
    ASSERT(perms_result.has_value(), "Listed permissions");
    ASSERT(perms_result.value().size() >= 9, "Found default permissions");
    
    TEST("3. Get specific permission");
    auto perm_result = persistence.get_permission("perm_db_read");
    ASSERT(perm_result.has_value(), "Retrieved permission");
    ASSERT(perm_result.value().permission_name == "database:read", "Permission name correct");
    
    TEST("4. Get user permissions (via roles)");
    // First assign a role
    auto assign_result = persistence.assign_role_to_user(user1_id, "role_admin");
    ASSERT(assign_result.has_value(), "Assigned admin role to alice");
    
    auto user_perms_result = persistence.get_user_permissions(user1_id);
    if (!user_perms_result.has_value()) {
        std::cerr << "  Error: " << user_perms_result.error().message << std::endl;
    }
    ASSERT(user_perms_result.has_value(), "Retrieved user permissions");
    ASSERT(user_perms_result.value().size() >= 5, "Admin has multiple permissions");
    
    TEST("5. Grant database permission");
    auto grant_result = persistence.grant_database_permission(
        "db_test_123", "user", user2_id, "perm_db_read", user1_id);
    ASSERT(grant_result.has_value(), "Granted database permission");
    
    TEST("6. Get database permissions for user");
    auto db_perms_result = persistence.get_database_permissions("db_test_123", user2_id);
    ASSERT(db_perms_result.has_value(), "Retrieved database permissions");
    ASSERT(db_perms_result.value().size() == 1, "User has 1 database permission");
    
    TEST("7. Check database permission");
    auto check_result = persistence.check_database_permission("db_test_123", user2_id, "database:read");
    ASSERT(check_result.has_value() && check_result.value(), "User has read permission");
    
    auto check_no_result = persistence.check_database_permission("db_test_123", user2_id, "database:write");
    ASSERT(check_no_result.has_value() && !check_no_result.value(), "User doesn't have write permission");
    
    TEST("8. Revoke database permission");
    auto revoke_result = persistence.revoke_database_permission(
        "db_test_123", "user", user2_id, "perm_db_read");
    ASSERT(revoke_result.has_value(), "Revoked database permission");
    
    auto db_perms_after = persistence.get_database_permissions("db_test_123", user2_id);
    ASSERT(db_perms_after.has_value() && db_perms_after.value().empty(), "No permissions remaining");
    
    // === API KEY TESTS ===
    
    TEST("9. Create API key");
    std::vector<std::string> scopes = {"read", "write"};
    int64_t expires_at = std::time(nullptr) + 86400; // 24 hours
    auto key_result = persistence.create_api_key(
        user1_id, "hashed_key_abc123", "Production Key", "jvdb_abc", scopes, expires_at);
    ASSERT(key_result.has_value(), "Created API key");
    std::string api_key_id = key_result.value();
    
    TEST("10. Get API key by ID");
    auto get_key_result = persistence.get_api_key_by_id(api_key_id);
    ASSERT(get_key_result.has_value(), "Retrieved API key by ID");
    ASSERT(get_key_result.value().key_name == "Production Key", "Key name matches");
    ASSERT(get_key_result.value().is_active, "Key is active");
    
    TEST("11. Get API key by prefix");
    auto prefix_result = persistence.get_api_key_by_prefix("jvdb_abc");
    ASSERT(prefix_result.has_value(), "Retrieved API key by prefix");
    ASSERT(prefix_result.value().api_key_id == api_key_id, "Same key retrieved");
    
    TEST("12. List user API keys");
    auto list_keys_result = persistence.list_user_api_keys(user1_id);
    ASSERT(list_keys_result.has_value(), "Listed user API keys");
    ASSERT(list_keys_result.value().size() == 1, "User has 1 API key");
    
    TEST("13. Update API key usage");
    auto usage_result = persistence.update_api_key_usage(api_key_id);
    ASSERT(usage_result.has_value(), "Updated API key usage");
    
    auto key_after_use = persistence.get_api_key_by_id(api_key_id);
    ASSERT(key_after_use.has_value() && key_after_use.value().usage_count == 1, "Usage count is 1");
    
    TEST("14. Revoke API key");
    auto revoke_key_result = persistence.revoke_api_key(api_key_id);
    ASSERT(revoke_key_result.has_value(), "Revoked API key");
    
    auto key_after_revoke = persistence.get_api_key_by_id(api_key_id);
    ASSERT(key_after_revoke.has_value() && !key_after_revoke.value().is_active, "Key is inactive");
    
    // === AUTH TOKEN TESTS ===
    
    TEST("15. Create auth token");
    int64_t token_expires = std::time(nullptr) + 3600; // 1 hour
    auto token_result = persistence.create_auth_token(
        user1_id, "hashed_token_xyz", "192.168.1.1", "Mozilla/5.0", token_expires);
    if (!token_result.has_value()) {
        std::cerr << "  Error: " << token_result.error().message << std::endl;
    }
    ASSERT(token_result.has_value(), "Created auth token");
    std::string token_id = token_result.value();
    
    TEST("16. Get auth token");
    auto get_token_result = persistence.get_auth_token(token_id);
    ASSERT(get_token_result.has_value(), "Retrieved auth token");
    ASSERT(get_token_result.value().user_id == user1_id, "Token belongs to alice");
    ASSERT(get_token_result.value().is_valid, "Token is valid");
    
    TEST("17. Update token last used");
    auto update_token_result = persistence.update_token_last_used(token_id);
    ASSERT(update_token_result.has_value(), "Updated token last used");
    
    TEST("18. Invalidate specific token");
    auto invalidate_result = persistence.invalidate_auth_token(token_id);
    ASSERT(invalidate_result.has_value(), "Invalidated token");
    
    auto token_after_invalid = persistence.get_auth_token(token_id);
    ASSERT(token_after_invalid.has_value() && !token_after_invalid.value().is_valid, "Token is invalid");
    
    TEST("19. Create multiple tokens for user");
    auto token2_result = persistence.create_auth_token(
        user1_id, "hashed_token_2", "192.168.1.2", "Chrome", token_expires);
    auto token3_result = persistence.create_auth_token(
        user1_id, "hashed_token_3", "192.168.1.3", "Safari", token_expires);
    ASSERT(token2_result.has_value() && token3_result.has_value(), "Created additional tokens");
    
    TEST("20. Invalidate all user tokens");
    auto invalidate_all_result = persistence.invalidate_user_tokens(user1_id);
    ASSERT(invalidate_all_result.has_value(), "Invalidated all user tokens");
    
    TEST("21. Cleanup expired tokens");
    auto cleanup_tokens_result = persistence.cleanup_expired_tokens();
    ASSERT(cleanup_tokens_result.has_value(), "Cleaned up expired tokens");
    
    // === SESSION TESTS ===
    
    // Create a valid token for session
    auto valid_token_result = persistence.create_auth_token(
        user2_id, "session_token_hash", "192.168.1.100", "Firefox", token_expires);
    ASSERT(valid_token_result.has_value(), "Created token for session");
    std::string valid_token_id = valid_token_result.value();
    
    TEST("22. Create session");
    int64_t session_expires = std::time(nullptr) + 7200; // 2 hours
    auto session_result = persistence.create_session(
        user2_id, valid_token_id, "192.168.1.100", session_expires);
    ASSERT(session_result.has_value(), "Created session");
    std::string session_id = session_result.value();
    
    TEST("23. Get session");
    auto get_session_result = persistence.get_session(session_id);
    ASSERT(get_session_result.has_value(), "Retrieved session");
    ASSERT(get_session_result.value().user_id == user2_id, "Session belongs to bob");
    ASSERT(get_session_result.value().is_active, "Session is active");
    
    TEST("24. Update session activity");
    auto update_session_result = persistence.update_session_activity(session_id);
    ASSERT(update_session_result.has_value(), "Updated session activity");
    
    auto session_after_update = persistence.get_session(session_id);
    ASSERT(session_after_update.has_value(), "Retrieved session after update");
    
    TEST("25. End session");
    auto end_session_result = persistence.end_session(session_id);
    ASSERT(end_session_result.has_value(), "Ended session");
    
    auto session_after_end = persistence.get_session(session_id);
    ASSERT(session_after_end.has_value() && !session_after_end.value().is_active, "Session is inactive");
    
    TEST("26. Cleanup expired sessions");
    auto cleanup_sessions_result = persistence.cleanup_expired_sessions();
    ASSERT(cleanup_sessions_result.has_value(), "Cleaned up expired sessions");
    
    TEST("27. Close database");
    auto close_result = persistence.close();
    ASSERT(close_result.has_value(), "Database closed");
    
    std::cout << "\n=== ALL 27 TESTS PASSED ✓ ===" << std::endl;
    std::cout << "\nDatabase: " << test_dir << "/system.db" << std::endl;
    
    return 0;
}
