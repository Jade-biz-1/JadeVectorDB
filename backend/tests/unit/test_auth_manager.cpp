#include <gtest/gtest.h>
#include <memory>
#include <chrono>

#include "lib/auth.h"
#include "lib/error_handling.h"

using namespace jadevectordb;
using namespace std::chrono_literals;

/**
 * @brief Test fixture for AuthManager
 *
 * This fixture provides access to the singleton AuthManager instance
 * and helper methods for common test operations.
 */
class AuthManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get singleton instance
        auth_manager_ = AuthManager::get_instance();
        ASSERT_NE(auth_manager_, nullptr);
    }

    void TearDown() override {
        // Note: Cannot fully reset singleton, but tests should be independent
    }

    // Helper method to create a test user
    std::string create_test_user(const std::string& username,
                                 const std::string& email,
                                 const std::vector<std::string>& roles = {"role_user"}) {
        auto result = auth_manager_->create_user(username, email, roles);
        EXPECT_TRUE(result.has_value()) << "Failed to create user: " << username;
        return result.has_value() ? result.value() : "";
    }

    AuthManager* auth_manager_;
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(AuthManagerTest, SingletonInstance) {
    auto instance1 = AuthManager::get_instance();
    auto instance2 = AuthManager::get_instance();

    EXPECT_NE(instance1, nullptr);
    EXPECT_NE(instance2, nullptr);
    EXPECT_EQ(instance1, instance2);  // Should be same instance
}

// ============================================================================
// User Management Tests
// ============================================================================

TEST_F(AuthManagerTest, CreateUser_Success) {
    auto result = auth_manager_->create_user("testuser", "test@example.com", {"role_user"});

    ASSERT_TRUE(result.has_value());
    std::string user_id = result.value();
    EXPECT_FALSE(user_id.empty());

    // Verify user was created
    auto user_result = auth_manager_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().username, "testuser");
    EXPECT_EQ(user_result.value().email, "test@example.com");
    EXPECT_EQ(user_result.value().roles.size(), 1);
    EXPECT_EQ(user_result.value().roles[0], "role_user");
    EXPECT_TRUE(user_result.value().is_active);
}

TEST_F(AuthManagerTest, CreateUser_DuplicateEmail) {
    auto result1 = auth_manager_->create_user("user1", "duplicate@example.com", {"role_user"});
    ASSERT_TRUE(result1.has_value());

    // Try to create another user with same email
    auto result2 = auth_manager_->create_user("user2", "duplicate@example.com", {"role_user"});
    EXPECT_FALSE(result2.has_value());
    EXPECT_EQ(result2.error().code(), ErrorCode::ALREADY_EXISTS);
}

TEST_F(AuthManagerTest, CreateUser_InvalidRole) {
    auto result = auth_manager_->create_user("testuser", "test@example.com", {"invalid_role"});

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::INVALID_ARGUMENT);
}

TEST_F(AuthManagerTest, CreateUser_MultipleRoles) {
    std::vector<std::string> roles = {"role_user", "role_reader"};
    auto result = auth_manager_->create_user("multiuser", "multi@example.com", roles);

    ASSERT_TRUE(result.has_value());

    auto user_result = auth_manager_->get_user(result.value());
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().roles.size(), 2);
}

TEST_F(AuthManagerTest, CreateUser_AdminRole) {
    auto result = auth_manager_->create_user("adminuser", "admin@example.com", {"role_admin"});

    ASSERT_TRUE(result.has_value());

    auto user_result = auth_manager_->get_user(result.value());
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().roles.size(), 1);
    EXPECT_EQ(user_result.value().roles[0], "role_admin");
}

TEST_F(AuthManagerTest, GetUser_Success) {
    std::string user_id = create_test_user("getuser", "get@example.com", {"role_user"});

    auto result = auth_manager_->get_user(user_id);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().user_id, user_id);
    EXPECT_EQ(result.value().username, "getuser");
    EXPECT_EQ(result.value().email, "get@example.com");
}

TEST_F(AuthManagerTest, GetUser_NotFound) {
    auto result = auth_manager_->get_user("nonexistent-user-id-12345");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::NOT_FOUND);
}

TEST_F(AuthManagerTest, GetUserByUsername_Success) {
    std::string user_id = create_test_user("usernametest", "username@example.com", {"role_user"});

    auto result = auth_manager_->get_user_by_username("usernametest");

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().user_id, user_id);
    EXPECT_EQ(result.value().username, "usernametest");
}

TEST_F(AuthManagerTest, GetUserByUsername_NotFound) {
    auto result = auth_manager_->get_user_by_username("nonexistentusername");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::NOT_FOUND);
}

TEST_F(AuthManagerTest, ListUsers) {
    // Create several users
    create_test_user("listuser1", "list1@example.com", {"role_user"});
    create_test_user("listuser2", "list2@example.com", {"role_user"});
    create_test_user("listuser3", "list3@example.com", {"role_admin"});

    auto result = auth_manager_->list_users();

    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 3);
}

TEST_F(AuthManagerTest, UpdateUser_Roles) {
    std::string user_id = create_test_user("updateuser", "update@example.com", {"role_user"});

    // Update user roles
    std::vector<std::string> new_roles = {"role_admin", "role_user"};
    auto update_result = auth_manager_->update_user(user_id, new_roles);
    ASSERT_TRUE(update_result.has_value());

    // Verify roles were updated
    auto user_result = auth_manager_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().roles.size(), 2);
}

TEST_F(AuthManagerTest, UpdateUserDetails_Success) {
    std::string user_id = create_test_user("detailsuser", "details@example.com", {"role_user"});

    // Update user details
    auto update_result = auth_manager_->update_user_details(
        user_id,
        std::optional<std::string>("newusername"),
        std::optional<std::string>("newemail@example.com"),
        {"role_admin"}
    );
    ASSERT_TRUE(update_result.has_value());

    // Verify details were updated
    auto user_result = auth_manager_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().username, "newusername");
    EXPECT_EQ(user_result.value().email, "newemail@example.com");
    EXPECT_EQ(user_result.value().roles.size(), 1);
    EXPECT_EQ(user_result.value().roles[0], "role_admin");
}

TEST_F(AuthManagerTest, DeactivateUser_Success) {
    std::string user_id = create_test_user("deactivateuser", "deactivate@example.com", {"role_user"});

    auto deactivate_result = auth_manager_->deactivate_user(user_id);
    ASSERT_TRUE(deactivate_result.has_value());

    auto user_result = auth_manager_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_FALSE(user_result.value().is_active);
}

TEST_F(AuthManagerTest, ActivateUser_Success) {
    std::string user_id = create_test_user("activateuser", "activate@example.com", {"role_user"});

    // Deactivate first
    auth_manager_->deactivate_user(user_id);

    // Then activate
    auto activate_result = auth_manager_->activate_user(user_id);
    ASSERT_TRUE(activate_result.has_value());

    auto user_result = auth_manager_->get_user(user_id);
    ASSERT_TRUE(user_result.has_value());
    EXPECT_TRUE(user_result.value().is_active);
}

// ============================================================================
// Role Management Tests
// ============================================================================

TEST_F(AuthManagerTest, CreateRole_Success) {
    std::vector<std::string> permissions = {"custom:read", "custom:write"};
    auto result = auth_manager_->create_role("CustomRole", permissions);

    ASSERT_TRUE(result.has_value());
    std::string role_id = result.value();
    EXPECT_FALSE(role_id.empty());

    // Verify role was created
    auto role_result = auth_manager_->get_role(role_id);
    ASSERT_TRUE(role_result.has_value());
    EXPECT_EQ(role_result.value().name, "CustomRole");
    EXPECT_EQ(role_result.value().permissions.size(), 2);
}

TEST_F(AuthManagerTest, GetRole_DefaultRoles) {
    // Check default admin role
    auto admin_result = auth_manager_->get_role("role_admin");
    ASSERT_TRUE(admin_result.has_value());
    EXPECT_EQ(admin_result.value().name, "Administrator");
    EXPECT_GT(admin_result.value().permissions.size(), 0);

    // Check default user role
    auto user_result = auth_manager_->get_role("role_user");
    ASSERT_TRUE(user_result.has_value());
    EXPECT_EQ(user_result.value().name, "Standard User");

    // Check default reader role
    auto reader_result = auth_manager_->get_role("role_reader");
    ASSERT_TRUE(reader_result.has_value());
    EXPECT_EQ(reader_result.value().name, "Reader");
}

TEST_F(AuthManagerTest, UpdateRole_Success) {
    std::vector<std::string> initial_permissions = {"perm:read"};
    auto create_result = auth_manager_->create_role("UpdatableRole", initial_permissions);
    ASSERT_TRUE(create_result.has_value());

    std::string role_id = create_result.value();

    // Update role permissions
    std::vector<std::string> new_permissions = {"perm:read", "perm:write", "perm:delete"};
    auto update_result = auth_manager_->update_role(role_id, new_permissions);
    ASSERT_TRUE(update_result.has_value());

    // Verify permissions were updated
    auto role_result = auth_manager_->get_role(role_id);
    ASSERT_TRUE(role_result.has_value());
    EXPECT_EQ(role_result.value().permissions.size(), 3);
}

TEST_F(AuthManagerTest, DeleteRole_Success) {
    std::vector<std::string> permissions = {"temp:read"};
    auto create_result = auth_manager_->create_role("TempRole", permissions);
    ASSERT_TRUE(create_result.has_value());

    std::string role_id = create_result.value();

    // Delete role
    auto delete_result = auth_manager_->delete_role(role_id);
    ASSERT_TRUE(delete_result.has_value());

    // Verify role was deleted
    auto role_result = auth_manager_->get_role(role_id);
    EXPECT_FALSE(role_result.has_value());
}

// ============================================================================
// API Key Management Tests
// ============================================================================

TEST_F(AuthManagerTest, GenerateApiKey_Success) {
    std::string user_id = create_test_user("apikeyuser", "apikey@example.com", {"role_user"});

    std::vector<std::string> permissions = {"api:read", "api:write"};
    auto result = auth_manager_->generate_api_key(user_id, permissions, "Test API Key");

    ASSERT_TRUE(result.has_value());
    std::string api_key = result.value();
    EXPECT_FALSE(api_key.empty());
    EXPECT_GT(api_key.length(), 20);
}

TEST_F(AuthManagerTest, GenerateApiKey_WithDefaultDuration) {
    std::string user_id = create_test_user("apikeyuser2", "apikey2@example.com", {"role_user"});

    auto result = auth_manager_->generate_api_key(user_id);

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());
}

TEST_F(AuthManagerTest, GenerateApiKey_WithCustomDuration) {
    std::string user_id = create_test_user("apikeyuser3", "apikey3@example.com", {"role_user"});

    auto result = auth_manager_->generate_api_key(
        user_id,
        {},
        "Custom duration key",
        std::chrono::hours(24 * 7)  // 7 days
    );

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());
}

TEST_F(AuthManagerTest, ValidateApiKey_Success) {
    std::string user_id = create_test_user("validatekey", "validate@example.com", {"role_user"});

    auto generate_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(generate_result.has_value());

    std::string api_key = generate_result.value();

    auto validate_result = auth_manager_->validate_api_key(api_key);
    ASSERT_TRUE(validate_result.has_value());
    EXPECT_TRUE(validate_result.value());
}

TEST_F(AuthManagerTest, ValidateApiKey_InvalidKey) {
    auto result = auth_manager_->validate_api_key("invalid-api-key-12345");

    EXPECT_FALSE(result.has_value());
}

TEST_F(AuthManagerTest, GetUserFromApiKey_Success) {
    std::string user_id = create_test_user("getuserkey", "getuser@example.com", {"role_user"});

    auto generate_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(generate_result.has_value());

    std::string api_key = generate_result.value();

    auto result = auth_manager_->get_user_from_api_key(api_key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), user_id);
}

TEST_F(AuthManagerTest, GetPermissionsFromApiKey_Success) {
    std::string user_id = create_test_user("permkey", "perm@example.com", {"role_user"});

    std::vector<std::string> permissions = {"custom:read", "custom:write", "custom:delete"};
    auto generate_result = auth_manager_->generate_api_key(user_id, permissions, "Permission test");
    ASSERT_TRUE(generate_result.has_value());

    std::string api_key = generate_result.value();

    auto result = auth_manager_->get_permissions_from_api_key(api_key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 3);
}

TEST_F(AuthManagerTest, RevokeApiKey_Success) {
    std::string user_id = create_test_user("revokekey", "revoke@example.com", {"role_user"});

    auto generate_result = auth_manager_->generate_api_key(user_id);
    ASSERT_TRUE(generate_result.has_value());

    std::string api_key = generate_result.value();

    // Get key_id from list of API keys
    auto list_result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(list_result.has_value());
    ASSERT_GT(list_result.value().size(), 0);

    std::string key_id = list_result.value()[0].key_id;

    // Revoke the key
    auto revoke_result = auth_manager_->revoke_api_key(key_id);
    ASSERT_TRUE(revoke_result.has_value());

    // Verify key is no longer valid
    auto validate_result = auth_manager_->validate_api_key(api_key);
    EXPECT_FALSE(validate_result.has_value());
}

TEST_F(AuthManagerTest, ListApiKeys_Success) {
    std::string user_id1 = create_test_user("listapikey1", "listapi1@example.com", {"role_user"});
    std::string user_id2 = create_test_user("listapikey2", "listapi2@example.com", {"role_user"});

    auth_manager_->generate_api_key(user_id1);
    auth_manager_->generate_api_key(user_id1);
    auth_manager_->generate_api_key(user_id2);

    auto result = auth_manager_->list_api_keys();
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 3);
}

TEST_F(AuthManagerTest, ListApiKeysForUser_Success) {
    std::string user_id = create_test_user("userkeys", "userkeys@example.com", {"role_user"});

    // Generate multiple keys for the user
    auth_manager_->generate_api_key(user_id, {}, "Key 1");
    auth_manager_->generate_api_key(user_id, {}, "Key 2");
    auth_manager_->generate_api_key(user_id, {}, "Key 3");

    auto result = auth_manager_->list_api_keys_for_user(user_id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 3);
}

// ============================================================================
// Permission Checking Tests
// ============================================================================

TEST_F(AuthManagerTest, HasPermission_AdminUser) {
    std::string user_id = create_test_user("adminperm", "adminperm@example.com", {"role_admin"});

    // Admin should have all permissions
    auto result1 = auth_manager_->has_permission(user_id, "database:create");
    ASSERT_TRUE(result1.has_value());
    EXPECT_TRUE(result1.value());

    auto result2 = auth_manager_->has_permission(user_id, "user:manage");
    ASSERT_TRUE(result2.has_value());
    EXPECT_TRUE(result2.value());
}

TEST_F(AuthManagerTest, HasPermission_StandardUser) {
    std::string user_id = create_test_user("userperm", "userperm@example.com", {"role_user"});

    // User should have read permission
    auto result1 = auth_manager_->has_permission(user_id, "database:read");
    ASSERT_TRUE(result1.has_value());
    EXPECT_TRUE(result1.value());

    // User should NOT have user management permission
    auto result2 = auth_manager_->has_permission(user_id, "user:manage");
    ASSERT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

TEST_F(AuthManagerTest, HasPermission_ReaderUser) {
    std::string user_id = create_test_user("readerperm", "readerperm@example.com", {"role_reader"});

    // Reader should have read permission
    auto result1 = auth_manager_->has_permission(user_id, "database:read");
    ASSERT_TRUE(result1.has_value());
    EXPECT_TRUE(result1.value());

    // Reader should NOT have write permission
    auto result2 = auth_manager_->has_permission(user_id, "vector:add");
    ASSERT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

TEST_F(AuthManagerTest, HasPermissionWithApiKey_Success) {
    std::string user_id = create_test_user("apiperm", "apiperm@example.com", {"role_user"});

    std::vector<std::string> permissions = {"custom:action"};
    auto generate_result = auth_manager_->generate_api_key(user_id, permissions);
    ASSERT_TRUE(generate_result.has_value());

    std::string api_key = generate_result.value();

    auto result = auth_manager_->has_permission_with_api_key(api_key, "custom:action");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

TEST_F(AuthManagerTest, HasPermissionWithApiKey_NoPermission) {
    std::string user_id = create_test_user("apiperm2", "apiperm2@example.com", {"role_user"});

    std::vector<std::string> permissions = {"read:only"};
    auto generate_result = auth_manager_->generate_api_key(user_id, permissions);
    ASSERT_TRUE(generate_result.has_value());

    std::string api_key = generate_result.value();

    auto result = auth_manager_->has_permission_with_api_key(api_key, "write:action");
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value());
}

// ============================================================================
// Helper Method Tests
// ============================================================================

TEST_F(AuthManagerTest, HashApiKey_Deterministic) {
    std::string api_key = "test-api-key-12345";

    std::string hash1 = auth_manager_->hash_api_key(api_key);
    std::string hash2 = auth_manager_->hash_api_key(api_key);

    EXPECT_EQ(hash1, hash2);  // Same input should produce same hash
}

TEST_F(AuthManagerTest, HashApiKey_Different) {
    std::string key1 = "api-key-1";
    std::string key2 = "api-key-2";

    std::string hash1 = auth_manager_->hash_api_key(key1);
    std::string hash2 = auth_manager_->hash_api_key(key2);

    EXPECT_NE(hash1, hash2);  // Different inputs should produce different hashes
}

TEST_F(AuthManagerTest, GenerateRandomApiKey_Unique) {
    std::string key1 = auth_manager_->generate_random_api_key();
    std::string key2 = auth_manager_->generate_random_api_key();
    std::string key3 = auth_manager_->generate_random_api_key();

    EXPECT_FALSE(key1.empty());
    EXPECT_FALSE(key2.empty());
    EXPECT_FALSE(key3.empty());

    EXPECT_NE(key1, key2);
    EXPECT_NE(key2, key3);
    EXPECT_NE(key1, key3);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(AuthManagerTest, CreateUser_EmptyUsername) {
    auto result = auth_manager_->create_user("", "empty@example.com", {"role_user"});
    // Implementation may or may not allow empty usernames
    // This test documents the behavior
}

TEST_F(AuthManagerTest, CreateUser_EmptyEmail) {
    auto result = auth_manager_->create_user("testempty", "", {"role_user"});
    // Implementation may or may not allow empty emails
    // This test documents the behavior
}

TEST_F(AuthManagerTest, GenerateApiKey_NonexistentUser) {
    auto result = auth_manager_->generate_api_key("nonexistent-user-id-12345");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::NOT_FOUND);
}

TEST_F(AuthManagerTest, UpdateUser_NonexistentUser) {
    auto result = auth_manager_->update_user("nonexistent-id", {"role_user"});

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::NOT_FOUND);
}

TEST_F(AuthManagerTest, DeactivateUser_NonexistentUser) {
    auto result = auth_manager_->deactivate_user("nonexistent-id");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::NOT_FOUND);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
