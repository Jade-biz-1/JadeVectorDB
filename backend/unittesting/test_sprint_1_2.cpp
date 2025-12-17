// Comprehensive test for Sprint 1.2 RBAC methods
#include "services/sqlite_persistence_layer.h"
#include "models/auth.h"
#include "lib/logging.h"
#include <iostream>
#include <cassert>

using namespace jadevectordb;

#define TEST(name) std::cout << "\n" << name << "..." << std::endl
#define ASSERT(cond, msg) if (!(cond)) { std::cerr << "  ❌ FAILED: " << msg << std::endl; return 1; } else { std::cout << "  ✓ " << msg << std::endl; }

int main() {
    std::cout << "=== Sprint 1.2 RBAC Methods Test ===" << std::endl;
    
    // Initialize logging
    logging::LoggerManager::initialize(logging::LogLevel::WARN);
    
    // Create test data directory
    std::string test_dir = "/tmp/jadevectordb_test_sprint12";
    system(("rm -rf " + test_dir).c_str());
    system(("mkdir -p " + test_dir).c_str());
    
    SQLitePersistenceLayer persistence(test_dir);
    
    TEST("1. Initialize database");
    auto init_result = persistence.initialize();
    ASSERT(init_result.has_value(), "Database initialized");
    
    // === USER MANAGEMENT TESTS ===
    
    TEST("2. Create users");
    auto user1_result = persistence.create_user("alice", "alice@example.com", "hash1", "salt1");
    ASSERT(user1_result.has_value(), "Created user alice");
    std::string user1_id = user1_result.value();
    
    auto user2_result = persistence.create_user("bob", "bob@example.com", "hash2", "salt2");
    ASSERT(user2_result.has_value(), "Created user bob");
    std::string user2_id = user2_result.value();
    
    TEST("3. Get user by email");
    auto email_result = persistence.get_user_by_email("alice@example.com");
    if (!email_result.has_value()) {
        std::cerr << "  Error: " << email_result.error().message << std::endl;
    }
    ASSERT(email_result.has_value(), "Retrieved user by email");
    ASSERT(email_result.value().username == "alice", "Username matches");
    
    TEST("4. Update user");
    User updated_user = email_result.value();
    updated_user.is_active = false;
    auto update_result = persistence.update_user(user1_id, updated_user);
    ASSERT(update_result.has_value(), "User updated");
    
    auto check_result = persistence.get_user(user1_id);
    ASSERT(check_result.has_value() && !check_result.value().is_active, "User is now inactive");
    
    TEST("5. Increment failed login");
    auto inc_result = persistence.increment_failed_login(user1_id);
    ASSERT(inc_result.has_value(), "Failed login incremented");
    
    auto user_after_fail = persistence.get_user(user1_id);
    ASSERT(user_after_fail.has_value() && user_after_fail.value().failed_login_attempts == 1, "Failed login count is 1");
    
    TEST("6. Lock account");
    int64_t lock_until = std::time(nullptr) + 3600; // Lock for 1 hour
    auto lock_result = persistence.lock_account(user1_id, lock_until);
    ASSERT(lock_result.has_value(), "Account locked");
    
    auto locked_user = persistence.get_user(user1_id);
    ASSERT(locked_user.has_value() && locked_user.value().account_locked_until == lock_until, "Lock timestamp correct");
    
    TEST("7. Reset failed login");
    auto reset_result = persistence.reset_failed_login(user1_id);
    ASSERT(reset_result.has_value(), "Failed login reset");
    
    auto user_after_reset = persistence.get_user(user1_id);
    ASSERT(user_after_reset.has_value() && user_after_reset.value().failed_login_attempts == 0, "Failed login count is 0");
    ASSERT(user_after_reset.value().account_locked_until == 0, "Account unlocked");
    
    // === GROUP MANAGEMENT TESTS ===
    
    TEST("8. Create groups");
    auto group1_result = persistence.create_group("developers", "Development team", user1_id);
    ASSERT(group1_result.has_value(), "Created developers group");
    std::string group1_id = group1_result.value();
    
    auto group2_result = persistence.create_group("admins", "Admin team", user2_id);
    ASSERT(group2_result.has_value(), "Created admins group");
    std::string group2_id = group2_result.value();
    
    TEST("9. Get group");
    auto get_group_result = persistence.get_group(group1_id);
    ASSERT(get_group_result.has_value(), "Retrieved group");
    ASSERT(get_group_result.value().group_name == "developers", "Group name matches");
    
    TEST("10. List groups");
    auto list_groups_result = persistence.list_groups(10, 0);
    ASSERT(list_groups_result.has_value(), "Listed groups");
    ASSERT(list_groups_result.value().size() == 2, "Found 2 groups");
    
    TEST("11. Update group");
    Group updated_group = get_group_result.value();
    updated_group.description = "Updated description";
    auto update_group_result = persistence.update_group(group1_id, updated_group);
    ASSERT(update_group_result.has_value(), "Group updated");
    
    auto check_group = persistence.get_group(group1_id);
    ASSERT(check_group.has_value() && check_group.value().description == "Updated description", "Description updated");
    
    TEST("12. Add users to groups");
    auto add1_result = persistence.add_user_to_group(group1_id, user1_id);
    ASSERT(add1_result.has_value(), "Added alice to developers");
    
    auto add2_result = persistence.add_user_to_group(group1_id, user2_id);
    ASSERT(add2_result.has_value(), "Added bob to developers");
    
    auto add3_result = persistence.add_user_to_group(group2_id, user2_id);
    ASSERT(add3_result.has_value(), "Added bob to admins");
    
    TEST("13. Get group members");
    auto members_result = persistence.get_group_members(group1_id);
    ASSERT(members_result.has_value(), "Retrieved group members");
    ASSERT(members_result.value().size() == 2, "Group has 2 members");
    
    TEST("14. Get user groups");
    auto user_groups_result = persistence.get_user_groups(user2_id);
    ASSERT(user_groups_result.has_value(), "Retrieved user groups");
    ASSERT(user_groups_result.value().size() == 2, "Bob is in 2 groups");
    
    TEST("15. Remove user from group");
    auto remove_result = persistence.remove_user_from_group(group1_id, user2_id);
    ASSERT(remove_result.has_value(), "Removed bob from developers");
    
    auto members_after = persistence.get_group_members(group1_id);
    ASSERT(members_after.has_value() && members_after.value().size() == 1, "Group now has 1 member");
    
    // === ROLE MANAGEMENT TESTS ===
    
    TEST("16. Assign roles to users");
    auto assign1_result = persistence.assign_role_to_user(user1_id, "role_admin");
    ASSERT(assign1_result.has_value(), "Assigned admin role to alice");
    
    auto assign2_result = persistence.assign_role_to_user(user2_id, "role_user");
    ASSERT(assign2_result.has_value(), "Assigned user role to bob");
    
    TEST("17. Get user roles");
    auto user_roles_result = persistence.get_user_roles(user1_id);
    ASSERT(user_roles_result.has_value(), "Retrieved user roles");
    ASSERT(user_roles_result.value().size() == 1, "Alice has 1 role");
    ASSERT(user_roles_result.value()[0] == "role_admin", "Role is admin");
    
    TEST("18. Assign roles to groups");
    auto assign_group_result = persistence.assign_role_to_group(group1_id, "role_user");
    ASSERT(assign_group_result.has_value(), "Assigned user role to developers group");
    
    TEST("19. Get group roles");
    auto group_roles_result = persistence.get_group_roles(group1_id);
    ASSERT(group_roles_result.has_value(), "Retrieved group roles");
    ASSERT(group_roles_result.value().size() == 1, "Group has 1 role");
    
    TEST("20. Get role permissions");
    auto perms_result = persistence.get_role_permissions("role_admin");
    ASSERT(perms_result.has_value(), "Retrieved role permissions");
    ASSERT(perms_result.value().size() >= 5, "Admin role has multiple permissions");
    
    TEST("21. Revoke role from user");
    auto revoke1_result = persistence.revoke_role_from_user(user1_id, "role_admin");
    ASSERT(revoke1_result.has_value(), "Revoked admin role from alice");
    
    auto roles_after = persistence.get_user_roles(user1_id);
    ASSERT(roles_after.has_value() && roles_after.value().empty(), "Alice has no roles");
    
    TEST("22. Revoke role from group");
    auto revoke_group_result = persistence.revoke_role_from_group(group1_id, "role_user");
    ASSERT(revoke_group_result.has_value(), "Revoked user role from developers group");
    
    auto group_roles_after = persistence.get_group_roles(group1_id);
    ASSERT(group_roles_after.has_value() && group_roles_after.value().empty(), "Group has no roles");
    
    // === CLEANUP TESTS ===
    
    TEST("23. Delete group");
    auto delete_group_result = persistence.delete_group(group2_id);
    ASSERT(delete_group_result.has_value(), "Deleted admins group");
    
    auto list_after_delete = persistence.list_groups(10, 0);
    ASSERT(list_after_delete.has_value() && list_after_delete.value().size() == 1, "1 group remaining");
    
    TEST("24. Delete user");
    auto delete_user_result = persistence.delete_user(user2_id);
    ASSERT(delete_user_result.has_value(), "Deleted bob");
    
    auto users_after_delete = persistence.list_users(10, 0);
    ASSERT(users_after_delete.has_value() && users_after_delete.value().size() == 1, "1 user remaining");
    
    TEST("25. Close database");
    auto close_result = persistence.close();
    ASSERT(close_result.has_value(), "Database closed");
    
    std::cout << "\n=== ALL 25 TESTS PASSED ✓ ===" << std::endl;
    std::cout << "\nDatabase: " << test_dir << "/system.db" << std::endl;
    
    return 0;
}
