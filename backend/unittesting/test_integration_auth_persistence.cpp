#include "src/services/sqlite_persistence_layer.h"
#include "src/models/auth.h"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <thread>
#include <chrono>

using namespace jadevectordb;

#define TEST(name) std::cout << "\n" << name << "..." << std::endl
#define ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "  ❌ FAILED: " << message << std::endl; \
        return 1; \
    } \
    std::cout << "  ✓ " << message << std::endl

int main() {
    std::cout << "=== Sprint 1.5 Integration Test: Authentication + Persistence ===" << std::endl;
    
    // Use temporary directory for test database
    std::string test_dir = "/tmp/jadevectordb_test_integration";
    system(("rm -rf " + test_dir).c_str());
    system(("mkdir -p " + test_dir).c_str());
    
    // ===================================================================
    // PART 1: Direct Persistence Layer Testing (Already Working)
    // ===================================================================
    
    TEST("1. Initialize SQLitePersistenceLayer");
    SQLitePersistenceLayer persistence(test_dir);
    auto init_result = persistence.initialize();
    ASSERT(init_result.has_value(), "Persistence layer initialized");
    
    TEST("2. Register user via persistence");
    auto user_result = persistence.create_user("alice", "alice@test.com", "hash123", "salt123");
    ASSERT(user_result.has_value(), "User created via persistence");
    std::string alice_id = user_result.value();
    
    TEST("3. Retrieve user via persistence");
    auto get_user_result = persistence.get_user(alice_id);
    ASSERT(get_user_result.has_value(), "Retrieved user");
    ASSERT(get_user_result.value().username == "alice", "Username correct");
    ASSERT(get_user_result.value().email == "alice@test.com", "Email correct");
    
    TEST("4. Assign role to user");
    auto role_result = persistence.assign_role_to_user(alice_id, "role_user");
    ASSERT(role_result.has_value(), "Assigned role to user");
    
    TEST("5. Check user roles");
    auto roles_result = persistence.get_user_roles(alice_id);
    ASSERT(roles_result.has_value(), "Retrieved user roles");
    ASSERT(roles_result.value().size() == 1, "User has 1 role");
    ASSERT(roles_result.value()[0] == "role_user", "Role is 'user'");
    
    TEST("6. Create database metadata");
    auto db_result = persistence.store_database_metadata(
        "vectors_db",
        "Test database",
        alice_id,
        384,
        "HNSW",
        "{}"
    );
    ASSERT(db_result.has_value(), "Database metadata created");
    std::string db_id = db_result.value();
    
    TEST("7. Grant database permission");
    auto perm_result = persistence.grant_database_permission(
        db_id,
        "user",
        alice_id,
        "perm_db_read",
        alice_id  // granted_by must be a valid user_id
    );
    if (!perm_result.has_value()) {
        std::cerr << "  Error: " << perm_result.error().message << std::endl;
    }
    ASSERT(perm_result.has_value(), "Permission granted");
    
    TEST("8. Check database permission");
    auto check_perm = persistence.check_database_permission(db_id, alice_id, "database:read");
    ASSERT(check_perm.has_value(), "Permission check executed");
    ASSERT(check_perm.value() == true, "User has read permission");
    
    TEST("9. Create API key");
    int64_t expires_at = std::time(nullptr) + 86400; // 24 hours
    auto api_key_result = persistence.create_api_key(
        alice_id,
        "hash_of_key",
        "My API Key",
        "jvdb_abc",
        {"read", "write"},
        expires_at
    );
    ASSERT(api_key_result.has_value(), "API key created");
    std::string api_key_id = api_key_result.value();
    
    TEST("10. Retrieve API key");
    auto get_key_result = persistence.get_api_key_by_id(api_key_id);
    ASSERT(get_key_result.has_value(), "Retrieved API key");
    ASSERT(get_key_result.value().key_name == "My API Key", "API key name correct");
    ASSERT(get_key_result.value().is_active == true, "API key is active");
    
    TEST("11. Create auth token");
    int64_t token_expires = std::time(nullptr) + 3600; // 1 hour
    auto token_result = persistence.create_auth_token(
        alice_id,
        "token_hash_123",
        "192.168.1.100",
        "Mozilla/5.0",
        token_expires
    );
    ASSERT(token_result.has_value(), "Auth token created");
    std::string token_id = token_result.value();
    
    TEST("12. Create session");
    int64_t session_expires = std::time(nullptr) + 7200; // 2 hours
    auto session_result = persistence.create_session(
        alice_id,
        token_id,
        "192.168.1.100",
        session_expires
    );
    ASSERT(session_result.has_value(), "Session created");
    std::string session_id = session_result.value();
    
    TEST("13. Update session activity");
    auto update_result = persistence.update_session_activity(session_id);
    ASSERT(update_result.has_value(), "Session activity updated");
    
    TEST("14. Log audit events");
    auto audit1 = persistence.log_audit_event(
        alice_id, "login", "user", alice_id, "192.168.1.100", true, "Successful login"
    );
    ASSERT(audit1.has_value(), "Login event logged");
    
    auto audit2 = persistence.log_audit_event(
        alice_id, "create", "database", db_id, "192.168.1.100", true, "Created database"
    );
    ASSERT(audit2.has_value(), "Database creation event logged");
    
    TEST("15. Query audit logs");
    auto logs = persistence.get_audit_logs(100, 0, alice_id);
    ASSERT(logs.has_value(), "Retrieved audit logs");
    ASSERT(logs.value().size() == 2, "Found 2 audit entries");
    
    // ===================================================================
    // PART 2: Persistence Across Restart Simulation
    // ===================================================================
    
    TEST("16. Close and reopen persistence layer");
    auto close_result = persistence.close();
    ASSERT(close_result.has_value(), "Closed persistence layer");
    
    SQLitePersistenceLayer persistence2(test_dir);
    auto init2_result = persistence2.initialize();
    ASSERT(init2_result.has_value(), "Reopened persistence layer");
    
    TEST("17. Verify user persisted");
    auto verify_user = persistence2.get_user(alice_id);
    ASSERT(verify_user.has_value(), "User still exists after restart");
    ASSERT(verify_user.value().username == "alice", "Username persisted");
    ASSERT(verify_user.value().email == "alice@test.com", "Email persisted");
    
    TEST("18. Verify roles persisted");
    auto verify_roles = persistence2.get_user_roles(alice_id);
    ASSERT(verify_roles.has_value(), "Roles retrieved after restart");
    ASSERT(verify_roles.value().size() == 1, "Role count persisted");
    ASSERT(verify_roles.value()[0] == "role_user", "Role value persisted");
    
    TEST("19. Verify database metadata persisted");
    auto verify_db = persistence2.get_database_metadata(db_id);
    ASSERT(verify_db.has_value(), "Database metadata exists after restart");
    ASSERT(verify_db.value().name == "vectors_db", "Database name persisted");
    ASSERT(verify_db.value().owner_user_id == alice_id, "Database owner persisted");
    ASSERT(verify_db.value().vector_dimension == 384, "Vector dimension persisted");
    
    TEST("20. Verify permissions persisted");
    auto verify_perm = persistence2.check_database_permission(db_id, alice_id, "database:read");
    ASSERT(verify_perm.has_value(), "Permission check after restart");
    ASSERT(verify_perm.value() == true, "Permission persisted");
    
    TEST("21. Verify API key persisted");
    auto verify_key = persistence2.get_api_key_by_id(api_key_id);
    ASSERT(verify_key.has_value(), "API key exists after restart");
    ASSERT(verify_key.value().key_name == "My API Key", "API key name persisted");
    ASSERT(verify_key.value().is_active == true, "API key active status persisted");
    
    TEST("22. Verify session persisted");
    auto verify_session = persistence2.get_session(session_id);
    ASSERT(verify_session.has_value(), "Session exists after restart");
    ASSERT(verify_session.value().user_id == alice_id, "Session user persisted");
    ASSERT(verify_session.value().is_active == true, "Session active status persisted");
    
    TEST("23. Verify audit logs persisted");
    auto verify_logs = persistence2.get_audit_logs(100, 0);
    ASSERT(verify_logs.has_value(), "Audit logs retrieved after restart");
    ASSERT(verify_logs.value().size() >= 2, "Audit logs persisted");
    
    // ===================================================================
    // PART 3: Transaction Testing
    // ===================================================================
    
    TEST("24. Transaction rollback test");
    auto begin_tx = persistence2.begin_transaction();
    ASSERT(begin_tx.has_value(), "Transaction started");
    
    auto new_user = persistence2.create_user("bob", "bob@test.com", "hash456", "salt456");
    ASSERT(new_user.has_value(), "User created in transaction");
    std::string bob_id = new_user.value();
    
    auto rollback_tx = persistence2.rollback_transaction();
    ASSERT(rollback_tx.has_value(), "Transaction rolled back");
    
    auto verify_rollback = persistence2.get_user(bob_id);
    ASSERT(!verify_rollback.has_value(), "User rolled back (doesn't exist)");
    
    TEST("25. Transaction commit test");
    auto begin_tx2 = persistence2.begin_transaction();
    ASSERT(begin_tx2.has_value(), "Second transaction started");
    
    auto new_user2 = persistence2.create_user("carol", "carol@test.com", "hash789", "salt789");
    ASSERT(new_user2.has_value(), "User created in second transaction");
    std::string carol_id = new_user2.value();
    
    auto commit_tx = persistence2.commit_transaction();
    ASSERT(commit_tx.has_value(), "Transaction committed");
    
    auto verify_commit = persistence2.get_user(carol_id);
    ASSERT(verify_commit.has_value(), "User committed (exists)");
    ASSERT(verify_commit.value().username == "carol", "Committed user data correct");
    
    // ===================================================================
    // PART 4: Concurrent Access Testing
    // ===================================================================
    
    TEST("26. Concurrent user creation");
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&persistence2, &success_count, i]() {
            std::string username = "user_" + std::to_string(i);
            std::string email = username + "@test.com";
            auto result = persistence2.create_user(username, email, "hash", "salt");
            if (result.has_value()) {
                success_count++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    ASSERT(success_count == 10, "All concurrent user creations succeeded");
    
    TEST("27. List all users");
    auto all_users = persistence2.list_users(100, 0);
    ASSERT(all_users.has_value(), "Listed all users");
    ASSERT(all_users.value().size() >= 12, "Found at least 12 users (alice, carol, user_0-9)");
    
    TEST("28. Close database");
    auto final_close = persistence2.close();
    ASSERT(final_close.has_value(), "Database closed");
    
    std::cout << "\n=== ALL 28 INTEGRATION TESTS PASSED ✓ ===" << std::endl;
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "  ✓ Persistence layer CRUD operations" << std::endl;
    std::cout << "  ✓ Data survives restart simulation" << std::endl;
    std::cout << "  ✓ Transaction support (rollback & commit)" << std::endl;
    std::cout << "  ✓ Concurrent access handling" << std::endl;
    std::cout << "\nDatabase: " << test_dir << "/system.db" << std::endl;
    
    return 0;
}
