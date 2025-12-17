#include "src/services/sqlite_persistence_layer.h"
#include "src/models/auth.h"
#include <iostream>
#include <cassert>
#include <cstdlib>

using namespace jadevectordb;

#define TEST(name) std::cout << "\n" << name << "..." << std::endl
#define ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "  ❌ FAILED: " << message << std::endl; \
        return 1; \
    } \
    std::cout << "  ✓ " << message << std::endl

int main() {
    std::cout << "=== Sprint 1.4 Persistence Test ===" << std::endl;
    
    // Use temporary directory for test database
    std::string test_dir = "/tmp/jadevectordb_test_sprint14";
    system(("rm -rf " + test_dir).c_str());
    system(("mkdir -p " + test_dir).c_str());
    
    SQLitePersistenceLayer persistence(test_dir);
    
    TEST("1. Initialize database");
    auto init_result = persistence.initialize();
    ASSERT(init_result.has_value(), "Database initialized");
    
    // Create test users for ownership
    auto user1_result = persistence.create_user("alice", "alice@test.com", "hash1", "salt1");
    ASSERT(user1_result.has_value(), "Created user alice");
    std::string user1_id = user1_result.value();
    
    auto user2_result = persistence.create_user("bob", "bob@test.com", "hash2", "salt2");
    ASSERT(user2_result.has_value(), "Created user bob");
    std::string user2_id = user2_result.value();
    
    // === DATABASE METADATA TESTS ===
    
    TEST("2. Store database metadata");
    auto db_result = persistence.store_database_metadata(
        "test_vectors_db",
        "Test database for vector embeddings",
        user1_id,
        384,
        "HNSW",
        "{\"max_connections\": 16}"
    );
    if (!db_result.has_value()) {
        std::cerr << "  Error: " << db_result.error().message << std::endl;
    }
    ASSERT(db_result.has_value(), "Stored database metadata");
    std::string db_id = db_result.value();
    ASSERT(!db_id.empty(), "Database ID generated");
    
    TEST("3. Get database metadata");
    auto get_db_result = persistence.get_database_metadata(db_id);
    ASSERT(get_db_result.has_value(), "Retrieved database metadata");
    ASSERT(get_db_result.value().name == "test_vectors_db", "Database name correct");
    ASSERT(get_db_result.value().owner_user_id == user1_id, "Owner correct");
    ASSERT(get_db_result.value().vector_dimension == 384, "Dimension correct");
    ASSERT(get_db_result.value().index_type == "HNSW", "Index type correct");
    ASSERT(get_db_result.value().vector_count == 0, "Initial vector count is 0");
    ASSERT(get_db_result.value().index_count == 0, "Initial index count is 0");
    
    TEST("4. Create another database");
    auto db2_result = persistence.store_database_metadata(
        "semantic_search_db",
        "Database for semantic search",
        user2_id,
        768,
        "IVF",
        "{\"nlist\": 100}"
    );
    ASSERT(db2_result.has_value(), "Stored second database");
    std::string db2_id = db2_result.value();
    
    TEST("5. List database metadata");
    auto list_result = persistence.list_database_metadata(10, 0);
    ASSERT(list_result.has_value(), "Listed databases");
    ASSERT(list_result.value().size() == 2, "Found 2 databases");
    
    TEST("6. Update database stats");
    auto update_stats_result = persistence.update_database_stats(db_id, 1000, 5);
    ASSERT(update_stats_result.has_value(), "Updated database stats");
    
    auto updated_db = persistence.get_database_metadata(db_id);
    ASSERT(updated_db.has_value(), "Retrieved updated database");
    ASSERT(updated_db.value().vector_count == 1000, "Vector count updated");
    ASSERT(updated_db.value().index_count == 5, "Index count updated");
    
    TEST("7. Update database metadata");
    DatabaseMetadata updated_metadata = get_db_result.value();
    updated_metadata.description = "Updated description for test database";
    updated_metadata.metadata = "{\"max_connections\": 32}";
    
    auto update_result = persistence.update_database_metadata(db_id, updated_metadata);
    ASSERT(update_result.has_value(), "Updated database metadata");
    
    auto verify_update = persistence.get_database_metadata(db_id);
    ASSERT(verify_update.has_value(), "Retrieved updated metadata");
    ASSERT(verify_update.value().description == "Updated description for test database", "Description updated");
    
    TEST("8. Check database existence helper");
    auto exists_result = persistence.database_name_exists("test_vectors_db");
    ASSERT(exists_result.has_value(), "Checked database existence");
    ASSERT(exists_result.value() == true, "Database exists");
    
    auto not_exists = persistence.database_name_exists("nonexistent_db");
    ASSERT(not_exists.has_value(), "Checked non-existent database");
    ASSERT(not_exists.value() == false, "Database doesn't exist");
    
    // === AUDIT LOGGING TESTS ===
    
    TEST("9. Log audit event - database creation");
    auto audit1 = persistence.log_audit_event(
        user1_id,
        "create",
        "database",
        db_id,
        "192.168.1.100",
        true,
        "Created database: test_vectors_db"
    );
    ASSERT(audit1.has_value(), "Logged database creation event");
    
    TEST("10. Log audit event - login");
    auto audit2 = persistence.log_audit_event(
        user2_id,
        "login",
        "user",
        user2_id,
        "192.168.1.101",
        true,
        "User logged in successfully"
    );
    ASSERT(audit2.has_value(), "Logged login event");
    
    TEST("11. Log audit event - failed operation");
    auto audit3 = persistence.log_audit_event(
        user2_id,
        "delete",
        "database",
        db_id,
        "192.168.1.101",
        false,
        "Permission denied: user does not own database"
    );
    ASSERT(audit3.has_value(), "Logged failed delete event");
    
    TEST("12. Log audit event - vector store");
    auto audit4 = persistence.log_audit_event(
        user1_id,
        "store",
        "vector",
        "vec_12345",
        "192.168.1.100",
        true,
        "Stored vector in database " + db_id
    );
    ASSERT(audit4.has_value(), "Logged vector store event");
    
    TEST("13. Log audit event - search");
    auto audit5 = persistence.log_audit_event(
        user2_id,
        "search",
        "vector",
        db2_id,
        "192.168.1.101",
        true,
        "Performed similarity search, k=10"
    );
    ASSERT(audit5.has_value(), "Logged search event");
    
    TEST("14. Get all audit logs");
    auto all_logs = persistence.get_audit_logs(100, 0);
    ASSERT(all_logs.has_value(), "Retrieved all audit logs");
    ASSERT(all_logs.value().size() == 5, "Found 5 audit log entries");
    
    // Verify logs are in descending timestamp order (most recent first)
    if (all_logs.value().size() >= 2) {
        ASSERT(all_logs.value()[0].timestamp >= all_logs.value()[1].timestamp, 
               "Logs ordered by timestamp descending");
    }
    
    TEST("15. Get audit logs with limit and offset");
    auto limited_logs = persistence.get_audit_logs(2, 0);
    ASSERT(limited_logs.has_value(), "Retrieved limited audit logs");
    ASSERT(limited_logs.value().size() == 2, "Returned 2 entries");
    
    auto offset_logs = persistence.get_audit_logs(2, 2);
    ASSERT(offset_logs.has_value(), "Retrieved offset audit logs");
    ASSERT(offset_logs.value().size() == 2, "Returned 2 entries with offset");
    
    TEST("16. Get audit logs filtered by user");
    auto user1_logs = persistence.get_audit_logs(100, 0, user1_id);
    ASSERT(user1_logs.has_value(), "Retrieved user1 audit logs");
    ASSERT(user1_logs.value().size() == 2, "User1 has 2 log entries");
    
    // Verify all logs belong to user1
    for (const auto& log : user1_logs.value()) {
        ASSERT(log.user_id == user1_id, "Log belongs to user1");
    }
    
    TEST("17. Get audit logs filtered by action");
    auto login_logs = persistence.get_audit_logs(100, 0, "", "login");
    ASSERT(login_logs.has_value(), "Retrieved login audit logs");
    ASSERT(login_logs.value().size() == 1, "Found 1 login event");
    ASSERT(login_logs.value()[0].action == "login", "Action is login");
    
    TEST("18. Get audit logs filtered by user and action");
    auto user2_login = persistence.get_audit_logs(100, 0, user2_id, "login");
    ASSERT(user2_login.has_value(), "Retrieved user2 login logs");
    ASSERT(user2_login.value().size() == 1, "User2 has 1 login");
    ASSERT(user2_login.value()[0].user_id == user2_id, "Belongs to user2");
    ASSERT(user2_login.value()[0].action == "login", "Action is login");
    
    TEST("19. Verify audit log details");
    auto detail_logs = persistence.get_audit_logs(100, 0, "", "delete");
    ASSERT(detail_logs.has_value(), "Retrieved delete logs");
    ASSERT(detail_logs.value().size() == 1, "Found 1 delete attempt");
    ASSERT(detail_logs.value()[0].success == false, "Delete failed");
    ASSERT(!detail_logs.value()[0].details.empty(), "Has failure details");
    ASSERT(detail_logs.value()[0].resource_type == "database", "Resource type is database");
    
    TEST("20. Verify audit log for successful operations");
    auto create_logs = persistence.get_audit_logs(100, 0, "", "create");
    ASSERT(create_logs.has_value(), "Retrieved create logs");
    ASSERT(create_logs.value().size() == 1, "Found 1 create event");
    ASSERT(create_logs.value()[0].success == true, "Create succeeded");
    ASSERT(create_logs.value()[0].resource_type == "database", "Created a database");
    ASSERT(create_logs.value()[0].resource_id == db_id, "Correct database ID");
    
    // === INTEGRATION TESTS ===
    
    TEST("21. Delete database metadata");
    auto delete_result = persistence.delete_database_metadata(db2_id);
    ASSERT(delete_result.has_value(), "Deleted database metadata");
    
    auto verify_delete = persistence.get_database_metadata(db2_id);
    ASSERT(!verify_delete.has_value(), "Database metadata no longer exists");
    
    auto remaining_dbs = persistence.list_database_metadata(10, 0);
    ASSERT(remaining_dbs.has_value(), "Listed remaining databases");
    ASSERT(remaining_dbs.value().size() == 1, "Only 1 database remains");
    
    TEST("22. Transaction support with metadata");
    auto begin_result = persistence.begin_transaction();
    ASSERT(begin_result.has_value(), "Started transaction");
    
    auto db3_result = persistence.store_database_metadata(
        "transaction_test_db",
        "Test database for transactions",
        user1_id,
        512,
        "FLAT",
        "{}"
    );
    ASSERT(db3_result.has_value(), "Created database in transaction");
    std::string db3_id = db3_result.value();
    
    auto rollback_result = persistence.rollback_transaction();
    ASSERT(rollback_result.has_value(), "Rolled back transaction");
    
    auto check_rollback = persistence.get_database_metadata(db3_id);
    ASSERT(!check_rollback.has_value(), "Database rolled back (doesn't exist)");
    
    TEST("23. Transaction commit with audit log");
    auto begin2 = persistence.begin_transaction();
    ASSERT(begin2.has_value(), "Started second transaction");
    
    auto db4_result = persistence.store_database_metadata(
        "committed_db",
        "Database that will be committed",
        user1_id,
        256,
        "HNSW",
        "{}"
    );
    ASSERT(db4_result.has_value(), "Created database in transaction");
    std::string db4_id = db4_result.value();
    
    auto audit_in_transaction = persistence.log_audit_event(
        user1_id,
        "create",
        "database",
        db4_id,
        "192.168.1.100",
        true,
        "Created in transaction"
    );
    ASSERT(audit_in_transaction.has_value(), "Logged audit in transaction");
    
    auto commit_result = persistence.commit_transaction();
    ASSERT(commit_result.has_value(), "Committed transaction");
    
    auto verify_commit = persistence.get_database_metadata(db4_id);
    ASSERT(verify_commit.has_value(), "Database committed (exists)");
    ASSERT(verify_commit.value().name == "committed_db", "Correct database name");
    
    TEST("24. Close database");
    auto close_result = persistence.close();
    ASSERT(close_result.has_value(), "Database closed");
    
    std::cout << "\n=== ALL 24 TESTS PASSED ✓ ===" << std::endl;
    std::cout << "\nDatabase: " << test_dir << "/system.db" << std::endl;
    
    return 0;
}
