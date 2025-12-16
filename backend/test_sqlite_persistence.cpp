// Test program for SQLite Persistence Layer
#include "services/sqlite_persistence_layer.h"
#include "models/auth.h"
#include "lib/logging.h"
#include <iostream>
#include <cstdio>

using namespace jadevectordb;

int main() {
    std::cout << "=== SQLite Persistence Layer Test ===" << std::endl;
    
    // Initialize logging system first
    logging::LoggerManager::initialize(logging::LogLevel::INFO);
    
    // Create test data directory
    std::string test_dir = "/tmp/jadevectordb_test";
    system(("mkdir -p " + test_dir).c_str());
    
    // Initialize persistence layer
    SQLitePersistenceLayer persistence(test_dir);
    
    std::cout << "\n1. Initializing SQLite database..." << std::endl;
    auto init_result = persistence.initialize();
    if (!init_result.has_value()) {
        std::cerr << "ERROR: Failed to initialize: " << init_result.error().message << std::endl;
        return 1;
    }
    std::cout << "✓ Database initialized successfully" << std::endl;
    
    // Test user creation
    std::cout << "\n2. Creating test user..." << std::endl;
    auto user_result = persistence.create_user(
        "testuser",
        "test@example.com",
        "hashed_password_here",
        "random_salt_here"
    );
    
    if (!user_result.has_value()) {
        std::cerr << "ERROR: Failed to create user: " << user_result.error().message << std::endl;
        return 1;
    }
    std::string user_id = user_result.value();
    std::cout << "✓ User created with ID: " << user_id << std::endl;
    
    // Test user retrieval
    std::cout << "\n3. Retrieving user..." << std::endl;
    auto get_result = persistence.get_user(user_id);
    if (!get_result.has_value()) {
        std::cerr << "ERROR: Failed to get user: " << get_result.error().message << std::endl;
        return 1;
    }
    User user = get_result.value();
    std::cout << "✓ User retrieved:" << std::endl;
    std::cout << "  - Username: " << user.username << std::endl;
    std::cout << "  - Email: " << user.email << std::endl;
    std::cout << "  - Is Active: " << (user.is_active ? "Yes" : "No") << std::endl;
    
    // Test user retrieval by username
    std::cout << "\n4. Retrieving user by username..." << std::endl;
    auto by_username_result = persistence.get_user_by_username("testuser");
    if (!by_username_result.has_value()) {
        std::cerr << "ERROR: Failed to get user by username: " << by_username_result.error().message << std::endl;
        return 1;
    }
    std::cout << "✓ User retrieved by username" << std::endl;
    
    // Test user existence check
    std::cout << "\n5. Checking if user exists..." << std::endl;
    auto exists_result = persistence.user_exists("testuser");
    if (!exists_result.has_value()) {
        std::cerr << "ERROR: Failed to check user existence: " << exists_result.error().message << std::endl;
        return 1;
    }
    std::cout << "✓ User exists: " << (exists_result.value() ? "Yes" : "No") << std::endl;
    
    // Test email existence check
    std::cout << "\n6. Checking if email exists..." << std::endl;
    auto email_exists_result = persistence.email_exists("test@example.com");
    if (!email_exists_result.has_value()) {
        std::cerr << "ERROR: Failed to check email existence: " << email_exists_result.error().message << std::endl;
        return 1;
    }
    std::cout << "✓ Email exists: " << (email_exists_result.value() ? "Yes" : "No") << std::endl;
    
    // Test duplicate user creation (should fail)
    std::cout << "\n7. Testing duplicate user creation (should fail)..." << std::endl;
    auto dup_result = persistence.create_user(
        "testuser",
        "test2@example.com",
        "hashed_password_here",
        "random_salt_here"
    );
    if (dup_result.has_value()) {
        std::cerr << "ERROR: Duplicate user creation should have failed!" << std::endl;
        return 1;
    }
    std::cout << "✓ Duplicate user creation correctly rejected: " << dup_result.error().message << std::endl;
    
    // Clean up
    std::cout << "\n8. Closing database..." << std::endl;
    auto close_result = persistence.close();
    if (!close_result.has_value()) {
        std::cerr << "ERROR: Failed to close: " << close_result.error().message << std::endl;
        return 1;
    }
    std::cout << "✓ Database closed successfully" << std::endl;
    
    std::cout << "\n=== ALL TESTS PASSED ✓ ===" << std::endl;
    std::cout << "\nDatabase file created at: " << test_dir << "/system.db" << std::endl;
    std::cout << "You can inspect it with: sqlite3 " << test_dir << "/system.db" << std::endl;
    
    return 0;
}
