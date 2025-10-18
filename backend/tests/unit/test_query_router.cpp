#include <gtest/gtest.h>
#include <memory>

// Include the headers we want to test
#include "services/query_router.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test fixture for Query_router
class Query_routerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the service
        service_ = std::make_unique<Query_router>();
    }
    
    void TearDown() override {
        // Clean up
        service_.reset();
    }
    
    std::unique_ptr<Query_router> service_;
};

// Test that the service initializes correctly
TEST_F(Query_routerTest, InitializeService) {
    EXPECT_NE(service_, nullptr);
}

// Test that the service can be initialized successfully
TEST_F(Query_routerTest, ServiceInitialization) {
    auto result = service_->initialize();
    EXPECT_TRUE(result.has_value());
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
