#include <gtest/gtest.h>
#include <memory>

// Include the headers we want to test
#include "services/schema_validator.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test fixture for Schema_validator
class Schema_validatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the service
        service_ = std::make_unique<Schema_validator>();
    }
    
    void TearDown() override {
        // Clean up
        service_.reset();
    }
    
    std::unique_ptr<Schema_validator> service_;
};

// Test that the service initializes correctly
TEST_F(Schema_validatorTest, InitializeService) {
    EXPECT_NE(service_, nullptr);
}

// Test that the service can be initialized successfully
TEST_F(Schema_validatorTest, ServiceInitialization) {
    auto result = service_->initialize();
    EXPECT_TRUE(result.has_value());
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
