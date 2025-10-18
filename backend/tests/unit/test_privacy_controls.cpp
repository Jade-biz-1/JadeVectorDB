#include <gtest/gtest.h>
#include <memory>

// Include the headers we want to test
#include "services/privacy_controls.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test fixture for Privacy_controls
class Privacy_controlsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the service
        service_ = std::make_unique<Privacy_controls>();
    }
    
    void TearDown() override {
        // Clean up
        service_.reset();
    }
    
    std::unique_ptr<Privacy_controls> service_;
};

// Test that the service initializes correctly
TEST_F(Privacy_controlsTest, InitializeService) {
    EXPECT_NE(service_, nullptr);
}

// Test that the service can be initialized successfully
TEST_F(Privacy_controlsTest, ServiceInitialization) {
    auto result = service_->initialize();
    EXPECT_TRUE(result.has_value());
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
