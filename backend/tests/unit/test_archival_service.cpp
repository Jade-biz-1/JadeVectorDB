#include <gtest/gtest.h>
#include <memory>

// Include the headers we want to test
#include "services/archival_service.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test fixture for Archival_service
class Archival_serviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the service
        service_ = std::make_unique<Archival_service>();
    }
    
    void TearDown() override {
        // Clean up
        service_.reset();
    }
    
    std::unique_ptr<Archival_service> service_;
};

// Test that the service initializes correctly
TEST_F(Archival_serviceTest, InitializeService) {
    EXPECT_NE(service_, nullptr);
}

// Test that the service can be initialized successfully
TEST_F(Archival_serviceTest, ServiceInitialization) {
    auto result = service_->initialize();
    EXPECT_TRUE(result.has_value());
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
