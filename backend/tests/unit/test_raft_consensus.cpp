#include <gtest/gtest.h>
#include <memory>

// Include the headers we want to test
#include "services/raft_consensus.h"
#include "lib/error_handling.h"

using namespace jadevectordb;

// Test fixture for Raft_consensus
class Raft_consensusTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the service
        service_ = std::make_unique<Raft_consensus>();
    }
    
    void TearDown() override {
        // Clean up
        service_.reset();
    }
    
    std::unique_ptr<Raft_consensus> service_;
};

// Test that the service initializes correctly
TEST_F(Raft_consensusTest, InitializeService) {
    EXPECT_NE(service_, nullptr);
}

// Test that the service can be initialized successfully
TEST_F(Raft_consensusTest, ServiceInitialization) {
    auto result = service_->initialize();
    EXPECT_TRUE(result.has_value());
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
