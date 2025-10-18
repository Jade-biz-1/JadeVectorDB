#include <gtest/gtest.h>

// Simple test to verify that the coverage system works
// This test is intentionally basic to ensure the coverage infrastructure is working

class CoverageTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up for tests if needed
    }

    void TearDown() override {
        // Clean up after tests if needed
    }
};

// A simple test case to verify coverage measurement
TEST_F(CoverageTest, BasicFunctionalityTest) {
    // Just testing a simple calculation to ensure coverage is measured
    int a = 5;
    int b = 3;
    int result = a + b;
    
    EXPECT_EQ(result, 8);
    EXPECT_GT(result, 5);
    EXPECT_LT(result, 10);
}

// Another simple test to demonstrate different code paths
TEST_F(CoverageTest, ConditionalLogicTest) {
    int value = 10;
    
    if (value > 5) {
        value += 5;
    } else {
        value -= 5;
    }
    
    EXPECT_EQ(value, 15);
    
    // Test the else branch by changing the value
    int smallValue = 3;
    if (smallValue > 5) {
        smallValue += 5;
    } else {
        smallValue -= 5;
    }
    
    EXPECT_EQ(smallValue, -2);
}