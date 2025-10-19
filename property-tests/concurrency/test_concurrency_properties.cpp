// Test suite for concurrency properties
// Using the property-based testing framework

#include <gtest/gtest.h>
#include <vector>
#include <random>

#include "../property-tests/framework/property_test_framework.h"
#include "../property-tests/concurrency/concurrency_properties.h"

namespace property_tests {
namespace concurrency {

// Test fixture for concurrency property tests
class ConcurrencyPropertyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common resources for the tests
    }
    
    void TearDown() override {
        // Clean up any resources after tests
    }
};

// Test the thread safety property
TEST_F(ConcurrencyPropertyTest, ThreadSafety) {
    ConcurrentOperationSequenceGenerator gen(15, 32);  // Max 15 operations, max 32 dimensions
    PropertyTest<std::vector<ConcurrentOperation>> test(
        "Thread Safety Property",
        thread_safety_property,
        &gen,
        20  // Run 20 tests (lower number due to concurrency overhead)
    );
    
    test.run();
}

// Test the atomic operation property
TEST_F(ConcurrencyPropertyTest, AtomicOperation) {
    ArbitraryVectorGenerator gen(5, 50);  // 5-50 dimensions
    PropertyTest<std::vector<float>> test(
        "Atomic Operation Property",
        atomic_operation_property,
        &gen,
        40
    );
    
    test.run();
}

// Test the read-write consistency property
TEST_F(ConcurrencyPropertyTest, ReadWriteConsistency) {
    // This property doesn't take parameters, so we'll run it directly
    int num_tests = 10;  // Lower number due to threading overhead
    
    for (int i = 0; i < num_tests; ++i) {
        bool result = read_write_consistency_property();
        
        if (!result) {
            ADD_FAILURE() << "Read-write consistency property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Read-write consistency property passed " << num_tests << " tests";
}

} // namespace concurrency
} // namespace property_tests