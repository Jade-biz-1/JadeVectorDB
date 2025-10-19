// Test suite for consistency properties
// Using the property-based testing framework

#include <gtest/gtest.h>
#include <vector>
#include <random>

#include "../property-tests/framework/property_test_framework.h"
#include "../property-tests/consistency/consistency_properties.h"

namespace property_tests {
namespace consistency {

// Test fixture for consistency property tests
class ConsistencyPropertyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common resources for the tests
    }
    
    void TearDown() override {
        // Clean up any resources after tests
    }
};

// Test the atomicity property
TEST_F(ConsistencyPropertyTest, Atomicity) {
    TransactionGenerator gen(5, 32);  // Max 5 operations, max 32 dimensions
    PropertyTest<Transaction> test(
        "Atomicity Property",
        atomicity_property,
        &gen,
        30  // Run 30 tests (transactions)
    );
    
    test.run();
}

// Test the consistency property (vector validity)
TEST_F(ConsistencyPropertyTest, VectorConsistency) {
    ArbitraryVectorGenerator vector_gen(2, 100);  // 2-100 dimensions
    PropertyTest<std::vector<float>> test(
        "Vector Consistency Property",
        consistency_property,
        &vector_gen,
        50
    );
    
    test.run();
}

// Test the vector integrity property
TEST_F(ConsistencyPropertyTest, VectorIntegrity) {
    // Custom test for vector pairs
    std::mt19937 rng(std::random_device{}());
    int num_tests = 50;
    ArbitraryVectorGenerator gen(5, 20);
    
    for (int i = 0; i < num_tests; ++i) {
        auto original = gen.generate(rng);
        auto result = original;  // Same vector for integrity check
        
        bool result_prop = vector_integrity_property(original, result);
        
        if (!result_prop) {
            ADD_FAILURE() << "Vector integrity property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Vector integrity property passed " << num_tests << " tests";
}

// Test the size consistency property
TEST_F(ConsistencyPropertyTest, SizeConsistency) {
    // Custom test for database state pairs before/after transaction
    std::mt19937 rng(std::random_device{}());
    int num_tests = 30;
    DatabaseStateGenerator state_gen(20, 32);  // Max 20 vectors, max 32 dimensions
    TransactionGenerator tx_gen(5, 32);       // Max 5 operations, max 32 dimensions
    
    for (int i = 0; i < num_tests; ++i) {
        auto before_state = state_gen.generate(rng);
        auto transaction = tx_gen.generate(rng);
        
        // Apply transaction to get after state (simplified simulation)
        auto after_state = before_state;  // In reality, we'd apply the transaction operations
        // For this test, we'll just check that the property accepts valid inputs
        
        bool result = size_consistency_property(before_state, after_state, transaction);
        
        if (!result) {
            ADD_FAILURE() << "Size consistency property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Size consistency property passed " << num_tests << " tests";
}

} // namespace consistency
} // namespace property_tests