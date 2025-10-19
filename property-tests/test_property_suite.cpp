// Main property-based testing suite for JadeVectorDB
// Combines all property tests into a comprehensive test suite

#include <gtest/gtest.h>

// Include all test files
#include "vector-space/test_vector_space_properties.cpp"
#include "consistency/test_consistency_properties.cpp" 
#include "concurrency/test_concurrency_properties.cpp"
#include "distributed-system/test_distributed_invariants.cpp"

// This file acts as a combined test suite that will run all property-based tests
// when compiled and executed with Google Test

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  
  // Run all tests
  return RUN_ALL_TESTS();
}