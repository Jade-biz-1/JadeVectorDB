// Test suite for vector space properties
// Using the property-based testing framework

#include <gtest/gtest.h>
#include <vector>
#include <random>

#include "../property-tests/framework/property_test_framework.h"
#include "../property-tests/vector-space/vector_space_properties.h"

namespace property_tests {
namespace vector_space {

// Test fixture for vector space property tests
class VectorSpacePropertyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common resources for the tests
    }
    
    void TearDown() override {
        // Clean up any resources after tests
    }
};

// Test the dimension consistency property
TEST_F(VectorSpacePropertyTest, DimensionConsistency) {
    // Test with vectors of dimension 10
    ArbitraryVectorGenerator gen(10, 10);  // Fixed dimension of 10
    PropertyTest<std::vector<float>> test(
        "Dimension Consistency Property",
        [](const std::vector<float>& v) { return dimension_consistency_property(v, 10); },
        &gen,
        50  // Run 50 tests
    );
    
    test.run();
}

// Test the norm bounds property (for normalized vectors)
TEST_F(VectorSpacePropertyTest, NormBounds) {
    NormalizedVectorGenerator gen(128);  // 128-dimensional normalized vectors
    PropertyTest<std::vector<float>> test(
        "Norm Bounds Property for Normalized Vectors",
        norm_bounds_property,
        &gen,
        50
    );
    
    test.run();
}

// Test the distance non-negativity property
TEST_F(VectorSpacePropertyTest, DistanceNonNegativity) {
    ArbitraryVectorGenerator gen(5, 20);  // Vectors between 5 and 20 dimensions
    auto gen_pair = [](std::mt19937& rng) -> std::pair<std::vector<float>, std::vector<float>> {
        ArbitraryVectorGenerator inner_gen(5, 20);
        auto v1 = inner_gen.generate(rng);
        auto v2 = inner_gen.generate(rng);
        // Ensure same dimension
        int target_dim = std::min(v1.size(), v2.size());
        v1.resize(target_dim);
        v2.resize(target_dim);
        return std::make_pair(v1, v2);
    };
    
    // Create a custom test for pairs of vectors
    std::mt19937 rng(std::random_device{}());
    int num_tests = 50;
    
    for (int i = 0; i < num_tests; ++i) {
        auto pair = gen_pair(rng);
        bool result = distance_non_negativity_property(pair.first, pair.second);
        
        if (!result) {
            ADD_FAILURE() << "Distance non-negativity property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Distance non-negativity property passed " << num_tests << " tests";
}

// Test the distance identity property
TEST_F(VectorSpacePropertyTest, DistanceIdentity) {
    ArbitraryVectorGenerator gen(5, 20);
    PropertyTest<std::vector<float>> test(
        "Distance Identity Property",
        distance_identity_direct_property,
        &gen,
        50
    );
    
    test.run();
}

// Test the distance symmetry property
TEST_F(VectorSpacePropertyTest, DistanceSymmetry) {
    // Since our framework doesn't directly support pair generation yet, 
    // we'll implement a custom test for this property
    std::mt19937 rng(std::random_device{}());
    int num_tests = 50;
    ArbitraryVectorGenerator gen(5, 20);
    
    for (int i = 0; i < num_tests; ++i) {
        auto v1 = gen.generate(rng);
        auto v2 = gen.generate(rng);
        
        // Ensure same dimension
        int target_dim = std::min(v1.size(), v2.size());
        v1.resize(target_dim);
        v2.resize(target_dim);
        
        bool result = distance_symmetry_property(v1, v2);
        
        if (!result) {
            ADD_FAILURE() << "Distance symmetry property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Distance symmetry property passed " << num_tests << " tests";
}

// Test the triangle inequality property
TEST_F(VectorSpacePropertyTest, TriangleInequality) {
    // Custom implementation for three vectors
    std::mt19937 rng(std::random_device{}());
    int num_tests = 30;  // Lower number for triplets
    ArbitraryVectorGenerator gen(5, 15);
    
    for (int i = 0; i < num_tests; ++i) {
        auto a = gen.generate(rng);
        auto b = gen.generate(rng);
        auto c = gen.generate(rng);
        
        // Ensure same dimension
        int target_dim = std::min({a.size(), b.size(), c.size()});
        a.resize(target_dim);
        b.resize(target_dim);
        c.resize(target_dim);
        
        bool result = triangle_inequality_property(a, b, c);
        
        if (!result) {
            ADD_FAILURE() << "Triangle inequality property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Triangle inequality property passed " << num_tests << " tests";
}

// Test the cosine similarity bounds property
TEST_F(VectorSpacePropertyTest, CosineSimilarityBounds) {
    // Custom implementation for pairs of vectors
    std::mt19937 rng(std::random_device{}());
    int num_tests = 50;
    ArbitraryVectorGenerator gen(5, 20);
    
    for (int i = 0; i < num_tests; ++i) {
        auto v1 = gen.generate(rng);
        auto v2 = gen.generate(rng);
        
        // Ensure same dimension
        int target_dim = std::min(v1.size(), v2.size());
        v1.resize(target_dim);
        v2.resize(target_dim);
        
        bool result = cosine_similarity_bounds_property(v1, v2);
        
        if (!result) {
            ADD_FAILURE() << "Cosine similarity bounds property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Cosine similarity bounds property passed " << num_tests << " tests";
}

// Test the linear combination property
TEST_F(VectorSpacePropertyTest, LinearCombination) {
    // Custom implementation for pairs of vectors
    std::mt19937 rng(std::random_device{}());
    int num_tests = 50;
    ArbitraryVectorGenerator gen(5, 20);
    
    for (int i = 0; i < num_tests; ++i) {
        auto v1 = gen.generate(rng);
        auto v2 = gen.generate(rng);
        
        // Ensure same dimension
        int target_dim = std::min(v1.size(), v2.size());
        v1.resize(target_dim);
        v2.resize(target_dim);
        
        bool result = linear_combination_property(v1, v2);
        
        if (!result) {
            ADD_FAILURE() << "Linear combination property failed on test " << (i+1);
            return;
        }
    }
    
    SUCCEED() << "Linear combination property passed " << num_tests << " tests";
}

} // namespace vector_space
} // namespace property_tests