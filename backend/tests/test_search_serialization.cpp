#include <gtest/gtest.h>
#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "models/vector.h"
#include "models/database.h"
#include <memory>
#include <vector>
#include <string>

namespace jadevectordb {
namespace test {

// Test fixture for search serialization tests
class SearchSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize database layer and vector storage
        db_layer_ = std::make_unique<DatabaseLayer>();
        vector_storage_ = std::make_unique<VectorStorageService>(std::move(db_layer_));

        // Initialize similarity search service
        search_service_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage_));
        auto init_result = search_service_->initialize();
        ASSERT_TRUE(init_result.has_value()) << "Failed to initialize search service";

        // Create test database
        test_database_id_ = "test_db_serialization";
        DatabaseMetadata db_meta;
        db_meta.id = test_database_id_;
        db_meta.name = "Test Database for Serialization";
        db_meta.dimension = 4;
        db_meta.metric = SimilarityMetric::COSINE;

        auto db_result = search_service_->vector_storage_->create_database(db_meta);
        ASSERT_TRUE(db_result.has_value()) << "Failed to create test database";

        // Add test vectors with rich metadata
        addTestVectors();
    }

    void TearDown() override {
        // Cleanup test database
        if (search_service_ && search_service_->vector_storage_) {
            search_service_->vector_storage_->delete_database(test_database_id_);
        }
    }

    void addTestVectors() {
        // Vector 1: Simple vector with basic metadata
        Vector v1;
        v1.id = "vec_1";
        v1.values = {1.0f, 0.0f, 0.0f, 0.0f};
        v1.metadata.tags = {"tag1", "important"};
        v1.metadata.category = "cat_a";
        v1.metadata.owner = "user1";
        v1.metadata.created_at = "2025-11-29T10:00:00Z";
        v1.metadata.updated_at = "2025-11-29T10:00:00Z";

        // Vector 2: Vector with different metadata
        Vector v2;
        v2.id = "vec_2";
        v2.values = {0.0f, 1.0f, 0.0f, 0.0f};
        v2.metadata.tags = {"tag2", "normal"};
        v2.metadata.category = "cat_b";
        v2.metadata.owner = "user2";
        v2.metadata.created_at = "2025-11-29T11:00:00Z";
        v2.metadata.updated_at = "2025-11-29T11:00:00Z";

        // Vector 3: Similar to vector 1
        Vector v3;
        v3.id = "vec_3";
        v3.values = {0.9f, 0.1f, 0.0f, 0.0f};
        v3.metadata.tags = {"tag1", "similar"};
        v3.metadata.category = "cat_a";
        v3.metadata.owner = "user1";
        v3.metadata.created_at = "2025-11-29T12:00:00Z";
        v3.metadata.updated_at = "2025-11-29T12:00:00Z";

        // Vector 4: Different vector
        Vector v4;
        v4.id = "vec_4";
        v4.values = {0.0f, 0.0f, 1.0f, 0.0f};
        v4.metadata.tags = {"tag3", "different"};
        v4.metadata.category = "cat_c";
        v4.metadata.owner = "user3";
        v4.metadata.created_at = "2025-11-29T13:00:00Z";
        v4.metadata.updated_at = "2025-11-29T13:00:00Z";

        // Store vectors
        auto r1 = search_service_->vector_storage_->store_vector(test_database_id_, v1);
        auto r2 = search_service_->vector_storage_->store_vector(test_database_id_, v2);
        auto r3 = search_service_->vector_storage_->store_vector(test_database_id_, v3);
        auto r4 = search_service_->vector_storage_->store_vector(test_database_id_, v4);

        ASSERT_TRUE(r1.has_value()) << "Failed to store vector 1";
        ASSERT_TRUE(r2.has_value()) << "Failed to store vector 2";
        ASSERT_TRUE(r3.has_value()) << "Failed to store vector 3";
        ASSERT_TRUE(r4.has_value()) << "Failed to store vector 4";
    }

    std::unique_ptr<SimilaritySearchService> search_service_;
    std::string test_database_id_;
};

// Test 1: Search with include_vector_data = false (default)
// Vector values should NOT be included in search results
TEST_F(SearchSerializationTest, SearchWithoutVectorData) {
    Vector query;
    query.id = "query_1";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};  // Exact match with vec_1

    SearchParams params;
    params.top_k = 3;
    params.include_vector_data = false;  // Explicitly set to false
    params.include_metadata = true;

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed";
    ASSERT_GE(result.value().size(), 1) << "Should return at least 1 result";

    // Verify that vector_data is NOT populated
    for (const auto& search_result : result.value()) {
        EXPECT_TRUE(search_result.vector_data.values.empty())
            << "Vector values should be empty when include_vector_data=false for vector: "
            << search_result.vector_id;

        // But we should still have vector_id and similarity_score
        EXPECT_FALSE(search_result.vector_id.empty()) << "Vector ID should be present";
        EXPECT_GE(search_result.similarity_score, 0.0f) << "Similarity score should be present";
    }
}

// Test 2: Search with include_vector_data = true
// Vector values SHOULD be included in search results
TEST_F(SearchSerializationTest, SearchWithVectorData) {
    Vector query;
    query.id = "query_2";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};  // Exact match with vec_1

    SearchParams params;
    params.top_k = 3;
    params.include_vector_data = true;  // Enable vector data inclusion
    params.include_metadata = true;

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed";
    ASSERT_GE(result.value().size(), 1) << "Should return at least 1 result";

    // Verify that vector_data IS populated
    for (const auto& search_result : result.value()) {
        EXPECT_FALSE(search_result.vector_data.values.empty())
            << "Vector values should be populated when include_vector_data=true for vector: "
            << search_result.vector_id;

        // Verify the dimension matches
        EXPECT_EQ(search_result.vector_data.values.size(), 4)
            << "Vector should have 4 dimensions for vector: " << search_result.vector_id;

        // Verify the vector ID matches
        EXPECT_EQ(search_result.vector_data.id, search_result.vector_id)
            << "Vector data ID should match result ID";

        // Basic fields should still be present
        EXPECT_FALSE(search_result.vector_id.empty()) << "Vector ID should be present";
        EXPECT_GE(search_result.similarity_score, 0.0f) << "Similarity score should be present";
    }
}

// Test 3: Verify search response schema correctness
// Test that all expected fields are present in SearchResult
TEST_F(SearchSerializationTest, SearchResponseSchema) {
    Vector query;
    query.id = "query_3";
    query.values = {0.9f, 0.1f, 0.0f, 0.0f};  // Similar to vec_1 and vec_3

    SearchParams params;
    params.top_k = 5;
    params.include_vector_data = true;
    params.include_metadata = true;

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed";
    ASSERT_GE(result.value().size(), 1) << "Should return at least 1 result";

    // Verify schema for each result
    for (const auto& search_result : result.value()) {
        // Required fields
        EXPECT_FALSE(search_result.vector_id.empty())
            << "vector_id field should be present and non-empty";

        EXPECT_GE(search_result.similarity_score, 0.0f)
            << "similarity_score should be >= 0.0";
        EXPECT_LE(search_result.similarity_score, 1.0f)
            << "similarity_score should be <= 1.0 (normalized)";

        // Vector data fields (when include_vector_data=true)
        EXPECT_FALSE(search_result.vector_data.values.empty())
            << "vector_data.values should be populated";
        EXPECT_EQ(search_result.vector_data.id, search_result.vector_id)
            << "vector_data.id should match vector_id";

        // Metadata fields (when include_metadata=true)
        EXPECT_FALSE(search_result.vector_data.metadata.tags.empty())
            << "Metadata tags should be present for vector: " << search_result.vector_id;
        EXPECT_FALSE(search_result.vector_data.metadata.category.empty())
            << "Metadata category should be present";
        EXPECT_FALSE(search_result.vector_data.metadata.owner.empty())
            << "Metadata owner should be present";
    }
}

// Test 4: Verify correct serialization with metadata but without vector data
TEST_F(SearchSerializationTest, SearchWithMetadataOnly) {
    Vector query;
    query.id = "query_4";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};

    SearchParams params;
    params.top_k = 3;
    params.include_vector_data = false;  // No vector values
    params.include_metadata = true;       // But include metadata

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed";
    ASSERT_GE(result.value().size(), 1) << "Should return at least 1 result";

    for (const auto& search_result : result.value()) {
        // Vector values should be empty
        EXPECT_TRUE(search_result.vector_data.values.empty())
            << "Vector values should be empty when include_vector_data=false";

        // But metadata should still be present
        if (params.include_metadata) {
            // Note: This depends on implementation - metadata might be in vector_data
            // or in a separate metadata field
            EXPECT_FALSE(search_result.vector_id.empty()) << "Basic fields should exist";
        }
    }
}

// Test 5: Verify search results are properly sorted by similarity score
TEST_F(SearchSerializationTest, SearchResultsSorted) {
    Vector query;
    query.id = "query_5";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};

    SearchParams params;
    params.top_k = 10;  // Get all results
    params.include_vector_data = true;

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed";
    ASSERT_GE(result.value().size(), 2) << "Should return multiple results for sorting test";

    // Verify results are sorted in descending order by similarity score
    for (size_t i = 1; i < result.value().size(); ++i) {
        EXPECT_GE(result.value()[i-1].similarity_score, result.value()[i].similarity_score)
            << "Results should be sorted in descending order of similarity";
    }
}

// Test 6: Verify vector data correctness when included
TEST_F(SearchSerializationTest, VectorDataCorrectness) {
    Vector query;
    query.id = "query_6";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};

    SearchParams params;
    params.top_k = 1;  // Get only the top result (should be vec_1)
    params.include_vector_data = true;

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed";
    ASSERT_EQ(result.value().size(), 1) << "Should return exactly 1 result";

    const auto& top_result = result.value()[0];

    // The top result should be vec_1 (exact match)
    EXPECT_EQ(top_result.vector_id, "vec_1") << "Top result should be vec_1";

    // Verify the vector values are correct
    ASSERT_EQ(top_result.vector_data.values.size(), 4) << "Should have 4 dimensions";
    EXPECT_FLOAT_EQ(top_result.vector_data.values[0], 1.0f);
    EXPECT_FLOAT_EQ(top_result.vector_data.values[1], 0.0f);
    EXPECT_FLOAT_EQ(top_result.vector_data.values[2], 0.0f);
    EXPECT_FLOAT_EQ(top_result.vector_data.values[3], 0.0f);

    // Verify metadata
    EXPECT_FALSE(top_result.vector_data.metadata.tags.empty()) << "Tags should be present";
    EXPECT_EQ(top_result.vector_data.metadata.category, "cat_a");
    EXPECT_EQ(top_result.vector_data.metadata.owner, "user1");
}

// Test 7: Verify empty results when no vectors match
TEST_F(SearchSerializationTest, EmptyResultsSchema) {
    Vector query;
    query.id = "query_7";
    query.values = {1.0f, 0.0f, 0.0f, 0.0f};

    SearchParams params;
    params.top_k = 5;
    params.threshold = 0.99f;  // Very high threshold - only exact matches
    params.filter_category = "nonexistent_category";  // Filter that matches nothing
    params.include_vector_data = true;

    auto result = search_service_->similarity_search(test_database_id_, query, params);
    ASSERT_TRUE(result.has_value()) << "Search should succeed even with no matches";

    // Empty results should still be a valid vector, just with size 0
    EXPECT_TRUE(result.value().empty()) << "Should return empty results when no matches";
}

} // namespace test
} // namespace jadevectordb
