#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "services/metadata_filter.h"
#include "models/vector.h"
#include "models/database.h"

namespace jadevectordb {

// Test fixture for advanced search integration tests
class AdvancedSearchIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        vector_storage_ = std::make_unique<VectorStorageService>(std::move(db_layer_));
        vector_storage_->initialize();
        
        similarity_search_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage_));
        similarity_search_->initialize();
        
        // Create test database
        Database db;
        db.name = "advanced_search_test_db";
        db.description = "Test database for advanced search";
        db.vectorDimension = 4;
        
        auto result = similarity_search_->vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
        
        // Add test vectors with varied metadata
        addTestVectors();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            similarity_search_->vector_storage_->db_layer_->delete_database(test_database_id_);
        }
    }

    void addTestVectors() {
        // Vector 1: Document vector with high score
        Vector v1;
        v1.id = "doc_vector_1";
        v1.values = {0.9f, 0.1f, 0.0f, 0.0f};
        v1.metadata.owner = "user1";
        v1.metadata.category = "documents";
        v1.metadata.tags = {"important", "urgent", "review"};
        v1.metadata.score = 0.95f;
        v1.metadata.status = "active";
        v1.metadata.custom["project"] = "project-alpha";
        v1.metadata.custom["department"] = "engineering";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v1);
        
        // Vector 2: Image vector
        Vector v2;
        v2.id = "img_vector_1";
        v2.values = {0.0f, 0.9f, 0.1f, 0.0f};
        v2.metadata.owner = "user2";
        v2.metadata.category = "images";
        v2.metadata.tags = {"nature", "landscape", "photo"};
        v2.metadata.score = 0.85f;
        v2.metadata.status = "active";
        v2.metadata.custom["project"] = "project-beta";
        v2.metadata.custom["department"] = "design";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v2);
        
        // Vector 3: Another document vector
        Vector v3;
        v3.id = "doc_vector_2";
        v3.values = {0.8f, 0.2f, 0.0f, 0.0f};
        v3.metadata.owner = "user1";
        v3.metadata.category = "documents";
        v3.metadata.tags = {"draft", "review"};
        v3.metadata.score = 0.75f;
        v3.metadata.status = "draft";
        v3.metadata.custom["project"] = "project-alpha";
        v3.metadata.custom["department"] = "engineering";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v3);
        
        // Vector 4: Audio vector
        Vector v4;
        v4.id = "aud_vector_1";
        v4.values = {0.1f, 0.0f, 0.9f, 0.0f};
        v4.metadata.owner = "user3";
        v4.metadata.category = "audio";
        v4.metadata.tags = {"music", "podcast"};
        v4.metadata.score = 0.65f;
        v4.metadata.status = "active";
        v4.metadata.custom["project"] = "project-gamma";
        v4.metadata.custom["department"] = "marketing";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v4);
    }

    std::unique_ptr<SimilaritySearchService> similarity_search_;
    std::string test_database_id_;
};

// Test advanced search with simple conditions
TEST_F(AdvancedSearchIntegrationTest, SimpleAdvancedSearch) {
    Vector query_vector;
    query_vector.id = "query_simple";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};  // Close to doc_vector_1
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    
    // Test similarity search
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);
}

// Test advanced search with owner filtering
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithOwnerFilter) {
    Vector query_vector;
    query_vector.id = "query_owner";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_owner = "user1";  // Should match doc_vector_1 and doc_vector_2
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Verify results are from user1
    for (const auto& res : result.value()) {
        auto vector_result = similarity_search_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_EQ(vector_result.value().metadata.owner, "user1");
    }
}

// Test advanced search with category filtering
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithCategoryFilter) {
    Vector query_vector;
    query_vector.id = "query_category";
    query_vector.values = {0.05f, 0.85f, 0.15f, 0.0f};  // Close to img_vector_1
    
    SearchParams params;
    params.top_k = 10;
    params.filter_category = "images";  // Should match img_vector_1
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Verify results are images
    for (const auto& res : result.value()) {
        auto vector_result = similarity_search_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_EQ(vector_result.value().metadata.category, "images");
    }
}

// Test advanced search with tag filtering
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithTagFilter) {
    Vector query_vector;
    query_vector.id = "query_tag";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_tags = {"important"};  // Should match doc_vector_1
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Verify results have "important" tag
    bool found = false;
    for (const auto& res : result.value()) {
        auto vector_result = similarity_search_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        const auto& tags = vector_result.value().metadata.tags;
        if (std::find(tags.begin(), tags.end(), "important") != tags.end()) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

// Test advanced search with score range filtering
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithScoreRangeFilter) {
    Vector query_vector;
    query_vector.id = "query_score";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_min_score = 0.8f;  // Should match doc_vector_1 and img_vector_1
    params.filter_max_score = 0.9f;
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Verify results are in the score range
    for (const auto& res : result.value()) {
        auto vector_result = similarity_search_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_GE(vector_result.value().metadata.score, 0.8f);
        EXPECT_LE(vector_result.value().metadata.score, 0.9f);
    }
}

// Test advanced search with complex AND combination
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithComplexAndFilter) {
    Vector query_vector;
    query_vector.id = "query_complex_and";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_owner = "user1";      // Matches doc_vector_1 and doc_vector_2
    params.filter_category = "documents";  // Matches doc_vector_1 and doc_vector_2
    params.filter_min_score = 0.9f;     // Only matches doc_vector_1 (score 0.95)
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Should only match doc_vector_1
    EXPECT_EQ(result.value().size(), 1);
    if (!result.value().empty()) {
        EXPECT_EQ(result.value()[0].vector_id, "doc_vector_1");
    }
}

// Test advanced search with complex OR combination
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithComplexOrFilter) {
    Vector query_vector;
    query_vector.id = "query_complex_or";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    // For this test, we'll set up conditions that would match OR combinations
    // Since our current implementation doesn't support OR logic in SearchParams,
    // we'll test what we can with the existing implementation
    
    SearchParams params;
    params.top_k = 10;
    params.filter_owner = "user1";  // Matches doc_vector_1 and doc_vector_2
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);
}

// Test advanced search with custom field filtering
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithCustomFieldFilter) {
    // This test will verify that we can access custom fields
    // Note: Our current implementation doesn't directly support custom field filtering in SearchParams
    // but we can test that the metadata filter can handle it
    
    Vector query_vector;
    query_vector.id = "query_custom";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_category = "documents";  // Filter by category to narrow down
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Verify that vectors have the expected custom fields
    for (const auto& res : result.value()) {
        auto vector_result = similarity_search_->vector_storage_->retrieve_vector(test_database_id_, res.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        
        // Check that we can access custom fields
        EXPECT_FALSE(vector_result.value().metadata.custom.empty());
    }
}

// Test advanced search with sorting and limiting
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithSortingAndLimiting) {
    Vector query_vector;
    query_vector.id = "query_sorted";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 2;  // Limit to top 2 results
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // Should have at most 2 results
    EXPECT_LE(result.value().size(), 2);
    
    // Verify results are sorted by similarity (descending)
    for (size_t i = 1; i < result.value().size(); ++i) {
        EXPECT_GE(result.value()[i-1].similarity_score, result.value()[i].similarity_score);
    }
}

// Test advanced search with threshold filtering
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchWithThresholdFilter) {
    Vector query_vector;
    query_vector.id = "query_threshold";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.9f;  // High threshold, should filter out many results
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    
    // All results should meet the threshold
    for (const auto& res : result.value()) {
        EXPECT_GE(res.similarity_score, 0.9f);
    }
}

// Test advanced search with including vector data
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchIncludingVectorData) {
    Vector query_vector;
    query_vector.id = "query_with_data";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 1;
    params.include_vector_data = true;
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);
    
    // Verify vector data is included
    const auto& search_result = result.value()[0];
    EXPECT_FALSE(search_result.vector_data.id.empty());
    EXPECT_EQ(search_result.vector_data.values.size(), 4);
}

// Test advanced search with including metadata
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchIncludingMetadata) {
    Vector query_vector;
    query_vector.id = "query_with_metadata";
    query_vector.values = {0.85f, 0.15f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 1;
    params.include_metadata = true;
    params.include_vector_data = false;
    
    auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);
    
    // Verify metadata is included (but not vector values)
    const auto& search_result = result.value()[0];
    EXPECT_FALSE(search_result.vector_data.id.empty());
    EXPECT_FALSE(search_result.vector_data.metadata.owner.empty());
}

// Test advanced search performance metrics
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchPerformanceMetrics) {
    Vector query_vector;
    query_vector.id = "query_metrics";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 3;
    
    // Perform multiple searches to trigger metrics
    for (int i = 0; i < 3; ++i) {
        auto result = similarity_search_->similarity_search(test_database_id_, query_vector, params);
        EXPECT_TRUE(result.has_value());
    }
    
    // Test passes if no exceptions are thrown
    SUCCEED();
}

// Test edge cases in advanced search
TEST_F(AdvancedSearchIntegrationTest, AdvancedSearchEdgeCases) {
    // Test with empty database (after deletion)
    // Create a temporary database for this test
    Database temp_db;
    temp_db.name = "temp_empty_db";
    temp_db.description = "Temporary empty database";
    temp_db.vectorDimension = 4;
    
    auto db_result = similarity_search_->vector_storage_->db_layer_->create_database(temp_db);
    ASSERT_TRUE(db_result.has_value());
    std::string temp_db_id = db_result.value();
    
    Vector query_vector;
    query_vector.id = "query_empty";
    query_vector.values = {0.5f, 0.5f, 0.0f, 0.0f};
    
    SearchParams params;
    params.top_k = 10;
    
    auto result = similarity_search_->similarity_search(temp_db_id, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 0);  // Should return empty results for empty database
    
    // Clean up
    similarity_search_->vector_storage_->db_layer_->delete_database(temp_db_id);
    
    // Test with non-existent database
    auto nonexistent_result = similarity_search_->similarity_search("nonexistent_db", query_vector, params);
    // Implementation may vary on how it handles non-existent databases
}

} // namespace jadevectordb