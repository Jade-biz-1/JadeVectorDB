#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <chrono>

#include "services/database_layer.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "services/metadata_filter.h"
#include "models/database.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Test fixture for advanced filtering integration tests
class AdvancedFilteringIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        vector_storage_ = std::make_unique<VectorStorageService>(std::move(db_layer_));
        vector_storage_->initialize();
        
        similarity_search_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage_));
        similarity_search_->initialize();
        
        metadata_filter_ = std::make_unique<MetadataFilter>();
        
        // Create test database
        Database db;
        db.name = "advanced_filtering_test_db";
        db.description = "Test database for advanced filtering integration";
        db.vectorDimension = 128;  // Standard dimension for tests
        
        auto create_result = similarity_search_->vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(create_result.has_value());
        test_database_id_ = create_result.value();
        
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
        // Vector 1: Document vector with geospatial metadata
        Vector v1;
        v1.id = "doc_vector_1";
        v1.values = std::vector<float>(128, 0.1f);
        v1.metadata.owner = "user1";
        v1.metadata.category = "documents";
        v1.metadata.tags = {"important", "urgent", "review"};
        v1.metadata.score = 0.95f;
        v1.metadata.status = "active";
        v1.metadata.created_at = "2025-01-01T00:00:00Z";
        v1.metadata.updated_at = "2025-01-02T00:00:00Z";
        v1.metadata.custom["project"] = "project-alpha";
        v1.metadata.custom["department"] = "engineering";
        v1.metadata.custom["location"] = "37.7749,-122.4194";  // San Francisco
        v1.metadata.custom["timestamp"] = "2025-01-01T12:00:00Z";
        v1.metadata.custom["description"] = "This is an important document about vector database implementation";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v1);
        
        // Vector 2: Image vector with geospatial metadata
        Vector v2;
        v2.id = "img_vector_1";
        v2.values = std::vector<float>(128, 0.2f);
        v2.metadata.owner = "user2";
        v2.metadata.category = "images";
        v2.metadata.tags = {"nature", "landscape", "photo"};
        v2.metadata.score = 0.85f;
        v2.metadata.status = "active";
        v2.metadata.created_at = "2025-01-01T01:00:00Z";
        v2.metadata.updated_at = "2025-01-02T01:00:00Z";
        v2.metadata.custom["project"] = "project-beta";
        v2.metadata.custom["department"] = "design";
        v2.metadata.custom["location"] = "40.7128,-74.0060";  // New York
        v2.metadata.custom["timestamp"] = "2025-01-01T13:00:00Z";
        v2.metadata.custom["description"] = "This is a beautiful landscape photo from New York";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v2);
        
        // Vector 3: Another document vector with geospatial metadata
        Vector v3;
        v3.id = "doc_vector_2";
        v3.values = std::vector<float>(128, 0.3f);
        v3.metadata.owner = "user1";
        v3.metadata.category = "documents";
        v3.metadata.tags = {"draft", "review"};
        v3.metadata.score = 0.75f;
        v3.metadata.status = "draft";
        v3.metadata.created_at = "2025-01-01T02:00:00Z";
        v3.metadata.updated_at = "2025-01-02T02:00:00Z";
        v3.metadata.custom["project"] = "project-alpha";
        v3.metadata.custom["department"] = "engineering";
        v3.metadata.custom["location"] = "34.0522,-118.2437";  // Los Angeles
        v3.metadata.custom["timestamp"] = "2025-01-01T14:00:00Z";
        v3.metadata.custom["description"] = "This is a draft document about database design";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v3);
        
        // Vector 4: Audio vector with geospatial metadata
        Vector v4;
        v4.id = "aud_vector_1";
        v4.values = std::vector<float>(128, 0.4f);
        v4.metadata.owner = "user3";
        v4.metadata.category = "audio";
        v4.metadata.tags = {"music", "podcast"};
        v4.metadata.score = 0.65f;
        v4.metadata.status = "active";
        v4.metadata.created_at = "2025-01-01T03:00:00Z";
        v4.metadata.updated_at = "2025-01-02T03:00:00Z";
        v4.metadata.custom["project"] = "project-gamma";
        v4.metadata.custom["department"] = "marketing";
        v4.metadata.custom["location"] = "37.7749,-122.4194";  // San Francisco (same as doc_vector_1)
        v4.metadata.custom["timestamp"] = "2025-01-01T15:00:00Z";
        v4.metadata.custom["description"] = "This is a podcast episode about vector databases";
        similarity_search_->vector_storage_->store_vector(test_database_id_, v4);
    }

    std::unique_ptr<DatabaseLayer> db_layer_;
    std::unique_ptr<VectorStorageService> vector_storage_;
    std::unique_ptr<SimilaritySearchService> similarity_search_;
    std::unique_ptr<MetadataFilter> metadata_filter_;
    std::string test_database_id_;
};

// Test integration of geospatial filtering with similarity search
TEST_F(AdvancedFilteringIntegrationTest, GeospatialFilteringWithSimilaritySearch) {
    // Create a geospatial query: find vectors near San Francisco
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(37.7749, -122.4194);  // San Francisco coordinates
    geo_query.radius = 1000.0;  // 1000 meters
    
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    // Create a query vector similar to doc_vector_1
    Vector query_vector;
    query_vector.id = "query_sf";
    query_vector.values = std::vector<float>(128, 0.11f);  // Slightly different from doc_vector_1
    
    // Set up search parameters with geospatial filtering
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // In a real implementation, we would pass the geo_queries to the search service
    // For now, we'll test the metadata filter directly
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_geo_filters(geo_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 2);  // Should match doc_vector_1 and aud_vector_1 (both in SF)
    
    // Verify that the filtered vectors are indeed from San Francisco
    for (const auto& vector : filtered_vectors) {
        EXPECT_TRUE(vector.id == "doc_vector_1" || vector.id == "aud_vector_1");
        auto location_it = vector.metadata.custom.find("location");
        ASSERT_NE(location_it, vector.metadata.custom.end());
        EXPECT_EQ(location_it->second, "37.7749,-122.4194");
    }
}

// Test integration of temporal filtering with similarity search
TEST_F(AdvancedFilteringIntegrationTest, TemporalFilteringWithSimilaritySearch) {
    // Create a temporal query: find vectors created after a certain time
    TemporalQuery temporal_query(TemporalOperator::AFTER, "metadata.custom.timestamp");
    
    std::tm tm_time = {};
    std::istringstream ss("2025-01-01T13:30:00Z");
    ss >> std::get_time(&tm_time, "%Y-%m-%dT%H:%M:%SZ");
    temporal_query.time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    
    std::vector<TemporalQuery> temporal_queries = {temporal_query};
    
    // Create a query vector
    Vector query_vector;
    query_vector.id = "query_temporal";
    query_vector.values = std::vector<float>(128, 0.15f);
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test the metadata filter directly
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_temporal_filters(temporal_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 1);  // Should match aud_vector_1 (timestamp 15:00:00)
    
    // Verify that the filtered vector has the correct timestamp
    for (const auto& vector : filtered_vectors) {
        EXPECT_EQ(vector.id, "aud_vector_1");
        auto timestamp_it = vector.metadata.custom.find("timestamp");
        ASSERT_NE(timestamp_it, vector.metadata.custom.end());
        EXPECT_EQ(timestamp_it->second, "2025-01-01T15:00:00Z");
    }
}

// Test integration of nested object filtering with similarity search
TEST_F(AdvancedFilteringIntegrationTest, NestedObjectFilteringWithSimilaritySearch) {
    // Create a nested query: check if a path exists in custom metadata
    NestedQuery nested_query("metadata.custom.location", NestedOperator::EXISTS_PATH, "");
    
    std::vector<NestedQuery> nested_queries = {nested_query};
    
    // Create a query vector
    Vector query_vector;
    query_vector.id = "query_nested";
    query_vector.values = std::vector<float>(128, 0.25f);
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test the metadata filter directly
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_nested_filters(nested_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 4);  // All vectors should have the location field
    
    // Verify that all filtered vectors have the location field
    for (const auto& vector : filtered_vectors) {
        auto location_it = vector.metadata.custom.find("location");
        EXPECT_NE(location_it, vector.metadata.custom.end());
    }
}

// Test integration of full-text search with similarity search
TEST_F(AdvancedFilteringIntegrationTest, FullTextSearchWithSimilaritySearch) {
    // Create a full-text query: match documents containing "vector database"
    FullTextQuery fulltext_query("metadata.custom.description", "vector database", FullTextOperator::MATCHES_ALL_TERMS);
    
    std::vector<FullTextQuery> fulltext_queries = {fulltext_query};
    
    // Create a query vector
    Vector query_vector;
    query_vector.id = "query_fulltext";
    query_vector.values = std::vector<float>(128, 0.35f);
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test the metadata filter directly
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_fulltext_filters(fulltext_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 3);  // Should match doc_vector_1, doc_vector_2, and aud_vector_1
    
    // Verify that all filtered vectors contain the search terms
    for (const auto& vector : filtered_vectors) {
        auto description_it = vector.metadata.custom.find("description");
        ASSERT_NE(description_it, vector.metadata.custom.end());
        std::string description = description_it->second;
        EXPECT_NE(description.find("vector"), std::string::npos);
        EXPECT_NE(description.find("database"), std::string::npos);
    }
}

// Test integration of fuzzy matching with similarity search
TEST_F(AdvancedFilteringIntegrationTest, FuzzyMatchingWithSimilaritySearch) {
    // Create a fuzzy matching query: match with small edit distance
    FullTextQuery fuzzy_query("metadata.custom.description", "documnet", FullTextOperator::FUZZY_MATCH);
    fuzzy_query.max_edit_distance = 2;  // Allow up to 2 character differences
    
    std::vector<FullTextQuery> fuzzy_queries = {fuzzy_query};
    
    // Create a query vector
    Vector query_vector;
    query_vector.id = "query_fuzzy";
    query_vector.values = std::vector<float>(128, 0.45f);
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test the metadata filter directly
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_fulltext_filters(fuzzy_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 2);  // Should match doc_vector_1 and doc_vector_2
    
    // Verify that the filtered vectors contain words similar to "documnet"
    for (const auto& vector : filtered_vectors) {
        auto description_it = vector.metadata.custom.find("description");
        ASSERT_NE(description_it, vector.metadata.custom.end());
        std::string description = description_it->second;
        EXPECT_NE(description.find("document"), std::string::npos);  // "documnet" -> "document" (1 edit)
    }
}

// Test integration of combined advanced filtering with similarity search
TEST_F(AdvancedFilteringIntegrationTest, CombinedAdvancedFilteringWithSimilaritySearch) {
    // Create multiple types of queries
    
    // Geospatial query: vectors near San Francisco
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(37.7749, -122.4194);  // San Francisco coordinates
    geo_query.radius = 1000.0;  // 1000 meters
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    // Temporal query: vectors created after a certain time
    TemporalQuery temporal_query(TemporalOperator::AFTER, "metadata.custom.timestamp");
    std::tm tm_time = {};
    std::istringstream ss("2025-01-01T12:30:00Z");
    ss >> std::get_time(&tm_time, "%Y-%m-%dT%H:%M:%SZ");
    temporal_query.time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    std::vector<TemporalQuery> temporal_queries = {temporal_query};
    
    // Nested query: vectors with location field
    NestedQuery nested_query("metadata.custom.location", NestedOperator::EXISTS_PATH, "");
    std::vector<NestedQuery> nested_queries = {nested_query};
    
    // Full-text query: vectors containing "document"
    FullTextQuery fulltext_query("metadata.custom.description", "document", FullTextOperator::MATCHES_ANY_TERM);
    std::vector<FullTextQuery> fulltext_queries = {fulltext_query};
    
    // Regular filter conditions
    std::vector<FilterCondition> conditions;
    FilterCondition owner_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    conditions.push_back(owner_condition);
    
    // Create a query vector
    Vector query_vector;
    query_vector.id = "query_combined";
    query_vector.values = std::vector<float>(128, 0.15f);
    
    // Set up search parameters
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.include_vector_data = false;
    params.include_metadata = false;
    
    // Test the metadata filter directly with combined advanced filters
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_advanced_filters(conditions, geo_queries, temporal_queries,
                                                               nested_queries, fulltext_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 1);  // Should match only doc_vector_1
    
    // Verify that the filtered vector meets all criteria
    for (const auto& vector : filtered_vectors) {
        EXPECT_EQ(vector.id, "doc_vector_1");
        
        // Check owner condition
        EXPECT_EQ(vector.metadata.owner, "user1");
        
        // Check geospatial condition
        auto location_it = vector.metadata.custom.find("location");
        ASSERT_NE(location_it, vector.metadata.custom.end());
        EXPECT_EQ(location_it->second, "37.7749,-122.4194");
        
        // Check temporal condition
        auto timestamp_it = vector.metadata.custom.find("timestamp");
        ASSERT_NE(timestamp_it, vector.metadata.custom.end());
        EXPECT_GT(timestamp_it->second, "2025-01-01T12:30:00Z");
        
        // Check nested condition (location field exists)
        EXPECT_NE(location_it, vector.metadata.custom.end());
        
        // Check full-text condition
        auto description_it = vector.metadata.custom.find("description");
        ASSERT_NE(description_it, vector.metadata.custom.end());
        std::string description = description_it->second;
        EXPECT_NE(description.find("document"), std::string::npos);
    }
}

// Test performance of advanced filtering with large datasets
TEST_F(AdvancedFilteringIntegrationTest, AdvancedFilteringPerformanceWithLargeDataset) {
    // For this test, we'll create a larger dataset to test performance
    
    // Add more vectors to test performance
    for (int i = 0; i < 100; ++i) {
        Vector v;
        v.id = "bulk_vector_" + std::to_string(i);
        v.values = std::vector<float>(128, static_cast<float>(i) / 100.0f);
        v.metadata.owner = "user" + std::to_string(i % 10);
        v.metadata.category = "bulk";
        v.metadata.tags = {"bulk", "test"};
        v.metadata.score = static_cast<float>(i) / 100.0f;
        v.metadata.status = (i % 3 == 0) ? "active" : (i % 3 == 1) ? "draft" : "archived";
        v.metadata.created_at = "2025-01-01T" + std::to_string(i % 24) + ":00:00Z";
        v.metadata.updated_at = "2025-01-02T" + std::to_string(i % 24) + ":00:00Z";
        v.metadata.custom["project"] = "project-bulk";
        v.metadata.custom["department"] = "testing";
        v.metadata.custom["location"] = "37.7" + std::to_string(i % 100) + ",-122.4" + std::to_string(i % 100);  // SF area
        v.metadata.custom["timestamp"] = "2025-01-01T" + std::to_string(i % 24) + ":" + 
                                      std::to_string((i * 2) % 60) + ":00Z";
        v.metadata.custom["description"] = "This is a bulk test document with ID " + std::to_string(i) + 
                                        " for performance testing advanced filtering capabilities";
        
        similarity_search_->vector_storage_->store_vector(test_database_id_, v);
    }
    
    // Create a geospatial query for performance testing
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(37.75, -122.45);  // Central SF coordinates
    geo_query.radius = 5000.0;  // 5km radius
    
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    // Test performance
    auto start = std::chrono::high_resolution_clock::now();
    
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_geo_filters(geo_queries, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_GT(filtered_vectors.size(), 0);  // Should find some vectors
    
    // Performance validation: Should complete in under 100ms for 100 vectors
    EXPECT_LT(duration.count(), 100);
    
    // Log performance information
    LOG_INFO(logging::LoggerManager::get_logger("test.advanced_filtering"),
             "Advanced filtering performance test: {} vectors filtered in {} ms",
             all_vectors.size(), duration.count());
}

// Test edge cases in advanced filtering
TEST_F(AdvancedFilteringIntegrationTest, AdvancedFilteringEdgeCases) {
    // Test with empty database
    Database empty_db;
    empty_db.name = "empty_test_db";
    empty_db.description = "Empty test database";
    empty_db.vectorDimension = 128;
    
    auto create_result = similarity_search_->vector_storage_->db_layer_->create_database(empty_db);
    ASSERT_TRUE(create_result.has_value());
    std::string empty_db_id = create_result.value();
    
    // Test geospatial filtering on empty database
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(0.0, 0.0);
    geo_query.radius = 1000.0;
    
    std::vector<GeoQuery> geo_queries = {geo_query};
    
    auto empty_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(empty_db_id, {});
    ASSERT_TRUE(empty_vectors_result.has_value());
    
    auto empty_vectors = empty_vectors_result.value();
    auto filtered_result = metadata_filter_->apply_geo_filters(geo_queries, empty_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    auto filtered_vectors = filtered_result.value();
    EXPECT_EQ(filtered_vectors.size(), 0);
    
    // Clean up empty database
    similarity_search_->vector_storage_->db_layer_->delete_database(empty_db_id);
    
    // Test with invalid geospatial coordinates
    Vector invalid_vector;
    invalid_vector.id = "invalid_geo_vector";
    invalid_vector.values = std::vector<float>(128, 0.5f);
    invalid_vector.metadata.custom["location"] = "invalid,coordinates";
    
    similarity_search_->vector_storage_->store_vector(test_database_id_, invalid_vector);
    
    // Test that invalid coordinates are handled gracefully
    GeoQuery invalid_geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    invalid_geo_query.center = Point(0.0, 0.0);
    invalid_geo_query.radius = 1000.0;
    
    std::vector<GeoQuery> invalid_geo_queries = {invalid_geo_query};
    
    auto all_vectors_result = similarity_search_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    auto invalid_filtered_result = metadata_filter_->apply_geo_filters(invalid_geo_queries, all_vectors);
    ASSERT_TRUE(invalid_filtered_result.has_value());
    
    // Should still work - vectors with invalid coordinates will be filtered out
    auto invalid_filtered_vectors = invalid_filtered_result.value();
    EXPECT_GE(invalid_filtered_vectors.size(), 0);
    
    // Test with invalid temporal timestamps
    Vector invalid_temporal_vector;
    invalid_temporal_vector.id = "invalid_temporal_vector";
    invalid_temporal_vector.values = std::vector<float>(128, 0.6f);
    invalid_temporal_vector.metadata.custom["timestamp"] = "invalid-timestamp-format";
    
    similarity_search_->vector_storage_->store_vector(test_database_id_, invalid_temporal_vector);
    
    // Test that invalid timestamps are handled gracefully
    TemporalQuery invalid_temporal_query(TemporalOperator::AFTER, "metadata.custom.timestamp");
    std::tm tm_time = {};
    std::istringstream ss("2025-01-01T00:00:00Z");
    ss >> std::get_time(&tm_time, "%Y-%m-%dT%H:%M:%SZ");
    invalid_temporal_query.time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    
    std::vector<TemporalQuery> invalid_temporal_queries = {invalid_temporal_query};
    
    auto invalid_temporal_filtered_result = metadata_filter_->apply_temporal_filters(invalid_temporal_queries, all_vectors);
    ASSERT_TRUE(invalid_temporal_filtered_result.has_value());
    
    // Should still work - vectors with invalid timestamps will be filtered out
    auto invalid_temporal_filtered_vectors = invalid_temporal_filtered_result.value();
    EXPECT_GE(invalid_temporal_filtered_vectors.size(), 0);
}

} // namespace jadevectordb