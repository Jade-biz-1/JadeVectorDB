#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <random>

#include "services/similarity_search.h"
#include "services/vector_storage.h"
#include "services/database_layer.h"
#include "services/metadata_filter.h"
#include "services/schema_validator.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/auth.h"

namespace jadevectordb {

// Test fixture for end-to-end filtered similarity search tests
class FilteredSimilaritySearchE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize services
        db_layer_ = std::make_unique<DatabaseLayer>();
        db_layer_->initialize();
        
        vector_storage_ = std::make_unique<VectorStorageService>(std::move(db_layer_));
        vector_storage_->initialize();
        
        search_service_ = std::make_unique<SimilaritySearchService>(std::move(vector_storage_));
        search_service_->initialize();
        
        metadata_filter_ = std::make_unique<MetadataFilter>();
        
        // Create test database
        Database db;
        db.name = "e2e_filtered_search_test_db";
        db.description = "Test database for end-to-end filtered similarity search";
        db.vectorDimension = 4;
        
        auto result = search_service_->vector_storage_->db_layer_->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
        
        // Add diverse test vectors
        addTestVectors();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            search_service_->vector_storage_->db_layer_->delete_database(test_database_id_);
        }
    }

    void addTestVectors() {
        std::mt19937 rng(42); // Fixed seed for reproducible tests
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> tag_dist(1, 10);
        std::uniform_int_distribution<int> owner_dist(1, 5);
        std::uniform_int_distribution<int> category_dist(1, 3);
        std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);
        
        // Create 100 diverse test vectors
        for (int i = 0; i < 100; ++i) {
            Vector v;
            v.id = "vector_" + std::to_string(i);
            v.values = {value_dist(rng), value_dist(rng), value_dist(rng), value_dist(rng)};
            
            // Generate metadata
            v.metadata.owner = "user" + std::to_string(owner_dist(rng));
            v.metadata.category = "category" + std::to_string(category_dist(rng));
            v.metadata.status = (i % 3 == 0) ? "active" : (i % 3 == 1) ? "draft" : "archived";
            v.metadata.score = score_dist(rng);
            v.metadata.created_at = "2025-01-01T00:00:00Z";
            v.metadata.updated_at = "2025-01-01T00:00:00Z";
            
            // Generate tags
            int tag_count = tag_dist(rng) % 5 + 1;
            for (int j = 0; j < tag_count; ++j) {
                v.metadata.tags.push_back("tag" + std::to_string((i + j) % 20));
            }
            
            // Generate permissions
            v.metadata.permissions = {"read", "search"};
            
            // Generate custom fields
            v.metadata.custom["project"] = "project-" + std::to_string((i % 5) + 1);
            v.metadata.custom["department"] = "dept-" + std::to_string((i % 3) + 1);
            v.metadata.custom["priority"] = std::to_string(i % 5);
            
            // Store the vector
            search_service_->vector_storage_->store_vector(test_database_id_, v);
        }
        
        LOG_INFO(logging::LoggerManager::get_logger("E2ETest"), 
                 "Added 100 test vectors to database " << test_database_id_);
    }

    std::unique_ptr<SimilaritySearchService> search_service_;
    std::unique_ptr<MetadataFilter> metadata_filter_;
    std::string test_database_id_;
};

// Test basic filtered similarity search
TEST_F(FilteredSimilaritySearchE2ETest, BasicFilteredSearch) {
    // Create a query vector similar to some test vectors
    Vector query_vector;
    query_vector.id = "query_basic";
    query_vector.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 10;
    params.threshold = 0.0f;
    params.filter_owner = "user1";  // Filter by owner
    
    // Perform filtered similarity search
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify results
    EXPECT_LE(result.value().size(), 10);  // Should not exceed top_k
    
    // All results should be from user1
    for (const auto& search_result : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, search_result.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_EQ(vector_result.value().metadata.owner, "user1");
    }
    
    // Results should be sorted by similarity (descending)
    for (size_t i = 1; i < result.value().size(); ++i) {
        EXPECT_GE(result.value()[i-1].similarity_score, result.value()[i].similarity_score);
    }
}

// Test filtered search with category filter
TEST_F(FilteredSimilaritySearchE2ETest, CategoryFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_category";
    query_vector.values = {0.3f, 0.7f, 0.2f, 0.8f};
    
    SearchParams params;
    params.top_k = 5;
    params.filter_category = "category2";  // Filter by category
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify all results are from category2
    for (const auto& search_result : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, search_result.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_EQ(vector_result.value().metadata.category, "category2");
    }
}

// Test filtered search with tag filter
TEST_F(FilteredSimilaritySearchE2ETest, TagFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_tag";
    query_vector.values = {0.1f, 0.9f, 0.3f, 0.7f};
    
    SearchParams params;
    params.top_k = 15;
    params.filter_tags = {"tag5"};  // Filter by specific tag
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify all results have tag5
    for (const auto& search_result : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, search_result.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        
        bool has_tag5 = false;
        for (const auto& tag : vector_result.value().metadata.tags) {
            if (tag == "tag5") {
                has_tag5 = true;
                break;
            }
        }
        EXPECT_TRUE(has_tag5) << "Vector " << search_result.vector_id << " should have tag5";
    }
}

// Test filtered search with score range
TEST_F(FilteredSimilaritySearchE2ETest, ScoreRangeFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_score";
    query_vector.values = {0.6f, 0.4f, 0.8f, 0.2f};
    
    SearchParams params;
    params.top_k = 20;
    params.filter_min_score = 0.7f;  // High score filter
    params.filter_max_score = 0.9f;
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify all results are in the score range
    for (const auto& search_result : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, search_result.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        EXPECT_GE(vector_result.value().metadata.score, 0.7f);
        EXPECT_LE(vector_result.value().metadata.score, 0.9f);
    }
}

// Test filtered search with threshold
TEST_F(FilteredSimilaritySearchE2ETest, ThresholdFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_threshold";
    query_vector.values = {0.4f, 0.6f, 0.5f, 0.3f};
    
    SearchParams params;
    params.top_k = 25;
    params.threshold = 0.8f;  // High threshold
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify all results meet the threshold
    for (const auto& search_result : result.value()) {
        EXPECT_GE(search_result.similarity_score, 0.8f);
    }
}

// Test combined filters (AND combination)
TEST_F(FilteredSimilaritySearchE2ETest, CombinedFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_combined";
    query_vector.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 15;
    params.filter_owner = "user2";
    params.filter_category = "category1";
    params.filter_min_score = 0.6f;
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify all results match all filters
    for (const auto& search_result : result.value()) {
        auto vector_result = search_service_->vector_storage_->retrieve_vector(test_database_id_, search_result.vector_id);
        ASSERT_TRUE(vector_result.has_value());
        
        EXPECT_EQ(vector_result.value().metadata.owner, "user2");
        EXPECT_EQ(vector_result.value().metadata.category, "category1");
        EXPECT_GE(vector_result.value().metadata.score, 0.6f);
    }
}

// Test filtered search with including vector data
TEST_F(FilteredSimilaritySearchE2ETest, FilteredSearchWithVectorData) {
    Vector query_vector;
    query_vector.id = "query_with_data";
    query_vector.values = {0.7f, 0.3f, 0.6f, 0.4f};
    
    SearchParams params;
    params.top_k = 5;
    params.include_vector_data = true;
    params.filter_owner = "user3";
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify vector data is included in results
    for (const auto& search_result : result.value()) {
        EXPECT_FALSE(search_result.vector_data.id.empty());
        EXPECT_GT(search_result.vector_data.values.size(), 0);
        EXPECT_FALSE(search_result.vector_data.metadata.owner.empty());
    }
}

// Test filtered search with including metadata only
TEST_F(FilteredSimilaritySearchE2ETest, FilteredSearchWithMetadataOnly) {
    Vector query_vector;
    query_vector.id = "query_with_metadata";
    query_vector.values = {0.2f, 0.8f, 0.1f, 0.9f};
    
    SearchParams params;
    params.top_k = 5;
    params.include_metadata = true;
    params.include_vector_data = false;  // Only metadata, not vector values
    params.filter_category = "category3";
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Verify metadata is included but vector values are not
    for (const auto& search_result : result.value()) {
        EXPECT_FALSE(search_result.vector_data.id.empty());
        EXPECT_FALSE(search_result.vector_data.metadata.owner.empty());
        // Vector values should be empty when include_vector_data is false
        EXPECT_EQ(search_result.vector_data.values.size(), 0);
    }
}

// Test filtered search performance with large dataset
TEST_F(FilteredSimilaritySearchE2ETest, FilteredSearchPerformance) {
    Vector query_vector;
    query_vector.id = "query_performance";
    query_vector.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_owner = "user1";
    
    // Measure search time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_TRUE(result.has_value());
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    LOG_INFO(logging::LoggerManager::get_logger("E2ETest"), 
             "Filtered search took " << duration.count() << " ms for " << result.value().size() << " results");
    
    // Performance assertion (should complete in reasonable time)
    EXPECT_LT(duration.count(), 1000);  // Should complete in under 1 second
}

// Test filtered search with empty results
TEST_F(FilteredSimilaritySearchE2ETest, FilteredSearchEmptyResults) {
    Vector query_vector;
    query_vector.id = "query_empty";
    query_vector.values = {0.1f, 0.2f, 0.3f, 0.4f};
    
    SearchParams params;
    params.top_k = 10;
    params.filter_owner = "nonexistent_user";  // Filter that should match nothing
    
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    ASSERT_TRUE(result.has_value());
    
    // Should return empty results
    EXPECT_EQ(result.value().size(), 0);
}

// Test filtered search with custom field filtering (through metadata filter service)
TEST_F(FilteredSimilaritySearchE2ETest, CustomFieldFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_custom";
    query_vector.values = {0.4f, 0.6f, 0.2f, 0.8f};
    
    SearchParams params;
    params.top_k = 10;
    
    // For this test, we'll use the metadata filter service directly
    // since the current SearchParams doesn't support custom field filtering
    
    // Get all vectors first
    auto all_vectors_result = search_service_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    
    // Create a complex filter for custom fields
    ComplexFilter custom_filter;
    custom_filter.combination = FilterCombination::AND;
    
    FilterCondition project_condition("metadata.custom.project", FilterOperator::EQUALS, "project-1");
    custom_filter.conditions.push_back(project_condition);
    
    // Apply the filter
    auto filtered_result = metadata_filter_->apply_complex_filters(custom_filter, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    EXPECT_GT(filtered_result.value().size(), 0);
    
    // Verify all filtered vectors have the correct custom field value
    for (const auto& vector : filtered_result.value()) {
        auto project_it = vector.metadata.custom.find("project");
        ASSERT_NE(project_it, vector.metadata.custom.end());
        EXPECT_EQ(project_it->second, "project-1");
    }
}

// Test filtered search with complex AND/OR combinations
TEST_F(FilteredSimilaritySearchE2ETest, ComplexFilterCombinations) {
    Vector query_vector;
    query_vector.id = "query_complex";
    query_vector.values = {0.3f, 0.7f, 0.1f, 0.9f};
    
    // Test with metadata filter service for complex combinations
    auto all_vectors_result = search_service_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    
    // Create a complex filter: (owner = user1 AND category = category1) OR (owner = user2 AND category = category2)
    ComplexFilter complex_filter;
    complex_filter.combination = FilterCombination::OR;
    
    // First nested filter: owner = user1 AND category = category1
    auto nested_filter1 = std::make_unique<ComplexFilter>();
    nested_filter1->combination = FilterCombination::AND;
    
    FilterCondition owner1_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    FilterCondition category1_condition("metadata.category", FilterOperator::EQUALS, "category1");
    
    nested_filter1->conditions.push_back(owner1_condition);
    nested_filter1->conditions.push_back(category1_condition);
    
    // Second nested filter: owner = user2 AND category = category2
    auto nested_filter2 = std::make_unique<ComplexFilter>();
    nested_filter2->combination = FilterCombination::AND;
    
    FilterCondition owner2_condition("metadata.owner", FilterOperator::EQUALS, "user2");
    FilterCondition category2_condition("metadata.category", FilterOperator::EQUALS, "category2");
    
    nested_filter2->conditions.push_back(owner2_condition);
    nested_filter2->conditions.push_back(category2_condition);
    
    // Add nested filters to the main filter
    complex_filter.nested_filters.push_back(std::move(nested_filter1));
    complex_filter.nested_filters.push_back(std::move(nested_filter2));
    
    // Apply the complex filter
    auto filtered_result = metadata_filter_->apply_complex_filters(complex_filter, all_vectors);
    ASSERT_TRUE(filtered_result.has_value());
    
    // Verify results match the complex filter logic
    for (const auto& vector : filtered_result.value()) {
        bool matches_first_condition = (vector.metadata.owner == "user1" && vector.metadata.category == "category1");
        bool matches_second_condition = (vector.metadata.owner == "user2" && vector.metadata.category == "category2");
        
        EXPECT_TRUE(matches_first_condition || matches_second_condition) 
            << "Vector " << vector.id << " should match one of the filter conditions";
    }
}

// Test filtered search with array-type filters
TEST_F(FilteredSimilaritySearchE2ETest, ArrayTypeFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_array";
    query_vector.values = {0.6f, 0.2f, 0.8f, 0.4f};
    
    // Test with metadata filter service for array filtering
    auto all_vectors_result = search_service_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    
    // Create a filter for array field (tags)
    FilterCondition tags_condition("metadata.tags", FilterOperator::CONTAINS, "tag10");
    
    std::vector<FilterCondition> conditions = {tags_condition};
    auto filtered_result = metadata_filter_->apply_filters(conditions, all_vectors, FilterCombination::AND);
    ASSERT_TRUE(filtered_result.has_value());
    
    // Verify results have the specified tag
    for (const auto& vector : filtered_result.value()) {
        bool has_tag10 = false;
        for (const auto& tag : vector.metadata.tags) {
            if (tag == "tag10") {
                has_tag10 = true;
                break;
            }
        }
        EXPECT_TRUE(has_tag10) << "Vector " << vector.id << " should have tag10";
    }
}

// Test filtered search with regex pattern matching
TEST_F(FilteredSimilaritySearchE2ETest, RegexPatternFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_regex";
    query_vector.values = {0.1f, 0.9f, 0.2f, 0.8f};
    
    // Test with metadata filter service for regex filtering
    auto all_vectors_result = search_service_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    
    // Create a filter for regex pattern matching on owner field
    FilterCondition regex_condition("metadata.owner", FilterOperator::MATCHES_REGEX, "user[1-3]");
    
    std::vector<FilterCondition> conditions = {regex_condition};
    auto filtered_result = metadata_filter_->apply_filters(conditions, all_vectors, FilterCombination::AND);
    ASSERT_TRUE(filtered_result.has_value());
    
    // Verify results match the regex pattern
    for (const auto& vector : filtered_result.value()) {
        EXPECT_TRUE(vector.metadata.owner == "user1" || 
                   vector.metadata.owner == "user2" || 
                   vector.metadata.owner == "user3")
            << "Vector " << vector.id << " owner " << vector.metadata.owner << " should match regex pattern";
    }
}

// Test filtered search with multiple value filters (IN operator)
TEST_F(FilteredSimilaritySearchE2ETest, MultipleValueFilteredSearch) {
    Vector query_vector;
    query_vector.id = "query_multiple";
    query_vector.values = {0.5f, 0.5f, 0.5f, 0.5f};
    
    // Test with metadata filter service for IN operator
    auto all_vectors_result = search_service_->vector_storage_->retrieve_vectors(test_database_id_, {});
    ASSERT_TRUE(all_vectors_result.has_value());
    
    auto all_vectors = all_vectors_result.value();
    
    // Create a filter for multiple values (IN operator)
    FilterCondition in_condition("metadata.category", FilterOperator::IN, std::vector<std::string>{"category1", "category2"});
    
    std::vector<FilterCondition> conditions = {in_condition};
    auto filtered_result = metadata_filter_->apply_filters(conditions, all_vectors, FilterCombination::AND);
    ASSERT_TRUE(filtered_result.has_value());
    
    // Verify results have one of the specified categories
    for (const auto& vector : filtered_result.value()) {
        EXPECT_TRUE(vector.metadata.category == "category1" || vector.metadata.category == "category2")
            << "Vector " << vector.id << " category " << vector.metadata.category << " should be in allowed values";
    }
}

} // namespace jadevectordb