#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/metadata_filter.h"
#include "models/vector.h"

namespace jadevectordb {

// Test fixture for metadata filter tests
class MetadataFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        filter_ = std::make_unique<MetadataFilter>();
        
        // Create test vector with various metadata
        test_vector_.id = "test_vector";
        test_vector_.values = {0.1f, 0.2f, 0.3f, 0.4f};
        test_vector_.metadata.owner = "user1";
        test_vector_.metadata.category = "documents";
        test_vector_.metadata.status = "active";
        test_vector_.metadata.source = "upload";
        test_vector_.metadata.created_at = "2025-01-01T00:00:00Z";
        test_vector_.metadata.updated_at = "2025-01-02T00:00:00Z";
        test_vector_.metadata.tags = {"tag1", "important", "review"};
        test_vector_.metadata.permissions = {"read", "write"};
        test_vector_.metadata.score = 0.85f;
        test_vector_.metadata.custom["region"] = "us-east-1";
        test_vector_.metadata.custom["project"] = "project-a";
    }

    std::unique_ptr<MetadataFilter> filter_;
    Vector test_vector_;
};

// Test basic equality filtering
TEST_F(MetadataFilterTest, EqualityFilter) {
    FilterCondition condition("metadata.owner", FilterOperator::EQUALS, "user1");
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    // Test with non-matching value
    FilterCondition condition2("metadata.owner", FilterOperator::EQUALS, "user2");
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test inequality filtering
TEST_F(MetadataFilterTest, InequalityFilter) {
    FilterCondition condition("metadata.owner", FilterOperator::NOT_EQUALS, "user2");
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    // Test with matching value (should return false)
    FilterCondition condition2("metadata.owner", FilterOperator::NOT_EQUALS, "user1");
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test numeric comparisons
TEST_F(MetadataFilterTest, NumericComparison) {
    FilterCondition greater_than("metadata.score", FilterOperator::GREATER_THAN, "0.8");
    
    auto result = filter_->applies_to_vector(greater_than, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    FilterCondition less_than("metadata.score", FilterOperator::LESS_THAN, "0.9");
    auto result2 = filter_->applies_to_vector(less_than, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_TRUE(result2.value());
    
    FilterCondition equal_to("metadata.score", FilterOperator::EQUALS, "0.85");
    auto result3 = filter_->applies_to_vector(equal_to, test_vector_);
    EXPECT_TRUE(result3.has_value());
    EXPECT_TRUE(result3.value());
}

// Test string contains
TEST_F(MetadataFilterTest, StringContains) {
    FilterCondition condition("metadata.source", FilterOperator::CONTAINS, "upl");
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    FilterCondition condition2("metadata.source", FilterOperator::CONTAINS, "download");
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test array contains (tags)
TEST_F(MetadataFilterTest, ArrayContains) {
    // Testing tags array
    FilterCondition condition("metadata.tags", FilterOperator::CONTAINS, "important");
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    FilterCondition condition2("metadata.tags", FilterOperator::CONTAINS, "missing");
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test IN operator
TEST_F(MetadataFilterTest, InOperator) {
    FilterCondition condition("metadata.category", FilterOperator::IN, std::vector<std::string>{"documents", "images"});
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    FilterCondition condition2("metadata.category", FilterOperator::IN, std::vector<std::string>{"videos", "audio"});
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test NOT IN operator
TEST_F(MetadataFilterTest, NotInOperator) {
    FilterCondition condition("metadata.category", FilterOperator::NOT_IN, std::vector<std::string>{"videos", "audio"});
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    FilterCondition condition2("metadata.category", FilterOperator::NOT_IN, std::vector<std::string>{"documents", "audio"});
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test EXISTS operator
TEST_F(MetadataFilterTest, ExistsOperator) {
    FilterCondition condition("metadata.owner", FilterOperator::EXISTS, "");
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    // For non-existing field, this would return false (tested in context of the implementation)
}

// Test complex filter with AND combination
TEST_F(MetadataFilterTest, ComplexFilterAnd) {
    ComplexFilter complex_filter;
    complex_filter.combination = FilterCombination::AND;
    
    FilterCondition owner_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    FilterCondition category_condition("metadata.category", FilterOperator::EQUALS, "documents");
    
    complex_filter.conditions.push_back(owner_condition);
    complex_filter.conditions.push_back(category_condition);
    
    auto result = filter_->applies_to_vector(complex_filter, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    // Test with one failing condition
    FilterCondition failing_condition("metadata.category", FilterOperator::EQUALS, "videos");
    complex_filter.conditions[1] = failing_condition;
    
    auto result2 = filter_->applies_to_vector(complex_filter, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test complex filter with OR combination
TEST_F(MetadataFilterTest, ComplexFilterOr) {
    ComplexFilter complex_filter;
    complex_filter.combination = FilterCombination::OR;
    
    FilterCondition first_condition("metadata.category", FilterOperator::EQUALS, "videos");  // False
    FilterCondition second_condition("metadata.owner", FilterOperator::EQUALS, "user1");     // True
    
    complex_filter.conditions.push_back(first_condition);
    complex_filter.conditions.push_back(second_condition);
    
    auto result = filter_->applies_to_vector(complex_filter, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    // Test with both conditions failing
    FilterCondition failing_condition1("metadata.category", FilterOperator::EQUALS, "videos");
    FilterCondition failing_condition2("metadata.owner", FilterOperator::EQUALS, "user2");
    
    complex_filter.conditions[0] = failing_condition1;
    complex_filter.conditions[1] = failing_condition2;
    
    auto result2 = filter_->applies_to_vector(complex_filter, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_FALSE(result2.value());
}

// Test nested complex filters
TEST_F(MetadataFilterTest, NestedComplexFilter) {
    ComplexFilter outer_filter;
    outer_filter.combination = FilterCombination::AND;
    
    // Add a simple condition
    FilterCondition owner_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    outer_filter.conditions.push_back(owner_condition);
    
    // Create inner filter
    auto inner_filter = std::make_unique<ComplexFilter>();
    inner_filter->combination = FilterCombination::OR;
    
    FilterCondition category_condition("metadata.category", FilterOperator::EQUALS, "documents");
    FilterCondition status_condition("metadata.status", FilterOperator::EQUALS, "inactive");
    
    inner_filter->conditions.push_back(category_condition);  // True
    inner_filter->conditions.push_back(status_condition);    // False
    
    outer_filter.nested_filters.push_back(std::move(inner_filter));
    
    auto result = filter_->applies_to_vector(outer_filter, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());  // owner matches AND (category matches OR status matches) = True AND (True OR False) = True
}

// Test filtering multiple vectors
TEST_F(MetadataFilterTest, ApplyFiltersToVectorCollection) {
    std::vector<Vector> vectors;
    
    // Vector 1: matches filters
    Vector v1 = test_vector_;
    v1.id = "vector1";
    vectors.push_back(v1);
    
    // Vector 2: doesn't match filters
    Vector v2 = test_vector_;
    v2.id = "vector2";
    v2.metadata.owner = "user2";
    vectors.push_back(v2);
    
    // Vector 3: matches filters
    Vector v3 = test_vector_;
    v3.id = "vector3";
    v3.metadata.category = "documents";
    vectors.push_back(v3);
    
    // Apply filter for owner = user1
    std::vector<FilterCondition> conditions;
    FilterCondition owner_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    conditions.push_back(owner_condition);
    
    auto result = filter_->apply_filters(conditions, vectors, FilterCombination::AND);
    EXPECT_TRUE(result.has_value());
    
    auto filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 2);  // v1 and v3 should match
    EXPECT_EQ(filtered_vectors[0].id, "vector1");
    EXPECT_EQ(filtered_vectors[1].id, "vector3");
}

// Test validation
TEST_F(MetadataFilterTest, ValidateCondition) {
    FilterCondition valid_condition("metadata.owner", FilterOperator::EQUALS, "user1");
    auto valid_result = filter_->validate_condition(valid_condition);
    EXPECT_TRUE(valid_result.has_value());
    
    FilterCondition invalid_condition;  // Empty field
    auto invalid_result = filter_->validate_condition(invalid_condition);
    EXPECT_FALSE(invalid_result.has_value());
}

// Test range-based filtering
TEST_F(MetadataFilterTest, RangeFiltering) {
    // Test score range: >= 0.8 and <= 0.9
    std::vector<FilterCondition> conditions;
    
    FilterCondition min_condition("metadata.score", FilterOperator::GREATER_THAN_OR_EQUAL, "0.8");
    FilterCondition max_condition("metadata.score", FilterOperator::LESS_THAN_OR_EQUAL, "0.9");
    
    conditions.push_back(min_condition);
    conditions.push_back(max_condition);
    
    auto result = filter_->apply_filters(conditions, {test_vector_}, FilterCombination::AND);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 1);  // Our test vector has score 0.85, so it should match
}

// Test custom field filtering
TEST_F(MetadataFilterTest, CustomFieldFiltering) {
    FilterCondition condition("metadata.custom.region", FilterOperator::EQUALS, "us-east-1");
    
    auto result = filter_->applies_to_vector(condition, test_vector_);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    
    FilterCondition condition2("metadata.custom.project", FilterOperator::EQUALS, "project-a");
    auto result2 = filter_->applies_to_vector(condition2, test_vector_);
    EXPECT_TRUE(result2.has_value());
    EXPECT_TRUE(result2.value());
}

} // namespace jadevectordb