#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "services/metadata_filter.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;
using ::testing::Eq;
using ::testing::ByRef;

// Test fixture for MetadataFilter
class MetadataFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create metadata filter service
        metadata_filter_ = std::make_unique<MetadataFilter>();
        
        // Initialize the service
        metadata_filter_->initialize();
    }
    
    void TearDown() override {
        // Clean up
        metadata_filter_.reset();
    }
    
    std::unique_ptr<MetadataFilter> metadata_filter_;
};

// Test that the service initializes correctly
TEST_F(MetadataFilterTest, InitializeService) {
    // Service should already be initialized in SetUp
    EXPECT_NE(metadata_filter_, nullptr);
}

// Test applying simple metadata filters
TEST_F(MetadataFilterTest, ApplySimpleMetadataFilters) {
    // Create test vectors with metadata
    std::vector<Vector> test_vectors;
    
    // Vector 1: Finance category, high score, banking tag
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 0.1f, 0.2f, 0.3f};
    v1.metadata["category"] = "finance";
    v1.metadata["score"] = 0.95f;
    v1.metadata["tags"] = nlohmann::json::array({"banking", "investment"});
    v1.metadata["region"] = "north_america";
    test_vectors.push_back(v1);
    
    // Vector 2: Technology category, medium score, ai tag
    Vector v2;
    v2.id = "vector_2";
    v2.values = {0.1f, 1.0f, 0.2f, 0.8f};
    v2.metadata["category"] = "technology";
    v2.metadata["score"] = 0.75f;
    v2.metadata["tags"] = nlohmann::json::array({"ai", "ml"});
    v2.metadata["region"] = "europe";
    test_vectors.push_back(v2);
    
    // Vector 3: Finance category, medium score, trading tag
    Vector v3;
    v3.id = "vector_3";
    v3.values = {0.9f, 0.2f, 0.1f, 0.4f};
    v3.metadata["category"] = "finance";
    v3.metadata["score"] = 0.82f;
    v3.metadata["tags"] = nlohmann::json::array({"trading", "cryptocurrency"});
    v3.metadata["region"] = "north_america";
    test_vectors.push_back(v3);
    
    // Vector 4: Healthcare category, low score, research tag
    Vector v4;
    v4.id = "vector_4";
    v4.values = {0.2f, 0.1f, 1.0f, 0.3f};
    v4.metadata["category"] = "healthcare";
    v4.metadata["score"] = 0.65f;
    v4.metadata["tags"] = nlohmann::json::array({"research", "clinical"});
    v4.metadata["region"] = "asia";
    test_vectors.push_back(v4);
    
    // Test filter for finance category
    FilterCondition finance_condition;
    finance_condition.field = "metadata.category";
    finance_condition.op = FilterOperator::EQUALS;
    finance_condition.value = "finance";
    
    std::vector<FilterCondition> conditions = {finance_condition};
    
    auto result = metadata_filter_->apply_filters(conditions, test_vectors);
    ASSERT_TRUE(result.has_value());
    
    auto filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 2); // v1 and v3 are finance
    
    // Check that all filtered vectors have category "finance"
    for (const auto& v : filtered_vectors) {
        EXPECT_EQ(v.metadata["category"].get<std::string>(), "finance");
    }
    
    // Test filter for technology category with score >= 0.7
    FilterCondition tech_condition;
    tech_condition.field = "metadata.category";
    tech_condition.op = FilterOperator::EQUALS;
    tech_condition.value = "technology";
    
    FilterCondition score_condition;
    score_condition.field = "metadata.score";
    score_condition.op = FilterOperator::GREATER_THAN_OR_EQUAL;
    score_condition.value = "0.7";
    
    std::vector<FilterCondition> conditions2 = {tech_condition, score_condition};
    
    result = metadata_filter_->apply_filters(conditions2, test_vectors, FilterCombination::AND);
    ASSERT_TRUE(result.has_value());
    
    filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 1); // Only v2 is technology with score >= 0.7
    
    // Check that the filtered vector is v2
    EXPECT_EQ(filtered_vectors[0].id, "vector_2");
    EXPECT_EQ(filtered_vectors[0].metadata["category"].get<std::string>(), "technology");
    EXPECT_GE(filtered_vectors[0].metadata["score"].get<float>(), 0.7f);
}

// Test applying complex metadata filters with OR combination
TEST_F(MetadataFilterTest, ApplyComplexMetadataFiltersOR) {
    // Create test vectors with metadata
    std::vector<Vector> test_vectors;
    
    // Vector 1: Finance category, high score, banking tag
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 0.1f, 0.2f, 0.3f};
    v1.metadata["category"] = "finance";
    v1.metadata["score"] = 0.95f;
    v1.metadata["tags"] = nlohmann::json::array({"banking", "investment"});
    test_vectors.push_back(v1);
    
    // Vector 2: Technology category, medium score, ai tag
    Vector v2;
    v2.id = "vector_2";
    v2.values = {0.1f, 1.0f, 0.2f, 0.8f};
    v2.metadata["category"] = "technology";
    v2.metadata["score"] = 0.75f;
    v2.metadata["tags"] = nlohmann::json::array({"ai", "ml"});
    test_vectors.push_back(v2);
    
    // Vector 3: Finance category, medium score, trading tag
    Vector v3;
    v3.id = "vector_3";
    v3.values = {0.9f, 0.2f, 0.1f, 0.4f};
    v3.metadata["category"] = "finance";
    v3.metadata["score"] = 0.82f;
    v3.metadata["tags"] = nlohmann::json::array({"trading", "cryptocurrency"});
    test_vectors.push_back(v3);
    
    // Test filter for finance OR technology category
    FilterCondition finance_condition;
    finance_condition.field = "metadata.category";
    finance_condition.op = FilterOperator::EQUALS;
    finance_condition.value = "finance";
    
    FilterCondition tech_condition;
    tech_condition.field = "metadata.category";
    tech_condition.op = FilterOperator::EQUALS;
    tech_condition.value = "technology";
    
    std::vector<FilterCondition> conditions = {finance_condition, tech_condition};
    
    auto result = metadata_filter_->apply_filters(conditions, test_vectors, FilterCombination::OR);
    ASSERT_TRUE(result.has_value());
    
    auto filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 3); // v1, v2, and v3 are either finance or technology
    
    // Check that all filtered vectors have category "finance" or "technology"
    for (const auto& v : filtered_vectors) {
        std::string category = v.metadata["category"].get<std::string>();
        EXPECT_TRUE(category == "finance" || category == "technology");
    }
}

// Test applying array-type filters (tags)
TEST_F(MetadataFilterTest, ApplyArrayFilters) {
    // Create test vectors with tag arrays
    std::vector<Vector> test_vectors;
    
    // Vector 1: Has investment and trading tags
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 0.1f, 0.2f, 0.3f};
    v1.metadata["tags"] = nlohmann::json::array({"investment", "trading", "finance"});
    test_vectors.push_back(v1);
    
    // Vector 2: Has ai and ml tags
    Vector v2;
    v2.id = "vector_2";
    v2.values = {0.1f, 1.0f, 0.2f, 0.8f};
    v2.metadata["tags"] = nlohmann::json::array({"ai", "ml", "technology"});
    test_vectors.push_back(v2);
    
    // Vector 3: Has research and clinical tags
    Vector v3;
    v3.id = "vector_3";
    v3.values = {0.9f, 0.2f, 0.1f, 0.4f};
    v3.metadata["tags"] = nlohmann::json::array({"research", "clinical", "healthcare"});
    test_vectors.push_back(v3);
    
    // Test filter for vectors with "investment" tag (using CONTAINS)
    FilterCondition investment_condition;
    investment_condition.field = "metadata.tags";
    investment_condition.op = FilterOperator::CONTAINS;
    investment_condition.value = "investment";
    
    std::vector<FilterCondition> conditions = {investment_condition};
    
    auto result = metadata_filter_->apply_filters(conditions, test_vectors);
    ASSERT_TRUE(result.has_value());
    
    auto filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 1); // Only v1 has investment tag
    
    // Check that the filtered vector has the investment tag
    EXPECT_EQ(filtered_vectors[0].id, "vector_1");
    auto tags = filtered_vectors[0].metadata["tags"].get<std::vector<std::string>>();
    bool has_investment = false;
    for (const auto& tag : tags) {
        if (tag == "investment") {
            has_investment = true;
            break;
        }
    }
    EXPECT_TRUE(has_investment);
    
    // Test filter for vectors with "ai" or "ml" tags (using IN)
    FilterCondition ai_ml_condition;
    ai_ml_condition.field = "metadata.tags";
    ai_ml_condition.op = FilterOperator::IN;
    ai_ml_condition.values = {"ai", "ml"};
    
    std::vector<FilterCondition> conditions2 = {ai_ml_condition};
    
    result = metadata_filter_->apply_filters(conditions2, test_vectors);
    ASSERT_TRUE(result.has_value());
    
    filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 1); // Only v2 has ai or ml tags
    
    // Check that the filtered vector has ai or ml tag
    EXPECT_EQ(filtered_vectors[0].id, "vector_2");
    tags = filtered_vectors[0].metadata["tags"].get<std::vector<std::string>>();
    bool has_ai_or_ml = false;
    for (const auto& tag : tags) {
        if (tag == "ai" || tag == "ml") {
            has_ai_or_ml = true;
            break;
        }
    }
    EXPECT_TRUE(has_ai_or_ml);
}

// Test applying range filters
TEST_F(MetadataFilterTest, ApplyRangeFilters) {
    // Create test vectors with numeric metadata
    std::vector<Vector> test_vectors;
    
    // Vector 1: High score
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 0.1f, 0.2f, 0.3f};
    v1.metadata["score"] = 0.95f;
    v1.metadata["age"] = 25;
    test_vectors.push_back(v1);
    
    // Vector 2: Medium-high score
    Vector v2;
    v2.id = "vector_2";
    v2.values = {0.1f, 1.0f, 0.2f, 0.8f};
    v2.metadata["score"] = 0.75f;
    v2.metadata["age"] = 30;
    test_vectors.push_back(v2);
    
    // Vector 3: Medium score
    Vector v3;
    v3.id = "vector_3";
    v3.values = {0.9f, 0.2f, 0.1f, 0.4f};
    v3.metadata["score"] = 0.50f;
    v3.metadata["age"] = 35;
    test_vectors.push_back(v3);
    
    // Vector 4: Low score
    Vector v4;
    v4.id = "vector_4";
    v4.values = {0.2f, 0.1f, 1.0f, 0.3f};
    v4.metadata["score"] = 0.25f;
    v4.metadata["age"] = 40;
    test_vectors.push_back(v4);
    
    // Test filter for vectors with score >= 0.5
    FilterCondition score_condition;
    score_condition.field = "metadata.score";
    score_condition.op = FilterOperator::GREATER_THAN_OR_EQUAL;
    score_condition.value = "0.5";
    
    std::vector<FilterCondition> conditions = {score_condition};
    
    auto result = metadata_filter_->apply_filters(conditions, test_vectors);
    ASSERT_TRUE(result.has_value());
    
    auto filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 3); // v1, v2, and v3 have score >= 0.5
    
    // Check that all filtered vectors have score >= 0.5
    for (const auto& v : filtered_vectors) {
        float score = v.metadata["score"].get<float>();
        EXPECT_GE(score, 0.5f);
    }
    
    // Test filter for vectors with age between 25 and 35 (inclusive)
    FilterCondition age_min_condition;
    age_min_condition.field = "metadata.age";
    age_min_condition.op = FilterOperator::GREATER_THAN_OR_EQUAL;
    age_min_condition.value = "25";
    
    FilterCondition age_max_condition;
    age_max_condition.field = "metadata.age";
    age_max_condition.op = FilterOperator::LESS_THAN_OR_EQUAL;
    age_max_condition.value = "35";
    
    std::vector<FilterCondition> conditions2 = {age_min_condition, age_max_condition};
    
    result = metadata_filter_->apply_filters(conditions2, test_vectors, FilterCombination::AND);
    ASSERT_TRUE(result.has_value());
    
    filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 3); // v1, v2, and v3 have age between 25 and 35
    
    // Check that all filtered vectors have age between 25 and 35
    for (const auto& v : filtered_vectors) {
        int age = v.metadata["age"].get<int>();
        EXPECT_GE(age, 25);
        EXPECT_LE(age, 35);
    }
}

// Test applying complex nested filters
TEST_F(MetadataFilterTest, ApplyComplexNestedFilters) {
    // Create test vectors with complex metadata
    std::vector<Vector> test_vectors;
    
    // Vector 1: Finance category, high score, investment tag
    Vector v1;
    v1.id = "vector_1";
    v1.values = {1.0f, 0.1f, 0.2f, 0.3f};
    v1.metadata["category"] = "finance";
    v1.metadata["score"] = 0.95f;
    v1.metadata["tags"] = nlohmann::json::array({"investment", "trading"});
    v1.metadata["region"] = "north_america";
    test_vectors.push_back(v1);
    
    // Vector 2: Technology category, medium score, ai tag
    Vector v2;
    v2.id = "vector_2";
    v2.values = {0.1f, 1.0f, 0.2f, 0.8f};
    v2.metadata["category"] = "technology";
    v2.metadata["score"] = 0.75f;
    v2.metadata["tags"] = nlohmann::json::array({"ai", "ml"});
    v2.metadata["region"] = "europe";
    test_vectors.push_back(v2);
    
    // Vector 3: Finance category, low score, trading tag
    Vector v3;
    v3.id = "vector_3";
    v3.values = {0.9f, 0.2f, 0.1f, 0.4f};
    v3.metadata["category"] = "finance";
    v3.metadata["score"] = 0.30f;
    v3.metadata["tags"] = nlohmann::json::array({"trading", "cryptocurrency"});
    v3.metadata["region"] = "north_america";
    test_vectors.push_back(v3);
    
    // Vector 4: Healthcare category, medium-high score, research tag
    Vector v4;
    v4.id = "vector_4";
    v4.values = {0.2f, 0.1f, 1.0f, 0.3f};
    v4.metadata["category"] = "healthcare";
    v4.metadata["score"] = 0.80f;
    v4.metadata["tags"] = nlohmann::json::array({"research", "clinical"});
    v4.metadata["region"] = "asia";
    test_vectors.push_back(v4);
    
    // Create a complex filter:
    // (category = "finance" AND score >= 0.8) OR (category = "technology" AND tags CONTAINS "ai")
    ComplexFilter filter;
    filter.combination = FilterCombination::OR;
    
    // Create nested filter for finance vectors with high score
    auto finance_filter = std::make_unique<ComplexFilter>();
    finance_filter->combination = FilterCombination::AND;
    
    FilterCondition finance_cat_condition;
    finance_cat_condition.field = "metadata.category";
    finance_cat_condition.op = FilterOperator::EQUALS;
    finance_cat_condition.value = "finance";
    finance_filter->conditions.push_back(finance_cat_condition);
    
    FilterCondition high_score_condition;
    high_score_condition.field = "metadata.score";
    high_score_condition.op = FilterOperator::GREATER_THAN_OR_EQUAL;
    high_score_condition.value = "0.8";
    finance_filter->conditions.push_back(high_score_condition);
    
    filter.nested_filters.push_back(std::move(finance_filter));
    
    // Create nested filter for technology vectors with ai tag
    auto tech_filter = std::make_unique<ComplexFilter>();
    tech_filter->combination = FilterCombination::AND;
    
    FilterCondition tech_cat_condition;
    tech_cat_condition.field = "metadata.category";
    tech_cat_condition.op = FilterOperator::EQUALS;
    tech_cat_condition.value = "technology";
    tech_filter->conditions.push_back(tech_cat_condition);
    
    FilterCondition ai_tag_condition;
    ai_tag_condition.field = "metadata.tags";
    ai_tag_condition.op = FilterOperator::CONTAINS;
    ai_tag_condition.value = "ai";
    tech_filter->conditions.push_back(ai_tag_condition);
    
    filter.nested_filters.push_back(std::move(tech_filter));
    
    auto result = metadata_filter_->apply_complex_filters(filter, test_vectors);
    ASSERT_TRUE(result.has_value());
    
    auto filtered_vectors = result.value();
    EXPECT_EQ(filtered_vectors.size(), 2); // v1 (finance + high score) and v2 (technology + ai tag)
    
    // Check that filtered vectors match the criteria
    for (const auto& v : filtered_vectors) {
        std::string category = v.metadata["category"].get<std::string>();
        float score = v.metadata["score"].get<float>();
        auto tags = v.metadata["tags"].get<std::vector<std::string>>();
        
        bool is_high_finance = (category == "finance" && score >= 0.8f);
        bool is_ai_tech = (category == "technology");
        bool has_ai_tag = false;
        for (const auto& tag : tags) {
            if (tag == "ai") {
                has_ai_tag = true;
                break;
            }
        }
        
        EXPECT_TRUE(is_high_finance || (is_ai_tech && has_ai_tag));
    }
}

// Test filter validation
TEST_F(MetadataFilterTest, ValidateFilters) {
    // Test valid filter condition
    FilterCondition valid_condition;
    valid_condition.field = "metadata.category";
    valid_condition.op = FilterOperator::EQUALS;
    valid_condition.value = "finance";
    
    auto result = metadata_filter_->validate_condition(valid_condition);
    EXPECT_TRUE(result.has_value());
    
    // Test invalid filter condition - empty field
    FilterCondition invalid_condition1;
    invalid_condition1.field = "";
    invalid_condition1.op = FilterOperator::EQUALS;
    invalid_condition1.value = "finance";
    
    result = metadata_filter_->validate_condition(invalid_condition1);
    EXPECT_FALSE(result.has_value());
    
    // Test invalid filter condition - invalid operator value
    FilterCondition invalid_condition2;
    invalid_condition2.field = "metadata.category";
    invalid_condition2.op = static_cast<FilterOperator>(999); // Invalid operator
    invalid_condition2.value = "finance";
    
    result = metadata_filter_->validate_condition(invalid_condition2);
    EXPECT_FALSE(result.has_value());
    
    // Test valid complex filter
    ComplexFilter valid_filter;
    valid_filter.combination = FilterCombination::AND;
    valid_filter.conditions.push_back(valid_condition);
    
    result = metadata_filter_->validate_filter(valid_filter);
    EXPECT_TRUE(result.has_value());
    
    // Test invalid complex filter - empty conditions and nested filters
    ComplexFilter invalid_filter;
    invalid_filter.combination = FilterCombination::AND;
    // No conditions or nested filters
    
    result = metadata_filter_->validate_filter(invalid_filter);
    EXPECT_FALSE(result.has_value());
}

// Test field value extraction
TEST_F(MetadataFilterTest, GetFieldValue) {
    // Create a test vector with metadata
    Vector test_vector;
    test_vector.id = "test_vector";
    test_vector.values = {1.0f, 0.1f, 0.2f, 0.3f};
    test_vector.metadata["category"] = "finance";
    test_vector.metadata["score"] = 0.95f;
    test_vector.metadata["tags"] = nlohmann::json::array({"investment", "trading"});
    
    // Test extracting simple field values
    auto result = metadata_filter_->get_field_value("metadata.category", test_vector);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "finance");
    
    result = metadata_filter_->get_field_value("metadata.score", test_vector);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "0.95");
    
    // Test extracting array field values
    auto array_result = metadata_filter_->get_field_values("metadata.tags", test_vector);
    ASSERT_TRUE(array_result.has_value());
    auto tags = array_result.value();
    EXPECT_EQ(tags.size(), 2);
    EXPECT_EQ(tags[0], "investment");
    EXPECT_EQ(tags[1], "trading");
    
    // Test extracting non-existent field
    result = metadata_filter_->get_field_value("metadata.non_existent", test_vector);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "");
}

// Test filter evaluation methods
TEST_F(MetadataFilterTest, EvaluateFilters) {
    // Test string filter evaluation
    EXPECT_TRUE(metadata_filter_->evaluate_string_filter(FilterOperator::EQUALS, "finance", "finance"));
    EXPECT_FALSE(metadata_filter_->evaluate_string_filter(FilterOperator::EQUALS, "finance", "technology"));
    EXPECT_TRUE(metadata_filter_->evaluate_string_filter(FilterOperator::NOT_EQUALS, "finance", "technology"));
    EXPECT_FALSE(metadata_filter_->evaluate_string_filter(FilterOperator::NOT_EQUALS, "finance", "finance"));
    EXPECT_TRUE(metadata_filter_->evaluate_string_filter(FilterOperator::CONTAINS, "financial_data", "data"));
    EXPECT_FALSE(metadata_filter_->evaluate_string_filter(FilterOperator::CONTAINS, "financial_data", "market"));
    
    // Test number filter evaluation
    EXPECT_TRUE(metadata_filter_->evaluate_number_filter(FilterOperator::EQUALS, 0.95, 0.95));
    EXPECT_FALSE(metadata_filter_->evaluate_number_filter(FilterOperator::EQUALS, 0.95, 0.94));
    EXPECT_TRUE(metadata_filter_->evaluate_number_filter(FilterOperator::GREATER_THAN, 0.95, 0.90));
    EXPECT_FALSE(metadata_filter_->evaluate_number_filter(FilterOperator::GREATER_THAN, 0.90, 0.95));
    EXPECT_TRUE(metadata_filter_->evaluate_number_filter(FilterOperator::LESS_THAN, 0.90, 0.95));
    EXPECT_FALSE(metadata_filter_->evaluate_number_filter(FilterOperator::LESS_THAN, 0.95, 0.90));
    
    // Test array operations
    std::vector<std::string> test_array = {"investment", "trading", "finance"};
    EXPECT_TRUE(metadata_filter_->array_contains(test_array, "investment"));
    EXPECT_FALSE(metadata_filter_->array_contains(test_array, "ai"));
    
    std::vector<std::string> test_values = {"investment", "ai"};
    EXPECT_TRUE(metadata_filter_->array_contains_any(test_array, test_values));
    
    std::vector<std::string> all_values = {"investment", "trading"};
    EXPECT_TRUE(metadata_filter_->array_contains_all(test_array, all_values));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}