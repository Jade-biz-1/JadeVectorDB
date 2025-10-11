#ifndef JADEVECTORDB_METADATA_FILTER_H
#define JADEVECTORDB_METADATA_FILTER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <regex>
#include <chrono>

#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

namespace jadevectordb {

// Enum for different filter operators
enum class FilterOperator {
    EQUALS,
    NOT_EQUALS,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL,
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    CONTAINS,      // For string/array fields
    NOT_CONTAINS,
    IN,            // For array/enum fields
    NOT_IN,
    EXISTS,        // Check if field exists
    NOT_EXISTS,
    MATCHES_REGEX  // For pattern matching on strings
};

// Filter condition structure
struct FilterCondition {
    std::string field;           // The field to filter on (e.g., "metadata.tags", "metadata.score", "metadata.owner")
    FilterOperator op;           // The operator to use
    std::string value;           // The value to compare against
    std::vector<std::string> values; // For operators that take multiple values (IN, NOT_IN)
    
    FilterCondition() = default;
    FilterCondition(const std::string& f, FilterOperator o, const std::string& v) 
        : field(f), op(o), value(v) {}
    FilterCondition(const std::string& f, FilterOperator o, const std::vector<std::string>& vs) 
        : field(f), op(o), values(vs) {}
};

// Combination type for complex filters
enum class FilterCombination {
    AND,  // All conditions must match
    OR    // Any condition must match
};

// Complex filter combining multiple conditions
struct ComplexFilter {
    FilterCombination combination;                    // How to combine the conditions
    std::vector<FilterCondition> conditions;          // Individual filter conditions
    std::vector<std::unique_ptr<ComplexFilter>> nested_filters;  // Nested complex filters
    
    ComplexFilter(FilterCombination comb = FilterCombination::AND) : combination(comb) {}
};

// Performance optimization settings
struct FilterPerformanceSettings {
    bool enable_short_circuit_evaluation = true;  // Stop evaluating when result is determined
    bool enable_cache_frequent_filters = true;    // Cache results of frequently used filters
    size_t cache_max_size = 1000;                  // Maximum number of cached filter results
    std::chrono::milliseconds cache_ttl{30000};   // Cache TTL (30 seconds)
    bool enable_parallel_processing = true;       // Enable parallel processing for large datasets
    size_t parallel_processing_threshold = 1000;   // Minimum vectors for parallel processing
};

// Metadata filter service class
class MetadataFilter {
private:
    std::shared_ptr<logging::Logger> logger_;
    FilterPerformanceSettings performance_settings_;
    
    // Cache for frequent filter results
    mutable std::unordered_map<std::string, std::pair<std::vector<bool>, std::chrono::steady_clock::time_point>> filter_cache_;
    mutable std::unordered_map<std::string, size_t> filter_usage_count_;

public:
    explicit MetadataFilter(const FilterPerformanceSettings& settings = FilterPerformanceSettings());
    ~MetadataFilter() = default;
    
    // Performance settings management
    void set_performance_settings(const FilterPerformanceSettings& settings);
    FilterPerformanceSettings get_performance_settings() const;
    
    // Apply a single filter condition to a vector
    Result<bool> applies_to_vector(const FilterCondition& condition, const Vector& vector) const;
    
    // Apply a complex filter (with multiple conditions and combinations) to a vector
    Result<bool> applies_to_vector(const ComplexFilter& filter, const Vector& vector) const;
    
    // Apply filters to a collection of vectors
    Result<std::vector<Vector>> apply_filters(
        const std::vector<FilterCondition>& conditions,
        const std::vector<Vector>& vectors,
        FilterCombination combination = FilterCombination::AND) const;
    
    // Apply complex filters to a collection of vectors
    Result<std::vector<Vector>> apply_complex_filters(
        const ComplexFilter& filter,
        const std::vector<Vector>& vectors) const;
    
    // Validate a filter condition
    Result<void> validate_condition(const FilterCondition& condition) const;
    
    // Validate a complex filter
    Result<void> validate_filter(const ComplexFilter& filter) const;
    
    // Parse filter from a string representation (e.g., "metadata.owner=user1 AND metadata.score>=0.8")
    Result<ComplexFilter> parse_filter_from_string(const std::string& filter_string) const;
    
    // Create filter from JSON representation
    Result<ComplexFilter> parse_filter_from_json(const std::string& json_string) const;
    
    // Helper methods to extract values from Vector based on field path
    Result<std::string> get_field_value(const std::string& field_path, const Vector& vector) const;
    Result<std::vector<std::string>> get_field_values(const std::string& field_path, const Vector& vector) const;

private:
    // Helper methods for value comparison
    bool evaluate_string_filter(FilterOperator op, const std::string& value1, const std::string& value2) const;
    bool evaluate_number_filter(FilterOperator op, double value1, double value2) const;
    bool evaluate_boolean_filter(FilterOperator op, bool value1, bool value2) const;
    
    // Helper methods for array operations
    bool array_contains(const std::vector<std::string>& array, const std::string& value) const;
    bool array_contains_any(const std::vector<std::string>& array, const std::vector<std::string>& values) const;
    bool array_contains_all(const std::vector<std::string>& array, const std::vector<std::string>& values) const;
    
    // Parse field path and get the appropriate value
    std::vector<std::string> parse_field_path(const std::string& field_path) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_METADATA_FILTER_H