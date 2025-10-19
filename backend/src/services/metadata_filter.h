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
#include <cmath>

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

// Enum for geospatial operators
enum class GeospatialOperator {
    WITHIN_RADIUS,
    WITHIN_BOUNDING_BOX,
    INTERSECTS_GEOMETRY,
    NEAREST_NEIGHBORS
};

// Structures for geospatial data
struct Point {
    double latitude;
    double longitude;
    
    Point(double lat = 0.0, double lon = 0.0) : latitude(lat), longitude(lon) {}
};

struct BoundingBox {
    Point bottom_left;
    Point top_right;
    
    BoundingBox(const Point& bl = Point(), const Point& tr = Point()) 
        : bottom_left(bl), top_right(tr) {}
};

struct GeoQuery {
    GeospatialOperator op;
    Point center;      // For radius queries
    double radius;     // For radius queries (in meters)
    BoundingBox bbox;  // For bounding box queries
    std::string field; // The field to apply the geo query to
    
    GeoQuery(GeospatialOperator operation, const std::string& field_name) 
        : op(operation), field(field_name), radius(0.0) {}
};

// Additional structures for temporal filtering
enum class TemporalOperator {
    BEFORE,
    AFTER,
    BETWEEN,
    WITHIN_DURATION,
    AT_T,
    NOT_AT_T
};

struct TimeRange {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    
    TimeRange(const std::chrono::system_clock::time_point& s = std::chrono::system_clock::time_point{}, 
              const std::chrono::system_clock::time_point& e = std::chrono::system_clock::time_point{}) 
        : start(s), end(e) {}
};

struct TemporalQuery {
    TemporalOperator op;
    std::chrono::system_clock::time_point time_point;  // For BEFORE, AFTER, AT_T
    TimeRange time_range;                              // For BETWEEN operations
    std::chrono::milliseconds duration;                // For WITHIN_DURATION
    std::string field;                                 // The field to apply the temporal query to
    
    explicit TemporalQuery(TemporalOperator operation, const std::string& field_name) 
        : op(operation), field(field_name), duration(0) {}
};

// Additional structures for nested object filtering
enum class NestedOperator {
    EXISTS_PATH,
    MATCHES_PATH,
    COUNT_GREATER_THAN,
    COUNT_LESS_THAN,
    COUNT_EQUALS,
    SUBPATH_MATCHES
};

struct NestedQuery {
    std::string path;           // The nested path to query (e.g., "metadata.tags[0].name")
    NestedOperator op;          // The operator to apply at the nested path
    std::string value;          // The value to match
    std::vector<std::string> values; // For operators that take multiple values
    std::string nested_path;    // For subpath operations
    
    NestedQuery(const std::string& p, NestedOperator o, const std::string& v) 
        : path(p), op(o), value(v) {}
};

// For full-text search
enum class FullTextOperator {
    MATCHES_ALL_TERMS,
    MATCHES_ANY_TERM,
    MATCHES_PHRASE,
    MATCHES_REGEX,
    FUZZY_MATCH,
    ELASTICSEARCH_QUERY  // For integration with Elasticsearch
};

struct FullTextQuery {
    std::string field;              // The text field to search
    std::string query_text;         // The text to search for
    FullTextOperator op;            // The search operation to perform
    int max_edit_distance;          // For fuzzy matching
    bool case_sensitive;            // Whether the search is case sensitive
    std::vector<std::string> terms; // For multiple-term queries
    std::string elasticsearch_query; // Raw Elasticsearch query string for ES integration
    
    FullTextQuery(const std::string& f, const std::string& qt, FullTextOperator operation) 
        : field(f), query_text(qt), op(operation), max_edit_distance(2), case_sensitive(false) {}
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

// Elasticsearch configuration
struct ElasticsearchConfig {
    std::string host;
    int port;
    std::string index_name;
    std::string username;
    std::string password;
    bool use_ssl;
    
    ElasticsearchConfig() : host("localhost"), port(9200), index_name("jadevectordb"), 
                           use_ssl(false) {}
};

// Metadata filter service class
class MetadataFilter {
private:
    std::shared_ptr<logging::Logger> logger_;
    FilterPerformanceSettings performance_settings_;
    
    // Cache for frequent filter results
    mutable std::unordered_map<std::string, std::pair<std::vector<bool>, std::chrono::steady_clock::time_point>> filter_cache_;
    mutable std::unordered_map<std::string, size_t> filter_usage_count_;
    
    // Elasticsearch configuration
    ElasticsearchConfig elasticsearch_config_;

public:
    explicit MetadataFilter(const FilterPerformanceSettings& settings = FilterPerformanceSettings());
    ~MetadataFilter() = default;
    
    // Initialize the metadata filter service
    Result<void> initialize();
    
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
    
    // New advanced filtering methods
    // Geospatial filtering
    Result<bool> applies_to_vector(const GeoQuery& geo_query, const Vector& vector) const;
    Result<std::vector<Vector>> apply_geo_filters(const std::vector<GeoQuery>& geo_queries,
                                                  const std::vector<Vector>& vectors) const;
    
    // Temporal filtering
    Result<bool> applies_to_vector(const TemporalQuery& temporal_query, const Vector& vector) const;
    Result<std::vector<Vector>> apply_temporal_filters(const std::vector<TemporalQuery>& temporal_queries,
                                                       const std::vector<Vector>& vectors) const;
    
    // Nested object filtering
    Result<bool> applies_to_vector(const NestedQuery& nested_query, const Vector& vector) const;
    Result<std::vector<Vector>> apply_nested_filters(const std::vector<NestedQuery>& nested_queries,
                                                     const std::vector<Vector>& vectors) const;
    
    // Full-text search
    Result<bool> applies_to_vector(const FullTextQuery& fulltext_query, const Vector& vector) const;
    Result<std::vector<Vector>> apply_fulltext_filters(const std::vector<FullTextQuery>& fulltext_queries,
                                                       const std::vector<Vector>& vectors) const;
    
    // Fuzzy matching (as part of full-text search)
    Result<bool> applies_to_vector_fuzzy(const FullTextQuery& fuzzy_query, const Vector& vector) const;

    // Method to set Elasticsearch configuration
    void set_elasticsearch_config(const ElasticsearchConfig& config);

    // Method for Elasticsearch integration
    Result<std::vector<std::string>> search_with_elasticsearch(const std::string& elasticsearch_query) const;
    Result<bool> applies_to_vector_elasticsearch(const FullTextQuery& es_query, const Vector& vector) const;

    // Method to apply all types of advanced filters together
    Result<std::vector<Vector>> apply_advanced_filters(
        const std::vector<FilterCondition>& conditions,
        const std::vector<GeoQuery>& geo_queries,
        const std::vector<TemporalQuery>& temporal_queries,
        const std::vector<NestedQuery>& nested_queries,
        const std::vector<FullTextQuery>& fulltext_queries,
        const std::vector<Vector>& vectors) const;

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
    
    // Geospatial helper methods
    double calculate_haversine_distance(const Point& p1, const Point& p2) const;
    bool is_point_within_radius(const Point& point, const Point& center, double radius_meters) const;
    bool is_point_within_bounding_box(const Point& point, const BoundingBox& bbox) const;
    
    // Temporal helper methods
    bool evaluate_temporal_filter(const TemporalQuery& query, const Vector& vector) const;
    
    // Nested object helper methods
    Result<std::string> get_nested_field_value(const std::string& path, const Vector& vector) const;
    Result<std::vector<std::string>> get_nested_field_values(const std::string& path, const Vector& vector) const;
    
    // Full-text search helper methods
    bool evaluate_fulltext_filter(const FullTextQuery& query, const Vector& vector) const;
    double calculate_levenshtein_distance(const std::string& s1, const std::string& s2) const;
    std::vector<std::string> tokenize_text(const std::string& text, const std::string& delimiter = " ") const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_METADATA_FILTER_H