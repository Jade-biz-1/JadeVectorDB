#include "metadata_filter.h"
#include "models/vector.h"
#include <iostream>
#include <cassert>

using namespace jadevectordb;

void test_geospatial_filtering() {
    std::cout << "Testing geospatial filtering..." << std::endl;
    
    MetadataFilter filter;
    
    // Create a test vector with geospatial metadata
    Vector test_vector;
    test_vector.id = "1";
    test_vector.values = std::vector<float>{1.0f, 2.0f, 3.0f};
    test_vector.metadata.owner = "test";
    test_vector.metadata.tags = {"tag1", "tag2"};
    // For geospatial, we'll use the custom field to store location
    test_vector.metadata.custom["location"] = "37.7749,-122.4194";  // San Francisco coordinates
    
    // Create a geospatial query: within 1000m of San Francisco coordinates
    GeoQuery geo_query(GeospatialOperator::WITHIN_RADIUS, "metadata.custom.location");
    geo_query.center = Point(37.7749, -122.4194);  // Same as the vector location
    geo_query.radius = 1000.0;  // 1000 meters
    
    auto result = filter.applies_to_vector(geo_query, test_vector);
    assert(result.has_value() && result.value() == true);
    
    std::cout << "Geospatial filtering test passed!" << std::endl;
}

void test_temporal_filtering() {
    std::cout << "Testing temporal filtering..." << std::endl;
    
    MetadataFilter filter;
    
    // Create a test vector with temporal metadata
    Vector test_vector;
    test_vector.id = "1";
    test_vector.values = std::vector<float>{1.0f, 2.0f, 3.0f};
    test_vector.metadata.created_at = "2023-05-15 10:30:00";  // Date string
    
    // Create a temporal query: check if date is after a certain time
    TemporalQuery temporal_query(TemporalOperator::AFTER, "metadata.created_at");
    std::tm tm_time = {};
    std::istringstream ss("2023-05-01 00:00:00");
    ss >> std::get_time(&tm_time, "%Y-%m-%d %H:%M:%S");
    temporal_query.time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    
    auto result = filter.applies_to_vector(temporal_query, test_vector);
    assert(result.has_value() && result.value() == true);
    
    std::cout << "Temporal filtering test passed!" << std::endl;
}

void test_full_text_filtering() {
    std::cout << "Testing full-text filtering..." << std::endl;
    
    MetadataFilter filter;
    
    // Create a test vector with text metadata
    Vector test_vector;
    test_vector.id = "1";
    test_vector.values = std::vector<float>{1.0f, 2.0f, 3.0f};
    test_vector.metadata.custom["description"] = "This is a sample document for testing";
    
    // Create a full-text query
    FullTextQuery fulltext_query("metadata.custom.description", "sample", FullTextOperator::MATCHES_ANY_TERM);
    
    auto result = filter.applies_to_vector(fulltext_query, test_vector);
    assert(result.has_value() && result.value() == true);
    
    std::cout << "Full-text filtering test passed!" << std::endl;
}

void test_nested_filtering() {
    std::cout << "Testing nested filtering..." << std::endl;
    
    MetadataFilter filter;
    
    // Create a test vector with nested metadata
    Vector test_vector;
    test_vector.id = "1";
    test_vector.values = std::vector<float>{1.0f, 2.0f, 3.0f};
    test_vector.metadata.custom["category"] = "technology";
    
    // Create a nested query
    NestedQuery nested_query("metadata.custom.category", NestedOperator::MATCHES_PATH, "technology");
    
    auto result = filter.applies_to_vector(nested_query, test_vector);
    assert(result.has_value() && result.value() == true);
    
    std::cout << "Nested filtering test passed!" << std::endl;
}

int main() {
    std::cout << "Starting advanced filtering tests..." << std::endl;
    
    test_geospatial_filtering();
    test_temporal_filtering();
    test_full_text_filtering();
    test_nested_filtering();
    
    std::cout << "All advanced filtering tests passed!" << std::endl;
    return 0;
}