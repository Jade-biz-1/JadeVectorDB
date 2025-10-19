#include "metadata_filter.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_set>
#include <chrono>
#include <cmath>

namespace jadevectordb {

MetadataFilter::MetadataFilter(const FilterPerformanceSettings& settings) 
    : performance_settings_(settings) {
    logger_ = logging::LoggerManager::get_logger("MetadataFilter");
    // Initialize default Elasticsearch configuration
    elasticsearch_config_.host = "localhost";
    elasticsearch_config_.port = 9200;
    elasticsearch_config_.index_name = "jadevectordb";
    elasticsearch_config_.use_ssl = false;
}

Result<void> MetadataFilter::initialize() {
    // Perform any necessary initialization for the metadata filter service
    // Currently, the constructor already sets up basic configuration,
    // so this can be an empty implementation, but we can add validation if needed
    
    // For now, just log that initialization is complete
    if (logger_) {
        LOG_INFO(logger_, "MetadataFilter service initialized successfully");
    }
    
    return {};
}

void MetadataFilter::set_performance_settings(const FilterPerformanceSettings& settings) {
    performance_settings_ = settings;
}

FilterPerformanceSettings MetadataFilter::get_performance_settings() const {
    return performance_settings_;
}

Result<bool> MetadataFilter::applies_to_vector(const FilterCondition& condition, const Vector& vector) const {
    // Validate the condition first
    auto validation_result = validate_condition(condition);
    if (!validation_result.has_value()) {
        return false;
    }
    
    // Special handling for array-type fields like tags
    if ((condition.field == "metadata.tags" || condition.field == "metadata.permissions") && 
        (condition.op == FilterOperator::IN || condition.op == FilterOperator::NOT_IN ||
         condition.op == FilterOperator::CONTAINS || condition.op == FilterOperator::NOT_CONTAINS)) {
        
        auto array_result = get_field_values(condition.field, vector);
        if (!array_result.has_value()) {
            // If field doesn't exist, handle based on operator
            if (condition.op == FilterOperator::EXISTS) {
                return false;
            } else if (condition.op == FilterOperator::NOT_EXISTS) {
                return true;
            }
            return false;
        }
        
        const auto& array_values = array_result.value();
        
        switch (condition.op) {
            case FilterOperator::IN:
                return array_contains_any(array_values, condition.values);
                
            case FilterOperator::NOT_IN:
                return !array_contains_any(array_values, condition.values);
                
            case FilterOperator::CONTAINS:
                return array_contains(array_values, condition.value);
                
            case FilterOperator::NOT_CONTAINS:
                return !array_contains(array_values, condition.value);
                
            default:
                // For other operators, use the default behavior
                break;
        }
    }
    
    // Get the field value from the vector
    auto field_result = get_field_value(condition.field, vector);
    if (!field_result.has_value()) {
        // If field doesn't exist, handle based on operator
        if (condition.op == FilterOperator::EXISTS) {
            return false;  // Field doesn't exist
        } else if (condition.op == FilterOperator::NOT_EXISTS) {
            return true;   // Field doesn't exist, so "not exists" is true
        }
        return false; // For other operators, assume false if field doesn't exist
    }
    
    std::string field_value = field_result.value();
    
    // Handle different operators
    switch (condition.op) {
        case FilterOperator::EQUALS:
            return field_value == condition.value;
            
        case FilterOperator::NOT_EQUALS:
            return field_value != condition.value;
            
        case FilterOperator::GREATER_THAN:
        case FilterOperator::GREATER_THAN_OR_EQUAL:
        case FilterOperator::LESS_THAN:
        case FilterOperator::LESS_THAN_OR_EQUAL: {
            try {
                double field_num = std::stod(field_value);
                double cond_num = std::stod(condition.value);
                return evaluate_number_filter(condition.op, field_num, cond_num);
            } catch (const std::exception&) {
                // If conversion fails, the values aren't numeric, return false
                return false;
            }
        }
        
        case FilterOperator::CONTAINS:
            return field_value.find(condition.value) != std::string::npos;
            
        case FilterOperator::NOT_CONTAINS:
            return field_value.find(condition.value) == std::string::npos;
            
        case FilterOperator::EXISTS:
            return true; // We already confirmed it exists above
            
        case FilterOperator::NOT_EXISTS:
            return false; // We confirmed it exists above
            
        case FilterOperator::MATCHES_REGEX: {
            try {
                std::regex pattern(condition.value);
                return std::regex_match(field_value, pattern);
            } catch (const std::regex_error&) {
                LOG_ERROR(logger_, "Invalid regex pattern: " << condition.value);
                return false;
            }
        }
        
        case FilterOperator::IN: {
            return std::find(condition.values.begin(), condition.values.end(), field_value) != condition.values.end();
        }
        
        case FilterOperator::NOT_IN: {
            return std::find(condition.values.begin(), condition.values.end(), field_value) == condition.values.end();
        }
    }
    
    // Unknown operator
    LOG_WARN(logger_, "Unknown filter operator: " << static_cast<int>(condition.op));
    return false;
}

Result<bool> MetadataFilter::applies_to_vector(const ComplexFilter& filter, const Vector& vector) const {
    // Validate the filter first
    auto validation_result = validate_filter(filter);
    if (!validation_result.has_value()) {
        return false;
    }
    
    if (filter.conditions.empty() && filter.nested_filters.empty()) {
        // Empty filter matches everything
        return true;
    }
    
    bool result = true;
    
    if (filter.combination == FilterCombination::AND) {
        // For AND, all conditions must match
        
        // Check simple conditions
        for (const auto& condition : filter.conditions) {
            auto condition_result = applies_to_vector(condition, vector);
            if (!condition_result.has_value() || !condition_result.value()) {
                return false; // Short-circuit on first failure
            }
        }
        
        // Check nested complex filters
        for (const auto& nested_filter : filter.nested_filters) {
            auto nested_result = applies_to_vector(*nested_filter, vector);
            if (!nested_result.has_value() || !nested_result.value()) {
                return false; // Short-circuit on first failure
            }
        }
        
        return true;
    } else { // FilterCombination::OR
        // For OR, at least one condition must match
        
        // Check simple conditions first
        for (const auto& condition : filter.conditions) {
            auto condition_result = applies_to_vector(condition, vector);
            if (condition_result.has_value() && condition_result.value()) {
                return true; // Short-circuit on first success
            }
        }
        
        // Check nested complex filters
        for (const auto& nested_filter : filter.nested_filters) {
            auto nested_result = applies_to_vector(*nested_filter, vector);
            if (nested_result.has_value() && nested_result.value()) {
                return true; // Short-circuit on first success
            }
        }
        
        return false;
    }
}

Result<std::vector<Vector>> MetadataFilter::apply_filters(
    const std::vector<FilterCondition>& conditions,
    const std::vector<Vector>& vectors,
    FilterCombination combination) const {
    
    // Performance optimization: For empty conditions, return all vectors
    if (conditions.empty()) {
        return vectors;
    }
    
    // Performance optimization: For very large datasets, consider parallel processing
    if (performance_settings_.enable_parallel_processing && 
        vectors.size() > performance_settings_.parallel_processing_threshold) {
        LOG_DEBUG(logger_, "Using parallel processing for large dataset (" << vectors.size() << " vectors)");
        // In a full implementation, this would use parallel processing
        // For now, we'll continue with sequential processing
    }
    
    std::vector<Vector> result;
    
    // Check if we can use cached results for frequently used filters
    std::string cache_key;
    bool use_cache = performance_settings_.enable_cache_frequent_filters && conditions.size() == 1;
    
    if (use_cache) {
        const auto& condition = conditions[0];
        cache_key = condition.field + "_" + std::to_string(static_cast<int>(condition.op)) + "_" + 
                   condition.value;
        
        // Update usage count for cache management
        filter_usage_count_[cache_key]++;
        
        // Check if we have a valid cached result
        auto cache_it = filter_cache_.find(cache_key);
        if (cache_it != filter_cache_.end()) {
            auto now = std::chrono::steady_clock::now();
            if (now - cache_it->second.second < performance_settings_.cache_ttl) {
                // Use cached results
                const auto& cached_matches = cache_it->second.first;
                if (cached_matches.size() == vectors.size()) {
                    result.reserve(std::count(cached_matches.begin(), cached_matches.end(), true));
                    for (size_t i = 0; i < vectors.size(); ++i) {
                        if (cached_matches[i]) {
                            result.push_back(vectors[i]);
                        }
                    }
                    LOG_DEBUG(logger_, "Used cached filter results for " << cache_key);
                    return result;
                }
            } else {
                // Expired cache entry, remove it
                filter_cache_.erase(cache_it);
            }
        }
    }
    
    // Process vectors sequentially
    std::vector<bool> matches_cache;  // For caching individual vector matches
    bool should_cache = use_cache && filter_usage_count_[cache_key] > 5;  // Cache after 5 uses
    
    if (should_cache) {
        matches_cache.reserve(vectors.size());
    }
    
    for (const auto& vector : vectors) {
        bool should_include = true;
        
        if (combination == FilterCombination::AND) {
            // All conditions must match
            // Performance optimization: Short-circuit evaluation
            if (performance_settings_.enable_short_circuit_evaluation) {
                for (const auto& condition : conditions) {
                    auto condition_result = applies_to_vector(condition, vector);
                    if (!condition_result.has_value() || !condition_result.value()) {
                        should_include = false;
                        break; // Short-circuit on first failure
                    }
                }
            } else {
                // Evaluate all conditions without short-circuiting (for consistency checking)
                should_include = true;
                for (const auto& condition : conditions) {
                    auto condition_result = applies_to_vector(condition, vector);
                    if (!condition_result.has_value() || !condition_result.value()) {
                        should_include = false;
                    }
                }
            }
        } else { // OR
            // At least one condition must match
            // Performance optimization: Short-circuit evaluation
            if (performance_settings_.enable_short_circuit_evaluation) {
                should_include = false;
                for (const auto& condition : conditions) {
                    auto condition_result = applies_to_vector(condition, vector);
                    if (condition_result.has_value() && condition_result.value()) {
                        should_include = true;
                        break; // Short-circuit on first success
                    }
                }
            } else {
                // Evaluate all conditions without short-circuiting
                should_include = false;
                for (const auto& condition : conditions) {
                    auto condition_result = applies_to_vector(condition, vector);
                    if (condition_result.has_value() && condition_result.value()) {
                        should_include = true;
                    }
                }
            }
        }
        
        if (should_include) {
            result.push_back(vector);
        }
        
        // Update cache if needed
        if (should_cache) {
            matches_cache.push_back(should_include);
        }
    }
    
    // Store results in cache if applicable
    if (should_cache && !matches_cache.empty()) {
        auto now = std::chrono::steady_clock::now();
        filter_cache_[cache_key] = std::make_pair(std::move(matches_cache), now);
        
        // Clean up old cache entries if cache is too large
        if (filter_cache_.size() > performance_settings_.cache_max_size) {
            auto oldest_it = filter_cache_.begin();
            auto oldest_time = oldest_it->second.second;
            
            for (auto it = filter_cache_.begin(); it != filter_cache_.end(); ++it) {
                if (it->second.second < oldest_time) {
                    oldest_it = it;
                    oldest_time = it->second.second;
                }
            }
            
            filter_cache_.erase(oldest_it);
        }
        
        LOG_DEBUG(logger_, "Cached filter results for " << cache_key << " (" << matches_cache.size() << " vectors)");
    }
    
    return result;
}

Result<std::vector<Vector>> MetadataFilter::apply_complex_filters(
    const ComplexFilter& filter,
    const std::vector<Vector>& vectors) const {
    
    std::vector<Vector> result;
    
    for (const auto& vector : vectors) {
        auto applies_result = applies_to_vector(filter, vector);
        if (applies_result.has_value() && applies_result.value()) {
            result.push_back(vector);
        }
    }
    
    return result;
}

Result<void> MetadataFilter::validate_condition(const FilterCondition& condition) const {
    // Check if field path is valid
    if (condition.field.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Filter condition field cannot be empty");
    }
    
    // Check if operator is valid
    if (static_cast<int>(condition.op) < 0 || 
        static_cast<int>(condition.op) >= static_cast<int>(FilterOperator::MATCHES_REGEX) + 2) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid filter operator");
    }
    
    // For certain operators, value is required
    if ((condition.op == FilterOperator::EQUALS || condition.op == FilterOperator::NOT_EQUALS ||
         condition.op == FilterOperator::GREATER_THAN || condition.op == FilterOperator::GREATER_THAN_OR_EQUAL ||
         condition.op == FilterOperator::LESS_THAN || condition.op == FilterOperator::LESS_THAN_OR_EQUAL ||
         condition.op == FilterOperator::CONTAINS || condition.op == FilterOperator::NOT_CONTAINS ||
         condition.op == FilterOperator::MATCHES_REGEX) && condition.value.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Value required for operator " + 
                    std::to_string(static_cast<int>(condition.op)));
    }
    
    // For IN/NOT_IN operators, values list is required
    if ((condition.op == FilterOperator::IN || condition.op == FilterOperator::NOT_IN) && condition.values.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Values list required for IN/NOT_IN operators");
    }
    
    return {};
}

Result<void> MetadataFilter::validate_filter(const ComplexFilter& filter) const {
    // Validate all simple conditions
    for (const auto& condition : filter.conditions) {
        auto condition_validation = validate_condition(condition);
        if (!condition_validation.has_value()) {
            return condition_validation;
        }
    }
    
    // Validate all nested filters
    for (const auto& nested_filter : filter.nested_filters) {
        auto nested_validation = validate_filter(*nested_filter);
        if (!nested_validation.has_value()) {
            return nested_validation;
        }
    }
    
    return {};
}

Result<ComplexFilter> MetadataFilter::parse_filter_from_string(const std::string& filter_string) const {
    // This is a simplified implementation for parsing filter strings
    // In a full implementation, you would have a proper parser for filter query language
    
    // For now, return an error since this requires a more complex parser
    RETURN_ERROR(ErrorCode::NOT_IMPLEMENTED, "Filter string parsing not fully implemented");
}

Result<ComplexFilter> MetadataFilter::parse_filter_from_json(const std::string& json_string) const {
    // This is a simplified implementation for parsing JSON filters
    // In a full implementation, you would parse the JSON and convert to ComplexFilter
    
    // For now, return an error since this requires a JSON parser and conversion logic
    RETURN_ERROR(ErrorCode::NOT_IMPLEMENTED, "Filter JSON parsing not fully implemented");
}

Result<std::string> MetadataFilter::get_field_value(const std::string& field_path, const Vector& vector) const {
    // Parse field path like "metadata.owner" or "metadata.custom.field"
    auto path_parts = parse_field_path(field_path);
    
    if (path_parts.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid field path: " + field_path);
    }
    
    // Handle the field path based on its structure
    if (path_parts[0] == "metadata") {
        if (path_parts.size() < 2) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Incomplete metadata field path: " + field_path);
        }
        
        std::string sub_field = path_parts[1];
        
        if (sub_field == "owner") {
            return vector.metadata.owner;
        } else if (sub_field == "category") {
            return vector.metadata.category;
        } else if (sub_field == "status") {
            return vector.metadata.status;
        } else if (sub_field == "source") {
            return vector.metadata.source;
        } else if (sub_field == "created_at") {
            return vector.metadata.created_at;
        } else if (sub_field == "updated_at") {
            return vector.metadata.updated_at;
        } else if (sub_field == "score") {
            // Handle numeric fields - convert to string for comparison
            return std::to_string(vector.metadata.score);
        } else if (sub_field == "permissions") {
            // For array fields, we might return a joined string or handle specially
            std::string result;
            for (size_t i = 0; i < vector.metadata.permissions.size(); ++i) {
                if (i > 0) result += ",";
                result += vector.metadata.permissions[i];
            }
            return result;
        } else if (sub_field == "tags") {
            // For array fields like tags
            std::string result;
            for (size_t i = 0; i < vector.metadata.tags.size(); ++i) {
                if (i > 0) result += ",";
                result += vector.metadata.tags[i];
            }
            return result;
        } else if (sub_field == "custom" && path_parts.size() > 2) {
            // Handle nested custom fields like "metadata.custom.fieldname"
            std::string custom_key = path_parts[2];
            auto it = vector.metadata.custom.find(custom_key);
            if (it != vector.metadata.custom.end()) {
                return it->second; // Return the nlohmann::json as string
            }
        } else if (sub_field == "custom") {
            // Handle other custom fields
            if (path_parts.size() > 2) {
                std::string custom_key = path_parts[2];
                auto it = vector.metadata.custom.find(custom_key);
                if (it != vector.metadata.custom.end()) {
                    return it->second.get<std::string>();
                }
            }
        }
    }
    
    // If field is not found, return empty string
    // In a real implementation, you might want to return a special "not found" indicator
    return std::string{};
}

Result<std::vector<std::string>> MetadataFilter::get_field_values(const std::string& field_path, const Vector& vector) const {
    // Parse field path like "metadata.tags" or "metadata.permissions"
    auto path_parts = parse_field_path(field_path);
    
    if (path_parts.empty() || path_parts[0] != "metadata" || path_parts.size() < 2) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid array field path: " + field_path);
    }
    
    std::string sub_field = path_parts[1];
    
    if (sub_field == "tags") {
        return vector.metadata.tags;
    } else if (sub_field == "permissions") {
        return vector.metadata.permissions;
    }
    
    // If field is not an array field, return empty vector
    return std::vector<std::string>{};
}

bool MetadataFilter::evaluate_string_filter(FilterOperator op, const std::string& value1, const std::string& value2) const {
    switch (op) {
        case FilterOperator::EQUALS:
            return value1 == value2;
        case FilterOperator::NOT_EQUALS:
            return value1 != value2;
        case FilterOperator::CONTAINS:
            return value1.find(value2) != std::string::npos;
        case FilterOperator::NOT_CONTAINS:
            return value1.find(value2) == std::string::npos;
        default:
            return false;
    }
}

bool MetadataFilter::evaluate_number_filter(FilterOperator op, double value1, double value2) const {
    switch (op) {
        case FilterOperator::EQUALS:
        case FilterOperator::CONTAINS: // For numeric comparisons
            return value1 == value2;
        case FilterOperator::NOT_EQUALS:
            return value1 != value2;
        case FilterOperator::GREATER_THAN:
            return value1 > value2;
        case FilterOperator::GREATER_THAN_OR_EQUAL:
            return value1 >= value2;
        case FilterOperator::LESS_THAN:
            return value1 < value2;
        case FilterOperator::LESS_THAN_OR_EQUAL:
            return value1 <= value2;
        default:
            return false;
    }
}

bool MetadataFilter::evaluate_boolean_filter(FilterOperator op, bool value1, bool value2) const {
    switch (op) {
        case FilterOperator::EQUALS:
            return value1 == value2;
        case FilterOperator::NOT_EQUALS:
            return value1 != value2;
        default:
            return false;
    }
}

bool MetadataFilter::array_contains(const std::vector<std::string>& array, const std::string& value) const {
    return std::find(array.begin(), array.end(), value) != array.end();
}

bool MetadataFilter::array_contains_any(const std::vector<std::string>& array, const std::vector<std::string>& values) const {
    for (const auto& value : values) {
        if (std::find(array.begin(), array.end(), value) != array.end()) {
            return true;
        }
    }
    return false;
}

bool MetadataFilter::array_contains_all(const std::vector<std::string>& array, const std::vector<std::string>& values) const {
    for (const auto& value : values) {
        if (std::find(array.begin(), array.end(), value) == array.end()) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> MetadataFilter::parse_field_path(const std::string& field_path) const {
    std::vector<std::string> parts;
    std::string current_part;
    
    for (char c : field_path) {
        if (c == '.') {
            if (!current_part.empty()) {
                parts.push_back(current_part);
                current_part.clear();
            }
        } else {
            current_part += c;
        }
    }
    
    if (!current_part.empty()) {
        parts.push_back(current_part);
    }
    
    return parts;
}

// Geospatial filtering implementations
Result<bool> MetadataFilter::applies_to_vector(const GeoQuery& geo_query, const Vector& vector) const {
    // Get the location value from the vector's metadata
    auto field_result = get_field_value(geo_query.field, vector);
    if (!field_result.has_value()) {
        return false; // Field doesn't exist
    }
    
    std::string location_str = field_result.value();
    
    // Parse location string in format "latitude,longitude" or similar
    Point location_point;
    
    // Try to parse the location string (format: "lat,lon")
    size_t comma_pos = location_str.find(',');
    if (comma_pos != std::string::npos) {
        try {
            std::string lat_str = location_str.substr(0, comma_pos);
            std::string lon_str = location_str.substr(comma_pos + 1);
            
            location_point.latitude = std::stod(lat_str);
            location_point.longitude = std::stod(lon_str);
        } catch (const std::exception& e) {
            LOG_WARN(logger_, "Failed to parse location string: " << location_str << ", error: " << e.what());
            return false;
        }
    } else {
        // If we can't parse the location string, return false
        LOG_WARN(logger_, "Invalid location format: " << location_str);
        return false;
    }
    
    // Apply the geospatial operation
    switch (geo_query.op) {
        case GeospatialOperator::WITHIN_RADIUS:
            return is_point_within_radius(location_point, geo_query.center, geo_query.radius);
            
        case GeospatialOperator::WITHIN_BOUNDING_BOX:
            return is_point_within_bounding_box(location_point, geo_query.bbox);
            
        case GeospatialOperator::INTERSECTS_GEOMETRY:
            // For now, we'll implement a simplified version that checks if the point is within a bounding box
            // In a full implementation, this would handle more complex geometries
            return is_point_within_bounding_box(location_point, geo_query.bbox);
            
        case GeospatialOperator::NEAREST_NEIGHBORS:
            // This would typically be handled by the search system, not filtering
            // For now, return true as a placeholder
            LOG_WARN(logger_, "NEAREST_NEIGHBORS operator is not properly implemented for filtering");
            return true;
            
        default:
            LOG_WARN(logger_, "Unknown geospatial operator: " << static_cast<int>(geo_query.op));
            return false;
    }
}

Result<std::vector<Vector>> MetadataFilter::apply_geo_filters(
    const std::vector<GeoQuery>& geo_queries,
    const std::vector<Vector>& vectors) const {
    
    std::vector<Vector> result;
    
    for (const auto& vector : vectors) {
        bool should_include = true;
        
        // All geo queries must match (AND operation)
        for (const auto& geo_query : geo_queries) {
            auto geo_result = applies_to_vector(geo_query, vector);
            if (!geo_result.has_value() || !geo_result.value()) {
                should_include = false;
                break;
            }
        }
        
        if (should_include) {
            result.push_back(vector);
        }
    }
    
    return result;
}

Result<bool> MetadataFilter::applies_to_vector(const TemporalQuery& temporal_query, const Vector& vector) const {
    return evaluate_temporal_filter(temporal_query, vector);
}

Result<std::vector<Vector>> MetadataFilter::apply_temporal_filters(
    const std::vector<TemporalQuery>& temporal_queries,
    const std::vector<Vector>& vectors) const {
    
    std::vector<Vector> result;
    
    for (const auto& vector : vectors) {
        bool should_include = true;
        
        // All temporal queries must match (AND operation)
        for (const auto& temporal_query : temporal_queries) {
            auto temporal_result = applies_to_vector(temporal_query, vector);
            if (!temporal_result.has_value() || !temporal_result.value()) {
                should_include = false;
                break;
            }
        }
        
        if (should_include) {
            result.push_back(vector);
        }
    }
    
    return result;
}

Result<bool> MetadataFilter::applies_to_vector(const NestedQuery& nested_query, const Vector& vector) const {
    // For now, we'll use our existing get_nested_field_value helper
    auto field_result = get_nested_field_value(nested_query.path, vector);
    if (!field_result.has_value()) {
        return false;
    }
    
    std::string field_value = field_result.value();
    
    // Apply the operator based on the type of comparison
    switch (nested_query.op) {
        case NestedOperator::EXISTS_PATH:
            return !field_value.empty(); // If we got a value, the path exists
            
        case NestedOperator::MATCHES_PATH:
            // For string comparison
            return field_value == nested_query.value;
            
        case NestedOperator::COUNT_GREATER_THAN: {
            // Handle array/object counts
            // Parse the field value as JSON to get the count
            try {
                auto json_value = nlohmann::json::parse(field_value);
                size_t count = 0;
                
                if (json_value.is_array()) {
                    count = json_value.size();
                } else if (json_value.is_object()) {
                    count = json_value.size(); // For objects, count number of keys
                } else {
                    // If it's a primitive value, we can't count it meaningfully
                    return false;
                }
                
                size_t threshold = std::stoi(nested_query.value);
                return count > threshold;
            } catch (const std::exception& e) {
                LOG_WARN(logger_, "Failed to parse JSON for count comparison: " << e.what());
                return false;
            }
        }
            
        case NestedOperator::COUNT_LESS_THAN: {
            // Handle array/object counts
            try {
                auto json_value = nlohmann::json::parse(field_value);
                size_t count = 0;
                
                if (json_value.is_array()) {
                    count = json_value.size();
                } else if (json_value.is_object()) {
                    count = json_value.size();
                } else {
                    return false;
                }
                
                size_t threshold = std::stoi(nested_query.value);
                return count < threshold;
            } catch (const std::exception& e) {
                LOG_WARN(logger_, "Failed to parse JSON for count comparison: " << e.what());
                return false;
            }
        }
            
        case NestedOperator::COUNT_EQUALS: {
            // Handle array/object counts
            try {
                auto json_value = nlohmann::json::parse(field_value);
                size_t count = 0;
                
                if (json_value.is_array()) {
                    count = json_value.size();
                } else if (json_value.is_object()) {
                    count = json_value.size();
                } else {
                    return false;
                }
                
                size_t threshold = std::stoi(nested_query.value);
                return count == threshold;
            } catch (const std::exception& e) {
                LOG_WARN(logger_, "Failed to parse JSON for count comparison: " << e.what());
                return false;
            }
        }
            
        case NestedOperator::SUBPATH_MATCHES: {
            // This would handle nested subpath matching
            // For now, we'll use a simple approach - try to parse as JSON and search for the nested value
            try {
                auto json_value = nlohmann::json::parse(field_value);
                
                // For subpath matching, we need to navigate to the nested path within the current value
                if (!nested_query.nested_path.empty()) {
                    // This is a simplified implementation that treats nested_path as a simple property name
                    if (json_value.contains(nested_query.nested_path)) {
                        auto nested_value = json_value[nested_query.nested_path];
                        return nested_value.dump() == nested_query.value;
                    }
                } else {
                    // If no specific subpath, just check if the value matches
                    return field_value == nested_query.value;
                }
            } catch (const std::exception& e) {
                LOG_WARN(logger_, "Failed to parse JSON for subpath matching: " << e.what());
                return false;
            }
            return false;
        }
            
        default:
            LOG_WARN(logger_, "Unknown nested operator: " << static_cast<int>(nested_query.op));
            return false;
    }
}

Result<std::vector<Vector>> MetadataFilter::apply_nested_filters(
    const std::vector<NestedQuery>& nested_queries,
    const std::vector<Vector>& vectors) const {
    
    std::vector<Vector> result;
    
    for (const auto& vector : vectors) {
        bool should_include = true;
        
        // All nested queries must match (AND operation)
        for (const auto& nested_query : nested_queries) {
            auto nested_result = applies_to_vector(nested_query, vector);
            if (!nested_result.has_value() || !nested_result.value()) {
                should_include = false;
                break;
            }
        }
        
        if (should_include) {
            result.push_back(vector);
        }
    }
    
    return result;
}

Result<bool> MetadataFilter::applies_to_vector(const FullTextQuery& fulltext_query, const Vector& vector) const {
    // Handle Elasticsearch queries specially
    if (fulltext_query.op == FullTextOperator::ELASTICSEARCH_QUERY) {
        return applies_to_vector_elasticsearch(fulltext_query, vector);
    }
    
    return evaluate_fulltext_filter(fulltext_query, vector);
}

Result<std::vector<Vector>> MetadataFilter::apply_fulltext_filters(
    const std::vector<FullTextQuery>& fulltext_queries,
    const std::vector<Vector>& vectors) const {
    
    std::vector<Vector> result;
    
    for (const auto& vector : vectors) {
        bool should_include = true;
        
        // All fulltext queries must match (AND operation)
        for (const auto& fulltext_query : fulltext_queries) {
            auto fulltext_result = applies_to_vector(fulltext_query, vector);
            if (!fulltext_result.has_value() || !fulltext_result.value()) {
                should_include = false;
                break;
            }
        }
        
        if (should_include) {
            result.push_back(vector);
        }
    }
    
    return result;
}

// Fuzzy matching implementation (as part of full-text search)
Result<bool> MetadataFilter::applies_to_vector_fuzzy(const FullTextQuery& fuzzy_query, const Vector& vector) const {
    // Get the text field value to match against
    auto field_result = get_field_value(fuzzy_query.field, vector);
    if (!field_result.has_value()) {
        return false;
    }
    
    std::string field_value = field_result.value();
    
    // Use our fulltext evaluation with fuzzy matching
    switch (fuzzy_query.op) {
        case FullTextOperator::FUZZY_MATCH: {
            // Calculate edit distance between the field value and the search term
            // For simplicity, we'll check if any token in the field value is within the edit distance
            auto tokens = tokenize_text(field_value);
            
            for (const auto& token : tokens) {
                double distance = calculate_levenshtein_distance(fuzzy_query.query_text, token);
                
                if (static_cast<int>(distance) <= fuzzy_query.max_edit_distance) {
                    return true; // Found a match within the edit distance
                }
            }
            
            return false;
        }
        
        default:
            // For other operators, use the standard fulltext evaluation
            return evaluate_fulltext_filter(fuzzy_query, vector);
    }
}

// Private helper methods

// Geospatial helper methods
double MetadataFilter::calculate_haversine_distance(const Point& p1, const Point& p2) const {
    // Convert degrees to radians
    const double R = 6371000.0; // Earth's radius in meters
    double lat1_rad = p1.latitude * M_PI / 180.0;
    double lat2_rad = p2.latitude * M_PI / 180.0;
    double delta_lat_rad = (p2.latitude - p1.latitude) * M_PI / 180.0;
    double delta_lon_rad = (p2.longitude - p1.longitude) * M_PI / 180.0;
    
    // Haversine formula
    double a = sin(delta_lat_rad / 2) * sin(delta_lat_rad / 2) +
               cos(lat1_rad) * cos(lat2_rad) *
               sin(delta_lon_rad / 2) * sin(delta_lon_rad / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    
    return R * c; // Distance in meters
}

bool MetadataFilter::is_point_within_radius(const Point& point, const Point& center, double radius_meters) const {
    double distance = calculate_haversine_distance(point, center);
    return distance <= radius_meters;
}

bool MetadataFilter::is_point_within_bounding_box(const Point& point, const BoundingBox& bbox) const {
    return (point.latitude >= bbox.bottom_left.latitude && 
            point.latitude <= bbox.top_right.latitude &&
            point.longitude >= bbox.bottom_left.longitude && 
            point.longitude <= bbox.top_right.longitude);
}

// Temporal helper methods
bool MetadataFilter::evaluate_temporal_filter(const TemporalQuery& query, const Vector& vector) const {
    // Get the date/time field value as string and convert to time_point
    auto field_result = get_field_value(query.field, vector);
    if (!field_result.has_value()) {
        return false;
    }
    
    std::string date_str = field_result.value();
    
    // Try to parse the date string with various formats
    std::tm tm_time = {};
    bool parsed = false;
    
    // Try different common date formats in order of preference
    std::vector<std::string> formats = {
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",  // With microseconds
        "%Y-%m-%dT%H:%M:%S",     // ISO 8601 without Z
        "%Y-%m-%dT%H:%M:%SZ",    // ISO 8601 with Z
        "%Y-%m-%dT%H:%M:%S%z",   // ISO 8601 with timezone
        "%Y-%m-%d"               // Date only
    };
    
    for (const auto& format : formats) {
        std::istringstream ss(date_str);
        ss >> std::get_time(&tm_time, format.c_str());
        
        if (!ss.fail()) {
            // If format is date-only, set time to beginning of day
            if (format == "%Y-%m-%d") {
                tm_time.tm_hour = 0;
                tm_time.tm_min = 0;
                tm_time.tm_sec = 0;
            }
            parsed = true;
            break;
        }
    }
    
    if (!parsed) {
        // If we can't parse the date format, return false
        LOG_WARN(logger_, "Failed to parse date string: " << date_str);
        return false;
    }
    
    // Convert to time_point
    auto vector_time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm_time));
    
    switch (query.op) {
        case TemporalOperator::BEFORE:
            return vector_time_point < query.time_point;
            
        case TemporalOperator::AFTER:
            return vector_time_point > query.time_point;
            
        case TemporalOperator::BETWEEN:
            return vector_time_point >= query.time_range.start && 
                   vector_time_point <= query.time_range.end;
            
        case TemporalOperator::WITHIN_DURATION:
            return std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
                           vector_time_point - query.time_point).count()) <= 
                   query.duration.count();
            
        case TemporalOperator::AT_T:
            return vector_time_point == query.time_point;
            
        case TemporalOperator::NOT_AT_T:
            return vector_time_point != query.time_point;
            
        default:
            LOG_WARN(logger_, "Unknown temporal operator: " << static_cast<int>(query.op));
            return false;
    }
}

// Nested object helper methods
Result<std::string> MetadataFilter::get_nested_field_value(const std::string& path, const Vector& vector) const {
    // This is a simplified implementation for nested fields
    // It handles simple formats like "metadata.custom.fieldname"
    auto path_parts = parse_field_path(path);
    
    if (path_parts.size() < 2 || path_parts[0] != "metadata") {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid nested field path: " + path);
    }
    
    if (path_parts[1] == "custom" && path_parts.size() >= 3) {
        // Handle custom nested fields like "metadata.custom.fieldname"
        std::string custom_key = path_parts[2];
        auto it = vector.metadata.custom.find(custom_key);
        if (it != vector.metadata.custom.end()) {
            // For now, return the JSON as string
            return it->second.dump();
        }
    }
    
    // For other nested paths, use the standard get_field_value
    return get_field_value(path, vector);
}

Result<std::vector<std::string>> MetadataFilter::get_nested_field_values(const std::string& path, const Vector& vector) const {
    // Similar to get_nested_field_value but returns multiple values
    // This would handle nested arrays in custom metadata
    auto path_parts = parse_field_path(path);
    
    if (path_parts.size() < 2 || path_parts[0] != "metadata") {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid nested field path: " + path);
    }
    
    if (path_parts[1] == "custom" && path_parts.size() >= 3) {
        // Handle custom nested fields that might be arrays
        std::string custom_key = path_parts[2];
        auto it = vector.metadata.custom.find(custom_key);
        if (it != vector.metadata.custom.end() && it->second.is_array()) {
            std::vector<std::string> result;
            for (const auto& item : it->second) {
                result.push_back(item.dump());
            }
            return result;
        }
    }
    
    // For other nested paths, use the standard get_field_values
    return get_field_values(path, vector);
}

// Full-text search helper methods
bool MetadataFilter::evaluate_fulltext_filter(const FullTextQuery& query, const Vector& vector) const {
    // Get the text field value to search in
    auto field_result = get_field_value(query.field, vector);
    if (!field_result.has_value()) {
        return false;
    }
    
    std::string field_value = field_result.value();
    
    // Process based on the operator
    switch (query.op) {
        case FullTextOperator::MATCHES_ALL_TERMS: {
            auto terms = tokenize_text(query.query_text);
            for (const auto& term : terms) {
                if (query.case_sensitive) {
                    if (field_value.find(term) == std::string::npos) {
                        return false; // If any term is not found, return false
                    }
                } else {
                    // Case-insensitive search
                    std::string lower_field = field_value;
                    std::string lower_term = term;
                    std::transform(lower_field.begin(), lower_field.end(), lower_field.begin(), ::tolower);
                    std::transform(lower_term.begin(), lower_term.end(), lower_term.begin(), ::tolower);
                    
                    if (lower_field.find(lower_term) == std::string::npos) {
                        return false; // If any term is not found, return false
                    }
                }
            }
            return true; // All terms found
        }
        
        case FullTextOperator::MATCHES_ANY_TERM: {
            auto terms = tokenize_text(query.query_text);
            for (const auto& term : terms) {
                if (query.case_sensitive) {
                    if (field_value.find(term) != std::string::npos) {
                        return true; // If any term is found, return true
                    }
                } else {
                    // Case-insensitive search
                    std::string lower_field = field_value;
                    std::string lower_term = term;
                    std::transform(lower_field.begin(), lower_field.end(), lower_field.begin(), ::tolower);
                    std::transform(lower_term.begin(), lower_term.end(), lower_term.begin(), ::tolower);
                    
                    if (lower_field.find(lower_term) != std::string::npos) {
                        return true; // If any term is found, return true
                    }
                }
            }
            return false; // No terms found
        }
        
        case FullTextOperator::MATCHES_PHRASE: {
            if (query.case_sensitive) {
                return field_value.find(query.query_text) != std::string::npos;
            } else {
                // Case-insensitive phrase search
                std::string lower_field = field_value;
                std::string lower_phrase = query.query_text;
                std::transform(lower_field.begin(), lower_field.end(), lower_field.begin(), ::tolower);
                std::transform(lower_phrase.begin(), lower_phrase.end(), lower_phrase.begin(), ::tolower);
                
                return lower_field.find(lower_phrase) != std::string::npos;
            }
        }
        
        case FullTextOperator::MATCHES_REGEX: {
            try {
                std::regex pattern(query.query_text);
                return std::regex_search(field_value, pattern);
            } catch (const std::regex_error& e) {
                LOG_ERROR(logger_, "Invalid regex pattern: " << query.query_text << ", error: " << e.what());
                return false;
            }
        }
        
        case FullTextOperator::FUZZY_MATCH: {
            // For fuzzy matching, we'll use the dedicated method
            // This is a simplified implementation - a full implementation would use more sophisticated algorithms
            auto tokens = tokenize_text(field_value);
            
            for (const auto& token : tokens) {
                double distance = calculate_levenshtein_distance(query.query_text, token);
                
                if (distance <= query.max_edit_distance) {
                    return true;
                }
            }
            
            return false;
        }
        
        case FullTextOperator::ELASTICSEARCH_QUERY: {
            // For Elasticsearch queries, we would use the Elasticsearch service
            // This is a placeholder implementation that logs the intended behavior
            LOG_WARN(logger_, "Elasticsearch query not fully executed. Query: " << query.elasticsearch_query);
            // In a real implementation, this would execute the query against Elasticsearch
            // For now, we'll use a basic match as a placeholder
            return field_value.find(query.query_text) != std::string::npos;
        }
        
        default:
            LOG_WARN(logger_, "Unknown full-text operator: " << static_cast<int>(query.op));
            return false;
    }
}

double MetadataFilter::calculate_levenshtein_distance(const std::string& s1, const std::string& s2) const {
    size_t len1 = s1.length();
    size_t len2 = s2.length();
    
    // Create a 2D array to store distances
    std::vector<std::vector<size_t>> dp(len1 + 1, std::vector<size_t>(len2 + 1));
    
    // Initialize base cases
    for (size_t i = 0; i <= len1; ++i) {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= len2; ++j) {
        dp[0][j] = j;
    }
    
    // Fill the dp table
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];  // No operation needed
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j],    // Deletion
                                         dp[i][j - 1],    // Insertion
                                         dp[i - 1][j - 1]});  // Substitution
            }
        }
    }
    
    return static_cast<double>(dp[len1][len2]);
}

std::vector<std::string> MetadataFilter::tokenize_text(const std::string& text, const std::string& delimiter) const {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = text.find(delimiter);
    
    while (end != std::string::npos) {
        tokens.push_back(text.substr(start, end - start));
        start = end + delimiter.length();
        end = text.find(delimiter, start);
    }
    
    tokens.push_back(text.substr(start));
    return tokens;
}

// Combined advanced filtering method
Result<std::vector<Vector>> MetadataFilter::apply_advanced_filters(
    const std::vector<FilterCondition>& conditions,
    const std::vector<GeoQuery>& geo_queries,
    const std::vector<TemporalQuery>& temporal_queries,
    const std::vector<NestedQuery>& nested_queries,
    const std::vector<FullTextQuery>& fulltext_queries,
    const std::vector<Vector>& vectors) const {
    
    std::vector<Vector> result;
    
    for (const auto& vector : vectors) {
        bool should_include = true;
        
        // Check regular conditions
        if (should_include && !conditions.empty()) {
            for (const auto& condition : conditions) {
                auto condition_result = applies_to_vector(condition, vector);
                if (!condition_result.has_value() || !condition_result.value()) {
                    should_include = false;
                    break;
                }
            }
        }
        
        // Check geospatial queries
        if (should_include && !geo_queries.empty()) {
            for (const auto& geo_query : geo_queries) {
                auto geo_result = applies_to_vector(geo_query, vector);
                if (!geo_result.has_value() || !geo_result.value()) {
                    should_include = false;
                    break;
                }
            }
        }
        
        // Check temporal queries
        if (should_include && !temporal_queries.empty()) {
            for (const auto& temporal_query : temporal_queries) {
                auto temporal_result = applies_to_vector(temporal_query, vector);
                if (!temporal_result.has_value() || !temporal_result.value()) {
                    should_include = false;
                    break;
                }
            }
        }
        
        // Check nested queries
        if (should_include && !nested_queries.empty()) {
            for (const auto& nested_query : nested_queries) {
                auto nested_result = applies_to_vector(nested_query, vector);
                if (!nested_result.has_value() || !nested_result.value()) {
                    should_include = false;
                    break;
                }
            }
        }
        
        // Check full-text queries
        if (should_include && !fulltext_queries.empty()) {
            for (const auto& fulltext_query : fulltext_queries) {
                auto fulltext_result = applies_to_vector(fulltext_query, vector);
                if (!fulltext_result.has_value() || !fulltext_result.value()) {
                    should_include = false;
                    break;
                }
            }
        }
        
        if (should_include) {
            result.push_back(vector);
        }
    }
    
    return result;
}

// Elasticsearch integration methods
void MetadataFilter::set_elasticsearch_config(const ElasticsearchConfig& config) {
    elasticsearch_config_ = config;
}

Result<std::vector<std::string>> MetadataFilter::search_with_elasticsearch(const std::string& elasticsearch_query) const {
    // For now, we'll simulate the Elasticsearch query by returning an empty result
    // In a real implementation, this would:
    // 1. Construct an HTTP request to the Elasticsearch endpoint
    // 2. Use the elasticsearch_config_ to determine the host/port
    // 3. Send the query JSON to the configured Elasticsearch instance
    // 4. Parse the response and return matching document IDs
    
    // Log the query for debugging purposes
    LOG_DEBUG(logger_, "Elasticsearch query (simulated): " << elasticsearch_query);
    
    // In a real implementation, we would do something like:
    /*
    std::string url = "http://" + elasticsearch_config_.host + ":" + 
                      std::to_string(elasticsearch_config_.port) + 
                      "/" + elasticsearch_config_.index_name + "/_search";
    
    // Create HTTP client and send POST request with the query
    // Parse the JSON response and extract document IDs
    */
    
    // For now, return an empty result as a placeholder
    return std::vector<std::string>{};
}

Result<bool> MetadataFilter::applies_to_vector_elasticsearch(const FullTextQuery& es_query, const Vector& vector) const {
    // Get the field value to check against Elasticsearch query
    auto field_result = get_field_value(es_query.field, vector);
    if (!field_result.has_value()) {
        LOG_WARN(logger_, "Field not found for Elasticsearch filter: " << es_query.field);
        return false;
    }
    
    std::string field_value = field_result.value();
    
    // For now, we'll simulate the Elasticsearch behavior by doing a simple match
    // In a real implementation, this would send the query to actual Elasticsearch
    if (es_query.op == FullTextOperator::ELASTICSEARCH_QUERY) {
        // Parse the elasticsearch_query string and apply appropriate matching
        // For now, we'll do a basic check if the field contains the query text
        return field_value.find(es_query.query_text) != std::string::npos;
    } else {
        // If the operation is not Elasticsearch-specific, fall back to standard fulltext evaluation
        return evaluate_fulltext_filter(es_query, vector);
    }
}

} // namespace jadevectordb