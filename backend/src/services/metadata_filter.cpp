#include "metadata_filter.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_set>
#include <chrono>

namespace jadevectordb {

MetadataFilter::MetadataFilter(const FilterPerformanceSettings& settings) 
    : performance_settings_(settings) {
    logger_ = logging::LoggerManager::get_logger("MetadataFilter");
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

} // namespace jadevectordb