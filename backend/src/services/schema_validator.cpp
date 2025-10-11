#include "schema_validator.h"
#include <regex>
#include <sstream>
#include <shared_mutex>

namespace jadevectordb {

SchemaValidator::SchemaValidator() {
    logger_ = logging::LoggerManager::get_logger("SchemaValidator");
}

Result<void> SchemaValidator::register_schema(const DatabaseSchema& schema) {
    // Validate the schema definition first
    auto validation_result = validate_schema_definition(schema);
    if (!validation_result.is_valid) {
        std::string error_msg = "Schema validation failed:";
        for (const auto& error : validation_result.errors) {
            error_msg += " " + error + ";";
        }
        RETURN_ERROR(ErrorCode::INVALID_SCHEMA, error_msg);
    }
    
    // Register the schema
    std::unique_lock<std::shared_mutex> lock(schemas_mutex_);
    schemas_[schema.database_id] = schema;
    
    LOG_INFO(logger_, "Registered schema for database: " << schema.database_id);
    return {};
}

Result<DatabaseSchema> SchemaValidator::get_schema(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(schemas_mutex_);
    
    auto it = schemas_.find(database_id);
    if (it == schemas_.end()) {
        RETURN_ERROR(ErrorCode::SCHEMA_NOT_FOUND, "Schema not found for database: " + database_id);
    }
    
    return it->second;
}

Result<void> SchemaValidator::remove_schema(const std::string& database_id) {
    std::unique_lock<std::shared_mutex> lock(schemas_mutex_);
    
    auto it = schemas_.find(database_id);
    if (it == schemas_.end()) {
        RETURN_ERROR(ErrorCode::SCHEMA_NOT_FOUND, "Schema not found for database: " + database_id);
    }
    
    schemas_.erase(it);
    LOG_INFO(logger_, "Removed schema for database: " << database_id);
    return {};
}

SchemaValidationResult SchemaValidator::validate_vector_metadata(
    const std::string& database_id,
    const VectorMetadata& metadata) const {
    
    SchemaValidationResult result;
    
    // Get the schema for this database
    auto schema_result = get_schema(database_id);
    if (!schema_result.has_value()) {
        result.add_error("No schema registered for database: " + database_id);
        return result;
    }
    
    const auto& schema = schema_result.value();
    
    // Validate each field in the metadata against the schema
    // Handle predefined fields first
    auto predefined_fields = {
        std::make_pair("owner", metadata.owner),
        std::make_pair("category", metadata.category),
        std::make_pair("status", metadata.status),
        std::make_pair("source", metadata.source),
        std::make_pair("created_at", metadata.created_at),
        std::make_pair("updated_at", metadata.updated_at)
    };
    
    for (const auto& [field_name, field_value] : predefined_fields) {
        auto field_it = schema.metadata_schema.find(field_name);
        if (field_it != schema.metadata_schema.end()) {
            nlohmann::json json_value = field_value;
            auto field_result = validate_field(field_it->second, json_value);
            result.errors.insert(result.errors.end(), field_result.errors.begin(), field_result.errors.end());
            result.warnings.insert(result.warnings.end(), field_result.warnings.begin(), field_result.warnings.end());
            if (!field_result.is_valid) {
                result.is_valid = false;
            }
        }
    }
    
    // Validate score field
    auto score_field_it = schema.metadata_schema.find("score");
    if (score_field_it != schema.metadata_schema.end()) {
        nlohmann::json json_value = metadata.score;
        auto field_result = validate_field(score_field_it->second, json_value);
        result.errors.insert(result.errors.end(), field_result.errors.begin(), field_result.errors.end());
        result.warnings.insert(result.warnings.end(), field_result.warnings.begin(), field_result.warnings.end());
        if (!field_result.is_valid) {
            result.is_valid = false;
        }
    }
    
    // Validate array fields
    auto tags_field_it = schema.metadata_schema.find("tags");
    if (tags_field_it != schema.metadata_schema.end()) {
        nlohmann::json json_value = metadata.tags;
        auto field_result = validate_array_field(tags_field_it->second, json_value);
        result.errors.insert(result.errors.end(), field_result.errors.begin(), field_result.errors.end());
        result.warnings.insert(result.warnings.end(), field_result.warnings.begin(), field_result.warnings.end());
        if (!field_result.is_valid) {
            result.is_valid = false;
        }
    }
    
    auto permissions_field_it = schema.metadata_schema.find("permissions");
    if (permissions_field_it != schema.metadata_schema.end()) {
        nlohmann::json json_value = metadata.permissions;
        auto field_result = validate_array_field(permissions_field_it->second, json_value);
        result.errors.insert(result.errors.end(), field_result.errors.begin(), field_result.errors.end());
        result.warnings.insert(result.warnings.end(), field_result.warnings.begin(), field_result.warnings.end());
        if (!field_result.is_valid) {
            result.is_valid = false;
        }
    }
    
    // Validate custom fields
    for (const auto& [custom_key, custom_value] : metadata.custom) {
        auto custom_field_it = schema.metadata_schema.find("custom." + custom_key);
        if (custom_field_it != schema.metadata_schema.end()) {
            auto field_result = validate_field(custom_field_it->second, custom_value);
            result.errors.insert(result.errors.end(), field_result.errors.begin(), field_result.errors.end());
            result.warnings.insert(result.warnings.end(), field_result.warnings.begin(), field_result.warnings.end());
            if (!field_result.is_valid) {
                result.is_valid = false;
            }
        }
    }
    
    return result;
}

SchemaValidationResult SchemaValidator::validate_field(
    const SchemaField& field,
    const nlohmann::json& value) const {
    
    SchemaValidationResult result;
    
    // Check if field is required but missing
    if (field.required && value.is_null()) {
        result.add_error("Required field '" + field.name + "' is missing");
        return result;
    }
    
    // If value is null and not required, validation passes
    if (value.is_null() && !field.required) {
        return result;
    }
    
    // Validate based on field type
    switch (field.type) {
        case SchemaDataType::STRING: {
            if (!value.is_string()) {
                result.add_error("Field '" + field.name + "' must be a string, got " + std::string(value.type_name()));
                return result;
            }
            
            std::string str_value = value.get<std::string>();
            
            // Length validation
            if (field.min_length > 0 && str_value.length() < field.min_length) {
                result.add_error("Field '" + field.name + "' must be at least " + 
                               std::to_string(field.min_length) + " characters long");
            }
            
            if (field.max_length > 0 && str_value.length() > field.max_length) {
                result.add_error("Field '" + field.name + "' must be at most " + 
                               std::to_string(field.max_length) + " characters long");
            }
            
            // Pattern validation
            if (!field.pattern.empty() && !validate_pattern_field(field, str_value)) {
                result.add_error("Field '" + field.name + "' does not match required pattern");
            }
            
            // Enum validation
            if (!field.allowed_values.empty() && !validate_enum_field(field, str_value)) {
                result.add_error("Field '" + field.name + "' value '" + str_value + 
                               "' is not in allowed values list");
            }
            
            break;
        }
        
        case SchemaDataType::INTEGER: {
            if (!value.is_number_integer()) {
                result.add_error("Field '" + field.name + "' must be an integer, got " + std::string(value.type_name()));
                return result;
            }
            
            int int_value = value.get<int>();
            double double_value = static_cast<double>(int_value);
            
            // Range validation
            if (field.min_value != 0.0 || field.max_value != 0.0) {
                if (!validate_numeric_field(field, double_value)) {
                    result.add_error("Field '" + field.name + "' value " + std::to_string(int_value) + 
                                   " is outside allowed range [" + std::to_string(field.min_value) + 
                                   ", " + std::to_string(field.max_value) + "]");
                }
            }
            
            break;
        }
        
        case SchemaDataType::FLOAT: {
            if (!value.is_number_float() && !value.is_number_integer()) {
                result.add_error("Field '" + field.name + "' must be a number, got " + std::string(value.type_name()));
                return result;
            }
            
            double double_value = value.get<double>();
            
            // Range validation
            if (field.min_value != 0.0 || field.max_value != 0.0) {
                if (!validate_numeric_field(field, double_value)) {
                    result.add_error("Field '" + field.name + "' value " + std::to_string(double_value) + 
                                   " is outside allowed range [" + std::to_string(field.min_value) + 
                                   ", " + std::to_string(field.max_value) + "]");
                }
            }
            
            break;
        }
        
        case SchemaDataType::BOOLEAN: {
            if (!value.is_boolean()) {
                result.add_error("Field '" + field.name + "' must be a boolean, got " + std::string(value.type_name()));
                return result;
            }
            
            bool bool_value = value.get<bool>();
            
            // Boolean-specific validation (if any)
            if (!validate_boolean_field(field, bool_value)) {
                result.add_error("Field '" + field.name + "' failed boolean validation");
            }
            
            break;
        }
        
        case SchemaDataType::ARRAY: {
            auto array_result = validate_array_field(field, value);
            result.errors.insert(result.errors.end(), array_result.errors.begin(), array_result.errors.end());
            result.warnings.insert(result.warnings.end(), array_result.warnings.begin(), array_result.warnings.end());
            if (!array_result.is_valid) {
                result.is_valid = false;
            }
            break;
        }
        
        case SchemaDataType::OBJECT: {
            auto object_result = validate_object_field(field, value);
            result.errors.insert(result.errors.end(), object_result.errors.begin(), object_result.errors.end());
            result.warnings.insert(result.warnings.end(), object_result.warnings.begin(), object_result.warnings.end());
            if (!object_result.is_valid) {
                result.is_valid = false;
            }
            break;
        }
        
        case SchemaDataType::DATETIME: {
            if (!value.is_string()) {
                result.add_error("Field '" + field.name + "' must be a datetime string, got " + std::string(value.type_name()));
                return result;
            }
            
            std::string datetime_value = value.get<std::string>();
            
            // Basic datetime format validation (ISO 8601)
            std::regex datetime_pattern(R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$)");
            if (!std::regex_match(datetime_value, datetime_pattern)) {
                result.add_warning("Field '" + field.name + "' does not appear to be in ISO 8601 datetime format");
            }
            
            break;
        }
    }
    
    return result;
}

SchemaValidationResult SchemaValidator::validate_object_field(
    const SchemaField& field,
    const nlohmann::json& value) const {
    
    SchemaValidationResult result;
    
    if (!value.is_object()) {
        result.add_error("Field '" + field.name + "' must be an object, got " + std::string(value.type_name()));
        return result;
    }
    
    // Validate nested fields
    for (const auto& [nested_field_name, nested_field_def] : field.nested_fields) {
        auto nested_value_it = value.find(nested_field_name);
        
        if (nested_value_it != value.end()) {
            // Field exists, validate it
            auto nested_result = validate_field(nested_field_def, *nested_value_it);
            result.errors.insert(result.errors.end(), nested_result.errors.begin(), nested_result.errors.end());
            result.warnings.insert(result.warnings.end(), nested_result.warnings.begin(), nested_result.warnings.end());
            if (!nested_result.is_valid) {
                result.is_valid = false;
            }
        } else if (nested_field_def.required) {
            // Required field is missing
            result.add_error("Required nested field '" + nested_field_name + "' is missing in object '" + field.name + "'");
            result.is_valid = false;
        }
    }
    
    return result;
}

SchemaValidationResult SchemaValidator::validate_array_field(
    const SchemaField& field,
    const nlohmann::json& value) const {
    
    SchemaValidationResult result;
    
    if (!value.is_array()) {
        result.add_error("Field '" + field.name + "' must be an array, got " + std::string(value.type_name()));
        return result;
    }
    
    // Validate array items if schema is defined
    if (field.array_item_schema) {
        const auto& item_schema = *(field.array_item_schema);
        
        for (size_t i = 0; i < value.size(); ++i) {
            const auto& array_item = value[i];
            auto item_result = validate_field(item_schema, array_item);
            
            // Prefix error messages with array index
            for (const auto& error : item_result.errors) {
                result.add_error("Array item [" + std::to_string(i) + "]: " + error);
            }
            
            for (const auto& warning : item_result.warnings) {
                result.add_warning("Array item [" + std::to_string(i) + "]: " + warning);
            }
            
            if (!item_result.is_valid) {
                result.is_valid = false;
            }
        }
    }
    
    return result;
}

DatabaseSchema SchemaValidator::create_default_schema(const std::string& database_id) const {
    DatabaseSchema schema;
    schema.database_id = database_id;
    schema.name = "Default schema for " + database_id;
    schema.created_at = std::chrono::system_clock::now();
    schema.updated_at = std::chrono::system_clock::now();
    
    // Add default fields with basic validation rules
    schema.metadata_schema["owner"] = SchemaField("owner", SchemaDataType::STRING, false);
    schema.metadata_schema["owner"].max_length = 255;
    
    schema.metadata_schema["category"] = SchemaField("category", SchemaDataType::STRING, false);
    schema.metadata_schema["category"].max_length = 100;
    
    schema.metadata_schema["status"] = SchemaField("status", SchemaDataType::STRING, false);
    schema.metadata_schema["status"].allowed_values = {"active", "draft", "archived", "deleted"};
    
    schema.metadata_schema["source"] = SchemaField("source", SchemaDataType::STRING, false);
    schema.metadata_schema["source"].max_length = 100;
    
    schema.metadata_schema["created_at"] = SchemaField("created_at", SchemaDataType::DATETIME, false);
    schema.metadata_schema["updated_at"] = SchemaField("updated_at", SchemaDataType::DATETIME, false);
    
    schema.metadata_schema["score"] = SchemaField("score", SchemaDataType::FLOAT, false);
    schema.metadata_schema["score"].min_value = 0.0;
    schema.metadata_schema["score"].max_value = 1.0;
    
    // Array fields
    schema.metadata_schema["tags"] = SchemaField("tags", SchemaDataType::ARRAY, false);
    schema.metadata_schema["tags"].array_item_schema = std::make_unique<SchemaField>("tag_item", SchemaDataType::STRING);
    schema.metadata_schema["tags"].array_item_schema->max_length = 50;
    
    schema.metadata_schema["permissions"] = SchemaField("permissions", SchemaDataType::ARRAY, false);
    schema.metadata_schema["permissions"].array_item_schema = std::make_unique<SchemaField>("permission_item", SchemaDataType::STRING);
    schema.metadata_schema["permissions"].array_item_schema->max_length = 50;
    
    return schema;
}

SchemaValidationResult SchemaValidator::validate_schema_definition(const DatabaseSchema& schema) const {
    SchemaValidationResult result;
    
    // Validate database ID
    if (schema.database_id.empty()) {
        result.add_error("Database ID cannot be empty");
    }
    
    // Validate schema field definitions
    for (const auto& [field_name, field_def] : schema.metadata_schema) {
        // Validate field name
        if (field_name.empty()) {
            result.add_error("Field name cannot be empty");
            continue;
        }
        
        // Validate field constraints based on type
        switch (field_def.type) {
            case SchemaDataType::STRING:
                if (field_def.min_length > field_def.max_length && field_def.max_length > 0) {
                    result.add_error("Field '" + field_name + "': min_length cannot be greater than max_length");
                }
                break;
                
            case SchemaDataType::INTEGER:
            case SchemaDataType::FLOAT:
                if (field_def.min_value > field_def.max_value && field_def.max_value != 0.0) {
                    result.add_error("Field '" + field_name + "': min_value cannot be greater than max_value");
                }
                break;
                
            case SchemaDataType::ARRAY:
                // Arrays should have item schema defined
                if (!field_def.array_item_schema && 
                    (field_def.min_length > 0 || field_def.max_length > 0)) {
                    result.add_warning("Field '" + field_name + "': Array length constraints without item schema may be ineffective");
                }
                break;
                
            case SchemaDataType::OBJECT:
                // Objects should have nested fields defined
                if (field_def.nested_fields.empty()) {
                    result.add_warning("Field '" + field_name + "': Object field has no nested fields defined");
                }
                break;
                
            default:
                break;
        }
        
        // Validate ENUM values
        if (!field_def.allowed_values.empty()) {
            if (field_def.type != SchemaDataType::STRING) {
                result.add_error("Field '" + field_name + "': ENUM validation only supported for string fields");
            }
        }
        
        // Validate regex pattern
        if (!field_def.pattern.empty()) {
            if (field_def.type != SchemaDataType::STRING) {
                result.add_error("Field '" + field_name + "': Pattern validation only supported for string fields");
            } else {
                try {
                    std::regex pattern(field_def.pattern);
                } catch (const std::regex_error& e) {
                    result.add_error("Field '" + field_name + "': Invalid regex pattern - " + e.what());
                }
            }
        }
    }
    
    return result;
}

Result<void> SchemaValidator::update_schema(const std::string& database_id, const DatabaseSchema& new_schema) {
    // Validate the new schema
    auto validation_result = validate_schema_definition(new_schema);
    if (!validation_result.is_valid) {
        std::string error_msg = "Schema validation failed:";
        for (const auto& error : validation_result.errors) {
            error_msg += " " + error + ";";
        }
        RETURN_ERROR(ErrorCode::INVALID_SCHEMA, error_msg);
    }
    
    // Update the schema
    std::unique_lock<std::shared_mutex> lock(schemas_mutex_);
    
    auto it = schemas_.find(database_id);
    if (it == schemas_.end()) {
        RETURN_ERROR(ErrorCode::SCHEMA_NOT_FOUND, "Schema not found for database: " + database_id);
    }
    
    schemas_[database_id] = new_schema;
    LOG_INFO(logger_, "Updated schema for database: " << database_id);
    return {};
}

bool SchemaValidator::schema_exists(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(schemas_mutex_);
    return schemas_.find(database_id) != schemas_.end();
}

bool SchemaValidator::validate_string_field(const SchemaField& field, const std::string& value) const {
    // Length validation
    if (field.min_length > 0 && value.length() < field.min_length) {
        return false;
    }
    
    if (field.max_length > 0 && value.length() > field.max_length) {
        return false;
    }
    
    // Pattern validation
    if (!field.pattern.empty() && !validate_pattern_field(field, value)) {
        return false;
    }
    
    // Enum validation
    if (!field.allowed_values.empty() && !validate_enum_field(field, value)) {
        return false;
    }
    
    return true;
}

bool SchemaValidator::validate_numeric_field(const SchemaField& field, double value) const {
    if (field.min_value != 0.0 && value < field.min_value) {
        return false;
    }
    
    if (field.max_value != 0.0 && value > field.max_value) {
        return false;
    }
    
    return true;
}

bool SchemaValidator::validate_boolean_field(const SchemaField& field, bool value) const {
    // For boolean fields, basic validation always passes
    // Custom boolean validation logic could be added here if needed
    return true;
}

bool SchemaValidator::validate_enum_field(const SchemaField& field, const std::string& value) const {
    return std::find(field.allowed_values.begin(), field.allowed_values.end(), value) != field.allowed_values.end();
}

bool SchemaValidator::validate_pattern_field(const SchemaField& field, const std::string& value) const {
    try {
        std::regex pattern(field.pattern);
        return std::regex_match(value, pattern);
    } catch (const std::regex_error&) {
        // Invalid pattern - treat as validation failure
        return false;
    }
}

bool SchemaValidator::is_valid_json_type(const nlohmann::json& value, SchemaDataType expected_type) const {
    switch (expected_type) {
        case SchemaDataType::STRING:
            return value.is_string();
        case SchemaDataType::INTEGER:
            return value.is_number_integer();
        case SchemaDataType::FLOAT:
            return value.is_number_float() || value.is_number_integer();
        case SchemaDataType::BOOLEAN:
            return value.is_boolean();
        case SchemaDataType::ARRAY:
            return value.is_array();
        case SchemaDataType::OBJECT:
            return value.is_object();
        case SchemaDataType::DATETIME:
            return value.is_string(); // Datetime is stored as string
        default:
            return false;
    }
}

std::string SchemaValidator::get_field_path(const std::string& parent_path, const std::string& field_name) const {
    if (parent_path.empty()) {
        return field_name;
    }
    return parent_path + "." + field_name;
}

std::string SchemaValidator::schema_data_type_to_string(SchemaDataType type) const {
    switch (type) {
        case SchemaDataType::STRING: return "string";
        case SchemaDataType::INTEGER: return "integer";
        case SchemaDataType::FLOAT: return "float";
        case SchemaDataType::BOOLEAN: return "boolean";
        case SchemaDataType::ARRAY: return "array";
        case SchemaDataType::OBJECT: return "object";
        case SchemaDataType::DATETIME: return "datetime";
        default: return "unknown";
    }
}

} // namespace jadevectordb