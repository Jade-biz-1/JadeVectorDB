#ifndef JADEVECTORDB_SCHEMA_VALIDATOR_H
#define JADEVECTORDB_SCHEMA_VALIDATOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>

#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

namespace jadevectordb {

// Supported data types for schema validation
enum class SchemaDataType {
    STRING,
    INTEGER,
    FLOAT,
    BOOLEAN,
    ARRAY,
    OBJECT,
    DATETIME
};

// Schema field definition
struct SchemaField {
    std::string name;
    SchemaDataType type;
    bool required = false;
    std::variant<std::string, int, float, bool> default_value;
    std::vector<std::string> allowed_values;  // For ENUM-like validation
    double min_value = 0.0;                   // For numeric types
    double max_value = 0.0;                   // For numeric types
    size_t min_length = 0;                  // For string types
    size_t max_length = 0;                   // For string types
    std::string pattern;                     // Regex pattern for string validation
    std::unordered_map<std::string, SchemaField> nested_fields;  // For object types
    std::unique_ptr<SchemaField> array_item_schema;             // For array types
    
    SchemaField() = default;
    SchemaField(const std::string& n, SchemaDataType t, bool req = false) 
        : name(n), type(t), required(req) {}
};

// Database schema definition
struct DatabaseSchema {
    std::string database_id;
    std::string name;
    std::unordered_map<std::string, SchemaField> metadata_schema;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    DatabaseSchema() = default;
};

// Schema validation result
struct SchemaValidationResult {
    bool is_valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    
    void add_error(const std::string& error) {
        is_valid = false;
        errors.push_back(error);
    }
    
    void add_warning(const std::string& warning) {
        warnings.push_back(warning);
    }
};

// Schema validator service
class SchemaValidator {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::unordered_map<std::string, DatabaseSchema> schemas_;  // database_id -> schema
    mutable std::shared_mutex schemas_mutex_;

public:
    SchemaValidator();
    ~SchemaValidator() = default;
    
    // Register a schema for a database
    Result<void> register_schema(const DatabaseSchema& schema);
    
    // Get schema for a database
    Result<DatabaseSchema> get_schema(const std::string& database_id) const;
    
    // Remove schema for a database
    Result<void> remove_schema(const std::string& database_id);
    
    // Validate vector metadata against schema
    SchemaValidationResult validate_vector_metadata(
        const std::string& database_id,
        const VectorMetadata& metadata) const;
    
    // Validate a single field value
    SchemaValidationResult validate_field(
        const SchemaField& field,
        const nlohmann::json& value) const;
    
    // Validate nested object field
    SchemaValidationResult validate_object_field(
        const SchemaField& field,
        const nlohmann::json& value) const;
    
    // Validate array field
    SchemaValidationResult validate_array_field(
        const SchemaField& field,
        const nlohmann::json& value) const;
    
    // Create default schema (basic fields)
    DatabaseSchema create_default_schema(const std::string& database_id) const;
    
    // Validate schema definition itself
    SchemaValidationResult validate_schema_definition(const DatabaseSchema& schema) const;
    
    // Update schema for a database
    Result<void> update_schema(const std::string& database_id, const DatabaseSchema& new_schema);
    
    // Check if schema exists for database
    bool schema_exists(const std::string& database_id) const;

private:
    // Helper methods for specific validations
    bool validate_string_field(const SchemaField& field, const std::string& value) const;
    bool validate_numeric_field(const SchemaField& field, double value) const;
    bool validate_boolean_field(const SchemaField& field, bool value) const;
    bool validate_enum_field(const SchemaField& field, const std::string& value) const;
    bool validate_pattern_field(const SchemaField& field, const std::string& value) const;
    
    // Type checking helpers
    bool is_valid_json_type(const nlohmann::json& value, SchemaDataType expected_type) const;
    
    // Utility methods
    std::string get_field_path(const std::string& parent_path, const std::string& field_name) const;
    std::string schema_data_type_to_string(SchemaDataType type) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SCHEMA_VALIDATOR_H