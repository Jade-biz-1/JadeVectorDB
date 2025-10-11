#include "error_handling.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace jadevectordb {

// Create error info with source location
ErrorInfo ErrorHandler::create_error(ErrorCode code, const std::string& message,
                                   const std::source_location& location) {
    ErrorInfo error;
    error.code = code;
    error.message = message;
    error.category = get_error_category(code);
    error.severity = get_error_severity(code);
    error.function = location.function_name();
    error.file = location.file_name();
    error.line = location.line();
    error.timestamp = std::chrono::system_clock::now();
    return error;
}

// Create error with context
ErrorInfo ErrorHandler::create_error(ErrorCode code, const std::string& message,
                                   const std::map<std::string, std::string>& context,
                                   const std::source_location& location) {
    ErrorInfo error = create_error(code, message, location);
    error.context = context;
    return error;
}

// Create error with causes
ErrorInfo ErrorHandler::create_error(ErrorCode code, const std::string& message,
                                   const std::vector<ErrorInfo>& causes,
                                   const std::source_location& location) {
    ErrorInfo error = create_error(code, message, location);
    error.causes = causes;
    return error;
}

// Get error code as numeric value
int ErrorHandler::get_error_code_numeric(ErrorCode code) {
    return static_cast<int>(code);
}

// Get error message for code
std::string ErrorHandler::get_error_message(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS:
            return "Success";
        case ErrorCode::UNKNOWN_ERROR:
            return "Unknown error";
        case ErrorCode::INVALID_ARGUMENT:
            return "Invalid argument";
        case ErrorCode::OUT_OF_RANGE:
            return "Out of range";
        case ErrorCode::NOT_IMPLEMENTED:
            return "Not implemented";
        case ErrorCode::PERMISSION_DENIED:
            return "Permission denied";
        case ErrorCode::RESOURCE_EXHAUSTED:
            return "Resource exhausted";
        case ErrorCode::FAILED_PRECONDITION:
            return "Failed precondition";
        case ErrorCode::ABORTED:
            return "Aborted";
        case ErrorCode::OUT_OF_MEMORY:
            return "Out of memory";
        case ErrorCode::TIMEOUT:
            return "Timeout";
        case ErrorCode::DEADLINE_EXCEEDED:
            return "Deadline exceeded";
        case ErrorCode::NOT_FOUND:
            return "Not found";
        case ErrorCode::ALREADY_EXISTS:
            return "Already exists";
        case ErrorCode::CANCELLED:
            return "Cancelled";
        case ErrorCode::DATA_LOSS:
            return "Data loss";
        case ErrorCode::UNAUTHENTICATED:
            return "Unauthenticated";
        case ErrorCode::UNAVAILABLE:
            return "Unavailable";
        case ErrorCode::INTERNAL_ERROR:
            return "Internal error";
        case ErrorCode::INVALID_STATE:
            return "Invalid state";
            
        // Vector database specific errors
        case ErrorCode::VECTOR_DIMENSION_MISMATCH:
            return "Vector dimension mismatch";
        case ErrorCode::INVALID_VECTOR_ID:
            return "Invalid vector ID";
        case ErrorCode::DATABASE_NOT_FOUND:
            return "Database not found";
        case ErrorCode::INDEX_NOT_READY:
            return "Index not ready";
        case ErrorCode::SIMILARITY_SEARCH_FAILED:
            return "Similarity search failed";
        case ErrorCode::EMBEDDING_GENERATION_FAILED:
            return "Embedding generation failed";
        case ErrorCode::STORAGE_IO_ERROR:
            return "Storage I/O error";
        case ErrorCode::NETWORK_ERROR:
            return "Network error";
        case ErrorCode::SERIALIZATION_ERROR:
            return "Serialization error";
        case ErrorCode::DESERIALIZATION_ERROR:
            return "Deserialization error";
        case ErrorCode::MEMORY_MAPPING_ERROR:
            return "Memory mapping error";
        case ErrorCode::INDEX_BUILDING_ERROR:
            return "Index building error";
        case ErrorCode::QUERY_EXECUTION_ERROR:
            return "Query execution error";
        case ErrorCode::BATCH_PROCESSING_ERROR:
            return "Batch processing error";
        case ErrorCode::METADATA_VALIDATION_ERROR:
            return "Metadata validation error";
        case ErrorCode::CONFIGURATION_ERROR:
            return "Configuration error";
        case ErrorCode::AUTHENTICATION_ERROR:
            return "Authentication error";
        case ErrorCode::AUTHORIZATION_ERROR:
            return "Authorization error";
        case ErrorCode::RATE_LIMIT_EXCEEDED:
            return "Rate limit exceeded";
        case ErrorCode::QUOTA_EXCEEDED:
            return "Quota exceeded";
            
        default:
            return "Unknown error code";
    }
}

// Get error category for code
ErrorCategory ErrorHandler::get_error_category(ErrorCode code) {
    int code_value = static_cast<int>(code);
    
    if (code_value >= 100 && code_value < 200) {
        return ErrorCategory::DATABASE;
    } else if (code_value >= 1000 && code_value < 1100) {
        return ErrorCategory::NETWORK;
    } else if (code_value >= 2000 && code_value < 2100) {
        return ErrorCategory::SECURITY;
    } else if (code_value >= 3000 && code_value < 3100) {
        return ErrorCategory::SYSTEM;
    } else if (code_value >= 4000 && code_value < 4100) {
        return ErrorCategory::APPLICATION;
    }
    
    return ErrorCategory::GENERAL;
}

// Get error severity for code
ErrorSeverity ErrorHandler::get_error_severity(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS:
            return ErrorSeverity::INFO;
            
        case ErrorCode::INVALID_ARGUMENT:
        case ErrorCode::OUT_OF_RANGE:
        case ErrorCode::NOT_FOUND:
        case ErrorCode::ALREADY_EXISTS:
        case ErrorCode::TIMEOUT:
        case ErrorCode::VECTOR_DIMENSION_MISMATCH:
        case ErrorCode::INVALID_VECTOR_ID:
        case ErrorCode::INDEX_NOT_READY:
        case ErrorCode::RATE_LIMIT_EXCEEDED:
        case ErrorCode::QUOTA_EXCEEDED:
            return ErrorSeverity::WARNING;
            
        case ErrorCode::NOT_IMPLEMENTED:
        case ErrorCode::PERMISSION_DENIED:
        case ErrorCode::RESOURCE_EXHAUSTED:
        case ErrorCode::FAILED_PRECONDITION:
        case ErrorCode::ABORTED:
        case ErrorCode::OUT_OF_MEMORY:
        case ErrorCode::DEADLINE_EXCEEDED:
        case ErrorCode::CANCELLED:
        case ErrorCode::DATA_LOSS:
        case ErrorCode::UNAUTHENTICATED:
        case ErrorCode::UNAVAILABLE:
        case ErrorCode::INVALID_STATE:
        case ErrorCode::DATABASE_NOT_FOUND:
        case ErrorCode::SIMILARITY_SEARCH_FAILED:
        case ErrorCode::EMBEDDING_GENERATION_FAILED:
        case ErrorCode::STORAGE_IO_ERROR:
        case ErrorCode::NETWORK_ERROR:
        case ErrorCode::SERIALIZATION_ERROR:
        case ErrorCode::DESERIALIZATION_ERROR:
        case ErrorCode::MEMORY_MAPPING_ERROR:
        case ErrorCode::INDEX_BUILDING_ERROR:
        case ErrorCode::QUERY_EXECUTION_ERROR:
        case ErrorCode::BATCH_PROCESSING_ERROR:
        case ErrorCode::METADATA_VALIDATION_ERROR:
        case ErrorCode::CONFIGURATION_ERROR:
        case ErrorCode::AUTHENTICATION_ERROR:
        case ErrorCode::AUTHORIZATION_ERROR:
            return ErrorSeverity::ERROR;
            
        case ErrorCode::INTERNAL_ERROR:
            return ErrorSeverity::CRITICAL;
            
        case ErrorCode::UNKNOWN_ERROR:
        default:
            return ErrorSeverity::FATAL;
    }
}

// Format error info as string
std::string ErrorHandler::format_error(const ErrorInfo& error) {
    std::ostringstream oss;
    
    // Format timestamp
    auto time_t = std::chrono::system_clock::to_time_t(error.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        error.timestamp.time_since_epoch()) % 1000;
    
    oss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    
    // Add severity
    switch (error.severity) {
        case ErrorSeverity::INFO:
            oss << "[INFO] ";
            break;
        case ErrorSeverity::WARNING:
            oss << "[WARN] ";
            break;
        case ErrorSeverity::ERROR:
            oss << "[ERROR] ";
            break;
        case ErrorSeverity::CRITICAL:
            oss << "[CRITICAL] ";
            break;
        case ErrorSeverity::FATAL:
            oss << "[FATAL] ";
            break;
    }
    
    // Add error code and message
    oss << "[" << get_error_code_numeric(error.code) << "] " << error.message;
    
    // Add location information
    if (!error.file.empty() && error.line > 0) {
        oss << " (" << error.file << ":" << error.line;
        if (!error.function.empty()) {
            oss << " in " << error.function;
        }
        oss << ")";
    }
    
    // Add module and component if available
    if (!error.module.empty()) {
        oss << " [Module: " << error.module << "]";
    }
    if (!error.component.empty()) {
        oss << " [Component: " << error.component << "]";
    }
    
    // Add context information
    if (!error.context.empty()) {
        oss << " {";
        bool first = true;
        for (const auto& [key, value] : error.context) {
            if (!first) oss << ", ";
            oss << key << "=" << value;
            first = false;
        }
        oss << "}";
    }
    
    return oss.str();
}

// Chain errors (add cause to effect)
ErrorInfo ErrorHandler::chain_error(const ErrorInfo& cause, const ErrorInfo& effect) {
    ErrorInfo chained_error = effect;
    chained_error.causes.push_back(cause);
    return chained_error;
}

// Add context to existing error
ErrorInfo ErrorHandler::add_context(const ErrorInfo& error, const std::string& key, const std::string& value) {
    ErrorInfo contextual_error = error;
    contextual_error.context[key] = value;
    return contextual_error;
}

// Convert std::error_code to ErrorInfo
ErrorInfo ErrorHandler::from_std_error(const std::error_code& ec, 
                                      const std::source_location& location) {
    ErrorInfo error;
    error.code = ErrorCode::INTERNAL_ERROR;
    error.message = ec.message();
    error.category = ErrorCategory::SYSTEM;
    error.severity = ErrorSeverity::ERROR;
    error.function = location.function_name();
    error.file = location.file_name();
    error.line = location.line();
    error.timestamp = std::chrono::system_clock::now();
    error.context["std_error_category"] = ec.category().name();
    error.context["std_error_value"] = std::to_string(ec.value());
    return error;
}

// Convert exception to ErrorInfo
ErrorInfo ErrorHandler::from_exception(const std::exception& ex,
                                      const std::source_location& location) {
    ErrorInfo error;
    error.code = ErrorCode::UNKNOWN_ERROR;
    error.message = ex.what();
    error.category = ErrorCategory::GENERAL;
    error.severity = ErrorSeverity::ERROR;
    error.function = location.function_name();
    error.file = location.file_name();
    error.line = location.line();
    error.timestamp = std::chrono::system_clock::now();
    return error;
}

} // namespace jadevectordb