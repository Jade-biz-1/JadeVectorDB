#ifndef JADEVECTORDB_ERROR_HANDLING_H
#define JADEVECTORDB_ERROR_HANDLING_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <optional>
#include <system_error>
#include <chrono>
#include <source_location>
#include "expected.hpp"

namespace jadevectordb {

// Error codes for different error categories
enum class ErrorCode {
    // General errors
    SUCCESS = 0,
    UNKNOWN_ERROR = 1,
    INVALID_ARGUMENT = 2,
    OUT_OF_RANGE = 3,
    NOT_IMPLEMENTED = 4,
    PERMISSION_DENIED = 5,
    RESOURCE_EXHAUSTED = 6,
    FAILED_PRECONDITION = 7,
    ABORTED = 8,
    OUT_OF_MEMORY = 9,
    TIMEOUT = 10,
    DEADLINE_EXCEEDED = 11,
    NOT_FOUND = 12,
    ALREADY_EXISTS = 13,
    CANCELLED = 14,
    DATA_LOSS = 15,
    UNAUTHENTICATED = 16,
    UNAVAILABLE = 17,
    INTERNAL_ERROR = 18,
    INVALID_STATE = 19,
    
    // Vector database specific errors
    INITIALIZE_ERROR = 99,
    SERVICE_ERROR = 100,
    SERVICE_UNAVAILABLE = 101,
    VECTOR_DIMENSION_MISMATCH = 102,
    INVALID_VECTOR_ID = 103,
    DATABASE_NOT_FOUND = 104,
    INDEX_NOT_READY = 105,
    SIMILARITY_SEARCH_FAILED = 106,
    EMBEDDING_GENERATION_FAILED = 107,
    STORAGE_IO_ERROR = 108,
    NETWORK_ERROR = 109,
    SERIALIZATION_ERROR = 110,
    DESERIALIZATION_ERROR = 111,
    MEMORY_MAPPING_ERROR = 112,
    INDEX_BUILDING_ERROR = 113,
    QUERY_EXECUTION_ERROR = 114,
    BATCH_PROCESSING_ERROR = 115,
    METADATA_VALIDATION_ERROR = 116,
    CONFIGURATION_ERROR = 117,
    AUTHENTICATION_ERROR = 118,
    AUTHORIZATION_ERROR = 119,
    RATE_LIMIT_EXCEEDED = 120,
    QUOTA_EXCEEDED = 121
};

// Error category for grouping related errors
enum class ErrorCategory {
    GENERAL = 0,
    DATABASE = 1,
    NETWORK = 2,
    SECURITY = 3,
    SYSTEM = 4,
    APPLICATION = 5
};

// Error severity levels
enum class ErrorSeverity {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    CRITICAL = 3,
    FATAL = 4
};

// Extended error information
struct ErrorInfo {
    ErrorCode code;
    std::string message;
    ErrorCategory category;
    ErrorSeverity severity;
    std::string module;
    std::string component;
    std::string function;
    std::string file;
    int line;
    std::chrono::system_clock::time_point timestamp;
    std::map<std::string, std::string> context;
    std::vector<ErrorInfo> causes;
    std::optional<std::string> stack_trace;
    std::optional<std::string> suggestion;
    
    ErrorInfo() : code(ErrorCode::SUCCESS), category(ErrorCategory::GENERAL), 
                  severity(ErrorSeverity::INFO), line(0), timestamp(std::chrono::system_clock::now()) {}
    
    ErrorInfo(ErrorCode c, const std::string& msg) 
        : code(c), message(msg), category(ErrorCategory::GENERAL), 
          severity(ErrorSeverity::ERROR), line(0), timestamp(std::chrono::system_clock::now()) {}
};

// Result type alias using tl::expected
template<typename T>
using Result = tl::expected<T, ErrorInfo>;

// Error handling utilities
class ErrorHandler {
public:
    // Create error info with source location
    static ErrorInfo create_error(ErrorCode code, const std::string& message,
                                 const std::source_location& location = std::source_location::current());
    
    // Create error with context
    static ErrorInfo create_error(ErrorCode code, const std::string& message,
                                 const std::map<std::string, std::string>& context,
                                 const std::source_location& location = std::source_location::current());
    
    // Create error with causes
    static ErrorInfo create_error(ErrorCode code, const std::string& message,
                                 const std::vector<ErrorInfo>& causes,
                                 const std::source_location& location = std::source_location::current());
    
    // Get error code as numeric value
    static int get_error_code_numeric(ErrorCode code);
    
    // Get error message for code
    static std::string get_error_message(ErrorCode code);
    
    // Get error category for code
    static ErrorCategory get_error_category(ErrorCode code);
    
    // Get error severity for code
    static ErrorSeverity get_error_severity(ErrorCode code);
    
    // Format error info as string
    static std::string format_error(const ErrorInfo& error);
    
    // Chain errors (add cause to effect)
    static ErrorInfo chain_error(const ErrorInfo& cause, const ErrorInfo& effect);
    
    // Add context to existing error
    static ErrorInfo add_context(const ErrorInfo& error, const std::string& key, const std::string& value);
    
    // Convert std::error_code to ErrorInfo
    static ErrorInfo from_std_error(const std::error_code& ec, 
                                   const std::source_location& location = std::source_location::current());
    
    // Convert exception to ErrorInfo
    static ErrorInfo from_exception(const std::exception& ex,
                                   const std::source_location& location = std::source_location::current());
};

// Convenience macros for easier error creation
#define MAKE_ERROR(code, message) \
    jadevectordb::ErrorHandler::create_error(code, message, std::source_location::current())

#define MAKE_ERROR_WITH_CONTEXT(code, message, ctx) \
    jadevectordb::ErrorHandler::create_error(code, message, ctx, std::source_location::current())

#define RETURN_ERROR(code, message) \
    return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(code, message, std::source_location::current()))

#define RETURN_IF_ERROR(result) \
    do { \
        if (!(result).has_value()) { \
            return (result); \
        } \
    } while(0)

#define EXPECT_OR_RETURN(expr, code, message) \
    ({ \
        auto&& result = (expr); \
        if (!(result).has_value()) { \
            return tl::make_unexpected(jadevectordb::ErrorHandler::create_error(code, message, std::source_location::current())); \
        } \
        std::move((result).value()); \
    })

} // namespace jadevectordb

#endif // JADEVECTORDB_ERROR_HANDLING_H