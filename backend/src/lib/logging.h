#ifndef JADEVECTORDB_LOGGING_H
#define JADEVECTORDB_LOGGING_H

#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <thread>
#include <map>
#include <functional>
#include <atomic>

namespace jadevectordb {

// Logging infrastructure
namespace logging {

    // Log levels
    enum class LogLevel {
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARN = 3,
        ERROR = 4,
        FATAL = 5
    };
    
    // Convert LogLevel to string
    std::string log_level_to_string(LogLevel level);
    
    // Convert string to LogLevel
    LogLevel string_to_log_level(const std::string& level_str);
    
    // Log record structure
    struct LogRecord {
        LogLevel level;
        std::string timestamp;
        std::string logger_name;
        std::string file;
        int line;
        std::string function;
        std::string message;
        std::thread::id thread_id;
        std::string module;
        std::string component;
        std::map<std::string, std::string> context;
        
        LogRecord() : level(LogLevel::INFO), line(0), thread_id(std::this_thread::get_id()) {}
    };
    
    // Log formatter interface
    class LogFormatter {
    public:
        virtual ~LogFormatter() = default;
        virtual std::string format(const LogRecord& record) = 0;
    };
    
    // Simple text formatter
    class SimpleTextFormatter : public LogFormatter {
    public:
        std::string format(const LogRecord& record) override;
    };
    
    // JSON formatter
    class JsonFormatter : public LogFormatter {
    public:
        std::string format(const LogRecord& record) override;
    };
    
    // Log appender interface
    class LogAppender {
    protected:
        std::unique_ptr<LogFormatter> formatter_;
        LogLevel min_level_;
        std::atomic<bool> enabled_;
        
    public:
        LogAppender(std::unique_ptr<LogFormatter> formatter, LogLevel min_level = LogLevel::INFO);
        virtual ~LogAppender() = default;
        
        virtual void append(const LogRecord& record) = 0;
        virtual void flush() = 0;
        
        bool should_log(LogLevel level) const;
        void set_min_level(LogLevel level);
        LogLevel get_min_level() const;
        void set_enabled(bool enabled);
        bool is_enabled() const;
    };
    
    // Console appender
    class ConsoleAppender : public LogAppender {
    private:
        mutable std::mutex mutex_;
        std::ostream& output_stream_;
        
    public:
        explicit ConsoleAppender(std::unique_ptr<LogFormatter> formatter, 
                                LogLevel min_level = LogLevel::INFO,
                                std::ostream& output_stream = std::cout);
        void append(const LogRecord& record) override;
        void flush() override;
    };
    
    // File appender
    class FileAppender : public LogAppender {
    private:
        mutable std::mutex mutex_;
        std::string filename_;
        std::ofstream file_stream_;
        std::atomic<size_t> current_file_size_;
        size_t max_file_size_;
        int max_backup_files_;
        bool rotate_on_open_;
        
    public:
        FileAppender(std::unique_ptr<LogFormatter> formatter,
                    const std::string& filename,
                    LogLevel min_level = LogLevel::INFO,
                    size_t max_file_size = 10 * 1024 * 1024, // 10MB default
                    int max_backup_files = 5,
                    bool rotate_on_open = false);
        ~FileAppender();
        
        void append(const LogRecord& record) override;
        void flush() override;
        
        const std::string& get_filename() const;
        void set_max_file_size(size_t size);
        size_t get_max_file_size() const;
        void set_max_backup_files(int count);
        int get_max_backup_files() const;
        
    private:
        void rotate_files();
        bool open_file();
    };
    
    // Rotating file appender
    class RotatingFileAppender : public FileAppender {
    public:
        RotatingFileAppender(std::unique_ptr<LogFormatter> formatter,
                            const std::string& filename,
                            LogLevel min_level = LogLevel::INFO,
                            size_t max_file_size = 10 * 1024 * 1024, // 10MB default
                            int max_backup_files = 5);
    };
    
    // Logger class
    class Logger {
    private:
        std::string name_;
        std::vector<std::shared_ptr<LogAppender>> appenders_;
        mutable std::mutex appenders_mutex_;
        std::atomic<LogLevel> min_level_;
        std::atomic<bool> enabled_;
        
    public:
        explicit Logger(const std::string& name, LogLevel min_level = LogLevel::INFO);
        
        void add_appender(std::shared_ptr<LogAppender> appender);
        void remove_appender(const std::string& appender_name);
        std::vector<std::shared_ptr<LogAppender>> get_appenders() const;
        
        void set_min_level(LogLevel level);
        LogLevel get_min_level() const;
        void set_enabled(bool enabled);
        bool is_enabled() const;
        
        const std::string& get_name() const;
        
        // Logging methods
        void log(LogLevel level, const std::string& message, 
                const std::string& file = "", int line = 0, 
                const std::string& function = "",
                const std::map<std::string, std::string>& context = {});
        
        void trace(const std::string& message, 
                  const std::string& file = "", int line = 0, 
                  const std::string& function = "",
                  const std::map<std::string, std::string>& context = {});
                  
        void debug(const std::string& message, 
                  const std::string& file = "", int line = 0, 
                  const std::string& function = "",
                  const std::map<std::string, std::string>& context = {});
                  
        void info(const std::string& message, 
                 const std::string& file = "", int line = 0, 
                 const std::string& function = "",
                 const std::map<std::string, std::string>& context = {});
                 
        void warn(const std::string& message, 
                 const std::string& file = "", int line = 0, 
                 const std::string& function = "",
                 const std::map<std::string, std::string>& context = {});
                 
        void error(const std::string& message, 
                  const std::string& file = "", int line = 0, 
                  const std::string& function = "",
                  const std::map<std::string, std::string>& context = {});
                  
        void fatal(const std::string& message, 
                  const std::string& file = "", int line = 0, 
                  const std::string& function = "",
                  const std::map<std::string, std::string>& context = {});
        
    private:
        bool should_log(LogLevel level) const;
        std::string get_timestamp() const;
    };
    
    // Logger manager
    class LoggerManager {
    private:
        static std::map<std::string, std::shared_ptr<Logger>> loggers_;
        static std::mutex loggers_mutex_;
        static std::shared_ptr<Logger> default_logger_;
        static std::atomic<bool> initialized_;
        static LogLevel global_min_level_;
        
    public:
        // Initialize the logger manager
        static void initialize(LogLevel global_min_level = LogLevel::INFO);
        
        // Get or create a logger by name
        static std::shared_ptr<Logger> get_logger(const std::string& name);
        
        // Get the default logger
        static std::shared_ptr<Logger> get_default_logger();
        
        // Set the default logger
        static void set_default_logger(std::shared_ptr<Logger> logger);
        
        // Set global minimum log level
        static void set_global_min_level(LogLevel level);
        static LogLevel get_global_min_level();
        
        // Shutdown and cleanup
        static void shutdown();
        
        // Check if initialized
        static bool is_initialized();
        
    private:
        LoggerManager() = default;
    };
    
    // Macro helpers for easier logging
    #define LOG_TRACE(logger, message) \
        do { \
            if (logger && logger->should_log(::jadevectordb::logging::LogLevel::TRACE)) { \
                std::ostringstream oss; \
                oss << message; \
                logger->trace(oss.str(), __FILE__, __LINE__, __FUNCTION__); \
            } \
        } while(0)
        
    #define LOG_DEBUG(logger, message) \
        do { \
            if (logger && logger->should_log(::jadevectordb::logging::LogLevel::DEBUG)) { \
                std::ostringstream oss; \
                oss << message; \
                logger->debug(oss.str(), __FILE__, __LINE__, __FUNCTION__); \
            } \
        } while(0)
        
    #define LOG_INFO(logger, message) \
        do { \
            if (logger && logger->should_log(::jadevectordb::logging::LogLevel::INFO)) { \
                std::ostringstream oss; \
                oss << message; \
                logger->info(oss.str(), __FILE__, __LINE__, __FUNCTION__); \
            } \
        } while(0)
        
    #define LOG_WARN(logger, message) \
        do { \
            if (logger && logger->should_log(::jadevectordb::logging::LogLevel::WARN)) { \
                std::ostringstream oss; \
                oss << message; \
                logger->warn(oss.str(), __FILE__, __LINE__, __FUNCTION__); \
            } \
        } while(0)
        
    #define LOG_ERROR(logger, message) \
        do { \
            if (logger && logger->should_log(::jadevectordb::logging::LogLevel::ERROR)) { \
                std::ostringstream oss; \
                oss << message; \
                logger->error(oss.str(), __FILE__, __LINE__, __FUNCTION__); \
            } \
        } while(0)
        
    #define LOG_FATAL(logger, message) \
        do { \
            if (logger && logger->should_log(::jadevectordb::logging::LogLevel::FATAL)) { \
                std::ostringstream oss; \
                oss << message; \
                logger->fatal(oss.str(), __FILE__, __LINE__, __FUNCTION__); \
            } \
        } while(0)
        
    // Convenience macros using the default logger
    #define LOG_TRACE_DEFAULT(message) \
        LOG_TRACE(::jadevectordb::logging::LoggerManager::get_default_logger(), message)
        
    #define LOG_DEBUG_DEFAULT(message) \
        LOG_DEBUG(::jadevectordb::logging::LoggerManager::get_default_logger(), message)
        
    #define LOG_INFO_DEFAULT(message) \
        LOG_INFO(::jadevectordb::logging::LoggerManager::get_default_logger(), message)
        
    #define LOG_WARN_DEFAULT(message) \
        LOG_WARN(::jadevectordb::logging::LoggerManager::get_default_logger(), message)
        
    #define LOG_ERROR_DEFAULT(message) \
        LOG_ERROR(::jadevectordb::logging::LoggerManager::get_default_logger(), message)
        
    #define LOG_FATAL_DEFAULT(message) \
        LOG_FATAL(::jadevectordb::logging::LoggerManager::get_default_logger(), message)

} // namespace logging

} // namespace jadevectordb

#endif // JADEVECTORDB_LOGGING_H