#include "logging.h"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <system_error>

namespace jadevectordb {

namespace logging {

    // Utility functions
    std::string log_level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARN: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
    
    LogLevel string_to_log_level(const std::string& level_str) {
        std::string upper_level = level_str;
        std::transform(upper_level.begin(), upper_level.end(), upper_level.begin(), ::toupper);
        
        if (upper_level == "TRACE") return LogLevel::TRACE;
        if (upper_level == "DEBUG") return LogLevel::DEBUG;
        if (upper_level == "INFO") return LogLevel::INFO;
        if (upper_level == "WARN") return LogLevel::WARN;
        if (upper_level == "ERROR") return LogLevel::ERROR;
        if (upper_level == "FATAL") return LogLevel::FATAL;
        
        return LogLevel::INFO; // Default
    }
    
    // SimpleTextFormatter implementation
    std::string SimpleTextFormatter::format(const LogRecord& record) {
        std::ostringstream oss;
        
        // Format: [TIMESTAMP] [LEVEL] [LOGGER] [THREAD_ID] MESSAGE
        oss << "[" << record.timestamp << "] "
            << "[" << log_level_to_string(record.level) << "] "
            << "[" << record.logger_name << "] "
            << "[" << std::hex << record.thread_id << std::dec << "] ";
            
        if (!record.module.empty()) {
            oss << "[" << record.module << "] ";
        }
        
        if (!record.component.empty()) {
            oss << "[" << record.component << "] ";
        }
        
        oss << record.message;
        
        // Add file and line information if available
        if (!record.file.empty() && record.line > 0) {
            oss << " (" << record.file << ":" << record.line;
            if (!record.function.empty()) {
                oss << " in " << record.function;
            }
            oss << ")";
        }
        
        // Add context information
        if (!record.context.empty()) {
            oss << " {";
            bool first = true;
            for (const auto& [key, value] : record.context) {
                if (!first) oss << ", ";
                oss << key << "=" << value;
                first = false;
            }
            oss << "}";
        }
        
        return oss.str();
    }
    
    // JsonFormatter implementation
    std::string JsonFormatter::format(const LogRecord& record) {
        std::ostringstream oss;
        
        oss << "{";
        oss << "\"timestamp\":\"" << record.timestamp << "\",";
        oss << "\"level\":\"" << log_level_to_string(record.level) << "\",";
        oss << "\"logger\":\"" << record.logger_name << "\",";
        oss << "\"thread_id\":\"" << std::hex << record.thread_id << std::dec << "\",";
        
        if (!record.module.empty()) {
            oss << "\"module\":\"" << record.module << "\",";
        }
        
        if (!record.component.empty()) {
            oss << "\"component\":\"" << record.component << "\",";
        }
        
        oss << "\"message\":\"" << record.message << "\"";
        
        // Add file and line information if available
        if (!record.file.empty() && record.line > 0) {
            oss << ",\"file\":\"" << record.file << "\"";
            oss << ",\"line\":" << record.line;
            if (!record.function.empty()) {
                oss << ",\"function\":\"" << record.function << "\"";
            }
        }
        
        // Add context information
        if (!record.context.empty()) {
            oss << ",\"context\":{";
            bool first = true;
            for (const auto& [key, value] : record.context) {
                if (!first) oss << ",";
                oss << "\"" << key << "\":\"" << value << "\"";
                first = false;
            }
            oss << "}";
        }
        
        oss << "}";
        
        return oss.str();
    }
    
    // LogAppender implementation
    LogAppender::LogAppender(std::unique_ptr<LogFormatter> formatter, LogLevel min_level)
        : formatter_(std::move(formatter)), min_level_(min_level), enabled_(true) {
    }
    
    bool LogAppender::should_log(LogLevel level) const {
        return enabled_.load(std::memory_order_relaxed) && 
               static_cast<int>(level) >= static_cast<int>(min_level_.load(std::memory_order_relaxed));
    }
    
    void LogAppender::set_min_level(LogLevel level) {
        min_level_.store(level, std::memory_order_relaxed);
    }
    
    LogLevel LogAppender::get_min_level() const {
        return min_level_.load(std::memory_order_relaxed);
    }
    
    void LogAppender::set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_relaxed);
    }
    
    bool LogAppender::is_enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }
    
    // ConsoleAppender implementation
    ConsoleAppender::ConsoleAppender(std::unique_ptr<LogFormatter> formatter, 
                                    LogLevel min_level,
                                    std::ostream& output_stream)
        : LogAppender(std::move(formatter), min_level), output_stream_(output_stream) {
    }
    
    void ConsoleAppender::append(const LogRecord& record) {
        if (!should_log(record.level)) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        output_stream_ << formatter_->format(record) << std::endl;
    }
    
    void ConsoleAppender::flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        output_stream_.flush();
    }
    
    // FileAppender implementation
    FileAppender::FileAppender(std::unique_ptr<LogFormatter> formatter,
                              const std::string& filename,
                              LogLevel min_level,
                              size_t max_file_size,
                              int max_backup_files,
                              bool rotate_on_open)
        : LogAppender(std::move(formatter), min_level),
          filename_(filename),
          current_file_size_(0),
          max_file_size_(max_file_size),
          max_backup_files_(max_backup_files),
          rotate_on_open_(rotate_on_open) {
        if (rotate_on_open_) {
            rotate_files();
        }
        open_file();
    }
    
    FileAppender::~FileAppender() {
        flush();
    }
    
    void FileAppender::append(const LogRecord& record) {
        if (!should_log(record.level)) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!file_stream_.is_open()) {
            if (!open_file()) {
                return;
            }
        }
        
        std::string formatted_message = formatter_->format(record);
        file_stream_ << formatted_message << std::endl;
        
        // Update file size
        current_file_size_.fetch_add(formatted_message.length() + 1, std::memory_order_relaxed);
        
        // Check if rotation is needed
        if (current_file_size_.load(std::memory_order_relaxed) > max_file_size_) {
            rotate_files();
            open_file();
        }
    }
    
    void FileAppender::flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_stream_.is_open()) {
            file_stream_.flush();
        }
    }
    
    const std::string& FileAppender::get_filename() const {
        return filename_;
    }
    
    void FileAppender::set_max_file_size(size_t size) {
        max_file_size_ = size;
    }
    
    size_t FileAppender::get_max_file_size() const {
        return max_file_size_;
    }
    
    void FileAppender::set_max_backup_files(int count) {
        max_backup_files_ = count;
    }
    
    int FileAppender::get_max_backup_files() const {
        return max_backup_files_;
    }
    
    void FileAppender::rotate_files() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Close current file
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
        
        // Rotate existing files
        for (int i = max_backup_files_ - 1; i >= 0; --i) {
            std::string old_name = filename_;
            std::string new_name = filename_;
            
            if (i > 0) {
                old_name += "." + std::to_string(i);
            } else {
                old_name = filename_;
            }
            
            new_name += "." + std::to_string(i + 1);
            
            // Rename files
            std::error_code ec;
            std::filesystem::rename(old_name, new_name, ec);
        }
        
        // Reset file size counter
        current_file_size_.store(0, std::memory_order_relaxed);
    }
    
    bool FileAppender::open_file() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Ensure directory exists
        std::filesystem::path file_path(filename_);
        if (file_path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(file_path.parent_path(), ec);
        }
        
        file_stream_.open(filename_, std::ios::app);
        return file_stream_.is_open();
    }
    
    // RotatingFileAppender implementation
    RotatingFileAppender::RotatingFileAppender(std::unique_ptr<LogFormatter> formatter,
                                              const std::string& filename,
                                              LogLevel min_level,
                                              size_t max_file_size,
                                              int max_backup_files)
        : FileAppender(std::move(formatter), filename, min_level, max_file_size, max_backup_files, true) {
    }
    
    // Logger implementation
    Logger::Logger(const std::string& name, LogLevel min_level)
        : name_(name), min_level_(min_level), enabled_(true) {
    }
    
    void Logger::add_appender(std::shared_ptr<LogAppender> appender) {
        std::lock_guard<std::mutex> lock(appenders_mutex_);
        appenders_.push_back(std::move(appender));
    }
    
    void Logger::remove_appender(const std::string& appender_name) {
        std::lock_guard<std::mutex> lock(appenders_mutex_);
        appenders_.erase(
            std::remove_if(appenders_.begin(), appenders_.end(),
                          [&appender_name](const std::shared_ptr<LogAppender>& appender) {
                              // This is a simplified implementation
                              // In a real implementation, we would need a way to identify appenders
                              return false;
                          }),
            appenders_.end());
    }
    
    std::vector<std::shared_ptr<LogAppender>> Logger::get_appenders() const {
        std::lock_guard<std::mutex> lock(appenders_mutex_);
        return appenders_;
    }
    
    void Logger::set_min_level(LogLevel level) {
        min_level_.store(level, std::memory_order_relaxed);
    }
    
    LogLevel Logger::get_min_level() const {
        return min_level_.load(std::memory_order_relaxed);
    }
    
    void Logger::set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_relaxed);
    }
    
    bool Logger::is_enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }
    
    const std::string& Logger::get_name() const {
        return name_;
    }
    
    void Logger::log(LogLevel level, const std::string& message, 
                    const std::string& file, int line, 
                    const std::string& function,
                    const std::map<std::string, std::string>& context) {
        if (!should_log(level) || !is_enabled()) return;
        
        LogRecord record;
        record.level = level;
        record.timestamp = get_timestamp();
        record.logger_name = name_;
        record.file = file;
        record.line = line;
        record.function = function;
        record.message = message;
        record.thread_id = std::this_thread::get_id();
        record.context = context;
        
        // Distribute to all appenders
        std::lock_guard<std::mutex> lock(appenders_mutex_);
        for (const auto& appender : appenders_) {
            if (appender && appender->should_log(level)) {
                appender->append(record);
            }
        }
    }
    
    void Logger::trace(const std::string& message, 
                      const std::string& file, int line, 
                      const std::string& function,
                      const std::map<std::string, std::string>& context) {
        log(LogLevel::TRACE, message, file, line, function, context);
    }
    
    void Logger::debug(const std::string& message, 
                       const std::string& file, int line, 
                       const std::string& function,
                       const std::map<std::string, std::string>& context) {
        log(LogLevel::DEBUG, message, file, line, function, context);
    }
    
    void Logger::info(const std::string& message, 
                      const std::string& file, int line, 
                      const std::string& function,
                      const std::map<std::string, std::string>& context) {
        log(LogLevel::INFO, message, file, line, function, context);
    }
    
    void Logger::warn(const std::string& message, 
                     const std::string& file, int line, 
                     const std::string& function,
                     const std::map<std::string, std::string>& context) {
        log(LogLevel::WARN, message, file, line, function, context);
    }
    
    void Logger::error(const std::string& message, 
                      const std::string& file, int line, 
                      const std::string& function,
                      const std::map<std::string, std::string>& context) {
        log(LogLevel::ERROR, message, file, line, function, context);
    }
    
    void Logger::fatal(const std::string& message, 
                      const std::string& file, int line, 
                      const std::string& function,
                      const std::map<std::string, std::string>& context) {
        log(LogLevel::FATAL, message, file, line, function, context);
    }
    
    bool Logger::should_log(LogLevel level) const {
        return enabled_.load(std::memory_order_relaxed) && 
               static_cast<int>(level) >= static_cast<int>(min_level_.load(std::memory_order_relaxed));
    }
    
    std::string Logger::get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        
        return oss.str();
    }
    
    // LoggerManager implementation
    std::map<std::string, std::shared_ptr<Logger>> LoggerManager::loggers_;
    std::mutex LoggerManager::loggers_mutex_;
    std::shared_ptr<Logger> LoggerManager::default_logger_;
    std::atomic<bool> LoggerManager::initialized_{false};
    LogLevel LoggerManager::global_min_level_ = LogLevel::INFO;
    
    void LoggerManager::initialize(LogLevel global_min_level) {
        if (initialized_.exchange(true)) {
            return; // Already initialized
        }
        
        global_min_level_ = global_min_level;
        
        // Create default logger with console appender
        default_logger_ = std::make_shared<Logger>("default", global_min_level);
        
        // Add console appender
        auto console_formatter = std::make_unique<SimpleTextFormatter>();
        auto console_appender = std::make_shared<ConsoleAppender>(
            std::move(console_formatter), global_min_level);
        default_logger_->add_appender(console_appender);
        
        // Add file appender
        auto json_formatter = std::make_unique<JsonFormatter>();
        auto file_appender = std::make_shared<RotatingFileAppender>(
            std::move(json_formatter), "logs/jadevectordb.log", global_min_level,
            10 * 1024 * 1024, 5); // 10MB max file size, 5 backup files
        default_logger_->add_appender(file_appender);
    }
    
    std::shared_ptr<Logger> LoggerManager::get_logger(const std::string& name) {
        if (!initialized_) {
            initialize();
        }
        
        std::lock_guard<std::mutex> lock(loggers_mutex_);
        
        auto it = loggers_.find(name);
        if (it != loggers_.end()) {
            return it->second;
        }
        
        // Create new logger
        auto logger = std::make_shared<Logger>(name, global_min_level_);
        loggers_[name] = logger;
        
        return logger;
    }
    
    std::shared_ptr<Logger> LoggerManager::get_default_logger() {
        if (!initialized_) {
            initialize();
        }
        
        return default_logger_;
    }
    
    void LoggerManager::set_default_logger(std::shared_ptr<Logger> logger) {
        if (initialized_) {
            std::lock_guard<std::mutex> lock(loggers_mutex_);
            default_logger_ = std::move(logger);
        }
    }
    
    void LoggerManager::set_global_min_level(LogLevel level) {
        global_min_level_ = level;
    }
    
    LogLevel LoggerManager::get_global_min_level() {
        return global_min_level_;
    }
    
    void LoggerManager::shutdown() {
        std::lock_guard<std::mutex> lock(loggers_mutex_);
        loggers_.clear();
        default_logger_.reset();
        initialized_.store(false);
    }
    
    bool LoggerManager::is_initialized() {
        return initialized_.load();
    }

} // namespace logging

} // namespace jadevectordb