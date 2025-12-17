#ifndef JADEVECTORDB_SIGNAL_HANDLER_H
#define JADEVECTORDB_SIGNAL_HANDLER_H

#include <csignal>
#include <atomic>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>
#include "lib/logging.h"

namespace jadevectordb {

/**
 * SignalHandler - Manages graceful shutdown on SIGTERM/SIGINT
 * 
 * Allows registering shutdown callbacks that will be invoked when
 * the process receives termination signals.
 */
class SignalHandler {
private:
    static std::atomic<bool> shutdown_requested_;
    static std::vector<std::function<void()>> shutdown_callbacks_;
    static std::mutex callbacks_mutex_;
    static std::shared_ptr<logging::Logger> logger_;
    
    static void signal_handler(int signal) {
        if (!logger_) {
            logger_ = logging::LoggerManager::get_logger("SignalHandler");
        }
        
        const char* signal_name = (signal == SIGTERM) ? "SIGTERM" : 
                                  (signal == SIGINT) ? "SIGINT" : "SIGNAL";
        
        LOG_INFO(logger_, "Received " << signal_name << " - initiating graceful shutdown");
        
        shutdown_requested_ = true;
        
        // Execute all registered callbacks
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (auto& callback : shutdown_callbacks_) {
            try {
                callback();
            } catch (const std::exception& e) {
                LOG_ERROR(logger_, "Error in shutdown callback: " << e.what());
            }
        }
        
        LOG_INFO(logger_, "Graceful shutdown completed");
    }
    
public:
    /**
     * Initialize signal handlers for SIGTERM and SIGINT
     */
    static void initialize() {
        logger_ = logging::LoggerManager::get_logger("SignalHandler");
        
        std::signal(SIGTERM, signal_handler);
        std::signal(SIGINT, signal_handler);
        
        LOG_INFO(logger_, "Signal handlers installed for SIGTERM and SIGINT");
    }
    
    /**
     * Register a callback to be executed on shutdown
     * @param callback Function to call during shutdown
     */
    static void register_shutdown_callback(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        shutdown_callbacks_.push_back(callback);
    }
    
    /**
     * Check if shutdown has been requested
     */
    static bool is_shutdown_requested() {
        return shutdown_requested_;
    }
    
    /**
     * Reset shutdown state (useful for testing)
     */
    static void reset() {
        shutdown_requested_ = false;
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        shutdown_callbacks_.clear();
    }
};

// Static member initialization
std::atomic<bool> SignalHandler::shutdown_requested_(false);
std::vector<std::function<void()>> SignalHandler::shutdown_callbacks_;
std::mutex SignalHandler::callbacks_mutex_;
std::shared_ptr<logging::Logger> SignalHandler::logger_;

} // namespace jadevectordb

#endif // JADEVECTORDB_SIGNAL_HANDLER_H
