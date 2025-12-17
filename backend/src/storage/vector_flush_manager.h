#ifndef JADEVECTORDB_VECTOR_FLUSH_MANAGER_H
#define JADEVECTORDB_VECTOR_FLUSH_MANAGER_H

#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include "storage/memory_mapped_vector_store.h"
#include "lib/logging.h"

namespace jadevectordb {

/**
 * VectorFlushManager - Manages periodic flushing of memory-mapped vector storage
 * 
 * Runs a background thread that periodically flushes all vector stores to disk
 * to ensure durability while minimizing performance impact.
 */
class VectorFlushManager {
private:
    std::shared_ptr<MemoryMappedVectorStore> vector_store_;
    std::shared_ptr<logging::Logger> logger_;
    
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> flush_thread_;
    std::chrono::seconds flush_interval_;
    
    void flush_loop() {
        LOG_INFO(logger_, "VectorFlushManager started with " << flush_interval_.count() << "s interval");
        
        while (running_) {
            // Sleep for interval, checking every second for shutdown
            for (int i = 0; i < flush_interval_.count() && running_; i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            
            if (!running_) break;
            
            try {
                // Async flush (don't block on msync)
                vector_store_->flush_all(false);
                LOG_DEBUG(logger_, "Periodic flush completed");
            } catch (const std::exception& e) {
                LOG_ERROR(logger_, "Error during periodic flush: " << e.what());
            }
        }
        
        // Final synchronous flush on shutdown
        try {
            LOG_INFO(logger_, "Performing final synchronous flush before shutdown");
            vector_store_->flush_all(true);
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error during final flush: " << e.what());
        }
        
        LOG_INFO(logger_, "VectorFlushManager stopped");
    }
    
public:
    /**
     * Constructor
     * @param vector_store The vector store to flush periodically
     * @param flush_interval_seconds How often to flush (default: 5 seconds)
     */
    explicit VectorFlushManager(
        std::shared_ptr<MemoryMappedVectorStore> vector_store,
        int flush_interval_seconds = 5)
        : vector_store_(vector_store)
        , running_(false)
        , flush_interval_(flush_interval_seconds) {
        
        logger_ = logging::LoggerManager::get_logger("VectorFlushManager");
    }
    
    /**
     * Start the flush manager background thread
     */
    void start() {
        if (running_) {
            LOG_WARN(logger_, "VectorFlushManager already running");
            return;
        }
        
        running_ = true;
        flush_thread_ = std::make_unique<std::thread>(&VectorFlushManager::flush_loop, this);
        
        LOG_INFO(logger_, "VectorFlushManager started");
    }
    
    /**
     * Stop the flush manager gracefully
     * Performs a final synchronous flush before stopping
     */
    void stop() {
        if (!running_) {
            return;
        }
        
        LOG_INFO(logger_, "Stopping VectorFlushManager...");
        running_ = false;
        
        if (flush_thread_ && flush_thread_->joinable()) {
            flush_thread_->join();
        }
    }
    
    /**
     * Destructor - ensures clean shutdown
     */
    ~VectorFlushManager() {
        stop();
    }
    
    /**
     * Trigger an immediate flush (in addition to periodic flushes)
     * @param sync If true, blocks until flush completes (msync with MS_SYNC)
     */
    void flush_now(bool sync = false) {
        try {
            vector_store_->flush_all(sync);
            LOG_INFO(logger_, "Manual flush completed (sync=" << sync << ")");
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error during manual flush: " << e.what());
        }
    }
    
    /**
     * Check if the flush manager is running
     */
    bool is_running() const {
        return running_;
    }
    
    /**
     * Get the configured flush interval
     */
    std::chrono::seconds get_flush_interval() const {
        return flush_interval_;
    }
};

} // namespace jadevectordb

#endif // JADEVECTORDB_VECTOR_FLUSH_MANAGER_H
