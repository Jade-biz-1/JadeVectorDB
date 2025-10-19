#ifndef JADEVECTORDB_GPU_MEMORY_MANAGER_H
#define JADEVECTORDB_GPU_MEMORY_MANAGER_H

#include "gpu_acceleration.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace jadevectordb {
namespace gpu_memory {

    /**
     * @brief Manages GPU memory allocations and deallocations
     * 
     * This class provides a memory management system for GPU memory,
     * including allocation, deallocation, and potential caching of memory blocks
     * to reduce allocation overhead.
     */
    class GPUDeviceMemoryManager {
    private:
        std::shared_ptr<hardware::IDevice> device_;
        std::mutex memory_mutex_;
        
        // For potential memory caching/buffer pooling in the future
        std::unordered_map<size_t, std::vector<void*>> cached_blocks_;
        size_t total_allocated_;
        size_t total_cached_;
        
    public:
        explicit GPUDeviceMemoryManager(std::shared_ptr<hardware::IDevice> device);
        ~GPUDeviceMemoryManager();
        
        /**
         * @brief Allocate memory on the GPU device
         * @param size Size in bytes to allocate
         * @return Pointer to allocated GPU memory
         */
        void* allocate(size_t size);
        
        /**
         * @brief Deallocate memory on the GPU device
         * @param ptr Pointer to memory to deallocate
         */
        void deallocate(void* ptr);
        
        /**
         * @brief Get memory statistics
         * @return Pair with total allocated and total cached memory in bytes
         */
        std::pair<size_t, size_t> get_memory_stats() const;
        
        /**
         * @brief Clear cached memory blocks (free up memory)
         */
        void clear_cache();
        
        /**
         * @brief Copy data from host memory to GPU memory
         * @param dst Destination pointer on GPU
         * @param src Source pointer on host
         * @param size Size of data to copy in bytes
         */
        void copy_to_device(void* dst, const void* src, size_t size);
        
        /**
         * @brief Copy data from GPU memory to host memory
         * @param dst Destination pointer on host
         * @param src Source pointer on GPU
         * @param size Size of data to copy in bytes
         */
        void copy_to_host(void* dst, const void* src, size_t size);
    };

    /**
     * @brief RAII wrapper for GPU memory
     * 
     * This class provides automatic memory management for GPU memory,
     * automatically deallocating when the object goes out of scope.
     */
    class GPUDeviceMemory {
    private:
        void* ptr_;
        size_t size_;
        std::shared_ptr<GPUDeviceMemoryManager> memory_manager_;
        
    public:
        GPUDeviceMemory(std::shared_ptr<GPUDeviceMemoryManager> memory_manager, size_t size);
        ~GPUDeviceMemory();
        
        // Move constructor and assignment operator
        GPUDeviceMemory(GPUDeviceMemory&& other) noexcept;
        GPUDeviceMemory& operator=(GPUDeviceMemory&& other) noexcept;
        
        // Disable copy operations
        GPUDeviceMemory(const GPUDeviceMemory&) = delete;
        GPUDeviceMemory& operator=(const GPUDeviceMemory&) = delete;
        
        void* get() const { return ptr_; }
        size_t size() const { return size_; }
    };

    /**
     * @brief Memory transfer utility for different memory spaces
     * 
     * Provides utilities for transferring data between host and device memory.
     */
    class MemoryTransferManager {
    private:
        std::shared_ptr<hardware::IDevice> device_;
        
    public:
        explicit MemoryTransferManager(std::shared_ptr<hardware::IDevice> device);
        
        /**
         * @brief Transfer vector of floats from host to device memory
         * @param host_data Vector of floats on host
         * @return GPU memory object containing the data
         */
        GPUDeviceMemory transfer_to_device(const std::vector<float>& host_data);
        
        /**
         * @brief Transfer vector of floats from host to device memory
         * @param host_data Pointer to float array on host
         * @param size Number of elements to transfer
         * @return GPU memory object containing the data
         */
        GPUDeviceMemory transfer_to_device(const float* host_data, size_t size);
        
        /**
         * @brief Transfer data from device to host memory
         * @param device_mem GPU memory object
         * @return Vector of floats on host containing the data
         */
        std::vector<float> transfer_to_host(const GPUDeviceMemory& device_mem);
    };

} // namespace gpu_memory
} // namespace jadevectordb

#endif // JADEVECTORDB_GPU_MEMORY_MANAGER_H