#ifndef JADEVECTORDB_MEMORY_POOL_H
#define JADEVECTORDB_MEMORY_POOL_H

#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <unordered_map>

namespace jadevectordb {

// Memory pool utilities with SIMD-aligned allocations
namespace memory_pool {

    // Constants for memory alignment
    constexpr size_t SIMD_ALIGNMENT = 64;  // 64-byte alignment for AVX-512
    constexpr size_t CACHE_LINE_SIZE = 64; // Typical cache line size
    constexpr size_t DEFAULT_BLOCK_SIZE = 1024 * 1024; // 1MB default block size
    
    // Memory block structure
    struct MemoryBlock {
        void* ptr;           // Pointer to allocated memory
        size_t size;         // Size of the block
        size_t used;         // Amount of memory used in the block
        bool is_aligned;     // Whether the block is SIMD-aligned
        
        MemoryBlock() : ptr(nullptr), size(0), used(0), is_aligned(false) {}
        MemoryBlock(void* p, size_t s, bool aligned = false) 
            : ptr(p), size(s), used(0), is_aligned(aligned) {}
    };
    
    // Thread-local memory pool
    class ThreadLocalMemoryPool {
    private:
        std::vector<MemoryBlock> blocks_;
        size_t current_block_index_;
        size_t block_size_;
        std::atomic<size_t> total_allocated_;
        std::atomic<size_t> total_used_;
        
    public:
        explicit ThreadLocalMemoryPool(size_t block_size = DEFAULT_BLOCK_SIZE);
        ~ThreadLocalMemoryPool();
        
        // Allocate memory with optional alignment
        void* allocate(size_t size, size_t alignment = SIMD_ALIGNMENT);
        
        // Allocate SIMD-aligned memory
        void* allocate_simd(size_t size);
        
        // Allocate cache-line-aligned memory
        void* allocate_cache_line(size_t size);
        
        // Reset the pool (does not deallocate memory, just resets usage counters)
        void reset();
        
        // Get memory usage statistics
        size_t get_total_allocated() const;
        size_t get_total_used() const;
        size_t get_utilization_ratio() const; // Returns percentage * 100
        
        // Pre-allocate blocks to reduce allocation overhead
        void preallocate_blocks(size_t num_blocks);
        
    private:
        // Internal helper methods
        void add_new_block(size_t min_size);
        bool fits_in_current_block(size_t size) const;
        MemoryBlock create_aligned_block(size_t size, size_t alignment);
    };
    
    // Global memory pool manager
    class GlobalMemoryPool {
    private:
        static std::unordered_map<std::thread::id, std::unique_ptr<ThreadLocalMemoryPool>> thread_pools_;
        static std::mutex pools_mutex_;
        static std::atomic<size_t> global_allocated_;
        static std::atomic<size_t> global_used_;
        static size_t default_block_size_;
        
    public:
        // Get thread-local memory pool for current thread
        static ThreadLocalMemoryPool* get_thread_pool();
        
        // Get thread-local memory pool for specific thread
        static ThreadLocalMemoryPool* get_thread_pool(std::thread::id thread_id);
        
        // Create thread-local memory pool for current thread
        static ThreadLocalMemoryPool* create_thread_pool(size_t block_size = DEFAULT_BLOCK_SIZE);
        
        // Destroy thread-local memory pool for current thread
        static void destroy_thread_pool();
        
        // Destroy thread-local memory pool for specific thread
        static void destroy_thread_pool(std::thread::id thread_id);
        
        // Get global memory usage statistics
        static size_t get_global_allocated();
        static size_t get_global_used();
        static size_t get_global_utilization_ratio(); // Returns percentage * 100
        
        // Set default block size for new pools
        static void set_default_block_size(size_t block_size);
        static size_t get_default_block_size();
        
        // Cleanup all thread pools
        static void cleanup_all_pools();
        
    private:
        // Private constructor to prevent instantiation
        GlobalMemoryPool() = default;
    };
    
    // Memory pool RAII wrapper
    class MemoryPoolGuard {
    private:
        ThreadLocalMemoryPool* pool_;
        size_t initial_used_;
        
    public:
        explicit MemoryPoolGuard(ThreadLocalMemoryPool* pool = nullptr);
        ~MemoryPoolGuard();
        
        // Allocate memory through the guarded pool
        void* allocate(size_t size, size_t alignment = SIMD_ALIGNMENT);
        void* allocate_simd(size_t size);
        void* allocate_cache_line(size_t size);
        
        // Get the amount of memory allocated during guard's lifetime
        size_t get_allocated_during_scope() const;
    };
    
    // Utility functions for memory alignment
    namespace utils {
        
        // Check if pointer is aligned to specified boundary
        bool is_aligned(void* ptr, size_t alignment = SIMD_ALIGNMENT);
        
        // Align size to specified boundary
        size_t align_size(size_t size, size_t alignment = SIMD_ALIGNMENT);
        
        // Align pointer to specified boundary
        void* align_pointer(void* ptr, size_t alignment = SIMD_ALIGNMENT);
        
        // Calculate padding needed for alignment
        size_t calculate_padding(void* ptr, size_t alignment = SIMD_ALIGNMENT);
        
        // Allocate aligned memory using system allocator
        void* allocate_aligned(size_t size, size_t alignment = SIMD_ALIGNMENT);
        
        // Deallocate aligned memory
        void deallocate_aligned(void* ptr);
        
        // Allocate SIMD-aligned memory
        void* allocate_simd(size_t size);
        
        // Allocate cache-line-aligned memory
        void* allocate_cache_line(size_t size);
        
        // Memory copying with alignment awareness
        void copy_aligned(void* dest, const void* src, size_t size);
        
        // Memory filling with alignment awareness
        void fill_aligned(void* dest, int value, size_t size);
        
        // Memory zeroing with alignment awareness
        void zero_aligned(void* dest, size_t size);
        
        // Prefetch memory for read
        void prefetch_read(const void* ptr, size_t size);
        
        // Prefetch memory for write
        void prefetch_write(void* ptr, size_t size);
        
    } // namespace utils
    
    // Memory pool performance counters
    class MemoryPoolCounters {
    private:
        static std::atomic<size_t> allocation_count_;
        static std::atomic<size_t> deallocation_count_;
        static std::atomic<size_t> allocation_failure_count_;
        static std::atomic<size_t> reallocation_count_;
        static std::atomic<size_t> cache_hit_count_;
        static std::atomic<size_t> cache_miss_count_;
        
    public:
        // Increment counters
        static void increment_allocation();
        static void increment_deallocation();
        static void increment_allocation_failure();
        static void increment_reallocation();
        static void increment_cache_hit();
        static void increment_cache_miss();
        
        // Get counter values
        static size_t get_allocation_count();
        static size_t get_deallocation_count();
        static size_t get_allocation_failure_count();
        static size_t get_reallocation_count();
        static size_t get_cache_hit_count();
        static size_t get_cache_miss_count();
        static size_t get_cache_hit_ratio(); // Returns percentage * 100
        
        // Reset all counters
        static void reset_counters();
    };
    
    // Memory pool configuration
    struct MemoryPoolConfig {
        size_t default_block_size = DEFAULT_BLOCK_SIZE;
        size_t max_blocks_per_pool = 1024;
        bool enable_prefetching = true;
        size_t prefetch_distance = 256; // Bytes to prefetch ahead
        bool enable_statistics = true;
        size_t garbage_collection_threshold = 1024 * 1024; // 1MB threshold for GC
    };
    
    // Get current memory pool configuration
    const MemoryPoolConfig& get_memory_pool_config();
    
    // Set memory pool configuration
    void set_memory_pool_config(const MemoryPoolConfig& config);

} // namespace memory_pool

} // namespace jadevectordb

#endif // JADEVECTORDB_MEMORY_POOL_H