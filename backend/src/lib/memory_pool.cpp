#include "memory_pool.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <algorithm>
#include <stdexcept>
#include <xmmintrin.h> // For prefetch instructions

#ifdef _WIN32
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

namespace jadevectordb {

namespace memory_pool {

    // ThreadLocalMemoryPool implementation
    ThreadLocalMemoryPool::ThreadLocalMemoryPool(size_t block_size)
        : current_block_index_(0), block_size_(block_size), 
          total_allocated_(0), total_used_(0) {
    }
    
    ThreadLocalMemoryPool::~ThreadLocalMemoryPool() {
        // Deallocate all memory blocks
        for (const auto& block : blocks_) {
            if (block.ptr) {
                if (block.is_aligned) {
                    utils::deallocate_aligned(block.ptr);
                } else {
                    std::free(block.ptr);
                }
            }
        }
    }
    
    void* ThreadLocalMemoryPool::allocate(size_t size, size_t alignment) {
        // Align the requested size
        size_t aligned_size = utils::align_size(size, alignment);
        
        // Check if we can fit in the current block
        if (fits_in_current_block(aligned_size)) {
            MemoryBlock& current_block = blocks_[current_block_index_];
            
            // Align the pointer within the block
            void* aligned_ptr = utils::align_pointer(
                static_cast<char*>(current_block.ptr) + current_block.used, 
                alignment);
            
            // Calculate actual offset and update used count
            size_t actual_offset = static_cast<char*>(aligned_ptr) - 
                                  static_cast<char*>(current_block.ptr);
            current_block.used = actual_offset + aligned_size;
            
            total_used_.fetch_add(aligned_size, std::memory_order_relaxed);
            return aligned_ptr;
        }
        
        // Need a new block
        add_new_block(aligned_size);
        
        // Now allocate from the new block
        if (!blocks_.empty() && fits_in_current_block(aligned_size)) {
            MemoryBlock& current_block = blocks_[current_block_index_];
            
            // Align the pointer within the block
            void* aligned_ptr = utils::align_pointer(current_block.ptr, alignment);
            
            // Update used count
            current_block.used = aligned_size;
            
            total_used_.fetch_add(aligned_size, std::memory_order_relaxed);
            return aligned_ptr;
        }
        
        // If we still can't allocate, fall back to system allocation
        return utils::allocate_aligned(size, alignment);
    }
    
    void* ThreadLocalMemoryPool::allocate_simd(size_t size) {
        return allocate(size, SIMD_ALIGNMENT);
    }
    
    void* ThreadLocalMemoryPool::allocate_cache_line(size_t size) {
        return allocate(size, CACHE_LINE_SIZE);
    }
    
    void ThreadLocalMemoryPool::reset() {
        for (auto& block : blocks_) {
            block.used = 0;
        }
        current_block_index_ = 0;
        total_used_.store(0, std::memory_order_relaxed);
    }
    
    size_t ThreadLocalMemoryPool::get_total_allocated() const {
        return total_allocated_.load(std::memory_order_relaxed);
    }
    
    size_t ThreadLocalMemoryPool::get_total_used() const {
        return total_used_.load(std::memory_order_relaxed);
    }
    
    size_t ThreadLocalMemoryPool::get_utilization_ratio() const {
        size_t allocated = get_total_allocated();
        if (allocated == 0) return 0;
        return (get_total_used() * 10000) / allocated; // Return percentage * 100
    }
    
    void ThreadLocalMemoryPool::preallocate_blocks(size_t num_blocks) {
        for (size_t i = 0; i < num_blocks; ++i) {
            add_new_block(0);
        }
    }
    
    void ThreadLocalMemoryPool::add_new_block(size_t min_size) {
        size_t block_size = std::max(block_size_, min_size);
        
        // Create a new aligned block
        MemoryBlock new_block = create_aligned_block(block_size, SIMD_ALIGNMENT);
        
        // Add to blocks vector
        blocks_.push_back(new_block);
        current_block_index_ = blocks_.size() - 1;
        
        // Update total allocated
        total_allocated_.fetch_add(block_size, std::memory_order_relaxed);
    }
    
    bool ThreadLocalMemoryPool::fits_in_current_block(size_t size) const {
        if (blocks_.empty()) return false;
        if (current_block_index_ >= blocks_.size()) return false;
        
        const MemoryBlock& current_block = blocks_[current_block_index_];
        return (current_block.size - current_block.used) >= size;
    }
    
    MemoryBlock ThreadLocalMemoryPool::create_aligned_block(size_t size, size_t alignment) {
        void* ptr = utils::allocate_aligned(size, alignment);
        return MemoryBlock(ptr, size, true);
    }
    
    // GlobalMemoryPool implementation
    std::unordered_map<std::thread::id, std::unique_ptr<ThreadLocalMemoryPool>> GlobalMemoryPool::thread_pools_;
    std::mutex GlobalMemoryPool::pools_mutex_;
    std::atomic<size_t> GlobalMemoryPool::global_allocated_{0};
    std::atomic<size_t> GlobalMemoryPool::global_used_{0};
    size_t GlobalMemoryPool::default_block_size_ = DEFAULT_BLOCK_SIZE;
    
    ThreadLocalMemoryPool* GlobalMemoryPool::get_thread_pool() {
        std::thread::id this_id = std::this_thread::get_id();
        
        std::lock_guard<std::mutex> lock(pools_mutex_);
        auto it = thread_pools_.find(this_id);
        if (it != thread_pools_.end()) {
            return it->second.get();
        }
        
        // Create a new pool for this thread
        return create_thread_pool();
    }
    
    ThreadLocalMemoryPool* GlobalMemoryPool::get_thread_pool(std::thread::id thread_id) {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        auto it = thread_pools_.find(thread_id);
        return (it != thread_pools_.end()) ? it->second.get() : nullptr;
    }
    
    ThreadLocalMemoryPool* GlobalMemoryPool::create_thread_pool(size_t block_size) {
        std::thread::id this_id = std::this_thread::get_id();
        
        std::lock_guard<std::mutex> lock(pools_mutex_);
        auto it = thread_pools_.find(this_id);
        if (it != thread_pools_.end()) {
            return it->second.get();
        }
        
        // Create a new pool
        auto pool = std::make_unique<ThreadLocalMemoryPool>(block_size);
        ThreadLocalMemoryPool* pool_ptr = pool.get();
        thread_pools_[this_id] = std::move(pool);
        
        return pool_ptr;
    }
    
    void GlobalMemoryPool::destroy_thread_pool() {
        std::thread::id this_id = std::this_thread::get_id();
        destroy_thread_pool(this_id);
    }
    
    void GlobalMemoryPool::destroy_thread_pool(std::thread::id thread_id) {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        thread_pools_.erase(thread_id);
    }
    
    size_t GlobalMemoryPool::get_global_allocated() {
        return global_allocated_.load(std::memory_order_relaxed);
    }
    
    size_t GlobalMemoryPool::get_global_used() {
        return global_used_.load(std::memory_order_relaxed);
    }
    
    size_t GlobalMemoryPool::get_global_utilization_ratio() {
        size_t allocated = get_global_allocated();
        if (allocated == 0) return 0;
        return (get_global_used() * 10000) / allocated; // Return percentage * 100
    }
    
    void GlobalMemoryPool::set_default_block_size(size_t block_size) {
        default_block_size_ = block_size;
    }
    
    size_t GlobalMemoryPool::get_default_block_size() {
        return default_block_size_;
    }
    
    void GlobalMemoryPool::cleanup_all_pools() {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        thread_pools_.clear();
    }
    
    // MemoryPoolGuard implementation
    MemoryPoolGuard::MemoryPoolGuard(ThreadLocalMemoryPool* pool)
        : pool_(pool ? pool : GlobalMemoryPool::get_thread_pool()) {
        initial_used_ = pool_->get_total_used();
    }
    
    MemoryPoolGuard::~MemoryPoolGuard() {
        // Optionally reset the pool when guard goes out of scope
        // pool_->reset();
    }
    
    void* MemoryPoolGuard::allocate(size_t size, size_t alignment) {
        return pool_->allocate(size, alignment);
    }
    
    void* MemoryPoolGuard::allocate_simd(size_t size) {
        return pool_->allocate_simd(size);
    }
    
    void* MemoryPoolGuard::allocate_cache_line(size_t size) {
        return pool_->allocate_cache_line(size);
    }
    
    size_t MemoryPoolGuard::get_allocated_during_scope() const {
        return pool_->get_total_used() - initial_used_;
    }
    
    // Utility functions implementation
    namespace utils {
        
        bool is_aligned(void* ptr, size_t alignment) {
            return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
        }
        
        size_t align_size(size_t size, size_t alignment) {
            return (size + alignment - 1) & ~(alignment - 1);
        }
        
        void* align_pointer(void* ptr, size_t alignment) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
            return reinterpret_cast<void*>(aligned_addr);
        }
        
        size_t calculate_padding(void* ptr, size_t alignment) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
            return aligned_addr - addr;
        }
        
        void* allocate_aligned(size_t size, size_t alignment) {
#ifdef _WIN32
            return _aligned_malloc(size, alignment);
#else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, size) != 0) {
                return nullptr;
            }
            return ptr;
#endif
        }
        
        void deallocate_aligned(void* ptr) {
            if (!ptr) return;
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
        
        void* allocate_simd(size_t size) {
            return allocate_aligned(size, SIMD_ALIGNMENT);
        }
        
        void* allocate_cache_line(size_t size) {
            return allocate_aligned(size, CACHE_LINE_SIZE);
        }
        
        void copy_aligned(void* dest, const void* src, size_t size) {
            std::memcpy(dest, src, size);
        }
        
        void fill_aligned(void* dest, int value, size_t size) {
            std::memset(dest, value, size);
        }
        
        void zero_aligned(void* dest, size_t size) {
            std::memset(dest, 0, size);
        }
        
        void prefetch_read(const void* ptr, size_t size) {
            // Prefetch in chunks
            const char* p = static_cast<const char*>(ptr);
            size_t chunks = (size + 63) / 64; // 64 bytes per cache line
            
            for (size_t i = 0; i < chunks; ++i) {
                _mm_prefetch(p + i * 64, _MM_HINT_T0);
            }
        }
        
        void prefetch_write(void* ptr, size_t size) {
            // For write prefetching, we can use the same approach
            char* p = static_cast<char*>(ptr);
            size_t chunks = (size + 63) / 64; // 64 bytes per cache line
            
            for (size_t i = 0; i < chunks; ++i) {
                _mm_prefetch(p + i * 64, _MM_HINT_T0);
            }
        }
        
    } // namespace utils
    
    // MemoryPoolCounters implementation
    std::atomic<size_t> MemoryPoolCounters::allocation_count_{0};
    std::atomic<size_t> MemoryPoolCounters::deallocation_count_{0};
    std::atomic<size_t> MemoryPoolCounters::allocation_failure_count_{0};
    std::atomic<size_t> MemoryPoolCounters::reallocation_count_{0};
    std::atomic<size_t> MemoryPoolCounters::cache_hit_count_{0};
    std::atomic<size_t> MemoryPoolCounters::cache_miss_count_{0};
    
    void MemoryPoolCounters::increment_allocation() {
        allocation_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void MemoryPoolCounters::increment_deallocation() {
        deallocation_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void MemoryPoolCounters::increment_allocation_failure() {
        allocation_failure_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void MemoryPoolCounters::increment_reallocation() {
        reallocation_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void MemoryPoolCounters::increment_cache_hit() {
        cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void MemoryPoolCounters::increment_cache_miss() {
        cache_miss_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_allocation_count() {
        return allocation_count_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_deallocation_count() {
        return deallocation_count_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_allocation_failure_count() {
        return allocation_failure_count_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_reallocation_count() {
        return reallocation_count_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_cache_hit_count() {
        return cache_hit_count_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_cache_miss_count() {
        return cache_miss_count_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryPoolCounters::get_cache_hit_ratio() {
        size_t hits = get_cache_hit_count();
        size_t misses = get_cache_miss_count();
        size_t total = hits + misses;
        
        if (total == 0) return 0;
        return (hits * 10000) / total; // Return percentage * 100
    }
    
    void MemoryPoolCounters::reset_counters() {
        allocation_count_.store(0, std::memory_order_relaxed);
        deallocation_count_.store(0, std::memory_order_relaxed);
        allocation_failure_count_.store(0, std::memory_order_relaxed);
        reallocation_count_.store(0, std::memory_order_relaxed);
        cache_hit_count_.store(0, std::memory_order_relaxed);
        cache_miss_count_.store(0, std::memory_order_relaxed);
    }
    
    // Memory pool configuration
    static MemoryPoolConfig current_config_;
    
    const MemoryPoolConfig& get_memory_pool_config() {
        return current_config_;
    }
    
    void set_memory_pool_config(const MemoryPoolConfig& config) {
        current_config_ = config;
    }

} // namespace memory_pool

} // namespace jadevectordb