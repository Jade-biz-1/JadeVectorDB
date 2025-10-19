#include "gpu_memory_manager.h"
#include <stdexcept>

namespace jadevectordb {
namespace gpu_memory {

GPUDeviceMemoryManager::GPUDeviceMemoryManager(std::shared_ptr<hardware::IDevice> device)
    : device_(device), total_allocated_(0), total_cached_(0) {
    if (!device_ || !device_->is_available()) {
        throw std::runtime_error("Invalid or unavailable device for GPU memory management");
    }
}

GPUDeviceMemoryManager::~GPUDeviceMemoryManager() {
    // Clear any cached blocks
    clear_cache();
}

void* GPUDeviceMemoryManager::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    // In a more advanced implementation, we might try to reuse cached blocks here
    // For now, directly allocate from the device
    void* ptr = device_->allocate(size);
    if (ptr) {
        total_allocated_ += size;
    }
    
    return ptr;
}

void GPUDeviceMemoryManager::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    device_->deallocate(ptr);
    // In a real implementation with caching, we might cache blocks here instead of deallocating
}

std::pair<size_t, size_t> GPUDeviceMemoryManager::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return {total_allocated_, total_cached_};
}

void GPUDeviceMemoryManager::clear_cache() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    // Deallocate all cached blocks
    for (auto& pair : cached_blocks_) {
        for (void* block : pair.second) {
            device_->deallocate(block);
            total_allocated_ -= pair.first;
        }
        pair.second.clear();
    }
    total_cached_ = 0;
}

void GPUDeviceMemoryManager::copy_to_device(void* dst, const void* src, size_t size) {
    device_->copy_to_device(dst, src, size);
}

void GPUDeviceMemoryManager::copy_to_host(void* dst, const void* src, size_t size) {
    device_->copy_to_host(dst, src, size);
}

// GPUDeviceMemory implementation
GPUDeviceMemory::GPUDeviceMemory(std::shared_ptr<GPUDeviceMemoryManager> memory_manager, size_t size)
    : ptr_(nullptr), size_(size), memory_manager_(memory_manager) {
    
    if (size_ > 0) {
        ptr_ = memory_manager_->allocate(size_);
        if (!ptr_ && size_ > 0) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
}

GPUDeviceMemory::~GPUDeviceMemory() {
    if (ptr_ && memory_manager_) {
        memory_manager_->deallocate(ptr_);
    }
}

GPUDeviceMemory::GPUDeviceMemory(GPUDeviceMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), memory_manager_(std::move(other.memory_manager_)) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

GPUDeviceMemory& GPUDeviceMemory::operator=(GPUDeviceMemory&& other) noexcept {
    if (this != &other) {
        // Deallocate current memory if valid
        if (ptr_ && memory_manager_) {
            memory_manager_->deallocate(ptr_);
        }
        
        // Move resources from other
        ptr_ = other.ptr_;
        size_ = other.size_;
        memory_manager_ = std::move(other.memory_manager_);
        
        // Reset other
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

// MemoryTransferManager implementation
MemoryTransferManager::MemoryTransferManager(std::shared_ptr<hardware::IDevice> device)
    : device_(device) {
}

GPUDeviceMemory MemoryTransferManager::transfer_to_device(const std::vector<float>& host_data) {
    if (host_data.empty()) {
        return GPUDeviceMemory(std::make_shared<GPUDeviceMemoryManager>(device_), 0);
    }
    
    auto memory_manager = std::make_shared<GPUDeviceMemoryManager>(device_);
    GPUDeviceMemory device_mem(memory_manager, host_data.size() * sizeof(float));
    
    device_->copy_to_device(device_mem.get(), host_data.data(), device_mem.size());
    
    return device_mem;
}

GPUDeviceMemory MemoryTransferManager::transfer_to_device(const float* host_data, size_t size) {
    if (!host_data || size == 0) {
        return GPUDeviceMemory(std::make_shared<GPUDeviceMemoryManager>(device_), 0);
    }
    
    auto memory_manager = std::make_shared<GPUDeviceMemoryManager>(device_);
    GPUDeviceMemory device_mem(memory_manager, size * sizeof(float));
    
    device_->copy_to_device(device_mem.get(), host_data, device_mem.size());
    
    return device_mem;
}

std::vector<float> MemoryTransferManager::transfer_to_host(const GPUDeviceMemory& device_mem) {
    if (!device_mem.get() || device_mem.size() == 0) {
        return std::vector<float>();
    }
    
    size_t float_count = device_mem.size() / sizeof(float);
    std::vector<float> host_data(float_count);
    
    device_->copy_to_host(host_data.data(), device_mem.get(), device_mem.size());
    
    return host_data;
}

} // namespace gpu_memory
} // namespace jadevectordb