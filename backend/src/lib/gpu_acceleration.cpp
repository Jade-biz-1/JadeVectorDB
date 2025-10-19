#include "gpu_acceleration.h"
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/sysinfo.h>  // For getting system memory info on Linux

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace jadevectordb {
namespace hardware {

// CPUDevice implementation
CPUDevice::CPUDevice() {
    // Constructor implementation
}

size_t CPUDevice::get_memory_size() const {
    // Get available system memory
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<size_t>(info.freeram) * info.mem_unit;
    }
    return 0; // Return 0 if unable to get memory info
}

void* CPUDevice::allocate(size_t size) {
    if (size == 0) return nullptr;
    void* ptr = std::malloc(size);
    return ptr;
}

void CPUDevice::deallocate(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

void CPUDevice::copy_to_device(void* dst, const void* src, size_t size) {
    // For CPU device, this is just a regular memory copy
    std::memcpy(dst, src, size);
}

void CPUDevice::copy_to_host(void* dst, const void* src, size_t size) {
    // For CPU device, this is just a regular memory copy
    std::memcpy(dst, src, size);
}

void CPUDevice::synchronize() {
    // CPU operations are synchronous by default
}

#ifdef CUDA_AVAILABLE
// CUDADevice implementation
CUDADevice::CUDADevice(int device_id) : cuda_available_(false), device_id_(device_id), memory_size_(0) {
    initialize();
}

void CUDADevice::initialize() {
    cudaError_t error = cudaSetDevice(device_id_);
    if (error != cudaSuccess) {
        cuda_available_ = false;
        return;
    }
    
    // Check device properties
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, device_id_);
    if (error != cudaSuccess) {
        cuda_available_ = false;
        return;
    }
    
    device_name_ = prop.name;
    memory_size_ = static_cast<size_t>(prop.totalGlobalMem);
    cuda_available_ = true;
}

void* CUDADevice::allocate(size_t size) {
    if (!cuda_available_ || size == 0) return nullptr;
    
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void CUDADevice::deallocate(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void CUDADevice::copy_to_device(void* dst, const void* src, size_t size) {
    if (!cuda_available_ || !dst || !src || size == 0) return;
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void CUDADevice::copy_to_host(void* dst, const void* src, size_t size) {
    if (!cuda_available_ || !dst || !src || size == 0) return;
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void CUDADevice::synchronize() {
    if (cuda_available_) {
        cudaDeviceSynchronize();
    }
}
#endif  // CUDA_AVAILABLE

// DeviceManager implementation
DeviceManager::DeviceManager() {
    cpu_device_ = std::make_shared<CPUDevice>();
    available_devices_.push_back(cpu_device_);
}

DeviceManager& DeviceManager::get_instance() {
    static DeviceManager instance;
    return instance;
}

void DeviceManager::initialize(DeviceType preferred_device_type) {
    available_devices_.clear();
    available_devices_.push_back(cpu_device_);
    
    // Try to initialize other devices based on preference
    // For now, only CPU is added by default
    
#ifdef CUDA_AVAILABLE
    // Try to add CUDA device if preferred
    if (preferred_device_type == DeviceType::CUDA) {
        auto cuda_device = std::make_shared<CUDADevice>();
        if (cuda_device->is_available()) {
            available_devices_.push_back(cuda_device);
        }
    }
#endif
    
    // Set active device based on preference and availability
    active_device_ = cpu_device_; // Default to CPU if no other device is available
    
    for (const auto& device : available_devices_) {
        if (device->get_device_type() == preferred_device_type && device->is_available()) {
            active_device_ = device;
            break;
        }
    }
    
    // If preferred device is not available, use first available device (other than CPU)
    if (active_device_->get_device_type() != preferred_device_type) {
        for (const auto& device : available_devices_) {
            if (device->get_device_type() != DeviceType::CPU && device->is_available()) {
                active_device_ = device;
                break;
            }
        }
    }
}

std::shared_ptr<IDevice> DeviceManager::get_active_device() const {
    return active_device_;
}

const std::vector<std::shared_ptr<IDevice>>& DeviceManager::get_available_devices() const {
    return available_devices_;
}

std::shared_ptr<IDevice> DeviceManager::find_device(DeviceType type) const {
    for (const auto& device : available_devices_) {
        if (device->get_device_type() == type) {
            return device;
        }
    }
    return nullptr;
}

bool DeviceManager::is_device_type_available(DeviceType type) const {
    return find_device(type) != nullptr;
}

} // namespace hardware
} // namespace jadevectordb