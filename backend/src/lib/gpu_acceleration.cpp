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

#ifdef OPENCL_AVAILABLE
#include <CL/cl.h>
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

#ifdef OPENCL_AVAILABLE
// OpenCLDevice implementation
OpenCLDevice::OpenCLDevice(cl_device_id cl_dev_id, cl_platform_id platform_id)
    : cl_device_id_(cl_dev_id), cl_platform_id_(platform_id), opencl_available_(false),
      memory_size_(0), device_name_(""), vendor_name_("") {
    initialize();
}

void OpenCLDevice::initialize() {
    cl_int error;

    // Get device name
    char name_buffer[256];
    error = clGetDeviceInfo(cl_device_id_, CL_DEVICE_NAME, sizeof(name_buffer), name_buffer, nullptr);
    if (error == CL_SUCCESS) {
        device_name_ = std::string(name_buffer);
    }

    // Get vendor name
    char vendor_buffer[256];
    error = clGetDeviceInfo(cl_device_id_, CL_DEVICE_VENDOR, sizeof(vendor_buffer), vendor_buffer, nullptr);
    if (error == CL_SUCCESS) {
        vendor_name_ = std::string(vendor_buffer);
    }

    // Get memory size
    cl_ulong mem_size;
    error = clGetDeviceInfo(cl_device_id_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
    if (error == CL_SUCCESS) {
        memory_size_ = static_cast<size_t>(mem_size);
    }

    // Check if device is available
    cl_bool available;
    error = clGetDeviceInfo(cl_device_id_, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr);
    if (error == CL_SUCCESS) {
        opencl_available_ = (available == CL_TRUE);
    }

    // Create context and queue for this device
    context_ = clCreateContext(nullptr, 1, &cl_device_id_, nullptr, nullptr, &error);
    if (error == CL_SUCCESS) {
        queue_ = clCreateCommandQueue(context_, cl_device_id_, 0, &error);
        if (error != CL_SUCCESS) {
            opencl_available_ = false;
        }
    } else {
        opencl_available_ = false;
    }
}

void* OpenCLDevice::allocate(size_t size) {
    if (!opencl_available_ || size == 0) return nullptr;

    cl_int error;
    cl_mem mem_buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &error);
    if (error != CL_SUCCESS) {
        return nullptr;
    }

    return static_cast<void*>(mem_buffer);
}

void OpenCLDevice::deallocate(void* ptr) {
    if (ptr && opencl_available_) {
        cl_mem mem_buffer = static_cast<cl_mem>(ptr);
        clReleaseMemObject(mem_buffer);
    }
}

void OpenCLDevice::copy_to_device(void* dst, const void* src, size_t size) {
    if (!opencl_available_ || !dst || !src || size == 0) return;

    cl_mem dst_buffer = static_cast<cl_mem>(dst);
    cl_int error = clEnqueueWriteBuffer(queue_, dst_buffer, CL_TRUE, 0, size, src, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        // Handle error appropriately
    }
}

void OpenCLDevice::copy_to_host(void* dst, const void* src, size_t size) {
    if (!opencl_available_ || !dst || !src || size == 0) return;

    cl_mem src_buffer = static_cast<cl_mem>(src);
    cl_int error = clEnqueueReadBuffer(queue_, src_buffer, CL_TRUE, 0, size, dst, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        // Handle error appropriately
    }
}

void OpenCLDevice::synchronize() {
    if (opencl_available_ && queue_) {
        clFinish(queue_);
    }
}
#endif // OPENCL_AVAILABLE

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

#ifdef CUDA_AVAILABLE
    // Try to add CUDA device if preferred or if CUDA is available
    if (preferred_device_type == DeviceType::CUDA || preferred_device_type == DeviceType::UNAVAILABLE) {
        auto cuda_device = std::make_shared<CUDADevice>();
        if (cuda_device->is_available()) {
            available_devices_.push_back(cuda_device);
        }
    }
#endif

#ifdef OPENCL_AVAILABLE
    // Try to add OpenCL devices if preferred or if available
    if (preferred_device_type == DeviceType::OPENCL || preferred_device_type == DeviceType::UNAVAILABLE) {
        cl_uint platform_count = 0;
        cl_int error = clGetPlatformIDs(0, nullptr, &platform_count);

        if (error == CL_SUCCESS && platform_count > 0) {
            std::vector<cl_platform_id> platforms(platform_count);
            error = clGetPlatformIDs(platform_count, platforms.data(), nullptr);

            if (error == CL_SUCCESS) {
                for (cl_uint p = 0; p < platform_count; ++p) {
                    cl_uint device_count = 0;
                    error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);

                    if (error == CL_SUCCESS && device_count > 0) {
                        std::vector<cl_device_id> cl_devices(device_count);
                        error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, device_count, cl_devices.data(), nullptr);

                        if (error == CL_SUCCESS) {
                            for (cl_uint d = 0; d < device_count; ++d) {
                                auto opencl_device = std::make_shared<OpenCLDevice>(cl_devices[d], platforms[p]);
                                if (opencl_device->is_available()) {
                                    available_devices_.push_back(opencl_device);
                                }
                            }
                        }
                    }
                }
            }
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