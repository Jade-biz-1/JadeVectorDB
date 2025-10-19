#include "gpu_detection.h"
#include <vector>
#include <string>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <vector_types.h>
#endif

#ifdef OPENCL_AVAILABLE
// OpenCL headers would go here
// #include <CL/cl.h>
#endif

namespace jadevectordb {
namespace gpu_detection {

std::vector<GPUDeviceInfo> detect_cuda_gpus() {
    std::vector<GPUDeviceInfo> devices;
    
#ifdef CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        // No CUDA devices available
        return devices;
    }
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        GPUDeviceInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.memory_size = static_cast<size_t>(prop.totalGlobalMem);
        info.is_available = true;
        
        // Format compute capability as "x.y"
        info.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        
        devices.push_back(info);
    }
#endif // CUDA_AVAILABLE
    
    return devices;
}

std::vector<GPUDeviceInfo> detect_opencl_gpus() {
    std::vector<GPUDeviceInfo> devices;
    
    // NOTE: This is a placeholder since we don't have OpenCL fully integrated
    // In a real implementation, we would query OpenCL platforms and devices
    
#ifdef OPENCL_AVAILABLE
    // Placeholder for OpenCL implementation
    // This would involve:
    // 1. Enumerating OpenCL platforms
    // 2. Querying devices on each platform
    // 3. Identifying GPU devices specifically
    // 4. Gathering device information
#endif
    
    return devices;
}

std::vector<GPUDeviceInfo> get_available_acceleration_hardware() {
    std::vector<GPUDeviceInfo> all_devices;
    
    // Add CUDA devices
    auto cuda_devices = detect_cuda_gpus();
    all_devices.insert(all_devices.end(), cuda_devices.begin(), cuda_devices.end());
    
    // Add OpenCL devices
    auto opencl_devices = detect_opencl_gpus();
    all_devices.insert(all_devices.end(), opencl_devices.begin(), opencl_devices.end());
    
    return all_devices;
}

bool is_gpu_available() {
    auto devices = get_available_acceleration_hardware();
    return !devices.empty();
}

bool initialize_gpu_environment() {
#ifdef CUDA_AVAILABLE
    // Try to initialize CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    if (device_count > 0) {
        // Set the first device as the current device
        cudaSetDevice(0);
    }
#endif
    
    return true;
}

void cleanup_gpu_environment() {
#ifdef CUDA_AVAILABLE
    // Reset the device
    int current_device = 0;
    cudaGetDevice(&current_device);
    cudaDeviceReset();
#endif
}

} // namespace gpu_detection
} // namespace jadevectordb