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

#ifdef OPENCL_AVAILABLE
    cl_uint platform_count = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &platform_count);

    if (error != CL_SUCCESS || platform_count == 0) {
        // No OpenCL platforms available
        return devices;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    error = clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    if (error != CL_SUCCESS) {
        return devices;
    }

    for (cl_uint p = 0; p < platform_count; ++p) {
        cl_uint device_count = 0;
        error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);

        if (error != CL_SUCCESS || device_count == 0) {
            continue; // No GPU devices on this platform
        }

        std::vector<cl_device_id> cl_devices(device_count);
        error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, device_count, cl_devices.data(), nullptr);

        if (error != CL_SUCCESS) {
            continue;
        }

        for (cl_uint d = 0; d < device_count; ++d) {
            GPUDeviceInfo info;
            info.device_id = devices.size(); // Unique ID across all devices

            // Get device name
            char name_buffer[256];
            size_t name_size;
            clGetDeviceInfo(cl_devices[d], CL_DEVICE_NAME, sizeof(name_buffer), name_buffer, &name_size);
            info.name = std::string(name_buffer, name_size);

            // Get global memory size
            cl_ulong global_mem_size;
            clGetDeviceInfo(cl_devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
            info.memory_size = static_cast<size_t>(global_mem_size);

            // Get compute capability equivalent for OpenCL
            char version_buffer[256];
            size_t version_size;
            clGetDeviceInfo(cl_devices[d], CL_DEVICE_VERSION, sizeof(version_buffer), version_buffer, &version_size);
            info.compute_capability = std::string(version_buffer, version_size); // OpenCL version

            // Check if the device is available
            cl_bool available;
            clGetDeviceInfo(cl_devices[d], CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr);
            info.is_available = (available == CL_TRUE);

            devices.push_back(info);
        }
    }
#endif // OPENCL_AVAILABLE

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