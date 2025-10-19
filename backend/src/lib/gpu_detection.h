#ifndef JADEVECTORDB_GPU_DETECTION_H
#define JADEVECTORDB_GPU_DETECTION_H

#include <string>
#include <vector>

namespace jadevectordb {
namespace gpu_detection {

    struct GPUDeviceInfo {
        int device_id;
        std::string name;
        size_t memory_size;  // in bytes
        std::string compute_capability;  // for CUDA
        bool is_available;
    };

    /**
     * @brief Detect available NVIDIA GPUs using CUDA
     * 
     * This function checks for NVIDIA GPUs by attempting to initialize CUDA
     * and querying the available devices.
     * 
     * @return Vector of GPUDeviceInfo structs for each detected GPU
     */
    std::vector<GPUDeviceInfo> detect_cuda_gpus();

    /**
     * @brief Detect available GPUs using OpenCL
     * 
     * This function checks for GPUs supporting OpenCL (NVIDIA, AMD, Intel).
     * 
     * @return Vector of GPUDeviceInfo structs for each detected GPU
     */
    std::vector<GPUDeviceInfo> detect_opencl_gpus();

    /**
     * @brief Get a summary of available hardware for acceleration
     * 
     * This function combines detection results from multiple APIs to provide
     * a comprehensive view of available acceleration hardware.
     * 
     * @return Vector of GPUDeviceInfo structs for all detected acceleration devices
     */
    std::vector<GPUDeviceInfo> get_available_acceleration_hardware();

    /**
     * @brief Check if any GPU is available for computation
     * 
     * @return True if at least one GPU is available, false otherwise
     */
    bool is_gpu_available();

    /**
     * @brief Initialize GPU environment
     * 
     * Performs necessary initialization steps for the GPU environment.
     * This might include initializing CUDA context, setting device, etc.
     * 
     * @return True if initialization was successful, false otherwise
     */
    bool initialize_gpu_environment();

    /**
     * @brief Cleanup GPU environment
     * 
     * Performs cleanup operations for the GPU environment.
     */
    void cleanup_gpu_environment();

} // namespace gpu_detection
} // namespace jadevectordb

#endif // JADEVECTORDB_GPU_DETECTION_H