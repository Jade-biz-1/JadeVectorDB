#ifndef JADEVECTORDB_GPU_ACCELERATION_H
#define JADEVECTORDB_GPU_ACCELERATION_H

#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace jadevectordb {
namespace hardware {

    // Enum to identify the type of device
    enum class DeviceType {
        CPU,
        CUDA,      // NVIDIA GPUs
        OPENCL,    // AMD/NVIDIA/Intel GPUs and CPUs
        UNAVAILABLE
    };

    /**
     * @brief Interface for hardware device abstraction
     * 
     * This class provides an abstraction layer for different hardware devices
     * (CPU, GPU) to allow for flexible switching between computation backends.
     */
    class IDevice {
    public:
        virtual ~IDevice() = default;
        
        /**
         * @brief Get the type of device this represents
         */
        virtual DeviceType get_device_type() const = 0;
        
        /**
         * @brief Check if the device is available and ready to use
         */
        virtual bool is_available() const = 0;
        
        /**
         * @brief Get a human-readable name for the device
         */
        virtual std::string get_device_name() const = 0;
        
        /**
         * @brief Get the available memory size in bytes
         */
        virtual size_t get_memory_size() const = 0;
        
        /**
         * @brief Allocate memory on the device
         * @param size Size in bytes to allocate
         * @return Pointer to allocated memory
         */
        virtual void* allocate(size_t size) = 0;
        
        /**
         * @brief Deallocate memory on the device
         * @param ptr Pointer to memory to deallocate
         */
        virtual void deallocate(void* ptr) = 0;
        
        /**
         * @brief Copy data from host memory to device memory
         * @param dst Destination pointer on device
         * @param src Source pointer on host
         * @param size Size of data to copy in bytes
         */
        virtual void copy_to_device(void* dst, const void* src, size_t size) = 0;
        
        /**
         * @brief Copy data from device memory to host memory
         * @param dst Destination pointer on host
         * @param src Source pointer on device
         * @param size Size of data to copy in bytes
         */
        virtual void copy_to_host(void* dst, const void* src, size_t size) = 0;
        
        virtual void synchronize() = 0;  // Wait for all operations to complete
    };
    
    /**
     * @brief Implementation for CPU device
     * 
     * This implementation performs operations on the CPU. It serves as the fallback
     * when no GPU is available.
     */
    class CPUDevice : public IDevice {
    public:
        CPUDevice();
        
        DeviceType get_device_type() const override { return DeviceType::CPU; }
        
        bool is_available() const override { return true; }  // CPU is always available
        
        std::string get_device_name() const override { return "CPU"; }
        
        size_t get_memory_size() const override;
        
        void* allocate(size_t size) override;
        
        void deallocate(void* ptr) override;
        
        void copy_to_device(void* dst, const void* src, size_t size) override;
        
        void copy_to_host(void* dst, const void* src, size_t size) override;
        
        void synchronize() override;
    };
    
    #ifdef CUDA_AVAILABLE
    /**
     * @brief Implementation for CUDA device (NVIDIA GPUs)
     * 
     * This implementation offloads operations to NVIDIA GPUs using CUDA.
     */
    class CUDADevice : public IDevice {
    private:
        bool cuda_available_;
        int device_id_;
        size_t memory_size_;
        std::string device_name_;
        
    public:
        CUDADevice(int device_id = 0);
        
        DeviceType get_device_type() const override { return DeviceType::CUDA; }
        
        bool is_available() const override { return cuda_available_; }
        
        std::string get_device_name() const override { return device_name_; }
        
        size_t get_memory_size() const override { return memory_size_; }
        
        void* allocate(size_t size) override;
        
        void deallocate(void* ptr) override;
        
        void copy_to_device(void* dst, const void* src, size_t size) override;
        
        void copy_to_host(void* dst, const void* src, size_t size) override;
        
        void synchronize() override;
        
    private:
        void initialize();
    };
    #endif  // CUDA_AVAILABLE

#ifdef OPENCL_AVAILABLE
#include <CL/cl.h>

    /**
     * @brief Implementation for OpenCL device (AMD/NVIDIA/Intel GPUs and CPUs)
     *
     * This implementation offloads operations to devices supporting OpenCL.
     */
    class OpenCLDevice : public IDevice {
    private:
        cl_device_id cl_device_id_;
        cl_platform_id cl_platform_id_;
        cl_context context_;
        cl_command_queue queue_;
        bool opencl_available_;
        size_t memory_size_;
        std::string device_name_;
        std::string vendor_name_;

    public:
        OpenCLDevice(cl_device_id cl_dev_id, cl_platform_id platform_id);

        DeviceType get_device_type() const override { return DeviceType::OPENCL; }

        bool is_available() const override { return opencl_available_; }

        std::string get_device_name() const override { return device_name_; }

        size_t get_memory_size() const override { return memory_size_; }

        void* allocate(size_t size) override;

        void deallocate(void* ptr) override;

        void copy_to_device(void* dst, const void* src, size_t size) override;

        void copy_to_host(void* dst, const void* src, size_t size) override;

        void synchronize() override;

    private:
        void initialize();
    };
#endif  // OPENCL_AVAILABLE

    /**
     * @brief Device manager to handle device selection and fallback
     *
     * This class manages available devices and provides the best available
     * device based on configuration and availability.
     */
    class DeviceManager {
    private:
        std::vector<std::shared_ptr<IDevice>> available_devices_;
        std::shared_ptr<IDevice> active_device_;
        std::shared_ptr<CPUDevice> cpu_device_;
        
        DeviceManager();  // Singleton
        
    public:
        static DeviceManager& get_instance();
        
        /**
         * @brief Initialize the device manager and detect available devices
         * @param preferred_device_type The preferred type of device to use
         */
        void initialize(DeviceType preferred_device_type = DeviceType::CPU);
        
        /**
         * @brief Get the currently active device
         */
        std::shared_ptr<IDevice> get_active_device() const;
        
        /**
         * @brief Get list of all available devices
         */
        const std::vector<std::shared_ptr<IDevice>>& get_available_devices() const;
        
        /**
         * @brief Find a specific type of device
         */
        std::shared_ptr<IDevice> find_device(DeviceType type) const;
        
        /**
         * @brief Check if a specific type of device is available
         */
        bool is_device_type_available(DeviceType type) const;
    };

} // namespace hardware
} // namespace jadevectordb

#endif // JADEVECTORDB_GPU_ACCELERATION_H