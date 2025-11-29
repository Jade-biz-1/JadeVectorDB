#include "vector_operations.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#ifdef CUDA_AVAILABLE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace jadevectordb {
namespace vector_ops {

// CPUVectorOperations implementation
float CPUVectorOperations::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for cosine similarity");
    }
    
    if (a.empty()) {
        return 0.0f;
    }
    
    float dot_product_result = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product_result += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;  // Undefined cosine similarity when one vector is zero
    }
    
    return dot_product_result / (norm_a * norm_b);
}

float CPUVectorOperations::euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for euclidean distance");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

float CPUVectorOperations::dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }
    
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

float CPUVectorOperations::l2_norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

std::vector<float> CPUVectorOperations::normalize(const std::vector<float>& vec) {
    float norm = l2_norm(vec);
    if (norm == 0.0f) {
        return vec;  // Return zero vector as is
    }
    
    std::vector<float> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] / norm;
    }
    
    return result;
}

std::vector<float> CPUVectorOperations::batch_cosine_similarity(
    const std::vector<float>& query, 
    const std::vector<std::vector<float>>& vectors) {
    
    std::vector<float> similarities;
    similarities.reserve(vectors.size());
    
    for (const auto& vec : vectors) {
        similarities.push_back(cosine_similarity(query, vec));
    }
    
    return similarities;
}

std::vector<float> CPUVectorOperations::batch_euclidean_distance(
    const std::vector<float>& query, 
    const std::vector<std::vector<float>>& vectors) {
    
    std::vector<float> distances;
    distances.reserve(vectors.size());
    
    for (const auto& vec : vectors) {
        distances.push_back(euclidean_distance(query, vec));
    }
    
    return distances;
}

#ifdef CUDA_AVAILABLE
// CUDAVectorOperations implementation
CUDAVectorOperations::CUDAVectorOperations(std::shared_ptr<hardware::IDevice> device)
    : device_(device) {
    // Ensure the device is valid and available
    if (!device_ || !device_->is_available()) {
        throw std::runtime_error("Invalid or unavailable device for CUDA operations");
    }

    // Verify that the device is actually a CUDA device
    if (device_->get_device_type() != hardware::DeviceType::CUDA) {
        throw std::runtime_error("CUDAVectorOperations expects a CUDA device");
    }
}

float CUDAVectorOperations::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for cosine similarity");
    }

    if (a.empty()) {
        return 0.0f;
    }

    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory
    size_t num_elements = a.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_a = device_->allocate(size_bytes);
    void* device_b = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float) * 3); // For dot, norm_a, norm_b

    if (!device_a || !device_b || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().cosine_similarity(a, b);
    }

    // Copy vectors to device
    device_->copy_to_device(device_a, a.data(), size_bytes);
    device_->copy_to_device(device_b, b.data(), size_bytes);

    // In a full implementation, we would run a CUDA kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_a = a;
    std::vector<float> cpu_b = b;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().cosine_similarity(cpu_a, cpu_b);

    // Clean up device memory
    device_->deallocate(device_a);
    device_->deallocate(device_b);
    device_->deallocate(device_result);

    return result;
}

float CUDAVectorOperations::euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for euclidean distance");
    }

    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory
    size_t num_elements = a.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_a = device_->allocate(size_bytes);
    void* device_b = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float));

    if (!device_a || !device_b || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().euclidean_distance(a, b);
    }

    // Copy vectors to device
    device_->copy_to_device(device_a, a.data(), size_bytes);
    device_->copy_to_device(device_b, b.data(), size_bytes);

    // In a full implementation, we would run a CUDA kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_a = a;
    std::vector<float> cpu_b = b;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().euclidean_distance(cpu_a, cpu_b);

    // Clean up device memory
    device_->deallocate(device_a);
    device_->deallocate(device_b);
    device_->deallocate(device_result);

    return result;
}

float CUDAVectorOperations::dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory
    size_t num_elements = a.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_a = device_->allocate(size_bytes);
    void* device_b = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float));

    if (!device_a || !device_b || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().dot_product(a, b);
    }

    // Copy vectors to device
    device_->copy_to_device(device_a, a.data(), size_bytes);
    device_->copy_to_device(device_b, b.data(), size_bytes);

    // In a full implementation, we would run a CUDA kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_a = a;
    std::vector<float> cpu_b = b;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().dot_product(cpu_a, cpu_b);

    // Clean up device memory
    device_->deallocate(device_a);
    device_->deallocate(device_b);
    device_->deallocate(device_result);

    return result;
}

float CUDAVectorOperations::l2_norm(const std::vector<float>& vec) {
    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    size_t num_elements = vec.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_vec = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float));

    if (!device_vec || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().l2_norm(vec);
    }

    // Copy vector to device
    device_->copy_to_device(device_vec, vec.data(), size_bytes);

    // In a full implementation, we would run a CUDA kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_vec = vec;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().l2_norm(cpu_vec);

    // Clean up device memory
    device_->deallocate(device_vec);
    device_->deallocate(device_result);

    return result;
}

std::vector<float> CUDAVectorOperations::normalize(const std::vector<float>& vec) {
    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    size_t num_elements = vec.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_vec = device_->allocate(size_bytes);
    void* device_result = device_->allocate(size_bytes); // For normalized vector

    if (!device_vec || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().normalize(vec);
    }

    // Copy vector to device
    device_->copy_to_device(device_vec, vec.data(), size_bytes);

    // In a full implementation, we would run CUDA kernels here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_vec = vec;

    // Perform calculation using CPU implementation
    auto result = CPUVectorOperations().normalize(cpu_vec);

    // Clean up device memory
    device_->deallocate(device_vec);
    device_->deallocate(device_result);

    return result;
}

std::vector<float> CUDAVectorOperations::batch_cosine_similarity(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& vectors) {

    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory for query
    size_t query_size = query.size();
    size_t query_bytes = query_size * sizeof(float);
    void* device_query = device_->allocate(query_bytes);

    if (!device_query) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().batch_cosine_similarity(query, vectors);
    }

    device_->copy_to_device(device_query, query.data(), query_bytes);

    // In a full implementation, we would transfer all vectors to GPU and run the batch operation
    // For now, copy back to CPU to perform calculation
    std::vector<float> result = CPUVectorOperations().batch_cosine_similarity(query, vectors);

    // Clean up device memory
    device_->deallocate(device_query);

    return result;
}

std::vector<float> CUDAVectorOperations::batch_euclidean_distance(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& vectors) {

    // In a real implementation, we would use cuBLAS or custom CUDA kernels
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory for query
    size_t query_size = query.size();
    size_t query_bytes = query_size * sizeof(float);
    void* device_query = device_->allocate(query_bytes);

    if (!device_query) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().batch_euclidean_distance(query, vectors);
    }

    device_->copy_to_device(device_query, query.data(), query_bytes);

    // In a full implementation, we would transfer all vectors to GPU and run the batch operation
    // For now, copy back to CPU to perform calculation
    std::vector<float> result = CPUVectorOperations().batch_euclidean_distance(query, vectors);

    // Clean up device memory
    device_->deallocate(device_query);

    return result;
}
#endif

#ifdef OPENCL_AVAILABLE
// OpenCLVectorOperations implementation
OpenCLVectorOperations::OpenCLVectorOperations(std::shared_ptr<hardware::IDevice> device)
    : device_(device) {
    // Ensure the device is valid and available
    if (!device_ || !device_->is_available()) {
        throw std::runtime_error("Invalid or unavailable device for OpenCL operations");
    }

    // Verify that the device is actually an OpenCL device
    if (device_->get_device_type() != hardware::DeviceType::OPENCL) {
        throw std::runtime_error("OpenCLVectorOperations expects an OpenCL device");
    }
}

float OpenCLVectorOperations::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for cosine similarity");
    }

    if (a.empty()) {
        return 0.0f;
    }

    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory
    size_t num_elements = a.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_a = device_->allocate(size_bytes);
    void* device_b = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float) * 3); // For dot, norm_a, norm_b

    if (!device_a || !device_b || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().cosine_similarity(a, b);
    }

    // Copy vectors to device
    device_->copy_to_device(device_a, a.data(), size_bytes);
    device_->copy_to_device(device_b, b.data(), size_bytes);

    // In a full implementation, we would create and run an OpenCL kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_a = a;
    std::vector<float> cpu_b = b;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().cosine_similarity(cpu_a, cpu_b);

    // Clean up device memory
    device_->deallocate(device_a);
    device_->deallocate(device_b);
    device_->deallocate(device_result);

    return result;
}

float OpenCLVectorOperations::euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for euclidean distance");
    }

    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory
    size_t num_elements = a.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_a = device_->allocate(size_bytes);
    void* device_b = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float));

    if (!device_a || !device_b || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().euclidean_distance(a, b);
    }

    // Copy vectors to device
    device_->copy_to_device(device_a, a.data(), size_bytes);
    device_->copy_to_device(device_b, b.data(), size_bytes);

    // In a full implementation, we would create and run an OpenCL kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_a = a;
    std::vector<float> cpu_b = b;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().euclidean_distance(cpu_a, cpu_b);

    // Clean up device memory
    device_->deallocate(device_a);
    device_->deallocate(device_b);
    device_->deallocate(device_result);

    return result;
}

float OpenCLVectorOperations::dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory
    size_t num_elements = a.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_a = device_->allocate(size_bytes);
    void* device_b = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float));

    if (!device_a || !device_b || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().dot_product(a, b);
    }

    // Copy vectors to device
    device_->copy_to_device(device_a, a.data(), size_bytes);
    device_->copy_to_device(device_b, b.data(), size_bytes);

    // In a full implementation, we would create and run an OpenCL kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_a = a;
    std::vector<float> cpu_b = b;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().dot_product(cpu_a, cpu_b);

    // Clean up device memory
    device_->deallocate(device_a);
    device_->deallocate(device_b);
    device_->deallocate(device_result);

    return result;
}

float OpenCLVectorOperations::l2_norm(const std::vector<float>& vec) {
    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    size_t num_elements = vec.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_vec = device_->allocate(size_bytes);
    void* device_result = device_->allocate(sizeof(float));

    if (!device_vec || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().l2_norm(vec);
    }

    // Copy vector to device
    device_->copy_to_device(device_vec, vec.data(), size_bytes);

    // In a full implementation, we would create and run an OpenCL kernel here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_vec = vec;

    // Perform calculation using CPU implementation
    float result = CPUVectorOperations().l2_norm(cpu_vec);

    // Clean up device memory
    device_->deallocate(device_vec);
    device_->deallocate(device_result);

    return result;
}

std::vector<float> OpenCLVectorOperations::normalize(const std::vector<float>& vec) {
    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    size_t num_elements = vec.size();
    size_t size_bytes = num_elements * sizeof(float);

    void* device_vec = device_->allocate(size_bytes);
    void* device_result = device_->allocate(size_bytes); // For normalized vector

    if (!device_vec || !device_result) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().normalize(vec);
    }

    // Copy vector to device
    device_->copy_to_device(device_vec, vec.data(), size_bytes);

    // In a full implementation, we would create and run OpenCL kernels here
    // For now, copy back to CPU to perform calculation
    std::vector<float> cpu_vec = vec;

    // Perform calculation using CPU implementation
    auto result = CPUVectorOperations().normalize(cpu_vec);

    // Clean up device memory
    device_->deallocate(device_vec);
    device_->deallocate(device_result);

    return result;
}

std::vector<float> OpenCLVectorOperations::batch_cosine_similarity(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& vectors) {

    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory for query
    size_t query_size = query.size();
    size_t query_bytes = query_size * sizeof(float);
    void* device_query = device_->allocate(query_bytes);

    if (!device_query) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().batch_cosine_similarity(query, vectors);
    }

    device_->copy_to_device(device_query, query.data(), query_bytes);

    // In a full implementation, we would transfer all vectors to GPU and run the batch operation
    // For now, copy back to CPU to perform calculation
    std::vector<float> result = CPUVectorOperations().batch_cosine_similarity(query, vectors);

    // Clean up device memory
    device_->deallocate(device_query);

    return result;
}

std::vector<float> OpenCLVectorOperations::batch_euclidean_distance(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& vectors) {

    // In a real implementation, we would use OpenCL kernels for computation
    // For now, we'll use the GPU device's memory management to perform the operation

    // Allocate device memory for query
    size_t query_size = query.size();
    size_t query_bytes = query_size * sizeof(float);
    void* device_query = device_->allocate(query_bytes);

    if (!device_query) {
        // Fall back to CPU if GPU allocation fails
        return CPUVectorOperations().batch_euclidean_distance(query, vectors);
    }

    device_->copy_to_device(device_query, query.data(), query_bytes);

    // In a full implementation, we would transfer all vectors to GPU and run the batch operation
    // For now, copy back to CPU to perform calculation
    std::vector<float> result = CPUVectorOperations().batch_euclidean_distance(query, vectors);

    // Clean up device memory
    device_->deallocate(device_query);

    return result;
}
#endif

// VectorOperationsFactory implementation
std::shared_ptr<IVectorOperations> VectorOperationsFactory::create_operations(
    hardware::DeviceType preferred_type) {

    // Try to get the device manager to check what's available
    auto& device_manager = hardware::DeviceManager::get_instance();

    // Initialize with the preferred type
    device_manager.initialize(preferred_type);

    auto active_device = device_manager.get_active_device();

    // Depending on the active device, return the appropriate implementation
    if (active_device->get_device_type() == hardware::DeviceType::CUDA) {
#ifdef CUDA_AVAILABLE
        return std::make_shared<CUDAVectorOperations>(active_device);
#else
        // If CUDA is preferred but not available/compiled, fall back to CPU
        return std::make_shared<CPUVectorOperations>();
#endif
    }
#ifdef OPENCL_AVAILABLE
    else if (active_device->get_device_type() == hardware::DeviceType::OPENCL) {
        return std::make_shared<OpenCLVectorOperations>(active_device);
    }
#endif
    else {
        // Default to CPU implementation
        return std::make_shared<CPUVectorOperations>();
    }
}

} // namespace vector_ops
} // namespace jadevectordb