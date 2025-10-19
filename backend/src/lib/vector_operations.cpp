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
}

float CUDAVectorOperations::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for cosine similarity");
    }
    
    if (a.empty()) {
        return 0.0f;
    }
    
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.cosine_similarity(a, b);
}

float CUDAVectorOperations::euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for euclidean distance");
    }
    
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.euclidean_distance(a, b);
}

float CUDAVectorOperations::dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }
    
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.dot_product(a, b);
}

float CUDAVectorOperations::l2_norm(const std::vector<float>& vec) {
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.l2_norm(vec);
}

std::vector<float> CUDAVectorOperations::normalize(const std::vector<float>& vec) {
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.normalize(vec);
}

std::vector<float> CUDAVectorOperations::batch_cosine_similarity(
    const std::vector<float>& query, 
    const std::vector<std::vector<float>>& vectors) {
    
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.batch_cosine_similarity(query, vectors);
}

std::vector<float> CUDAVectorOperations::batch_euclidean_distance(
    const std::vector<float>& query, 
    const std::vector<std::vector<float>>& vectors) {
    
    // For a real implementation, we'd use cuBLAS or custom kernels
    // For now, fallback to CPU implementation when CUDA isn't fully implemented
    CPUVectorOperations cpu_ops;
    return cpu_ops.batch_euclidean_distance(query, vectors);
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
    } else {
        // Default to CPU implementation
        return std::make_shared<CPUVectorOperations>();
    }
}

} // namespace vector_ops
} // namespace jadevectordb