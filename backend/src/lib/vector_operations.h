#ifndef JADEVECTORDB_VECTOR_OPERATIONS_H
#define JADEVECTORDB_VECTOR_OPERATIONS_H

#include "gpu_acceleration.h"
#include <vector>
#include <memory>

namespace jadevectordb {
namespace vector_ops {

    /**
     * @brief Interface for vector operations with hardware acceleration support
     * 
     * This class provides an interface for vector operations that can be
     * accelerated using different hardware (CPU, GPU).
     */
    class IVectorOperations {
    public:
        virtual ~IVectorOperations() = default;
        
        /**
         * @brief Compute cosine similarity between two vectors
         * @param a First vector
         * @param b Second vector
         * @return Cosine similarity value
         */
        virtual float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) = 0;
        
        /**
         * @brief Compute Euclidean distance between two vectors
         * @param a First vector
         * @param b Second vector
         * @return Euclidean distance value
         */
        virtual float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) = 0;
        
        /**
         * @brief Compute dot product of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Dot product value
         */
        virtual float dot_product(const std::vector<float>& a, const std::vector<float>& b) = 0;
        
        /**
         * @brief Compute L2 norm of a vector
         * @param vec Input vector
         * @return L2 norm value
         */
        virtual float l2_norm(const std::vector<float>& vec) = 0;
        
        /**
         * @brief Normalize a vector to unit length
         * @param vec Input vector to normalize
         * @return Normalized vector
         */
        virtual std::vector<float> normalize(const std::vector<float>& vec) = 0;
        
        /**
         * @brief Compute batch cosine similarities between a query vector and multiple vectors
         * @param query Query vector
         * @param vectors List of vectors to compare against
         * @return Vector of similarity scores
         */
        virtual std::vector<float> batch_cosine_similarity(
            const std::vector<float>& query, 
            const std::vector<std::vector<float>>& vectors) = 0;
        
        /**
         * @brief Compute batch Euclidean distances between a query vector and multiple vectors
         * @param query Query vector
         * @param vectors List of vectors to compare against
         * @return Vector of distance values
         */
        virtual std::vector<float> batch_euclidean_distance(
            const std::vector<float>& query, 
            const std::vector<std::vector<float>>& vectors) = 0;
    };
    
    /**
     * @brief CPU implementation of vector operations
     * 
     * This implementation performs all operations on the CPU using standard algorithms.
     */
    class CPUVectorOperations : public IVectorOperations {
    public:
        float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) override;
        
        float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) override;
        
        float dot_product(const std::vector<float>& a, const std::vector<float>& b) override;
        
        float l2_norm(const std::vector<float>& vec) override;
        
        std::vector<float> normalize(const std::vector<float>& vec) override;
        
        std::vector<float> batch_cosine_similarity(
            const std::vector<float>& query, 
            const std::vector<std::vector<float>>& vectors) override;
        
        std::vector<float> batch_euclidean_distance(
            const std::vector<float>& query, 
            const std::vector<std::vector<float>>& vectors) override;
    };
    
#ifdef CUDA_AVAILABLE
    /**
     * @brief GPU implementation of vector operations using CUDA
     * 
     * This implementation offloads operations to the GPU using CUDA kernels.
     */
    class CUDAVectorOperations : public IVectorOperations {
    private:
        std::shared_ptr<hardware::IDevice> device_;
        
    public:
        explicit CUDAVectorOperations(std::shared_ptr<hardware::IDevice> device);
        
        float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) override;
        
        float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) override;
        
        float dot_product(const std::vector<float>& a, const std::vector<float>& b) override;
        
        float l2_norm(const std::vector<float>& vec) override;
        
        std::vector<float> normalize(const std::vector<float>& vec) override;
        
        std::vector<float> batch_cosine_similarity(
            const std::vector<float>& query, 
            const std::vector<std::vector<float>>& vectors) override;
        
        std::vector<float> batch_euclidean_distance(
            const std::vector<float>& query, 
            const std::vector<std::vector<float>>& vectors) override;
    };
#endif

#ifdef OPENCL_AVAILABLE
    /**
     * @brief GPU implementation of vector operations using OpenCL
     *
     * This implementation offloads operations to the GPU using OpenCL kernels.
     */
    class OpenCLVectorOperations : public IVectorOperations {
    private:
        std::shared_ptr<hardware::IDevice> device_;

    public:
        explicit OpenCLVectorOperations(std::shared_ptr<hardware::IDevice> device);

        float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) override;

        float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) override;

        float dot_product(const std::vector<float>& a, const std::vector<float>& b) override;

        float l2_norm(const std::vector<float>& vec) override;

        std::vector<float> normalize(const std::vector<float>& vec) override;

        std::vector<float> batch_cosine_similarity(
            const std::vector<float>& query,
            const std::vector<std::vector<float>>& vectors) override;

        std::vector<float> batch_euclidean_distance(
            const std::vector<float>& query,
            const std::vector<std::vector<float>>& vectors) override;
    };
#endif

    /**
     * @brief Factory for creating vector operation implementations
     *
     * This class provides a way to get the best available vector operations
     * implementation based on available hardware.
     */
    class VectorOperationsFactory {
    public:
        /**
         * @brief Get the best available vector operations implementation
         * @param preferred_type Preferred hardware type (will fall back if not available)
         * @return Shared pointer to IVectorOperations implementation
         */
        static std::shared_ptr<IVectorOperations> create_operations(
            hardware::DeviceType preferred_type = hardware::DeviceType::CPU);
    };
    
} // namespace vector_ops
} // namespace jadevectordb

#endif // JADEVECTORDB_VECTOR_OPERATIONS_H