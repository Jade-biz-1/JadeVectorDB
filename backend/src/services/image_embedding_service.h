#ifndef JADEVECTORDB_IMAGE_EMBEDDING_SERVICE_H
#define JADEVECTORDB_IMAGE_EMBEDDING_SERVICE_H

#include "models/embedding_model.h"
#include <string>
#include <vector>
#include <memory>

namespace jadevectordb {

/**
 * @brief Service to generate embeddings from images using internal models
 * 
 * This service handles the specific use case of generating embeddings from image input
 * using the various embedding providers registered in the system.
 */
class ImageEmbeddingService {
public:
    struct ImageConfig {
        std::string model_name;          // Name of the embedding model to use
        int target_width;                // Target width for image resizing
        int target_height;               // Target height for image resizing
        bool normalize_result;           // Whether to normalize the output embedding
        std::string input_format;        // Expected input format (e.g., "RGB", "BGR")
        std::string preprocessing_method; // How to preprocess the image (resize, crop, etc.)
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
public:
    explicit ImageEmbeddingService();
    ~ImageEmbeddingService() = default;
    
    // Generate embedding from an image file
    Result<std::vector<float>> generate_embedding(const std::string& image_path, 
                                                const ImageConfig& config);
    
    // Generate embedding from image data in memory
    Result<std::vector<float>> generate_embedding_from_data(const std::vector<uint8_t>& image_data, 
                                                          const ImageConfig& config);
    
    // Generate embeddings from multiple image files (batch processing)
    Result<std::vector<std::vector<float>>> generate_batch_embeddings(
        const std::vector<std::string>& image_paths, 
        const ImageConfig& config);
    
    // Generate embeddings from image with specific model
    Result<std::vector<float>> generate_embedding_with_model(const std::string& image_path, 
                                                           const std::string& model_id,
                                                           const ImageConfig& config);
    
    // Validate image input before processing
    Result<bool> validate_image_input(const std::string& image_path, const ImageConfig& config) const;
    
    // Get the dimension of embeddings produced by a specific model
    Result<int> get_embedding_dimension(const std::string& model_id) const;
    
private:
    // Preprocess image input (resize, normalize, etc.)
    Result<std::vector<float>> preprocess_image(const std::string& image_path, 
                                              const ImageConfig& config) const;
    
    // Load and decode an image from file
    Result<std::vector<uint8_t>> load_image_data(const std::string& image_path) const;
    
    // Resize image to target dimensions
    Result<std::vector<uint8_t>> resize_image(const std::vector<uint8_t>& image_data,
                                            int original_width, int original_height,
                                            int target_width, int target_height) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_IMAGE_EMBEDDING_SERVICE_H