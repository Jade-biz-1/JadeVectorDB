#ifndef JADEVECTORDB_EMBEDDING_SERVICE_H
#define JADEVECTORDB_EMBEDDING_SERVICE_H

#include "services/embedding_provider.h"
#include "models/embedding_model.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace jadevectordb {

/**
 * @brief Service to manage all embedding providers and coordinate embedding generation
 * 
 * This service acts as the main interface for embedding generation, handling
 * provider selection, caching, and integration with other services.
 */
class EmbeddingService {
public:
    struct GenerationConfig {
        std::string provider_id;         // ID of the provider to use
        std::string model_name;          // Name of the model to use
        std::vector<std::string> input;  // Multiple inputs if needed for batch processing
        bool normalize_output;           // Whether to normalize the output embeddings
        float dimensions_scale_factor;   // Factor to adjust dimensions if needed
    };

private:
    std::unordered_map<std::string, std::unique_ptr<IEmbeddingProvider>> providers_;
    std::shared_ptr<logging::Logger> logger_;
    
public:
    EmbeddingService();
    ~EmbeddingService() = default;
    
    // Register an embedding provider with the service
    void register_provider(const std::string& provider_id, 
                          std::unique_ptr<IEmbeddingProvider> provider);
    
    // Generate embedding from text input
    Result<std::vector<float>> generate_text_embedding(const std::string& text, 
                                                     const GenerationConfig& config);
    
    // Generate embedding from image input  
    Result<std::vector<float>> generate_image_embedding(const std::string& image_path, 
                                                      const GenerationConfig& config);
    
    // Get available providers
    std::vector<std::string> get_available_providers() const;
    
    // Check if a provider is ready
    bool is_provider_ready(const std::string& provider_id) const;
    
    // Get model information for a provider
    Result<EmbeddingModel> get_provider_model_info(const std::string& provider_id) const;
    
    // Validate provider configuration
    bool validate_provider_config(const std::string& provider_id) const;

    // Batch generation of embeddings from multiple texts
    Result<std::vector<std::vector<float>>> generate_batch_text_embeddings(
        const std::vector<std::string>& texts, const GenerationConfig& config);

    // Integration method for direct storage after embedding generation
    Result<std::string> generate_and_store_embedding(const std::string& input, 
                                                   const std::string& input_type,
                                                   const GenerationConfig& config,
                                                   const std::string& database_id);
    
private:
    IEmbeddingProvider* get_provider(const std::string& provider_id) const;
    Result<std::vector<float>> post_process_embedding(const std::vector<float>& embedding, 
                                                    const GenerationConfig& config);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_EMBEDDING_SERVICE_H