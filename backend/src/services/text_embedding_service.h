#ifndef JADEVECTORDB_TEXT_EMBEDDING_SERVICE_H
#define JADEVECTORDB_TEXT_EMBEDDING_SERVICE_H

#include "models/embedding_model.h"
#include <string>
#include <vector>
#include <memory>

namespace jadevectordb {

/**
 * @brief Service to generate embeddings from raw text using internal models
 * 
 * This service handles the specific use case of generating embeddings from text input
 * using the various embedding providers registered in the system.
 */
class TextEmbeddingService {
public:
    struct TextConfig {
        std::string model_name;          // Name of the embedding model to use
        int max_tokens;                  // Maximum number of tokens to process
        bool normalize_result;           // Whether to normalize the output embedding
        std::vector<std::string> prefixes; // Optional prefixes to add to the text
        std::string truncation_strategy; // Strategy for handling long texts (e.g., "start", "end", "middle")
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
public:
    explicit TextEmbeddingService();
    ~TextEmbeddingService() = default;
    
    // Generate embedding from a single text input
    Result<std::vector<float>> generate_embedding(const std::string& text, 
                                                const TextConfig& config);
    
    // Generate embeddings from multiple text inputs (batch processing)
    Result<std::vector<std::vector<float>>> generate_batch_embeddings(
        const std::vector<std::string>& texts, 
        const TextConfig& config);
    
    // Generate embeddings from a text with specific model
    Result<std::vector<float>> generate_embedding_with_model(const std::string& text, 
                                                           const std::string& model_id,
                                                           const TextConfig& config);
    
    // Validate text input before processing
    Result<bool> validate_text_input(const std::string& text, const TextConfig& config) const;
    
    // Get the dimension of embeddings produced by a specific model
    Result<int> get_embedding_dimension(const std::string& model_id) const;
    
private:
    // Preprocess text input (tokenization, cleaning, etc.)
    Result<std::string> preprocess_text(const std::string& text, const TextConfig& config) const;
    
    // Truncate text according to the specified strategy if it exceeds max tokens
    std::string truncate_text(const std::string& text, 
                            const TextConfig& config, 
                            int max_chars) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_TEXT_EMBEDDING_SERVICE_H