#ifndef JADEVECTORDB_HF_EMBEDDING_PROVIDER_H
#define JADEVECTORDB_HF_EMBEDDING_PROVIDER_H

#include "services/embedding_provider.h"
#include <string>
#include <vector>

namespace jadevectordb {

/**
 * @brief Implementation of IEmbeddingProvider for Hugging Face models
 * 
 * This class provides embedding generation using models from Hugging Face Hub.
 * It handles downloading, caching, and running Hugging Face models locally.
 */
class HuggingFaceEmbeddingProvider : public IEmbeddingProvider {
public:
    struct Config {
        std::string model_name;           // e.g., "sentence-transformers/all-MiniLM-L6-v2"
        std::string cache_dir;            // Directory to cache downloaded models
        std::string token;                // Hugging Face token (if required for private models)
        int max_tokens;                   // Maximum input tokens for the model
        bool normalize_embeddings;        // Whether to normalize output embeddings
        int pooling_strategy;             // How to pool token embeddings (0=mean, 1=cls)
        std::string device;               // Device to run the model on ("cpu", "cuda", etc.)
    };

private:
    Config config_;
    EmbeddingModel model_info_;
    bool initialized_;
    
public:
    explicit HuggingFaceEmbeddingProvider(const Config& config);
    virtual ~HuggingFaceEmbeddingProvider() = default;

    // IEmbeddingProvider interface implementation
    Result<std::vector<float>> generate_text_embedding(const std::string& text) override;
    Result<std::vector<float>> generate_image_embedding(const std::string& image_path) override;
    bool is_ready() const override;
    EmbeddingModel get_model_info() const override;
    bool validate_config() const override;

private:
    Result<std::vector<float>> preprocess_and_tokenize(const std::string& text);
    Result<std::vector<float>> run_inference(const std::vector<int>& input_ids, 
                                           const std::vector<int>& attention_mask);
    Result<std::vector<float>> download_and_load_model();
    void initialize_model_info();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_HF_EMBEDDING_PROVIDER_H