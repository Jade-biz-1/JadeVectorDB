#ifndef JADEVECTORDB_LOCAL_API_EMBEDDING_PROVIDER_H
#define JADEVECTORDB_LOCAL_API_EMBEDDING_PROVIDER_H

#include "services/embedding_provider.h"
#include <string>
#include <vector>
#include <memory>

namespace jadevectordb {

/**
 * @brief Implementation of IEmbeddingProvider for local API models
 * 
 * This class provides embedding generation by connecting to local inference servers
 * like Ollama or other local embedding APIs.
 */
class LocalAPIEmbeddingProvider : public IEmbeddingProvider {
public:
    struct Config {
        std::string api_endpoint;         // e.g., "http://localhost:11434/api/embeddings"
        std::string model_name;          // Name of the model to use
        std::string api_key;             // API key if required
        int timeout_seconds;             // Request timeout
        std::string provider_name;       // Name of the local provider (ollama, vllm, etc.)
    };

private:
    Config config_;
    EmbeddingModel model_info_;
    bool initialized_;
    
public:
    explicit LocalAPIEmbeddingProvider(const Config& config);
    virtual ~LocalAPIEmbeddingProvider() = default;

    // IEmbeddingProvider interface implementation
    Result<std::vector<float>> generate_text_embedding(const std::string& text) override;
    Result<std::vector<float>> generate_image_embedding(const std::string& image_path) override;
    bool is_ready() const override;
    EmbeddingModel get_model_info() const override;
    bool validate_config() const override;

private:
    Result<std::vector<float>> make_api_request(const std::string& input, const std::string& input_type);
    Result<std::string> get_provider_specific_payload(const std::string& input, const std::string& input_type);
    Result<bool> check_api_availability();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_LOCAL_API_EMBEDDING_PROVIDER_H