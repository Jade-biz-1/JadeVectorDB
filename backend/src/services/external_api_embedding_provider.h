#ifndef JADEVECTORDB_EXTERNAL_API_EMBEDDING_PROVIDER_H
#define JADEVECTORDB_EXTERNAL_API_EMBEDDING_PROVIDER_H

#include "services/embedding_provider.h"
#include <string>
#include <vector>

namespace jadevectordb {

/**
 * @brief Implementation of IEmbeddingProvider for external APIs
 * 
 * This class provides embedding generation by connecting to external services
 * like OpenAI, Google, Cohere, etc.
 */
class ExternalAPIEmbeddingProvider : public IEmbeddingProvider {
public:
    enum class ProviderType {
        OPENAI,
        GOOGLE,
        COHERE,
        AZURE_OPENAI,
        HUGGING_FACE_API
    };

    struct Config {
        ProviderType provider_type;
        std::string api_endpoint;        // API endpoint URL
        std::string api_key;             // API key for authentication
        std::string model_name;          // Name of the model to use
        int timeout_seconds;             // Request timeout
        std::string organization;        // Organization ID (for some providers)
        std::string project;             // Project ID (for some providers)
    };

private:
    Config config_;
    EmbeddingModel model_info_;
    bool initialized_;
    
public:
    explicit ExternalAPIEmbeddingProvider(const Config& config);
    virtual ~ExternalAPIEmbeddingProvider() = default;

    // IEmbeddingProvider interface implementation
    Result<std::vector<float>> generate_text_embedding(const std::string& text) override;
    Result<std::vector<float>> generate_image_embedding(const std::string& image_path) override;
    bool is_ready() const override;
    EmbeddingModel get_model_info() const override;
    bool validate_config() const override;

private:
    Result<std::vector<float>> make_api_request(const std::string& input, const std::string& input_type);
    std::string get_provider_specific_payload(const std::string& input, const std::string& input_type);
    std::string get_provider_specific_headers();
    Result<bool> check_api_availability();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_EXTERNAL_API_EMBEDDING_PROVIDER_H