#ifndef JADEVECTORDB_EMBEDDING_PROVIDER_H
#define JADEVECTORDB_EMBEDDING_PROVIDER_H

#include <string>
#include <vector>
#include <memory>
#include "models/embedding_model.h"

namespace jadevectordb {

// Forward declaration of Result type
template<typename T>
struct Result {
    bool success;
    T value;
    std::string error;
    
    Result(T val) : success(true), value(val), error("") {}
    Result(bool success, T val, const std::string& err) : success(success), value(val), error(err) {}
    Result(const std::string& err) : success(false), error(err) {}
};

/**
 * @brief Interface for embedding providers
 * 
 * This interface defines the contract that all embedding providers must implement
 * to generate vector embeddings from input data.
 */
class IEmbeddingProvider {
public:
    virtual ~IEmbeddingProvider() = default;

    /**
     * @brief Generate embeddings from text input
     * @param text The input text to convert to embeddings
     * @return A vector of float values representing the embedding
     */
    virtual Result<std::vector<float>> generate_text_embedding(const std::string& text) = 0;

    /**
     * @brief Generate embeddings from image input
     * @param image_path Path to the image file
     * @return A vector of float values representing the embedding
     */
    virtual Result<std::vector<float>> generate_image_embedding(const std::string& image_path) = 0;

    /**
     * @brief Check if the provider is available and ready to generate embeddings
     * @return True if the provider is ready, false otherwise
     */
    virtual bool is_ready() const = 0;

    /**
     * @brief Get information about the embedding model used
     * @return EmbeddingModel struct with model information
     */
    virtual EmbeddingModel get_model_info() const = 0;

    /**
     * @brief Validate the configuration for this provider
     * @return True if configuration is valid, false otherwise
     */
    virtual bool validate_config() const = 0;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_EMBEDDING_PROVIDER_H