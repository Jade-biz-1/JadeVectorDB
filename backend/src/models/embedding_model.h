#ifndef JADEVECTORDB_EMBEDDING_MODEL_H
#define JADEVECTORDB_EMBEDDING_MODEL_H

#include <string>
#include <map>

namespace jadevectordb {

struct EmbeddingModel {
    std::string modelId;
    std::string name;  // BERT, ResNet, etc.
    std::string version;
    std::string provider;  // huggingface, torchvision, etc.
    std::string inputType;  // text, image, etc.
    int outputDimension;
    std::map<std::string, std::string> parameters;
    std::string status;  // active, inactive, failed
    
    // Constructors
    EmbeddingModel() : outputDimension(0), status("active") {}
    
    // Methods for validation
    bool validate() const {
        return !modelId.empty() && 
               !name.empty() && 
               !provider.empty() && 
               outputDimension > 0 &&
               (status == "active" || status == "inactive" || status == "failed");
    }
    
    // Set model parameters
    void setParameter(const std::string& key, const std::string& value) {
        parameters[key] = value;
    }
    
    // Get model parameter
    std::string getParameter(const std::string& key) const {
        auto it = parameters.find(key);
        return it != parameters.end() ? it->second : "";
    }
};

} // namespace jadevectordb

#endif // JADEVECTORDB_EMBEDDING_MODEL_H