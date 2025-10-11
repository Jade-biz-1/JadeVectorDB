#ifndef JADEVECTORDB_VECTOR_H
#define JADEVECTORDB_VECTOR_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>

namespace jadevectordb {

struct Vector {
    std::string id;
    std::vector<float> values;
    struct Metadata {
        std::string source;
        std::string created_at;
        std::string updated_at;
        std::vector<std::string> tags;
        std::string owner;
        std::vector<std::string> permissions;
        std::string category;
        float score;
        std::string status;  // active, archived, deleted
        std::map<std::string, nlohmann::json> custom;
    } metadata;
    
    struct IndexInfo {
        std::string type;  // HNSW, IVF, LSH, etc.
        std::string version;
        std::map<std::string, nlohmann::json> parameters;
    } index;
    
    struct EmbeddingModelInfo {
        std::string name;
        std::string version;
        std::string provider;
        std::string input_type;  // text, image, etc.
    } embedding_model;
    
    std::string shard;  // Shard identifier for distributed storage
    std::vector<std::string> replicas;  // List of node identifiers where replicas are stored
    int version;
    bool deleted;
    
    // Constructors
    Vector() : version(1), deleted(false) {}
    
    // Methods for validation
    bool validate() const {
        return !id.empty() && 
               !values.empty() &&
               (metadata.status == "active" || 
                metadata.status == "archived" || 
                metadata.status == "deleted");
    }
    
    size_t getDimension() const {
        return values.size();
    }
};

// Utility functions
inline bool operator==(const Vector& lhs, const Vector& rhs) {
    return lhs.id == rhs.id && 
           lhs.values == rhs.values && 
           lhs.metadata.status == rhs.metadata.status;
}

} // namespace jadevectordb

#endif // JADEVECTORDB_VECTOR_H