#ifndef JADEVECTORDB_DATABASE_H
#define JADEVECTORDB_DATABASE_H

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace jadevectordb {

struct Database {
    std::string databaseId;
    std::string name;
    std::string description;
    std::string owner;  // Add owner field
    int vectorDimension;
    std::string indexType;  // HNSW, IVF, LSH, etc.
    std::map<std::string, std::string> indexParameters;
    
    struct ShardingConfig {
        std::string strategy;  // hash, range, vector-based
        int numShards;
    } sharding;
    
    struct ReplicationConfig {
        int factor;
        bool sync;
    } replication;
    
    struct EmbeddingModel {
        std::string name;
        std::string version;
        std::string provider;
        std::string inputType;
        int outputDimension;
        std::map<std::string, std::string> parameters;
        std::string status;  // active, inactive, failed
    };
    std::vector<EmbeddingModel> embeddingModels;
    
    // Schema definition for metadata
    std::map<std::string, std::string> metadataSchema;
    
    struct RetentionPolicy {
        int maxAgeDays;
        bool archiveOnExpire;
    };
    std::unique_ptr<RetentionPolicy> retentionPolicy;
    
    struct AccessControl {
        std::vector<std::string> roles;
        std::vector<std::string> defaultPermissions;
    };
    AccessControl accessControl;
    
    std::string created_at;
    std::string updated_at;
    
    // Constructors
    Database() : vectorDimension(0), indexType("HNSW"), 
                 sharding({"hash", 1}), replication({1, true}) {}
    
    // Copy constructor
    Database(const Database& other) 
        : databaseId(other.databaseId)
        , name(other.name)
        , description(other.description)
        , owner(other.owner)
        , vectorDimension(other.vectorDimension)
        , indexType(other.indexType)
        , indexParameters(other.indexParameters)
        , sharding(other.sharding)
        , replication(other.replication)
        , embeddingModels(other.embeddingModels)
        , metadataSchema(other.metadataSchema)
        , accessControl(other.accessControl)
        , created_at(other.created_at)
        , updated_at(other.updated_at)
    {
        if (other.retentionPolicy) {
            retentionPolicy = std::make_unique<RetentionPolicy>(*other.retentionPolicy);
        }
    }
    
    // Copy assignment operator
    Database& operator=(const Database& other) {
        if (this != &other) {
            databaseId = other.databaseId;
            name = other.name;
            description = other.description;
            owner = other.owner;
            vectorDimension = other.vectorDimension;
            indexType = other.indexType;
            indexParameters = other.indexParameters;
            sharding = other.sharding;
            replication = other.replication;
            embeddingModels = other.embeddingModels;
            metadataSchema = other.metadataSchema;
            accessControl = other.accessControl;
            created_at = other.created_at;
            updated_at = other.updated_at;
            
            if (other.retentionPolicy) {
                retentionPolicy = std::make_unique<RetentionPolicy>(*other.retentionPolicy);
            } else {
                retentionPolicy.reset();
            }
        }
        return *this;
    }
    
    // Move constructor
    Database(Database&&) = default;
    
    // Move assignment operator
    Database& operator=(Database&&) = default;
    
    // Methods for validation
    bool validate() const {
        return !databaseId.empty() && 
               !name.empty() && 
               vectorDimension > 0;
    }
    
    void addEmbeddingModel(const EmbeddingModel& model) {
        embeddingModels.push_back(model);
    }
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DATABASE_H