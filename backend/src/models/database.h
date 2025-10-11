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