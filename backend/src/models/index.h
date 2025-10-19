#ifndef JADEVECTORDB_INDEX_H
#define JADEVECTORDB_INDEX_H

#include <string>
#include <map>
#include <memory>

namespace jadevectordb {

struct Index {
    std::string indexId;
    std::string databaseId;
    std::string type;  // HNSW, IVF, LSH, Flat, PQ, OPQ, SQ, Composite Index
    std::map<std::string, std::string> parameters;  // Index-specific parameters
    std::string status;  // building, ready, failed
    std::string created_at;
    std::string updated_at;
    
    // Index statistics
    struct Stats {
        int vectorCount;
        long long sizeBytes;
        int buildTimeMs;
    };
    std::unique_ptr<Stats> stats;
    
    // Constructors
    Index() : type("HNSW"), status("building") {}
    
    // Methods for validation
    bool validate() const {
        return !indexId.empty() && 
               !databaseId.empty() &&
               !type.empty() &&
               (status == "building" || status == "ready" || status == "failed");
    }
    
    void updateStatus(const std::string& newStatus) {
        status = newStatus;
    }
};

} // namespace jadevectordb

#endif // JADEVECTORDB_INDEX_H