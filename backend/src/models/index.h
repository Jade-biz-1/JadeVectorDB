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

    // Copy constructor
    Index(const Index& other)
        : indexId(other.indexId)
        , databaseId(other.databaseId)
        , type(other.type)
        , parameters(other.parameters)
        , status(other.status)
        , created_at(other.created_at)
        , updated_at(other.updated_at)
    {
        if (other.stats) {
            stats = std::make_unique<Stats>(*other.stats);
        }
    }

    // Copy assignment operator
    Index& operator=(const Index& other) {
        if (this != &other) {
            indexId = other.indexId;
            databaseId = other.databaseId;
            type = other.type;
            parameters = other.parameters;
            status = other.status;
            created_at = other.created_at;
            updated_at = other.updated_at;

            if (other.stats) {
                stats = std::make_unique<Stats>(*other.stats);
            } else {
                stats.reset();
            }
        }
        return *this;
    }

    // Move constructor
    Index(Index&&) = default;

    // Move assignment operator
    Index& operator=(Index&&) = default;

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