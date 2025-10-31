#ifndef JADEVECTORDB_INDEX_SERVICE_H
#define JADEVECTORDB_INDEX_SERVICE_H

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "lib/error_handling.h"
#include "models/index.h"
#include "models/database.h"

namespace jadevectordb {

// Index management service
class IndexService {
private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Storage for index configurations
    std::map<std::string, std::map<std::string, Index>> database_indexes_; // database_id -> (index_id -> index)
    std::mutex index_mutex_;

public:
    IndexService();
    ~IndexService() = default;
    
    // Create a new index for a database
    Result<std::string> create_index(const Database& database, const Index& index_config);
    
    // Get index by ID
    Result<Index> get_index(const std::string& database_id, const std::string& index_id);
    
    // List all indexes for a database
    Result<std::vector<Index>> list_indexes(const std::string& database_id);
    
    // Update index configuration
    Result<void> update_index(const std::string& database_id, const std::string& index_id, const Index& new_config);
    
    // Delete an index
    Result<void> delete_index(const std::string& database_id, const std::string& index_id);
    
    // Build/rebuild an index
    Result<void> build_index(const std::string& database_id, const std::string& index_id);
    
    // Check if index is ready for queries
    Result<bool> is_index_ready(const std::string& database_id, const std::string& index_id);
    
    // Get index statistics
    Result<std::map<std::string, double>> get_index_stats(const std::string& database_id, const std::string& index_id);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_INDEX_SERVICE_H