#ifndef JADEVECTORDB_VECTOR_STORAGE_SERVICE_H
#define JADEVECTORDB_VECTOR_STORAGE_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <shared_mutex>

#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/database_layer.h"

namespace jadevectordb {

class VectorStorageService {
private:
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::shared_ptr<logging::Logger> logger_;

public:
    explicit VectorStorageService(std::unique_ptr<DatabaseLayer> db_layer = nullptr);
    ~VectorStorageService() = default;

    // Initialize the service
    Result<void> initialize();

    // Store a single vector
    Result<void> store_vector(const std::string& database_id, const Vector& vector);

    // Store multiple vectors in a batch
    Result<void> batch_store_vectors(const std::string& database_id, 
                                   const std::vector<Vector>& vectors);

    // Retrieve a single vector by ID
    Result<Vector> retrieve_vector(const std::string& database_id, const std::string& vector_id) const;

    // Retrieve multiple vectors by ID
    Result<std::vector<Vector>> retrieve_vectors(const std::string& database_id, 
                                               const std::vector<std::string>& vector_ids) const;

    // Update an existing vector
    Result<void> update_vector(const std::string& database_id, const Vector& vector);

    // Delete a vector
    Result<void> delete_vector(const std::string& database_id, const std::string& vector_id);

    // Delete multiple vectors
    Result<void> batch_delete_vectors(const std::string& database_id,
                                    const std::vector<std::string>& vector_ids);

    // Check if a vector exists
    Result<bool> vector_exists(const std::string& database_id, const std::string& vector_id) const;

    // Get the number of vectors in a database
    Result<size_t> get_vector_count(const std::string& database_id) const;

    // Validate a vector before storing
    Result<void> validate_vector(const std::string& database_id, const Vector& vector) const;

    // Get all vector IDs in a database
    Result<std::vector<std::string>> get_all_vector_ids(const std::string& database_id) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_VECTOR_STORAGE_SERVICE_H