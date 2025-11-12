#ifndef JADEVECTORDB_GRPC_SERVICE_H
#define JADEVECTORDB_GRPC_SERVICE_H

#include <string>
#include <memory>

namespace jadevectordb {

// Minimal stub definitions when gRPC is not available
// This allows the code to compile without the full gRPC dependency

// Forward declarations for gRPC service implementations
class VectorDatabaseImpl;

// The gRPC service interface (stub version)
class VectorDatabaseService {
private:
    void* server_;  // Opaque pointer (not used in stub)
    VectorDatabaseImpl* service_impl_;  // Opaque pointer (not used in stub)
    std::string server_address_;

public:
    explicit VectorDatabaseService(const std::string& server_address);
    ~VectorDatabaseService();

    // Start the gRPC server
    bool start();

    // Stop the gRPC server
    void stop();

    // Wait for the server to shutdown
    void wait();

    // Check if the server is running
    bool is_running() const;

private:
    void setup_services();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_GRPC_SERVICE_H