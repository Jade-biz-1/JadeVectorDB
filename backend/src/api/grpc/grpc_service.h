#ifndef JADEVECTORDB_GRPC_SERVICE_H
#define JADEVECTORDB_GRPC_SERVICE_H

#include <string>
#include <memory>

#ifdef GRPC_AVAILABLE
#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#endif

namespace jadevectordb {

#ifdef GRPC_AVAILABLE

// Forward declarations for gRPC service implementations
class VectorDatabaseGrpcImpl;

// The gRPC service interface (full implementation when gRPC is available)
class VectorDatabaseService {
private:
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<VectorDatabaseGrpcImpl> service_impl_;
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

#else  // GRPC_AVAILABLE

// Minimal stub definitions when gRPC is not available
// This allows the code to compile without the full gRPC dependency

// The gRPC service interface (stub version)
class VectorDatabaseService {
private:
    void* server_;  // Opaque pointer (not used in stub)
    void* service_impl_;  // Opaque pointer (not used in stub)
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

#endif  // GRPC_AVAILABLE

} // namespace jadevectordb

#endif // JADEVECTORDB_GRPC_SERVICE_H