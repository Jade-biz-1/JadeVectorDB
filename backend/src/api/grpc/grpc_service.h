#ifndef JADEVECTORDB_GRPC_SERVICE_H
#define JADEVECTORDB_GRPC_SERVICE_H

#include <grpcpp/grpcpp.h>
#include <string>
#include <memory>

namespace jadevectordb {

// Forward declarations for gRPC service implementations
class VectorDatabaseImpl;

// The gRPC service interface
class VectorDatabaseService {
private:
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<VectorDatabaseImpl> service_impl_;
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

// The gRPC service implementation class
class VectorDatabaseImpl {
public:
    // Constructor and destructor
    VectorDatabaseImpl();
    virtual ~VectorDatabaseImpl() = default;
    
    // Core gRPC methods that need to be implemented
    // For now, using dummy signatures that will compile
    // These should be replaced with proper request/response types
    virtual grpc::Status CreateDatabase(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status DeleteDatabase(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status StoreVector(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status RetrieveVector(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status Search(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status CreateIndex(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status UpdateConfiguration(
        grpc::ServerContext* context) = 0;
    
    virtual grpc::Status GetStatus(
        grpc::ServerContext* context) = 0;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_GRPC_SERVICE_H