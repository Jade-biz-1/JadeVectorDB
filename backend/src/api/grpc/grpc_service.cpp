#include "grpc_service.h"
#include "lib/logging.h"
#include "lib/config.h"
#include <memory>

#ifdef GRPC_AVAILABLE

#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

using grpc::Server;
using grpc::ServerBuilder;

namespace jadevectordb {

// Simple gRPC service implementation placeholder
// This will be replaced with actual generated protobuf service implementation

class VectorDatabaseGrpcImpl {
public:
    VectorDatabaseGrpcImpl() {
        // Initialize with required services for gRPC handlers
        db_service_ = std::make_unique<DatabaseService>();
        vector_storage_service_ = std::make_unique<VectorStorageService>();
        similarity_search_service_ = std::make_unique<SimilaritySearchService>();
        // REMOVED: auth_manager_ - migrated to AuthenticationService
        // TODO: Add AuthenticationService if needed for gRPC authentication
    }

private:
    std::unique_ptr<DatabaseService> db_service_;
    std::unique_ptr<VectorStorageService> vector_storage_service_;
    std::unique_ptr<SimilaritySearchService> similarity_search_service_;
    // REMOVED: AuthManager* auth_manager_ - migrated to AuthenticationService
};


VectorDatabaseService::VectorDatabaseService(const std::string& server_address)
    : server_address_(server_address) {
    service_impl_ = std::make_unique<VectorDatabaseGrpcImpl>();
}

VectorDatabaseService::~VectorDatabaseService() {
    if (server_ && server_->IsRunning()) {
        server_->Shutdown();
    }
}

bool VectorDatabaseService::start() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());

    setup_services();

    // Register services - this would happen once we have actual generated services from .proto files
    // builder.RegisterService(service_impl_.get());  // Commenting out for now

    server_ = builder.BuildAndStart();
    if (!server_) {
        return false;
    }

    auto logger = logging::LoggerManager::get_logger("gRPC-Server");
    LOG_INFO(logger, "gRPC server listening on " << server_address_);

    return true;
}

void VectorDatabaseService::stop() {
    if (server_) {
        server_->Shutdown();
    }
}

void VectorDatabaseService::wait() {
    if (server_) {
        server_->Wait();
    }
}

bool VectorDatabaseService::is_running() const {
    return server_ && server_->IsRunning();
}

void VectorDatabaseService::setup_services() {
    // Set up gRPC services - this would involve registering generated protobuf services
    // For now, this is just a placeholder
}

} // namespace jadevectordb

#else  // GRPC_AVAILABLE

// Implement stub methods when gRPC is not available
namespace jadevectordb {

VectorDatabaseService::VectorDatabaseService(const std::string& server_address)
    : server_address_(server_address) {
    // For stub implementation, no actual service is created
    service_impl_ = nullptr;
}

VectorDatabaseService::~VectorDatabaseService() {
    // Nothing to clean up in stub implementation
}

bool VectorDatabaseService::start() {
    // In stub implementation, return false to indicate gRPC is not available
    return false;
}

void VectorDatabaseService::stop() {
    // Do nothing for stub implementation
}

void VectorDatabaseService::wait() {
    // Do nothing for stub implementation
}

bool VectorDatabaseService::is_running() const {
    // In stub implementation, always return false
    return false;
}

void VectorDatabaseService::setup_services() {
    // Do nothing for stub implementation
}

} // namespace jadevectordb

#endif  // GRPC_AVAILABLE