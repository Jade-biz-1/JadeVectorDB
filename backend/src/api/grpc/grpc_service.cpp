#include "grpc_service.h"
#include "lib/logging.h"
#include "lib/config.h"
#include <memory>

namespace jadevectordb {

VectorDatabaseService::VectorDatabaseService(const std::string& server_address)
    : server_address_(server_address) {
    service_impl_ = std::make_unique<VectorDatabaseImpl>();
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
    
    builder.RegisterService(service_impl_.get());
    
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
    // Register services with the gRPC server
    // This would involve defining protobuf service definitions
    // For now, we're just setting up the structure
}

// Implementation of the gRPC service methods
VectorDatabaseImpl::VectorDatabaseImpl() {
    // Initialize gRPC service implementation
}

} // namespace jadevectordb