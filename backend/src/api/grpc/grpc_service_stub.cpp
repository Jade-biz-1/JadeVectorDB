#include "grpc_service.h"

namespace jadevectordb {

// VectorDatabaseService stub implementation
VectorDatabaseService::VectorDatabaseService(const std::string& server_address)
    : server_address_(server_address) {
    // Stub constructor
}

VectorDatabaseService::~VectorDatabaseService() {
    // Stub destructor
    if (server_) {
        server_->Shutdown();
    }
}

bool VectorDatabaseService::start() {
    // Stub: Return false to indicate gRPC is not implemented
    return false;
}

void VectorDatabaseService::stop() {
    // Stub: Do nothing
    if (server_) {
        server_->Shutdown();
    }
}

void VectorDatabaseService::wait() {
    // Stub: Do nothing
    if (server_) {
        server_->Wait();
    }
}

bool VectorDatabaseService::is_running() const {
    // Stub: Always return false
    return false;
}

void VectorDatabaseService::setup_services() {
    // Stub: Do nothing
}

} // namespace jadevectordb
