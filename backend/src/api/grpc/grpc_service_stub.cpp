// Stub implementation when gRPC is not available
// This file provides no-op implementations to allow the project to build without gRPC

#include "grpc_service.h"

namespace jadevectordb {

// VectorDatabaseService stub implementation
VectorDatabaseService::VectorDatabaseService(const std::string& server_address)
    : server_(nullptr), service_impl_(nullptr), server_address_(server_address) {
    // Stub constructor
}

VectorDatabaseService::~VectorDatabaseService() {
    // Stub destructor - no cleanup needed since nothing is initialized
}

bool VectorDatabaseService::start() {
    // Stub: Return false to indicate gRPC is not available
    return false;
}

void VectorDatabaseService::stop() {
    // Stub: Do nothing
}

void VectorDatabaseService::wait() {
    // Stub: Do nothing
}

bool VectorDatabaseService::is_running() const {
    // Stub: Always return false
    return false;
}

void VectorDatabaseService::setup_services() {
    // Stub: Do nothing
}

} // namespace jadevectordb
