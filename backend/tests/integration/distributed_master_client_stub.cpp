// Stub implementation for distributed master client - provides minimal symbols for linking tests
#include <string>

namespace jadevectordb {

class DistributedMasterClient {
public:
    struct ReplicationRequest {
        ~ReplicationRequest() = default;
    };
    
    struct SearchRequest {
        ~SearchRequest() = default;
    };
    
    // Stub implementations
    bool is_worker_connected(const std::string&) const {
        return false;
    }
    
    bool replicate_data(const std::string&, const ReplicationRequest&) {
        return false;
    }
};

} // namespace jadevectordb

