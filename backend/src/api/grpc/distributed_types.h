#ifndef JADEVECTORDB_DISTRIBUTED_TYPES_H
#define JADEVECTORDB_DISTRIBUTED_TYPES_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace jadevectordb {

// Enums matching proto definitions
enum class ConsistencyLevel {
    CONSISTENCY_UNKNOWN = 0,
    STRONG = 1,
    QUORUM = 2,
    EVENTUAL = 3
};

enum class HealthStatus {
    HEALTH_UNKNOWN = 0,
    HEALTHY = 1,
    DEGRADED = 2,
    UNHEALTHY = 3,
    STARTING = 4,
    STOPPING = 5
};

enum class ShardState {
    SHARD_STATE_UNKNOWN = 0,
    INITIALIZING = 1,
    ACTIVE = 2,
    MIGRATING = 3,
    SYNCING = 4,
    READONLY = 5,
    OFFLINE = 6
};

enum class ReplicationType {
    REPLICATION_UNKNOWN = 0,
    FULL = 1,
    INCREMENTAL = 2,
    SNAPSHOT = 3
};

enum class LogEntryType {
    LOG_UNKNOWN = 0,
    LOG_CONFIG_CHANGE = 1,
    LOG_SHARD_ASSIGNMENT = 2,
    LOG_NODE_JOIN = 3,
    LOG_NODE_LEAVE = 4,
    LOG_WRITE_OPERATION = 5
};

// Structures matching proto messages
struct ResourceUsage {
    double cpu_usage_percent{0.0};
    int64_t memory_used_bytes{0};
    int64_t memory_total_bytes{0};
    int64_t disk_used_bytes{0};
    int64_t disk_total_bytes{0};
    int32_t active_connections{0};
};

struct ShardStatus {
    std::string shard_id;
    ShardState state{ShardState::SHARD_STATE_UNKNOWN};
    int64_t vector_count{0};
    int64_t size_bytes{0};
    bool is_primary{false};
};

struct ShardStats {
    std::string shard_id;
    int64_t vector_count{0};
    int64_t size_bytes{0};
    int64_t queries_processed{0};
    int64_t writes_processed{0};
    double avg_query_latency_ms{0.0};
    int64_t last_updated_timestamp{0};
};

struct ShardConfig {
    std::string index_type;
    int32_t vector_dimension{0};
    std::string metric_type;
    int32_t replication_factor{3};
    std::map<std::string, std::string> index_parameters;
};

struct LogEntry {
    int64_t term{0};
    int64_t index{0};
    LogEntryType type{LogEntryType::LOG_UNKNOWN};
    std::vector<uint8_t> data;
    int64_t timestamp{0};
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_TYPES_H
