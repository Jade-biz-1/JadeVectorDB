# Distributed Services API Documentation

> ⚠️ **IMPORTANT NOTICE**: Distributed features are currently **DISABLED** in the codebase (see `backend/src/main.cpp` lines 167-171). All distributed services are fully implemented but intentionally disabled to resolve vector storage issues and complete shard database implementation. These features are planned for **Phase 2 (Post-Launch)**. For production use, deploy in **single-node mode** using `docker-compose.yml`.

This document provides comprehensive documentation for JadeVectorDB's distributed services APIs, including clustering, sharding, replication, and distributed service management.

## Table of Contents

1. [Overview](#overview)
2. [Distributed Service Manager](#distributed-service-manager)
3. [Cluster Service](#cluster-service)
4. [Sharding Service](#sharding-service)
5. [Replication Service](#replication-service)
6. [Security Services](#security-services)
7. [Configuration Guide](#configuration-guide)
8. [Code Examples](#code-examples)

---

## Overview

JadeVectorDB provides a comprehensive distributed architecture that enables horizontal scaling, high availability, and fault tolerance. The distributed services are built on top of:

- **Cluster Service**: Manages cluster membership, leader election, and node coordination using Raft consensus
- **Sharding Service**: Distributes data across multiple nodes using various sharding strategies
- **Replication Service**: Ensures data durability through multi-node replication
- **Distributed Service Manager**: Coordinates all distributed services and provides a unified API

---

## Distributed Service Manager

The `DistributedServiceManager` is the central coordinator for all distributed services.

### Initialization

```cpp
#include "services/distributed_service_manager.h"

// Create the service manager
auto service_manager = std::make_unique<DistributedServiceManager>();

// Configure distributed services
DistributedConfig config;
config.cluster_host = "node1.example.com";
config.cluster_port = 8080;

// NOTE: Distributed features currently disabled (Phase 2)
// When enabled, requires shard database implementation
config.enable_sharding = false;  // TODO: Enable in Phase 2
config.enable_replication = false;  // TODO: Enable in Phase 2
config.enable_clustering = false;  // TODO: Enable in Phase 2

// Sharding configuration (for Phase 2)
config.sharding_config.strategy = "hash";  // or "range", "vector", "auto"
config.sharding_config.num_shards = 16;
config.sharding_config.node_list = {"node1", "node2", "node3", "node4"};
config.sharding_config.replication_factor = 3;

// Replication configuration
config.replication_config.default_replication_factor = 3;
config.replication_config.synchronous_replication = false;
config.replication_config.replication_timeout_ms = 5000;

// Initialize
auto result = service_manager->initialize(config);
if (result.has_value() && result.value()) {
    // Start services
    service_manager->start();
}
```

### Core Operations

#### Create Shards for Database

```cpp
Database db;
db.databaseId = "my-database";
db.dimensions = 512;

auto result = service_manager->create_shards_for_database(db);
if (result.has_value()) {
    std::cout << "Shards created successfully\n";
}
```

#### Get Shard for Vector

```cpp
std::string vector_id = "vector-12345";
std::string database_id = "my-database";

auto shard_result = service_manager->get_shard_for_vector(vector_id, database_id);
if (shard_result.has_value()) {
    std::string shard_id = shard_result.value();

    // Get the node hosting this shard
    auto node_result = service_manager->get_node_for_shard(shard_id);
    if (node_result.has_value()) {
        std::string node_id = node_result.value();
        std::cout << "Vector " << vector_id << " is on shard "
                  << shard_id << " hosted by " << node_id << "\n";
    }
}
```

#### Replicate Vector

```cpp
Vector vec;
vec.id = "vector-12345";
vec.values = {0.1f, 0.2f, 0.3f, /* ... */};

Database db;
db.databaseId = "my-database";

auto result = service_manager->replicate_vector(vec, db);
if (result.has_value()) {
    // Check replication status
    auto status = service_manager->is_vector_fully_replicated(vec.id);
    if (status.has_value() && status.value()) {
        std::cout << "Vector fully replicated\n";
    }
}
```

#### Cluster Operations

```cpp
// Join a cluster
auto join_result = service_manager->join_cluster("seed-node.example.com", 8080);

// Get cluster state
auto state_result = service_manager->get_cluster_state();
if (state_result.has_value()) {
    const auto& state = state_result.value();
    std::cout << "Cluster has " << state.nodes.size() << " nodes\n";
    std::cout << "Master node: " << state.master_node_id << "\n";
}

// Check cluster health
auto health_result = service_manager->is_cluster_healthy();
if (health_result.has_value() && health_result.value()) {
    std::cout << "Cluster is healthy\n";
}
```

#### Node Management

```cpp
// Add a node to the cluster
service_manager->add_node_to_cluster("new-node-5");

// Handle node failure
service_manager->handle_node_failure("failed-node-2");

// Remove a node
service_manager->remove_node_from_cluster("node-4");

// Rebalance shards after node changes
service_manager->rebalance_shards();
```

---

## Cluster Service

The `ClusterService` manages cluster membership and coordination.

### Basic Usage

```cpp
#include "services/cluster_service.h"

// Create cluster service
auto cluster_service = std::make_unique<ClusterService>("localhost", 8080);

// Initialize
cluster_service->initialize();

// Start the service
cluster_service->start();

// Join an existing cluster
cluster_service->join_cluster("seed.example.com", 8080);
```

### Leader Election

```cpp
// Trigger an election
cluster_service->trigger_election();

// Check if this node is the master
if (cluster_service->is_master()) {
    std::cout << "This node is the cluster master\n";
}

// Get master node information
auto master_result = cluster_service->get_master_node();
if (master_result.has_value()) {
    const auto& master = master_result.value();
    std::cout << "Master: " << master.node_id
              << " at " << master.host << ":" << master.port << "\n";
}
```

### Node Discovery

```cpp
// Get all nodes in the cluster
auto nodes_result = cluster_service->get_all_nodes();
if (nodes_result.has_value()) {
    for (const auto& node : nodes_result.value()) {
        std::cout << "Node: " << node.node_id
                  << " (Role: " << node.role
                  << ", Alive: " << (node.is_alive ? "yes" : "no")
                  << ")\n";
    }
}

// Check if a specific node is in the cluster
bool in_cluster = cluster_service->is_node_in_cluster("node-3");
```

### Health Monitoring

```cpp
// Check cluster health
auto health_result = cluster_service->check_cluster_health();
if (health_result.has_value() && health_result.value()) {
    std::cout << "Cluster is healthy\n";
} else {
    std::cerr << "Cluster health check failed\n";
}

// Get cluster statistics
auto stats_result = cluster_service->get_cluster_stats();
if (stats_result.has_value()) {
    const auto& stats = stats_result.value();
    for (const auto& [key, value] : stats) {
        std::cout << key << ": " << value << "\n";
    }
}
```

---

## Sharding Service

The `ShardingService` handles data distribution across nodes.

### Sharding Strategies

JadeVectorDB supports multiple sharding strategies:

1. **Hash-based**: Distributes data using consistent hashing
2. **Range-based**: Distributes data based on key ranges
3. **Vector-based**: Distributes similar vectors together
4. **Auto**: Automatically selects the best strategy

### Configuration

```cpp
#include "services/sharding_service.h"

auto sharding_service = std::make_unique<ShardingService>();

ShardingConfig config;
config.strategy = "hash";
config.num_shards = 16;
config.node_list = {"node-1", "node-2", "node-3", "node-4"};
config.hash_function = "murmur";
config.replication_factor = 3;

sharding_service->initialize(config);
```

### Shard Operations

```cpp
// Create shards for a database
Database db;
db.databaseId = "test-db";
sharding_service->create_shards_for_database(db);

// Determine shard for a vector
Vector vec;
vec.id = "vector-123";
auto shard_result = sharding_service->determine_shard(vec, db);
if (shard_result.has_value()) {
    const auto& shard_info = shard_result.value();
    std::cout << "Vector assigned to shard: " << shard_info.shard_id << "\n";
    std::cout << "Hosted on node: " << shard_info.node_id << "\n";
}

// Get all shards for a database
auto shards_result = sharding_service->get_shards_for_database("test-db");
if (shards_result.has_value()) {
    std::cout << "Database has " << shards_result.value().size() << " shards\n";
}
```

### Shard Management

```cpp
// Check if sharding is balanced
auto balance_result = sharding_service->is_balanced();
if (balance_result.has_value() && !balance_result.value()) {
    // Rebalance if needed
    sharding_service->rebalance_shards();
}

// Migrate a shard to another node
sharding_service->migrate_shard("shard-5", "target-node-3");

// Handle node failure
sharding_service->handle_node_failure("failed-node");

// Add/remove nodes
sharding_service->add_node_to_cluster("new-node-5");
sharding_service->remove_node_from_cluster("old-node-2");
```

---

## Replication Service

The `ReplicationService` ensures data durability through replication.

### Configuration

```cpp
#include "services/replication_service.h"

auto replication_service = std::make_unique<ReplicationService>();

ReplicationConfig config;
config.default_replication_factor = 3;
config.synchronous_replication = false;  // or true for sync replication
config.replication_timeout_ms = 5000;
config.replication_strategy = "simple";  // or "chain", "star"
config.enable_cross_region = true;
config.preferred_regions = {"us-east-1", "eu-west-1", "ap-south-1"};

replication_service->initialize(config);
```

### Replication Operations

```cpp
// Replicate a vector
Vector vec;
vec.id = "vector-456";
Database db;
db.databaseId = "my-db";

auto result = replication_service->replicate_vector(vec, db);

// Replicate to specific nodes
std::vector<std::string> target_nodes = {"node-1", "node-2", "node-3"};
replication_service->replicate_vector_to_nodes(vec, target_nodes);

// Update and replicate
vec.values[0] = 0.5f;  // Modify vector
replication_service->update_and_replicate(vec, db);

// Delete and replicate
replication_service->delete_and_replicate(vec.id, db);
```

### Replication Monitoring

```cpp
// Check replication status
auto status_result = replication_service->get_replication_status("vector-456");
if (status_result.has_value()) {
    const auto& status = status_result.value();
    std::cout << "Replicated to " << status.replica_nodes.size() << " nodes\n";
    std::cout << "Pending on " << status.pending_nodes.size() << " nodes\n";
}

// Check if fully replicated
auto fully_repl = replication_service->is_fully_replicated("vector-456");

// Get replica nodes
auto nodes_result = replication_service->get_replica_nodes("vector-456");
if (nodes_result.has_value()) {
    for (const auto& node : nodes_result.value()) {
        std::cout << "Replica on: " << node << "\n";
    }
}
```

### Failure Handling

```cpp
// Handle node failure (triggers re-replication)
replication_service->handle_node_failure("failed-node-2");

// Add new node and replicate data to it
replication_service->add_node_and_replicate("new-node-4");

// Force replication for entire database
replication_service->force_replication_for_database("my-db");
```

---

## Security Services

### Authentication Service

```cpp
#include "services/authentication_service.h"

auto auth_service = std::make_unique<AuthenticationService>();

AuthenticationConfig config;
config.token_expiry_seconds = 3600;  // 1 hour
config.max_failed_attempts = 5;
config.require_strong_passwords = true;

auth_service->initialize(config);

// Register user
std::vector<std::string> roles = {"user"};
auth_service->register_user("john.doe", "SecurePass123!", roles);

// Authenticate
auto token_result = auth_service->authenticate("john.doe", "SecurePass123!",
                                              "192.168.1.100");
if (token_result.has_value()) {
    const auto& token = token_result.value();
    std::cout << "Authentication successful. Token: " << token.token_value << "\n";

    // Validate token
    auto user_result = auth_service->validate_token(token.token_value);
    if (user_result.has_value()) {
        std::cout << "Token valid for user: " << user_result.value() << "\n";
    }
}
```

### Authorization Service

```cpp
#include "services/authorization_service.h"

auto authz_service = std::make_unique<AuthorizationService>();

AuthorizationConfig config;
config.enable_rbac = true;
config.enable_acl = true;

authz_service->initialize(config);

// Assign role to user
authz_service->assign_role_to_user("user-123", "admin");

// Check authorization
auto auth_result = authz_service->authorize(
    "user-123",              // user_id
    "database",              // resource_type
    "my-database",           // resource_id
    "delete"                 // action
);

if (auth_result.has_value()) {
    std::cout << "User authorized to delete database\n";
} else {
    std::cerr << "Access denied\n";
}
```

---

## Configuration Guide

### Complete Configuration Example

```cpp
// Distributed Service Configuration
DistributedConfig config;

// Cluster settings
config.cluster_host = "node1.example.com";
config.cluster_port = 8080;
config.seed_nodes = {"seed1:8080", "seed2:8080", "seed3:8080"};

// Enable services
config.enable_sharding = true;
config.enable_replication = true;
config.enable_clustering = true;

// Sharding configuration
config.sharding_config.strategy = "hash";
config.sharding_config.num_shards = 16;
config.sharding_config.node_list = {"node1", "node2", "node3", "node4"};
config.sharding_config.hash_function = "murmur";
config.sharding_config.replication_factor = 3;

// Replication configuration
config.replication_config.default_replication_factor = 3;
config.replication_config.synchronous_replication = false;
config.replication_config.replication_timeout_ms = 5000;
config.replication_config.replication_strategy = "simple";
config.replication_config.enable_cross_region = true;
config.replication_config.preferred_regions = {"us-east-1", "eu-west-1"};

// Routing configuration
config.routing_config.strategy = "round_robin";  // or "least_loaded", "consistent_hash"
config.routing_config.max_route_cache_size = 10000;
config.routing_config.route_ttl_seconds = 300;
config.routing_config.enable_adaptive_routing = true;
```

---

## Code Examples

### Complete Distributed Setup Example

```cpp
#include "services/distributed_service_manager.h"
#include <iostream>

int main() {
    // Create service manager
    auto service_manager = std::make_unique<DistributedServiceManager>();

    // Configure
    DistributedConfig config;
    config.cluster_host = "node1.example.com";
    config.cluster_port = 8080;
    config.enable_sharding = true;
    config.enable_replication = true;
    config.enable_clustering = true;

    config.sharding_config.strategy = "hash";
    config.sharding_config.num_shards = 16;
    config.replication_config.default_replication_factor = 3;

    // Initialize and start
    if (service_manager->initialize(config).value()) {
        service_manager->start();

        // Join cluster
        service_manager->join_cluster("seed.example.com", 8080);

        // Create database and shards
        Database db;
        db.databaseId = "prod-db";
        db.dimensions = 512;
        service_manager->create_shards_for_database(db);

        // Insert and replicate vectors
        Vector vec;
        vec.id = "vec-001";
        vec.values.resize(512, 1.0f);
        service_manager->replicate_vector(vec, db);

        // Monitor cluster health
        if (service_manager->is_cluster_healthy().value()) {
            std::cout << "Cluster is healthy\n";
        }

        // Get statistics
        auto stats = service_manager->get_distributed_stats();
        if (stats.has_value()) {
            for (const auto& [key, value] : stats.value()) {
                std::cout << key << ": " << value << "\n";
            }
        }
    }

    return 0;
}
```

---

## API Reference Summary

### DistributedServiceManager

| Method | Description | Returns |
|--------|-------------|---------|
| `initialize(config)` | Initialize distributed services | `Result<bool>` |
| `start()` | Start all services | `Result<bool>` |
| `stop()` | Stop all services | `Result<bool>` |
| `create_shards_for_database(db)` | Create shards for database | `Result<bool>` |
| `get_shard_for_vector(vec_id, db_id)` | Get shard assignment | `Result<string>` |
| `replicate_vector(vec, db)` | Replicate vector | `Result<bool>` |
| `join_cluster(host, port)` | Join cluster | `Result<bool>` |
| `add_node_to_cluster(node_id)` | Add node | `Result<bool>` |
| `handle_node_failure(node_id)` | Handle failure | `Result<bool>` |

### ClusterService

| Method | Description | Returns |
|--------|-------------|---------|
| `initialize()` | Initialize cluster service | `bool` |
| `start()` | Start cluster service | `bool` |
| `join_cluster(host, port)` | Join cluster | `Result<bool>` |
| `get_cluster_state()` | Get cluster state | `Result<ClusterState>` |
| `is_master()` | Check if master | `bool` |
| `get_all_nodes()` | Get all nodes | `Result<vector<ClusterNode>>` |

For complete API documentation, refer to the header files in `backend/src/services/`.
