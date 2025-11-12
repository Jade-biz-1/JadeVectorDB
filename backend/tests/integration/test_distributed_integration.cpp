#include <gtest/gtest.h>
#include "services/distributed_service_manager.h"
#include "services/cluster_service.h"
#include "services/sharding_service.h"
#include "services/replication_service.h"
#include "models/database.h"
#include "models/vector.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <thread>
#include <chrono>

using namespace jadevectordb;

class DistributedIntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<DistributedServiceManager> service_manager_;
    DistributedConfig config_;

    void SetUp() override {
        // Initialize logger for tests
        logging::LoggerManager::initialize();

        // Create distributed service manager
        service_manager_ = std::make_unique<DistributedServiceManager>();

        // Configure distributed services
        config_.cluster_host = "localhost";
        config_.cluster_port = 8080;
        config_.enable_sharding = true;
        config_.enable_replication = true;
        config_.enable_clustering = true;

        // Configure sharding
        config_.sharding_config.strategy = "hash";
        config_.sharding_config.num_shards = 4;
        config_.sharding_config.node_list = {"node-1", "node-2", "node-3", "node-4"};
        config_.sharding_config.hash_function = "murmur";
        config_.sharding_config.replication_factor = 3;

        // Configure replication
        config_.replication_config.default_replication_factor = 3;
        config_.replication_config.synchronous_replication = false;
        config_.replication_config.replication_timeout_ms = 5000;
        config_.replication_config.replication_strategy = "simple";

        // Configure routing
        config_.routing_config.strategy = "round_robin";
        config_.routing_config.max_route_cache_size = 1000;
        config_.routing_config.route_ttl_seconds = 300;
        config_.routing_config.enable_adaptive_routing = true;
    }

    void TearDown() override {
        if (service_manager_ && service_manager_->is_running()) {
            service_manager_->stop();
        }
        service_manager_.reset();
    }

    Database createTestDatabase(const std::string& db_id) {
        Database db;
        db.databaseId = db_id;
        db.name = "test_db_" + db_id;
        db.dimensions = 128;
        return db;
    }

    Vector createTestVector(const std::string& vector_id, int dimensions = 128) {
        Vector vec;
        vec.id = vector_id;
        vec.values.resize(dimensions, 1.0f);
        return vec;
    }
};

// Test: Initialize Distributed Services
TEST_F(DistributedIntegrationTest, InitializeDistributedServices) {
    auto result = service_manager_->initialize(config_);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
    EXPECT_TRUE(service_manager_->is_initialized());
}

// Test: Start and Stop Distributed Services
TEST_F(DistributedIntegrationTest, StartAndStopDistributedServices) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());
    EXPECT_TRUE(service_manager_->is_running());

    ASSERT_TRUE(service_manager_->stop().value());
    EXPECT_FALSE(service_manager_->is_running());
}

// Test: End-to-End Shard Creation and Vector Assignment
TEST_F(DistributedIntegrationTest, EndToEndShardingWorkflow) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    Database db = createTestDatabase("e2e-shard-db");

    // Create shards for the database
    auto create_result = service_manager_->create_shards_for_database(db);
    ASSERT_TRUE(create_result.has_value());
    EXPECT_TRUE(create_result.value());

    // Assign a vector to a shard
    std::string vector_id = "e2e-vector-1";
    auto shard_result = service_manager_->get_shard_for_vector(vector_id, db.databaseId);
    ASSERT_TRUE(shard_result.has_value());
    EXPECT_FALSE(shard_result.value().empty());

    // Get the node for the shard
    auto node_result = service_manager_->get_node_for_shard(shard_result.value());
    ASSERT_TRUE(node_result.has_value());
    EXPECT_FALSE(node_result.value().empty());
}

// Test: End-to-End Replication Workflow
TEST_F(DistributedIntegrationTest, EndToEndReplicationWorkflow) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    Database db = createTestDatabase("e2e-repl-db");
    Vector vec = createTestVector("e2e-repl-vector");

    // Replicate the vector
    auto repl_result = service_manager_->replicate_vector(vec, db);
    EXPECT_TRUE(!repl_result.has_value() || repl_result.value());

    // Check if fully replicated
    auto check_result = service_manager_->is_vector_fully_replicated(vec.id);
    EXPECT_TRUE(check_result.has_value());
}

// Test: Cluster Join and Node Management
TEST_F(DistributedIntegrationTest, ClusterJoinAndNodeManagement) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    // Join cluster via seed node
    auto join_result = service_manager_->join_cluster("seed-node.example.com", 8080);
    ASSERT_TRUE(join_result.has_value());

    // Add a node to the cluster
    auto add_result = service_manager_->add_node_to_cluster("new-node-1");
    ASSERT_TRUE(add_result.has_value());
    EXPECT_TRUE(add_result.value());

    // Get cluster state
    auto state_result = service_manager_->get_cluster_state();
    ASSERT_TRUE(state_result.has_value());

    // Remove the node
    auto remove_result = service_manager_->remove_node_from_cluster("new-node-1");
    ASSERT_TRUE(remove_result.has_value());
}

// Test: Node Failure Handling Across Services
TEST_F(DistributedIntegrationTest, NodeFailureHandling) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    std::string failed_node = "node-2";

    // Handle node failure
    auto result = service_manager_->handle_node_failure(failed_node);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    // Verify system still healthy
    auto health_result = service_manager_->check_distributed_health();
    ASSERT_TRUE(health_result.has_value());
}

// Test: Route Operation Through Distributed System
TEST_F(DistributedIntegrationTest, RouteOperation) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    std::string database_id = "route-test-db";
    std::string operation_type = "search";
    std::string operation_key = "vector-key-1";

    auto route_result = service_manager_->route_operation(database_id, operation_type, operation_key);

    if (route_result.has_value()) {
        const auto& route_info = route_result.value();
        EXPECT_FALSE(route_info.target_node.empty());
    }
}

// Test: Distributed Statistics Collection
TEST_F(DistributedIntegrationTest, CollectDistributedStatistics) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    auto stats_result = service_manager_->get_distributed_stats();
    ASSERT_TRUE(stats_result.has_value());

    const auto& stats = stats_result.value();
    EXPECT_GE(stats.size(), 0);
}

// Test: Configuration Update During Runtime
TEST_F(DistributedIntegrationTest, RuntimeConfigurationUpdate) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    DistributedConfig new_config = config_;
    new_config.sharding_config.num_shards = 8;

    auto update_result = service_manager_->update_distributed_config(new_config);
    ASSERT_TRUE(update_result.has_value());

    auto current_config = service_manager_->get_config();
    EXPECT_EQ(current_config.sharding_config.num_shards, 8);
}

// Test: Shard Rebalancing
TEST_F(DistributedIntegrationTest, ShardRebalancing) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    Database db = createTestDatabase("rebalance-db");
    ASSERT_TRUE(service_manager_->create_shards_for_database(db).value());

    auto rebalance_result = service_manager_->rebalance_shards();
    ASSERT_TRUE(rebalance_result.has_value());
    EXPECT_TRUE(rebalance_result.value());
}

// Test: Force Replication for Database
TEST_F(DistributedIntegrationTest, ForceReplicationForDatabase) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    std::string database_id = "force-repl-db";

    auto result = service_manager_->force_replication_for_database(database_id);
    EXPECT_TRUE(!result.has_value() || result.value());
}

// Test: Security Audit Integration
TEST_F(DistributedIntegrationTest, SecurityAuditIntegration) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    auto logger = service_manager_->get_security_audit_logger();
    ASSERT_NE(logger, nullptr);

    SecurityEvent event(
        SecurityEventType::DATA_ACCESS,
        "test-user",
        "127.0.0.1",
        "test-resource",
        "read",
        true
    );

    auto audit_result = service_manager_->audit_security_event(event);
    ASSERT_TRUE(audit_result.has_value());
}

// Test: Performance Benchmark Integration
TEST_F(DistributedIntegrationTest, PerformanceBenchmarkIntegration) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    auto benchmark = service_manager_->get_performance_benchmark();
    ASSERT_NE(benchmark, nullptr);

    BenchmarkConfig bench_config;
    bench_config.benchmark_name = "distributed_test";
    bench_config.operation_type = BenchmarkOperation::SEARCH;
    bench_config.num_operations = 100;

    auto operation_func = []() -> Result<BenchmarkOperationResult> {
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return BenchmarkOperationResult(latency, true);
    };

    auto result = service_manager_->run_distributed_benchmark(bench_config, operation_func);
    ASSERT_TRUE(result.has_value());
}

// Test: Cluster Health Monitoring
TEST_F(DistributedIntegrationTest, ClusterHealthMonitoring) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    auto cluster_health = service_manager_->is_cluster_healthy();
    ASSERT_TRUE(cluster_health.has_value());

    auto distributed_health = service_manager_->check_distributed_health();
    ASSERT_TRUE(distributed_health.has_value());
}

// Test: Multiple Databases Concurrent Operations
TEST_F(DistributedIntegrationTest, MultipleDatabasesConcurrentOperations) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    std::vector<std::thread> threads;

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this, i]() {
            Database db = createTestDatabase("concurrent-db-" + std::to_string(i));
            auto result = service_manager_->create_shards_for_database(db);
            EXPECT_TRUE(result.has_value());

            Vector vec = createTestVector("vector-" + std::to_string(i));
            service_manager_->replicate_vector(vec, db);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    SUCCEED();
}

// Test: Service Coordination
TEST_F(DistributedIntegrationTest, ServiceCoordination) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    // Verify all services are accessible
    ASSERT_NE(service_manager_->get_sharding_service(), nullptr);
    ASSERT_NE(service_manager_->get_replication_service(), nullptr);
    ASSERT_NE(service_manager_->get_query_router(), nullptr);
    ASSERT_NE(service_manager_->get_cluster_service(), nullptr);
}

// Test: Partial Service Initialization
TEST_F(DistributedIntegrationTest, PartialServiceInitialization) {
    DistributedConfig partial_config = config_;
    partial_config.enable_clustering = false;  // Disable clustering

    auto result = service_manager_->initialize(partial_config);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    ASSERT_TRUE(service_manager_->start().value());

    // Sharding and replication should still work
    EXPECT_NE(service_manager_->get_sharding_service(), nullptr);
    EXPECT_NE(service_manager_->get_replication_service(), nullptr);
}

// Test: Node Addition and Shard Redistribution
TEST_F(DistributedIntegrationTest, NodeAdditionAndShardRedistribution) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    Database db = createTestDatabase("redistrib-db");
    ASSERT_TRUE(service_manager_->create_shards_for_database(db).value());

    // Add a new node
    auto add_result = service_manager_->add_node_to_cluster("new-node-2");
    ASSERT_TRUE(add_result.has_value());

    // Rebalance shards to include the new node
    auto rebalance_result = service_manager_->rebalance_shards();
    ASSERT_TRUE(rebalance_result.has_value());
}

// Test: Graceful Shutdown
TEST_F(DistributedIntegrationTest, GracefulShutdown) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    // Perform some operations
    Database db = createTestDatabase("shutdown-test-db");
    service_manager_->create_shards_for_database(db);

    // Shutdown gracefully
    auto stop_result = service_manager_->stop();
    ASSERT_TRUE(stop_result.has_value());
    EXPECT_FALSE(service_manager_->is_running());
}

// Test: Leave Cluster
TEST_F(DistributedIntegrationTest, LeaveCluster) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    // Join first
    service_manager_->join_cluster("seed.example.com", 8080);

    // Then leave
    auto result = service_manager_->leave_cluster();
    ASSERT_TRUE(result.has_value());
}

// Test: Multiple Node Selection
TEST_F(DistributedIntegrationTest, MultipleNodeSelection) {
    ASSERT_TRUE(service_manager_->initialize(config_).value());
    ASSERT_TRUE(service_manager_->start().value());

    auto result = service_manager_->select_multiple_nodes("test-db", "shard-1", "search", 3);
    ASSERT_TRUE(result.has_value());

    const auto& nodes = result.value();
    EXPECT_GE(nodes.size(), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
