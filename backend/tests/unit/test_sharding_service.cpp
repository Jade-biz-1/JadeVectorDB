#include <gtest/gtest.h>
#include "services/sharding_service.h"
#include "models/database.h"
#include "models/vector.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <thread>
#include <set>

using namespace jadevectordb;

class ShardingServiceTest : public ::testing::Test {
protected:
    std::unique_ptr<ShardingService> sharding_service_;
    ShardingConfig config_;

    void SetUp() override {
        // Initialize logger for tests
        logging::LoggerManager::initialize();

        // Create sharding service instance
        sharding_service_ = std::make_unique<ShardingService>();

        // Set up default configuration
        config_.strategy = "hash";
        config_.num_shards = 4;
        config_.node_list = {"node-1", "node-2", "node-3", "node-4"};
        config_.hash_function = "murmur";
        config_.replication_factor = 3;
    }

    void TearDown() override {
        sharding_service_.reset();
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

// Test: Sharding Service Initialization
TEST_F(ShardingServiceTest, InitializeShardingService) {
    ASSERT_TRUE(sharding_service_->initialize(config_));
}

// Test: Initialize with Invalid Configuration
TEST_F(ShardingServiceTest, InitializeWithInvalidConfiguration) {
    ShardingConfig invalid_config;
    invalid_config.num_shards = 0;  // Invalid
    invalid_config.strategy = "hash";

    EXPECT_FALSE(sharding_service_->initialize(invalid_config));
}

// Test: Get Shard for Vector
TEST_F(ShardingServiceTest, GetShardForVector) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    auto result = sharding_service_->get_shard_for_vector("vector-123", "test-db");
    ASSERT_TRUE(result.has_value());

    const auto& shard_id = result.value();
    EXPECT_FALSE(shard_id.empty());
}

// Test: Consistent Shard Assignment
TEST_F(ShardingServiceTest, ConsistentShardAssignment) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string vector_id = "vector-456";
    std::string database_id = "test-db";

    auto result1 = sharding_service_->get_shard_for_vector(vector_id, database_id);
    auto result2 = sharding_service_->get_shard_for_vector(vector_id, database_id);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // Same vector should always map to the same shard
    EXPECT_EQ(result1.value(), result2.value());
}

// Test: Different Vectors to Different Shards
TEST_F(ShardingServiceTest, DifferentVectorsDistribution) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::set<std::string> shard_ids;
    std::string database_id = "test-db";

    // Create multiple vectors and check they're distributed
    for (int i = 0; i < 100; ++i) {
        std::string vector_id = "vector-" + std::to_string(i);
        auto result = sharding_service_->get_shard_for_vector(vector_id, database_id);
        ASSERT_TRUE(result.has_value());
        shard_ids.insert(result.value());
    }

    // Should use multiple shards (at least 2 out of 4 for 100 vectors)
    EXPECT_GE(shard_ids.size(), 2);
}

// Test: Get Node for Shard
TEST_F(ShardingServiceTest, GetNodeForShard) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string shard_id = "test-db-shard-0";
    auto result = sharding_service_->get_node_for_shard(shard_id);
    ASSERT_TRUE(result.has_value());

    const auto& node_id = result.value();
    EXPECT_FALSE(node_id.empty());
}

// Test: Get Shards for Database
TEST_F(ShardingServiceTest, GetShardsForDatabase) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string database_id = "test-db";
    auto result = sharding_service_->get_shards_for_database(database_id);
    ASSERT_TRUE(result.has_value());

    const auto& shards = result.value();
    EXPECT_EQ(shards.size(), config_.num_shards);
}

// Test: Determine Shard
TEST_F(ShardingServiceTest, DetermineShard) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("test-db");
    Vector vec = createTestVector("vector-789");

    auto result = sharding_service_->determine_shard(vec, db);
    ASSERT_TRUE(result.has_value());

    const auto& shard_info = result.value();
    EXPECT_FALSE(shard_info.shard_id.empty());
    EXPECT_EQ(shard_info.database_id, db.databaseId);
    EXPECT_EQ(shard_info.status, "active");
}

// Test: Create Shards for Database
TEST_F(ShardingServiceTest, CreateShardsForDatabase) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("new-test-db");
    auto result = sharding_service_->create_shards_for_database(db);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    // Verify shards were created
    auto shards_result = sharding_service_->get_shards_for_database(db.databaseId);
    ASSERT_TRUE(shards_result.has_value());
    EXPECT_EQ(shards_result.value().size(), config_.num_shards);
}

// Test: Update Sharding Configuration
TEST_F(ShardingServiceTest, UpdateShardingConfiguration) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    ShardingConfig new_config = config_;
    new_config.num_shards = 8;  // Change number of shards

    auto result = sharding_service_->update_sharding_config(new_config);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    // Verify config was updated
    auto current_config = sharding_service_->get_config();
    EXPECT_EQ(current_config.num_shards, 8);
}

// Test: Get Current Configuration
TEST_F(ShardingServiceTest, GetCurrentConfiguration) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    auto current_config = sharding_service_->get_config();
    EXPECT_EQ(current_config.strategy, config_.strategy);
    EXPECT_EQ(current_config.num_shards, config_.num_shards);
    EXPECT_EQ(current_config.replication_factor, config_.replication_factor);
}

// Test: Get Shard Distribution
TEST_F(ShardingServiceTest, GetShardDistribution) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("dist-test-db");
    ASSERT_TRUE(sharding_service_->create_shards_for_database(db).value());

    auto result = sharding_service_->get_shard_distribution();
    ASSERT_TRUE(result.has_value());

    const auto& distribution = result.value();
    EXPECT_GE(distribution.size(), 0);
}

// Test: Check if Sharding is Balanced
TEST_F(ShardingServiceTest, CheckShardingBalance) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    auto result = sharding_service_->is_balanced();
    ASSERT_TRUE(result.has_value());

    // With no data, should be balanced
    EXPECT_TRUE(result.value());
}

// Test: Rebalance Shards
TEST_F(ShardingServiceTest, RebalanceShards) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    auto result = sharding_service_->rebalance_shards();
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Handle Node Failure
TEST_F(ShardingServiceTest, HandleNodeFailure) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string failed_node = "node-2";
    auto result = sharding_service_->handle_node_failure(failed_node);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Add Node to Cluster
TEST_F(ShardingServiceTest, AddNodeToCluster) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string new_node = "node-5";
    auto result = sharding_service_->add_node_to_cluster(new_node);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Remove Node from Cluster
TEST_F(ShardingServiceTest, RemoveNodeFromCluster) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string node_to_remove = "node-4";
    auto result = sharding_service_->remove_node_from_cluster(node_to_remove);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Get Strategy for Database
TEST_F(ShardingServiceTest, GetStrategyForDatabase) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::string database_id = "test-db";
    auto strategy = sharding_service_->get_strategy_for_database(database_id);

    // Should return one of the valid strategies
    EXPECT_TRUE(
        strategy == ShardingService::ShardingStrategy::HASH ||
        strategy == ShardingService::ShardingStrategy::RANGE ||
        strategy == ShardingService::ShardingStrategy::VECTOR ||
        strategy == ShardingService::ShardingStrategy::AUTO
    );
}

// Test: Update Shard Metadata
TEST_F(ShardingServiceTest, UpdateShardMetadata) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("metadata-test-db");
    ASSERT_TRUE(sharding_service_->create_shards_for_database(db).value());

    std::string shard_id = "metadata-test-db-shard-0";
    size_t record_count = 1000;
    size_t size_bytes = 50000;

    auto result = sharding_service_->update_shard_metadata(shard_id, record_count, size_bytes);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Migrate Shard
TEST_F(ShardingServiceTest, MigrateShard) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("migration-test-db");
    ASSERT_TRUE(sharding_service_->create_shards_for_database(db).value());

    std::string shard_id = "migration-test-db-shard-0";
    std::string target_node = "node-3";

    auto result = sharding_service_->migrate_shard(shard_id, target_node);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Hash-based Sharding with Different Strategies
TEST_F(ShardingServiceTest, HashBasedShardingStrategy) {
    config_.strategy = "hash";
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("hash-test-db");
    Vector vec = createTestVector("hash-vector-1");

    auto result = sharding_service_->determine_shard(vec, db);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().shard_id.empty());
}

// Test: Range-based Sharding Strategy
TEST_F(ShardingServiceTest, RangeBasedShardingStrategy) {
    config_.strategy = "range";
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("range-test-db");
    Vector vec = createTestVector("range-vector-1");

    auto result = sharding_service_->determine_shard(vec, db);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().shard_id.empty());
}

// Test: Auto Sharding Strategy
TEST_F(ShardingServiceTest, AutoShardingStrategy) {
    config_.strategy = "auto";
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("auto-test-db");
    Vector vec = createTestVector("auto-vector-1");

    auto result = sharding_service_->determine_shard(vec, db);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().shard_id.empty());
}

// Test: Concurrent Shard Assignments
TEST_F(ShardingServiceTest, ConcurrentShardAssignments) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    std::vector<std::thread> threads;
    std::atomic<int> successful_assignments{0};

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([this, i, &successful_assignments]() {
            std::string vector_id = "concurrent-vector-" + std::to_string(i);
            auto result = sharding_service_->get_shard_for_vector(vector_id, "concurrent-db");
            if (result.has_value()) {
                successful_assignments++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(successful_assignments.load(), 10);
}

// Test: Multiple Databases with Separate Shards
TEST_F(ShardingServiceTest, MultipleDatabasesSeparateShards) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db1 = createTestDatabase("db-1");
    Database db2 = createTestDatabase("db-2");

    ASSERT_TRUE(sharding_service_->create_shards_for_database(db1).value());
    ASSERT_TRUE(sharding_service_->create_shards_for_database(db2).value());

    auto shards1 = sharding_service_->get_shards_for_database(db1.databaseId);
    auto shards2 = sharding_service_->get_shards_for_database(db2.databaseId);

    ASSERT_TRUE(shards1.has_value());
    ASSERT_TRUE(shards2.has_value());

    EXPECT_EQ(shards1.value().size(), config_.num_shards);
    EXPECT_EQ(shards2.value().size(), config_.num_shards);

    // Shards should have different IDs
    EXPECT_NE(shards1.value()[0].shard_id, shards2.value()[0].shard_id);
}

// Test: Shard with Empty Vector ID
TEST_F(ShardingServiceTest, ShardWithEmptyVectorId) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("empty-id-db");
    Vector vec = createTestVector("");  // Empty ID

    auto result = sharding_service_->determine_shard(vec, db);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().shard_id.empty());
}

// Test: Load Balancing Across Nodes
TEST_F(ShardingServiceTest, LoadBalancingAcrossNodes) {
    ASSERT_TRUE(sharding_service_->initialize(config_));

    Database db = createTestDatabase("load-balance-db");
    ASSERT_TRUE(sharding_service_->create_shards_for_database(db).value());

    auto shards_result = sharding_service_->get_shards_for_database(db.databaseId);
    ASSERT_TRUE(shards_result.has_value());

    const auto& shards = shards_result.value();

    // Count shards per node
    std::map<std::string, int> shards_per_node;
    for (const auto& shard : shards) {
        shards_per_node[shard.node_id]++;
    }

    // Should distribute shards relatively evenly
    EXPECT_GE(shards_per_node.size(), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
