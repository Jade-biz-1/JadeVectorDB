#include <gtest/gtest.h>
#include "services/cluster_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <thread>
#include <chrono>

using namespace jadevectordb;

class ClusterServiceTest : public ::testing::Test {
protected:
    std::unique_ptr<ClusterService> cluster_service_;

    void SetUp() override {
        // Initialize logger for tests
        logging::LoggerManager::initialize();

        // Create cluster service instance
        cluster_service_ = std::make_unique<ClusterService>("localhost", 8080);
    }

    void TearDown() override {
        if (cluster_service_) {
            cluster_service_->stop();
        }
        cluster_service_.reset();
    }
};

// Test: Cluster Service Initialization
TEST_F(ClusterServiceTest, InitializeClusterService) {
    ASSERT_TRUE(cluster_service_->initialize());
}

// Test: Cluster Service Start and Stop
TEST_F(ClusterServiceTest, StartAndStopClusterService) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Give it a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    cluster_service_->stop();

    // Verify that stop doesn't crash
    SUCCEED();
}

// Test: Join Cluster
TEST_F(ClusterServiceTest, JoinCluster) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->join_cluster("seed-node.example.com", 8080);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

// Test: Leave Cluster
TEST_F(ClusterServiceTest, LeaveCluster) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // First join the cluster
    auto join_result = cluster_service_->join_cluster("seed-node.example.com", 8080);
    ASSERT_TRUE(join_result.has_value());

    // Then leave
    auto leave_result = cluster_service_->leave_cluster();
    ASSERT_TRUE(leave_result.has_value());
    EXPECT_TRUE(leave_result.value());
}

// Test: Get Cluster State
TEST_F(ClusterServiceTest, GetClusterState) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->get_cluster_state();
    ASSERT_TRUE(result.has_value());

    const auto& state = result.value();
    EXPECT_GE(state.term, 0);
}

// Test: Check Initial Role is Follower
TEST_F(ClusterServiceTest, InitialRoleIsFollower) {
    ASSERT_TRUE(cluster_service_->initialize());

    auto role = cluster_service_->get_current_role();
    EXPECT_EQ(role, ClusterService::ClusterRole::FOLLOWER);
}

// Test: Is Master Check
TEST_F(ClusterServiceTest, IsMasterCheck) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Initially should not be master
    EXPECT_FALSE(cluster_service_->is_master());
}

// Test: Get Master Node
TEST_F(ClusterServiceTest, GetMasterNode) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->get_master_node();
    // May not have a master initially, so we just check it doesn't crash
    if (result.has_value()) {
        const auto& master = result.value();
        EXPECT_FALSE(master.node_id.empty());
    }
}

// Test: Get All Nodes
TEST_F(ClusterServiceTest, GetAllNodes) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->get_all_nodes();
    ASSERT_TRUE(result.has_value());

    // Initially may have no nodes or just this node
    const auto& nodes = result.value();
    EXPECT_GE(nodes.size(), 0);
}

// Test: Add Node to Cluster
TEST_F(ClusterServiceTest, AddNodeToCluster) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->add_node_to_cluster("test-node-1");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    // Verify node was added
    auto nodes_result = cluster_service_->get_all_nodes();
    ASSERT_TRUE(nodes_result.has_value());

    const auto& nodes = nodes_result.value();
    bool found = false;
    for (const auto& node : nodes) {
        if (node.node_id == "test-node-1") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

// Test: Remove Node from Cluster
TEST_F(ClusterServiceTest, RemoveNodeFromCluster) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // First add a node
    auto add_result = cluster_service_->add_node_to_cluster("test-node-2");
    ASSERT_TRUE(add_result.has_value());

    // Then remove it
    auto remove_result = cluster_service_->remove_node_from_cluster("test-node-2");
    ASSERT_TRUE(remove_result.has_value());
    EXPECT_TRUE(remove_result.value());

    // Verify node was removed
    auto nodes_result = cluster_service_->get_all_nodes();
    ASSERT_TRUE(nodes_result.has_value());

    const auto& nodes = nodes_result.value();
    bool found = false;
    for (const auto& node : nodes) {
        if (node.node_id == "test-node-2") {
            found = true;
            break;
        }
    }
    EXPECT_FALSE(found);
}

// Test: Handle Node Failure
TEST_F(ClusterServiceTest, HandleNodeFailure) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Add a node first
    auto add_result = cluster_service_->add_node_to_cluster("failing-node");
    ASSERT_TRUE(add_result.has_value());

    // Simulate node failure
    cluster_service_->handle_node_failure("failing-node");

    // Verify that the service handles it gracefully
    SUCCEED();
}

// Test: Check Cluster Health
TEST_F(ClusterServiceTest, CheckClusterHealth) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->check_cluster_health();
    ASSERT_TRUE(result.has_value());

    // Should be healthy with no nodes
    EXPECT_TRUE(result.value());
}

// Test: Get Cluster Statistics
TEST_F(ClusterServiceTest, GetClusterStatistics) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->get_cluster_stats();
    ASSERT_TRUE(result.has_value());

    const auto& stats = result.value();
    // Stats map should exist (may be empty)
    EXPECT_GE(stats.size(), 0);
}

// Test: Is Node In Cluster
TEST_F(ClusterServiceTest, IsNodeInCluster) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Add a node
    auto add_result = cluster_service_->add_node_to_cluster("test-node-3");
    ASSERT_TRUE(add_result.has_value());

    // Check if it's in the cluster
    EXPECT_TRUE(cluster_service_->is_node_in_cluster("test-node-3"));

    // Check for non-existent node
    EXPECT_FALSE(cluster_service_->is_node_in_cluster("non-existent-node"));
}

// Test: Send and Receive Heartbeat
TEST_F(ClusterServiceTest, HeartbeatMechanism) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Send a heartbeat
    cluster_service_->send_heartbeat();

    // Simulate receiving a heartbeat
    cluster_service_->receive_heartbeat("remote-node-1", 1);

    // Verify no crashes
    SUCCEED();
}

// Test: Request Vote
TEST_F(ClusterServiceTest, RequestVote) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto result = cluster_service_->request_vote("candidate-node", 1, 0, 0);
    ASSERT_TRUE(result.has_value());

    // The result depends on the state, just verify it doesn't crash
    SUCCEED();
}

// Test: Trigger Election
TEST_F(ClusterServiceTest, TriggerElection) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Trigger an election
    cluster_service_->trigger_election();

    // Give it time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify no crashes
    SUCCEED();
}

// Test: Register RPC Handlers
TEST_F(ClusterServiceTest, RegisterRPCHandlers) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Register RPC handlers
    cluster_service_->register_rpc_handlers();

    // Verify no crashes
    SUCCEED();
}

// Test: Multiple Start Calls
TEST_F(ClusterServiceTest, MultipleStartCalls) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Second start should return true without issue
    ASSERT_TRUE(cluster_service_->start());
}

// Test: Stop Without Start
TEST_F(ClusterServiceTest, StopWithoutStart) {
    ASSERT_TRUE(cluster_service_->initialize());

    // Stop without starting should not crash
    cluster_service_->stop();

    SUCCEED();
}

// Test: Concurrent Node Operations
TEST_F(ClusterServiceTest, ConcurrentNodeOperations) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    // Add multiple nodes concurrently
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this, i]() {
            std::string node_id = "concurrent-node-" + std::to_string(i);
            auto result = cluster_service_->add_node_to_cluster(node_id);
            EXPECT_TRUE(result.has_value());
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify nodes were added
    auto nodes_result = cluster_service_->get_all_nodes();
    ASSERT_TRUE(nodes_result.has_value());
    EXPECT_GE(nodes_result.value().size(), 5);
}

// Test: Cluster State After Node Addition
TEST_F(ClusterServiceTest, ClusterStateAfterNodeAddition) {
    ASSERT_TRUE(cluster_service_->initialize());
    ASSERT_TRUE(cluster_service_->start());

    auto initial_state = cluster_service_->get_cluster_state();
    ASSERT_TRUE(initial_state.has_value());
    size_t initial_node_count = initial_state.value().nodes.size();

    // Add a node
    auto add_result = cluster_service_->add_node_to_cluster("new-node");
    ASSERT_TRUE(add_result.has_value());

    auto new_state = cluster_service_->get_cluster_state();
    ASSERT_TRUE(new_state.has_value());

    // Should have one more node
    EXPECT_EQ(new_state.value().nodes.size(), initial_node_count + 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
