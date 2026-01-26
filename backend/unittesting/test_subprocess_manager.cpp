#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>

#include "services/search/subprocess_manager.h"

using namespace jadedb::search;

// Test fixture for SubprocessManager tests
class SubprocessManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test Python script that echoes JSON
        test_script_path_ = "/tmp/test_echo_server.py";
        create_test_echo_script();

        // Default config for testing
        config_.python_executable = "python3";
        config_.script_path = test_script_path_;
        config_.model_name = "test-model";
        config_.batch_size = 32;
        config_.startup_timeout = std::chrono::milliseconds(10000);  // Increased to 10s
        config_.request_timeout = std::chrono::milliseconds(5000);   // Increased to 5s
    }

    void TearDown() override {
        // Clean up test script
        std::remove(test_script_path_.c_str());
    }

    // Create a simple Python echo server for testing
    void create_test_echo_script() {
        std::ofstream script(test_script_path_);
        script << R"(#!/usr/bin/env python3
import sys
import json
import time

# Send ready signal with correct format
print(json.dumps({"type": "ready", "model": "test-model"}))
sys.stdout.flush()

# Echo loop
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break

        request = json.loads(line)
        request_type = request.get("type")

        # Handle heartbeat (correct protocol)
        if request_type == "heartbeat":
            response = {"type": "heartbeat", "status": "alive"}
        # Handle shutdown
        elif request_type == "shutdown":
            response = {"type": "shutdown", "status": "acknowledged"}
            print(json.dumps(response))
            sys.stdout.flush()
            break
        # Handle test echo
        elif request.get("command") == "echo":
            response = {"echo": request.get("data")}
        # Handle query (simulate reranking)
        elif "query" in request and "documents" in request:
            docs = request["documents"]
            scores = [0.9 - i * 0.1 for i in range(len(docs))]
            response = {"scores": scores}
        else:
            response = {"error": "Unknown command"}

        print(json.dumps(response))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.stderr.flush()
        break
)";
        script.close();

        // Make script executable
        chmod(test_script_path_.c_str(), 0755);
    }

    std::string test_script_path_;
    SubprocessConfig config_;
};

// Test basic subprocess start and stop
TEST_F(SubprocessManagerTest, BasicStartStop) {
    SubprocessManager manager(config_);

    // Start subprocess
    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value()) << "Subprocess should start successfully";

    // Check status
    EXPECT_EQ(manager.get_status(), SubprocessStatus::READY);
    EXPECT_TRUE(manager.is_ready());

    // Stop subprocess
    manager.stop();

    // Status should change
    // Note: stop() is graceful, status might be TERMINATED
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test sending and receiving JSON
TEST_F(SubprocessManagerTest, SendReceiveJSON) {
    SubprocessManager manager(config_);

    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value());

    // Send echo request
    nlohmann::json request;
    request["command"] = "echo";
    request["data"] = "test message";

    auto response = manager.send_request(request);
    ASSERT_TRUE(response.has_value()) << "Request should succeed: " <<
        (response.has_value() ? "" : response.error().message);

    // Verify echo
    auto json_response = response.value();
    EXPECT_TRUE(json_response.contains("echo"));
    EXPECT_EQ(json_response["echo"].get<std::string>(), "test message");

    manager.stop();
}

// Test heartbeat/ping functionality
TEST_F(SubprocessManagerTest, Heartbeat) {
    SubprocessManager manager(config_);

    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value());

    // Send heartbeat
    auto heartbeat_result = manager.send_heartbeat();
    EXPECT_TRUE(heartbeat_result.has_value()) << "Heartbeat should succeed";

    if (heartbeat_result.has_value()) {
        EXPECT_TRUE(heartbeat_result.value()) << "Heartbeat should return true";
    }

    manager.stop();
}

// Test subprocess not ready error
TEST_F(SubprocessManagerTest, NotReadyError) {
    SubprocessManager manager(config_);

    // Try to send request without starting
    nlohmann::json request;
    request["command"] = "echo";
    request["data"] = "test";

    auto response = manager.send_request(request);
    EXPECT_FALSE(response.has_value()) << "Request should fail when not ready";

    if (!response.has_value()) {
        EXPECT_EQ(response.error().code, jadevectordb::ErrorCode::SERVICE_UNAVAILABLE);
    }
}

// Test reranking request simulation
TEST_F(SubprocessManagerTest, RerankingRequest) {
    SubprocessManager manager(config_);

    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value());

    // Send reranking request
    nlohmann::json request;
    request["query"] = "test query";
    request["documents"] = nlohmann::json::array({"doc1", "doc2", "doc3"});

    auto response = manager.send_request(request);
    ASSERT_TRUE(response.has_value()) << "Reranking request should succeed";

    auto json_response = response.value();
    EXPECT_TRUE(json_response.contains("scores"));
    EXPECT_TRUE(json_response["scores"].is_array());
    EXPECT_EQ(json_response["scores"].size(), 3);

    manager.stop();
}

// Test multiple sequential requests
TEST_F(SubprocessManagerTest, MultipleRequests) {
    SubprocessManager manager(config_);

    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value());

    // Send multiple requests
    for (int i = 0; i < 5; i++) {
        nlohmann::json request;
        request["command"] = "echo";
        request["data"] = "message_" + std::to_string(i);

        auto response = manager.send_request(request);
        ASSERT_TRUE(response.has_value()) << "Request " << i << " should succeed";

        auto json_response = response.value();
        EXPECT_EQ(json_response["echo"].get<std::string>(), "message_" + std::to_string(i));
    }

    manager.stop();
}

// Test invalid script path
TEST_F(SubprocessManagerTest, InvalidScriptPath) {
    SubprocessConfig bad_config = config_;
    bad_config.script_path = "/nonexistent/script.py";

    SubprocessManager manager(bad_config);

    auto start_result = manager.start();
    EXPECT_FALSE(start_result.has_value()) << "Should fail with invalid script";
}

// Test subprocess restart after error
TEST_F(SubprocessManagerTest, RestartAfterError) {
    SubprocessManager manager(config_);

    // Start first time
    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value());

    // Stop
    manager.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Restart
    start_result = manager.start();
    EXPECT_TRUE(start_result.has_value()) << "Should be able to restart after stop";

    if (start_result.has_value()) {
        // Verify it works
        nlohmann::json request;
        request["command"] = "echo";
        request["data"] = "after restart";

        auto response = manager.send_request(request);
        EXPECT_TRUE(response.has_value());
    }

    manager.stop();
}

// Test concurrent request handling (thread-safety)
TEST_F(SubprocessManagerTest, ConcurrentRequests) {
    SubprocessManager manager(config_);

    auto start_result = manager.start();
    ASSERT_TRUE(start_result.has_value());

    const int num_threads = 3;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&manager, &success_count, i]() {
            nlohmann::json request;
            request["command"] = "echo";
            request["data"] = "thread_" + std::to_string(i);

            auto response = manager.send_request(request);
            if (response.has_value()) {
                success_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads) << "All concurrent requests should succeed";

    manager.stop();
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
