#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#include "services/embedding_service.h"
#include "services/custom_model_training_service.h"
#include "services/model_versioning_service.h"
#include "models/embedding_model.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Integration test fixture that tests interactions between multiple services
class AdvancedEmbeddingIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        embedding_service_ = std::make_unique<EmbeddingService>();
        training_service_ = std::make_unique<CustomModelTrainingService>();
        versioning_service_ = std::make_unique<ModelVersioningService>();
    }
    
    void TearDown() override {
        embedding_service_.reset();
        training_service_.reset();
        versioning_service_.reset();
    }
    
    std::unique_ptr<EmbeddingService> embedding_service_;
    std::unique_ptr<CustomModelTrainingService> training_service_;
    std::unique_ptr<ModelVersioningService> versioning_service_;
};

// Test integration between training service and versioning service
TEST_F(AdvancedEmbeddingIntegrationTest, TrainModelAndCreateVersion) {
    // Create a training configuration
    CustomModelTrainingConfig config;
    config.model_name = "integration_test_model";
    config.training_data_path = "/tmp/dummy_dataset.txt";  // This will trigger the dummy implementation
    config.output_path = "/tmp/output_model.onnx";
    config.epochs = 2;  // Small number for testing
    config.learning_rate = 0.001f;
    config.batch_size = 4;
    
    // Train the model
    auto training_result = training_service_->train_model(config);
    ASSERT_TRUE(training_result.success) << "Training should succeed";
    
    // Create a model version from the trained model
    ModelVersion version_info;
    version_info.version_id = training_result.value;  // Use training job ID as version ID
    version_info.model_id = config.model_name;
    version_info.version_number = "1.0.0";
    version_info.path_to_model = config.output_path;
    version_info.author = "integration_test";
    version_info.changelog = "Trained with integration test data";
    version_info.status = "inactive";  // Will be activated later
    
    bool version_created = versioning_service_->create_model_version(config.model_name, version_info);
    EXPECT_TRUE(version_created);
    
    // Verify the version was created
    ModelVersion retrieved = versioning_service_->get_model_version(config.model_name, "1.0.0");
    EXPECT_EQ(retrieved.version_id, version_info.version_id);
    EXPECT_EQ(retrieved.model_id, config.model_name);
    EXPECT_EQ(retrieved.status, "inactive");
    
    // Activate the model version
    bool activated = versioning_service_->activate_model_version(config.model_name, "1.0.0");
    EXPECT_TRUE(activated);
    
    // Verify the version is now active
    ModelVersion active_version = versioning_service_->get_model_version(config.model_name, "1.0.0");
    EXPECT_EQ(active_version.status, "active");
}

// Test A/B testing with simulated embedding requests
TEST_F(AdvancedEmbeddingIntegrationTest, ABTestingWithEmbeddingSelection) {
    // Create two model versions for A/B testing
    ModelVersion model_a_version;
    model_a_version.version_id = "model_a_v1";
    model_a_version.model_id = "model_a";
    model_a_version.version_number = "1.0.0";
    model_a_version.path_to_model = "/tmp/model_a.onnx";
    model_a_version.author = "test";
    model_a_version.status = "active";
    
    ModelVersion model_b_version;
    model_b_version.version_id = "model_b_v1";
    model_b_version.model_id = "model_b";
    model_b_version.version_number = "1.0.0";
    model_b_version.path_to_model = "/tmp/model_b.onnx";
    model_b_version.author = "test";
    model_b_version.status = "active";
    
    // Create the model versions
    bool a_created = versioning_service_->create_model_version("model_a", model_a_version);
    bool b_created = versioning_service_->create_model_version("model_b", model_b_version);
    EXPECT_TRUE(a_created);
    EXPECT_TRUE(b_created);
    
    // Create an A/B test with these models
    ABTestConfig config;
    config.test_id = "integration_ab_test";
    config.model_ids = {"model_a", "model_b"};
    config.traffic_split = {0.6f, 0.4f};  // 60/40 split
    config.test_name = "Integration A/B Test";
    config.description = "Testing A/B functionality with embedding selection";
    config.is_active = true;
    
    bool test_created = versioning_service_->create_ab_test(config);
    EXPECT_TRUE(test_created);
    
    // Simulate multiple requests and verify model selection follows the distribution
    std::vector<std::string> selected_models;
    for (int i = 0; i < 1000; ++i) {
        std::string selected = versioning_service_->select_model_for_ab_test("integration_ab_test");
        EXPECT_TRUE(selected == "model_a" || selected == "model_b");
        selected_models.push_back(selected);
    }
    
    // Count selections (allowing for some statistical variation)
    size_t model_a_count = std::count(selected_models.begin(), selected_models.end(), "model_a");
    size_t model_b_count = std::count(selected_models.begin(), selected_models.end(), "model_b");
    
    // With 1000 requests and 60/40 split, we expect roughly 600/400
    // Testing with reasonable tolerance for randomness
    EXPECT_GT(model_a_count, 550);  // More than 55%
    EXPECT_LT(model_a_count, 650);  // Less than 65%
    EXPECT_GT(model_b_count, 350);  // More than 35%
    EXPECT_LT(model_b_count, 450);  // Less than 45%
    EXPECT_EQ(model_a_count + model_b_count, 1000);
    
    // Record usage metrics for both models
    versioning_service_->record_model_usage("integration_ab_test", "model_a", 45.0, true);
    versioning_service_->record_model_usage("integration_ab_test", "model_a", 55.0, true);
    versioning_service_->record_model_usage("integration_ab_test", "model_b", 65.0, true);
    versioning_service_->record_model_usage("integration_ab_test", "model_b", 75.0, true);
    
    // Get the results and verify metrics were recorded
    ABTestResults results = versioning_service_->get_ab_test_results("integration_ab_test");
    EXPECT_EQ(results.test_id, "integration_ab_test");
    EXPECT_EQ(results.model_requests["model_a"], 2);  // Two requests for model_a
    EXPECT_EQ(results.model_requests["model_b"], 2);  // Two requests for model_b
}

// Test training and immediate use of the model (simulated)
TEST_F(AdvancedEmbeddingIntegrationTest, TrainAndUseModel) {
    // This test simulates the workflow of training a model and then using it
    // In a real system, we would use the actual trained model; here we verify the flow
    
    // Create training config
    CustomModelTrainingConfig config;
    config.model_name = "workflow_test_model";
    config.training_data_path = "/tmp/dummy_dataset.txt";
    config.output_path = "/tmp/workflow_model.onnx";
    config.epochs = 1;
    config.learning_rate = 0.001f;
    config.batch_size = 2;
    
    // Train the model
    auto training_result = training_service_->train_model(config);
    ASSERT_TRUE(training_result.success) << "Training should succeed";
    
    // Create version for the trained model
    ModelVersion version_info;
    version_info.version_id = training_result.value;
    version_info.model_id = config.model_name;
    version_info.version_number = "1.0.0";
    version_info.path_to_model = config.output_path;
    version_info.author = "workflow_test";
    version_info.status = "active";
    
    bool version_created = versioning_service_->create_model_version(config.model_name, version_info);
    EXPECT_TRUE(version_created);
    
    // In a real system, we would now integrate this model with the embedding service
    // For this test, we just verify the workflow components work together
    ModelVersion active_version = versioning_service_->get_model_version(config.model_name, "1.0.0");
    EXPECT_EQ(active_version.status, "active");
    EXPECT_EQ(active_version.path_to_model, config.output_path);
    
    // Verify we can list the versions
    std::vector<ModelVersion> versions = versioning_service_->list_model_versions(config.model_name);
    EXPECT_EQ(versions.size(), 1);
    EXPECT_EQ(versions[0].version_number, "1.0.0");
}
