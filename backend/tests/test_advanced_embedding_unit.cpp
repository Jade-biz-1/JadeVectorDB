#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "services/custom_model_training_service.h"
#include "services/model_versioning_service.h"

using namespace jadevectordb;
using ::testing::Return;
using ::testing::_;

// Test fixture for CustomModelTrainingService
class CustomModelTrainingServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        training_service_ = std::make_unique<CustomModelTrainingService>();
    }
    
    void TearDown() override {
        training_service_.reset();
    }
    
    std::unique_ptr<CustomModelTrainingService> training_service_;
};

// Test fixture for ModelVersioningService
class ModelVersioningServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        versioning_service_ = std::make_unique<ModelVersioningService>();
    }
    
    void TearDown() override {
        versioning_service_.reset();
    }
    
    std::unique_ptr<ModelVersioningService> versioning_service_;
};

// Test custom model training service initialization
TEST_F(CustomModelTrainingServiceTest, InitializeService) {
    EXPECT_NE(training_service_, nullptr);
}

// Test custom model training configuration validation
TEST_F(CustomModelTrainingServiceTest, ValidateConfig) {
    CustomModelTrainingConfig config;
    config.model_name = "test_model";
    config.training_data_path = "/path/to/training/data";
    config.output_path = "/path/to/output";
    config.epochs = 10;
    config.learning_rate = 0.001f;
    config.batch_size = 32;
    
    bool is_valid = training_service_->validate_config(config);
    EXPECT_TRUE(is_valid);
    
    // Test with invalid config (empty model name)
    config.model_name = "";
    is_valid = training_service_->validate_config(config);
    EXPECT_FALSE(is_valid);
    
    // Test with invalid config (negative epochs)
    config.model_name = "test_model";
    config.epochs = -1;
    is_valid = training_service_->validate_config(config);
    EXPECT_FALSE(is_valid);
}

// Test model versioning service initialization
TEST_F(ModelVersioningServiceTest, InitializeService) {
    EXPECT_NE(versioning_service_, nullptr);
}

// Test creating and retrieving model versions
TEST_F(ModelVersioningServiceTest, ModelVersioning) {
    // Create a model version
    ModelVersion version_info;
    version_info.version_id = "v1_0_0_12345";
    version_info.model_id = "test_model";
    version_info.version_number = "1.0.0";
    version_info.path_to_model = "/path/to/model_v1_0_0.onnx";
    version_info.author = "test_author";
    version_info.changelog = "Initial model version";
    version_info.status = "inactive";  // Not active initially
    
    bool created = versioning_service_->create_model_version("test_model", version_info);
    EXPECT_TRUE(created);
    
    // Retrieve the model version
    ModelVersion retrieved = versioning_service_->get_model_version("test_model", "1.0.0");
    EXPECT_EQ(retrieved.version_id, version_info.version_id);
    EXPECT_EQ(retrieved.model_id, version_info.model_id);
    EXPECT_EQ(retrieved.version_number, version_info.version_number);
    EXPECT_EQ(retrieved.path_to_model, version_info.path_to_model);
    EXPECT_EQ(retrieved.author, version_info.author);
    EXPECT_EQ(retrieved.changelog, version_info.changelog);
    EXPECT_EQ(retrieved.status, version_info.status);
    
    // List all versions for the model
    std::vector<ModelVersion> versions = versioning_service_->list_model_versions("test_model");
    EXPECT_EQ(versions.size(), 1);
    EXPECT_EQ(versions[0].version_number, "1.0.0");
    
    // Activate the model version
    bool activated = versioning_service_->activate_model_version("test_model", "1.0.0");
    EXPECT_TRUE(activated);
    
    // Verify the version is now active
    ModelVersion active_version = versioning_service_->get_model_version("test_model", "1.0.0");
    EXPECT_EQ(active_version.status, "active");
}

// Test A/B testing functionality
TEST_F(ModelVersioningServiceTest, ABTesting) {
    // Create an A/B test with two models
    ABTestConfig config;
    config.test_id = "ab_test_1";
    config.model_ids = {"model_a", "model_b"};
    config.traffic_split = {0.5f, 0.5f};  // 50/50 split
    config.test_name = "Model A vs Model B";
    config.description = "A/B test comparing Model A and Model B";
    config.is_active = false;  // Start inactive
    
    bool created = versioning_service_->create_ab_test(config);
    EXPECT_TRUE(created);
    
    // Retrieve the A/B test
    ABTestConfig retrieved = versioning_service_->get_ab_test("ab_test_1");
    EXPECT_EQ(retrieved.test_id, config.test_id);
    EXPECT_EQ(retrieved.model_ids, config.model_ids);
    EXPECT_EQ(retrieved.test_name, config.test_name);
    EXPECT_EQ(retrieved.description, config.description);
    EXPECT_EQ(retrieved.is_active, config.is_active);
    
    // Start the A/B test
    bool started = versioning_service_->start_ab_test("ab_test_1");
    EXPECT_TRUE(started);
    
    // Verify the test is now active
    ABTestConfig updated = versioning_service_->get_ab_test("ab_test_1");
    EXPECT_TRUE(updated.is_active);
    
    // Select models for the A/B test (simulate multiple requests)
    std::vector<std::string> selected_models;
    for (int i = 0; i < 100; ++i) {
        std::string selected = versioning_service_->select_model_for_ab_test("ab_test_1");
        EXPECT_TRUE(selected == "model_a" || selected == "model_b");
        selected_models.push_back(selected);
    }
    
    // Count how many times each model was selected (should be roughly 50/50)
    size_t model_a_count = std::count(selected_models.begin(), selected_models.end(), "model_a");
    size_t model_b_count = std::count(selected_models.begin(), selected_models.end(), "model_b");
    
    // Allow for some variation due to randomness
    EXPECT_GT(model_a_count, 30);  // More than 30%
    EXPECT_GT(model_b_count, 30);  // More than 30%
    EXPECT_EQ(model_a_count + model_b_count, 100);
    
    // Record usage for metrics
    versioning_service_->record_model_usage("ab_test_1", "model_a", 50.0, true);  // 50ms latency, success
    versioning_service_->record_model_usage("ab_test_1", "model_b", 75.0, true);  // 75ms latency, success
    
    // Get A/B test results
    ABTestResults results = versioning_service_->get_ab_test_results("ab_test_1");
    EXPECT_EQ(results.test_id, "ab_test_1");
    EXPECT_EQ(results.model_requests["model_a"], 1);  // One request recorded
    EXPECT_EQ(results.model_requests["model_b"], 1);  // One request recorded
}

// Test A/B testing with different traffic splits
TEST_F(ModelVersioningServiceTest, ABTestingWithDifferentSplits) {
    // Create an A/B test with uneven traffic splits
    ABTestConfig config;
    config.test_id = "ab_test_2";
    config.model_ids = {"model_x", "model_y", "model_z"};
    config.traffic_split = {0.7f, 0.2f, 0.1f};  // 70/20/10 split
    config.test_name = "Model X vs Y vs Z";
    config.description = "A/B test with uneven traffic distribution";
    config.is_active = true;
    
    bool created = versioning_service_->create_ab_test(config);
    EXPECT_TRUE(created);
    
    // Select models for the A/B test (simulate multiple requests)
    std::vector<std::string> selected_models;
    for (int i = 0; i < 1000; ++i) {
        std::string selected = versioning_service_->select_model_for_ab_test("ab_test_2");
        EXPECT_TRUE(selected == "model_x" || selected == "model_y" || selected == "model_z");
        selected_models.push_back(selected);
    }
    
    // Count how many times each model was selected (should be roughly 70/20/10)
    size_t model_x_count = std::count(selected_models.begin(), selected_models.end(), "model_x");
    size_t model_y_count = std::count(selected_models.begin(), selected_models.end(), "model_y");
    size_t model_z_count = std::count(selected_models.begin(), selected_models.end(), "model_z");
    
    // Allow for some variation due to randomness, but expect approximate distribution
    EXPECT_GT(model_x_count, 600);  // More than 60%
    EXPECT_LT(model_x_count, 800);  // Less than 80%
    EXPECT_GT(model_y_count, 100);  // More than 10%
    EXPECT_LT(model_y_count, 300);  // Less than 30%
    EXPECT_GT(model_z_count, 50);   // More than 5%
    EXPECT_LT(model_z_count, 150);  // Less than 15%
    EXPECT_EQ(model_x_count + model_y_count + model_z_count, 1000);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}