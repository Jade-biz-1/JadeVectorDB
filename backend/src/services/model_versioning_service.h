#ifndef JADEVECTORDB_MODEL_VERSIONING_SERVICE_H
#define JADEVECTORDB_MODEL_VERSIONING_SERVICE_H

#include "models/embedding_model.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace jadevectordb {

// Structure to represent a model version
struct ModelVersion {
    std::string version_id;                    // Unique identifier for this version
    std::string model_id;                      // ID of the parent model
    std::string version_number;                // Semantic version (e.g., "1.0.0")
    std::string path_to_model;                 // Path to the model file
    std::string author;                        // Who created this version
    std::string changelog;                     // Description of changes in this version
    std::chrono::system_clock::time_point creation_time;  // When this version was created
    std::string status;                        // active, inactive, deprecated, etc.
    std::unordered_map<std::string, std::string> metadata; // Additional metadata about the version
};

// Structure for A/B testing configuration
struct ABTestConfig {
    std::string test_id;                       // Unique identifier for the test
    std::vector<std::string> model_ids;        // IDs of models to include in the test
    std::vector<float> traffic_split;          // Traffic distribution (e.g., {0.5, 0.5} for 50/50 split)
    std::string test_name;                     // Human-readable name for the test
    std::string description;                   // Description of the test
    std::chrono::system_clock::time_point start_time; // When the test should start
    std::chrono::system_clock::time_point end_time;   // When the test should end
    bool is_active;                           // Whether the test is currently running
};

// Structure to represent A/B testing results
struct ABTestResults {
    std::string test_id;
    std::unordered_map<std::string, double> model_metrics;  // Metrics for each model (e.g., latency, accuracy)
    std::unordered_map<std::string, size_t> model_requests; // Number of requests handled by each model
    std::chrono::system_clock::time_point last_updated;
};

/**
 * @brief Service for managing model versions and A/B testing
 * 
 * This service handles versioning of embedding models and provides
 * A/B testing capabilities to compare different model versions.
 */
class ModelVersioningService {
public:
    ModelVersioningService();
    ~ModelVersioningService() = default;

    /**
     * @brief Create a new model version
     * @param model_id ID of the parent model
     * @param version_info Information about the new version
     * @return True if version was created successfully, false otherwise
     */
    bool create_model_version(const std::string& model_id, const ModelVersion& version_info);

    /**
     * @brief Get a specific version of a model
     * @param model_id ID of the model
     * @param version_number Version number to retrieve
     * @return ModelVersion struct with the requested version info
     */
    ModelVersion get_model_version(const std::string& model_id, const std::string& version_number) const;

    /**
     * @brief List all versions of a model
     * @param model_id ID of the model
     * @return Vector of ModelVersion structs
     */
    std::vector<ModelVersion> list_model_versions(const std::string& model_id) const;

    /**
     * @brief Activate a specific version of a model (make it the default)
     * @param model_id ID of the model
     * @param version_number Version number to activate
     * @return True if activation was successful, false otherwise
     */
    bool activate_model_version(const std::string& model_id, const std::string& version_number);

    /**
     * @brief Create a new A/B test
     * @param config Configuration for the A/B test
     * @return True if test was created successfully, false otherwise
     */
    bool create_ab_test(const ABTestConfig& config);

    /**
     * @brief Update an existing A/B test
     * @param test_id ID of the test to update
     * @param config Updated configuration for the A/B test
     * @return True if test was updated successfully, false otherwise
     */
    bool update_ab_test(const std::string& test_id, const ABTestConfig& config);

    /**
     * @brief Get an A/B test by ID
     * @param test_id ID of the test to retrieve
     * @return ABTestConfig struct with the test configuration
     */
    ABTestConfig get_ab_test(const std::string& test_id) const;

    /**
     * @brief Start an A/B test
     * @param test_id ID of the test to start
     * @return True if test was started successfully, false otherwise
     */
    bool start_ab_test(const std::string& test_id);

    /**
     * @brief Stop an A/B test
     * @param test_id ID of the test to stop
     * @return True if test was stopped successfully, false otherwise
     */
    bool stop_ab_test(const std::string& test_id);

    /**
     * @brief Get results for an A/B test
     * @param test_id ID of the test
     * @return ABTestResults struct with the test results
     */
    ABTestResults get_ab_test_results(const std::string& test_id) const;

    /**
     * @brief Select a model based on A/B test configuration
     * @param test_id ID of the A/B test
     * @return ID of the model selected for this request
     */
    std::string select_model_for_ab_test(const std::string& test_id);

    /**
     * @brief Record usage of a model in an A/B test
     * @param test_id ID of the A/B test
     * @param model_id ID of the model that was used
     * @param latency Latency of the model response in milliseconds
     * @param success Whether the model operation was successful
     */
    void record_model_usage(const std::string& test_id, 
                           const std::string& model_id, 
                           double latency, 
                           bool success);

private:
    // Store model versions: model_id -> version_number -> ModelVersion
    std::unordered_map<std::string, std::unordered_map<std::string, ModelVersion>> model_versions_;
    
    // Store active A/B tests
    std::unordered_map<std::string, ABTestConfig> active_tests_;
    
    // Store A/B test results
    std::unordered_map<std::string, ABTestResults> test_results_;
    
    // Random number generator for A/B test model selection
    mutable std::mt19937 rng_;
    
    // Internal method to check if a test is currently active
    bool is_test_active(const std::string& test_id) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_MODEL_VERSIONING_SERVICE_H