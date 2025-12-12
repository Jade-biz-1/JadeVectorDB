#include "model_versioning_service.h"
#include "models/vector.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <random>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <sstream>

namespace jadevectordb {

ModelVersioningService::ModelVersioningService() : rng_(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
    // Initialize the model versioning service
}

bool ModelVersioningService::create_model_version(const std::string& model_id, const ModelVersion& version_info) {
    if (model_id.empty() || version_info.version_number.empty()) {
        return false;
    }
    
    // Check if the model exists (in a real implementation, we might want to validate this)
    // For now, we just create the version
    
    model_versions_[model_id][version_info.version_number] = version_info;
    
    return true;
}

ModelVersion ModelVersioningService::get_model_version(const std::string& model_id, const std::string& version_number) const {
    auto model_it = model_versions_.find(model_id);
    if (model_it != model_versions_.end()) {
        auto version_it = model_it->second.find(version_number);
        if (version_it != model_it->second.end()) {
            return version_it->second;
        }
    }
    
    // Return an empty model version if not found
    return ModelVersion{};
}

std::vector<ModelVersion> ModelVersioningService::list_model_versions(const std::string& model_id) const {
    std::vector<ModelVersion> versions;
    
    auto model_it = model_versions_.find(model_id);
    if (model_it != model_versions_.end()) {
        for (const auto& pair : model_it->second) {
            versions.push_back(pair.second);
        }
    }
    
    return versions;
}

bool ModelVersioningService::activate_model_version(const std::string& model_id, const std::string& version_number) {
    auto model_it = model_versions_.find(model_id);
    if (model_it == model_versions_.end()) {
        return false; // Model doesn't exist
    }
    
    auto version_it = model_it->second.find(version_number);
    if (version_it == model_it->second.end()) {
        return false; // Version doesn't exist
    }
    
    // Deactivate all versions of this model
    for (auto& version_pair : model_it->second) {
        version_pair.second.status = "inactive";
    }
    
    // Activate the specified version
    version_it->second.status = "active";
    
    return true;
}

bool ModelVersioningService::create_ab_test(const ABTestConfig& config) {
    if (config.test_id.empty() || config.model_ids.empty() || config.traffic_split.empty()) {
        return false;
    }
    
    // Validate that the number of models matches the number of traffic splits
    if (config.model_ids.size() != config.traffic_split.size()) {
        return false;
    }
    
    // Normalize traffic splits to sum to 1.0
    float total_split = 0.0f;
    for (float split : config.traffic_split) {
        total_split += split;
    }
    
    if (total_split <= 0.0f) {
        return false;
    }
    
    // Create cumulative distribution for model selection
    std::vector<float> cumulative_splits(config.traffic_split.size());
    float running_sum = 0.0f;
    for (size_t i = 0; i < config.traffic_split.size(); ++i) {
        running_sum += config.traffic_split[i] / total_split;
        cumulative_splits[i] = running_sum;
    }
    
    // Make a copy of the config and update the traffic splits
    ABTestConfig normalized_config = config;
    normalized_config.traffic_split = cumulative_splits;
    
    // Store the test
    active_tests_[config.test_id] = normalized_config;
    
    // Initialize test results
    ABTestResults results;
    results.test_id = config.test_id;
    for (const auto& model_id : config.model_ids) {
        results.model_requests[model_id] = 0;
        results.model_metrics[model_id] = 0.0; // Default metric value
    }
    results.last_updated = std::chrono::system_clock::now();
    
    test_results_[config.test_id] = results;
    
    return true;
}

bool ModelVersioningService::update_ab_test(const std::string& test_id, const ABTestConfig& config) {
    auto it = active_tests_.find(test_id);
    if (it == active_tests_.end()) {
        return false; // Test doesn't exist
    }
    
    // Update the test configuration
    active_tests_[test_id] = config;
    
    return true;
}

ABTestConfig ModelVersioningService::get_ab_test(const std::string& test_id) const {
    auto it = active_tests_.find(test_id);
    if (it != active_tests_.end()) {
        return it->second;
    }
    
    // Return an empty config if not found
    return ABTestConfig{};
}

bool ModelVersioningService::start_ab_test(const std::string& test_id) {
    auto it = active_tests_.find(test_id);
    if (it == active_tests_.end()) {
        return false; // Test doesn't exist
    }
    
    it->second.is_active = true;
    it->second.start_time = std::chrono::system_clock::now();
    
    return true;
}

bool ModelVersioningService::stop_ab_test(const std::string& test_id) {
    auto it = active_tests_.find(test_id);
    if (it == active_tests_.end()) {
        return false; // Test doesn't exist
    }
    
    it->second.is_active = false;
    it->second.end_time = std::chrono::system_clock::now();
    
    return true;
}

ABTestResults ModelVersioningService::get_ab_test_results(const std::string& test_id) const {
    auto it = test_results_.find(test_id);
    if (it != test_results_.end()) {
        return it->second;
    }
    
    // Return empty results if not found
    return ABTestResults{};
}

std::string ModelVersioningService::select_model_for_ab_test(const std::string& test_id) {
    auto test_it = active_tests_.find(test_id);
    if (test_it == active_tests_.end() || !is_test_active(test_id)) {
        return ""; // Return empty string if test doesn't exist or isn't active
    }
    
    const auto& config = test_it->second;
    
    // Generate a random value between 0 and 1
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_val = dist(rng_);
    
    // Select model based on cumulative traffic distribution
    for (size_t i = 0; i < config.traffic_split.size(); ++i) {
        if (random_val <= config.traffic_split[i]) {
            return config.model_ids[i];
        }
    }
    
    // Fallback: return the last model
    return config.model_ids.back();
}

void ModelVersioningService::record_model_usage(const std::string& test_id, 
                                               const std::string& model_id, 
                                               double latency, 
                                               bool success) {
    auto it = test_results_.find(test_id);
    if (it != test_results_.end()) {
        // Record the request
        it->second.model_requests[model_id]++;
        
        // Update metrics (in a real implementation, you'd track more complex metrics)
        // For now, we'll just track a simple performance metric
        auto current_metric = it->second.model_metrics[model_id];
        auto new_requests = it->second.model_requests[model_id];
        
        // Calculate a simple metric (e.g., weighted average of inverse latency for performance)
        double new_metric = success ? (1.0 / (latency + 1.0)) : 0.0;
        
        // Update the metric as a weighted average
        it->second.model_metrics[model_id] = (current_metric * (new_requests - 1) + new_metric) / new_requests;
        
        it->second.last_updated = std::chrono::system_clock::now();
    }
}

bool ModelVersioningService::is_test_active(const std::string& test_id) const {
    auto it = active_tests_.find(test_id);
    if (it != active_tests_.end()) {
        const auto& config = it->second;
        
        // Check if the test is manually set as inactive
        if (!config.is_active) {
            return false;
        }
        
        // Check if the test has a start time in the future
        auto now = std::chrono::system_clock::now();
        if (config.start_time > now) {
            return false;
        }
        
        // Check if the test has an end time in the past
        if (config.end_time.time_since_epoch().count() != 0 && config.end_time < now) {
            return false;
        }
        
        return true;
    }
    
    return false; // Test doesn't exist
}

bool ModelVersioningService::check_version_compatibility(const std::string& model_id,
                                                        const std::string& version1,
                                                        const std::string& version2) const {
    auto model_it = model_versions_.find(model_id);
    if (model_it == model_versions_.end()) {
        return false; // Model doesn't exist
    }
    
    auto v1_it = model_it->second.find(version1);
    auto v2_it = model_it->second.find(version2);
    
    if (v1_it == model_it->second.end() || v2_it == model_it->second.end()) {
        return false; // One or both versions don't exist
    }
    
    // Parse semantic versions (e.g., "1.2.3")
    auto parse_version = [](const std::string& version) {
        std::vector<int> parts;
        std::stringstream ss(version);
        std::string part;
        while (std::getline(ss, part, '.')) {
            try {
                parts.push_back(std::stoi(part));
            } catch (...) {
                parts.push_back(0);
            }
        }
        while (parts.size() < 3) parts.push_back(0);
        return parts;
    };
    
    auto v1_parts = parse_version(version1);
    auto v2_parts = parse_version(version2);
    
    // Check compatibility: versions are compatible if major version is the same
    // This is a common convention in semantic versioning
    if (v1_parts[0] == v2_parts[0]) {
        return true; // Same major version = compatible
    }
    
    // Check metadata for explicit compatibility information
    if (v1_it->second.metadata.count("compatible_versions")) {
        // In a real implementation, we'd parse this metadata
        return true;
    }
    
    return false; // Different major versions = incompatible by default
}

bool ModelVersioningService::upgrade_vectors(std::vector<Vector>& vectors,
                                            const std::string& target_model_id,
                                            const std::string& target_version) {
    auto model_it = model_versions_.find(target_model_id);
    if (model_it == model_versions_.end()) {
        return false; // Target model doesn't exist
    }
    
    auto version_it = model_it->second.find(target_version);
    if (version_it == model_it->second.end()) {
        return false; // Target version doesn't exist
    }
    
    const ModelVersion& target_model = version_it->second;
    
    // Upgrade each vector
    for (auto& vector : vectors) {
        // Check if upgrade is needed
        if (vector.embedding_model.name == target_model_id &&
            vector.embedding_model.version == target_version) {
            continue; // Already on target version
        }
        
        // Check compatibility
        if (vector.embedding_model.name == target_model_id) {
            if (!check_version_compatibility(target_model_id, 
                                            vector.embedding_model.version, 
                                            target_version)) {
                // Incompatible versions - would need re-embedding
                // In a real implementation, this would trigger re-embedding
                continue;
            }
        }
        
        // Update vector's model information
        vector.embedding_model.name = target_model_id;
        vector.embedding_model.version = target_version;
        vector.embedding_model.provider = target_model.metadata.count("provider") ? 
                                         target_model.metadata.at("provider") : "";
        
        // Increment vector version to track the change
        vector.version++;
        
        // Update metadata timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        vector.metadata.updated_at = std::ctime(&time_t_now);
    }
    
    return true;
}

std::string ModelVersioningService::get_recommended_version(const std::string& model_id,
                                                           const std::string& input_type) const {
    auto model_it = model_versions_.find(model_id);
    if (model_it == model_versions_.end()) {
        return ""; // Model doesn't exist
    }
    
    // Find the latest active version that supports the input type
    std::string recommended_version;
    std::vector<int> highest_version = {0, 0, 0};
    
    for (const auto& [version_num, version_info] : model_it->second) {
        // Skip inactive versions
        if (version_info.status != "active") {
            continue;
        }
        
        // Check if this version supports the input type
        bool supports_input = true;
        if (version_info.metadata.count("supported_inputs")) {
            // In a real implementation, we'd parse the supported inputs
            // For now, assume all active versions support all input types
        }
        
        if (supports_input) {
            // Parse version number
            std::vector<int> version_parts;
            std::stringstream ss(version_num);
            std::string part;
            while (std::getline(ss, part, '.')) {
                try {
                    version_parts.push_back(std::stoi(part));
                } catch (...) {
                    version_parts.push_back(0);
                }
            }
            while (version_parts.size() < 3) version_parts.push_back(0);
            
            // Compare versions
            bool is_higher = false;
            for (size_t i = 0; i < 3; i++) {
                if (version_parts[i] > highest_version[i]) {
                    is_higher = true;
                    break;
                } else if (version_parts[i] < highest_version[i]) {
                    break;
                }
            }
            
            if (is_higher) {
                highest_version = version_parts;
                recommended_version = version_num;
            }
        }
    }
    
    return recommended_version;
}

} // namespace jadevectordb