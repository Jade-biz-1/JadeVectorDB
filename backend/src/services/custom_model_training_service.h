#ifndef JADEVECTORDB_CUSTOM_MODEL_TRAINING_SERVICE_H
#define JADEVECTORDB_CUSTOM_MODEL_TRAINING_SERVICE_H

#include "models/embedding_model.h"
#include "services/embedding_provider.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace jadevectordb {

// Configuration for custom model training
struct CustomModelTrainingConfig {
    std::string model_name;                    // Name to give the trained model
    std::string base_model_id;                 // ID of the base model to fine-tune (optional)
    std::string training_data_path;            // Path to training data
    std::string output_path;                   // Path where trained model will be saved
    std::unordered_map<std::string, std::string> hyperparameters; // Training hyperparameters
    int epochs;                                // Number of training epochs
    float learning_rate;                       // Learning rate for training
    int batch_size;                            // Batch size for training
    float validation_split;                    // Fraction of data to use for validation
    std::string save_format;                   // Format for saving the model (e.g., "onnx", "torchscript", "huggingface")
};

// Structure to represent training dataset
struct TrainingDataset {
    std::vector<std::string> texts;            // Text inputs (for text models)
    std::vector<std::string> image_paths;      // Image paths (for vision models)
    std::vector<std::vector<float>> labels;    // Target embeddings or labels
    std::vector<std::vector<float>> metadata;  // Additional metadata features
};

// Training progress callback function type
using TrainingProgressCallback = std::function<void(int epoch, int batch, float loss, float accuracy)>;

// Result of a training operation
template<typename T>
struct TrainingResult {
    bool success;
    T value;
    std::string error;
    float final_loss;
    float final_accuracy;
    
    TrainingResult(T val, float loss, float acc) 
        : success(true), value(val), error(""), final_loss(loss), final_accuracy(acc) {}
    
    TrainingResult(const std::string& err, T val = T{}) 
        : success(false), value(val), error(err), final_loss(-1.0f), final_accuracy(-1.0f) {}
};

/**
 * @brief Service for training custom embedding models
 * 
 * This service provides functionality for training custom embedding models
 * using provided training data. It supports fine-tuning existing models
 * or training new models from scratch.
 */
class CustomModelTrainingService {
public:
    CustomModelTrainingService();
    ~CustomModelTrainingService() = default;

    /**
     * @brief Train a new custom embedding model
     * @param config Configuration for the training process
     * @param progress_callback Optional callback to track training progress
     * @return Result containing the trained model ID on success
     */
    TrainingResult<std::string> train_model(
        const CustomModelTrainingConfig& config,
        TrainingProgressCallback progress_callback = nullptr);

    /**
     * @brief Fine-tune an existing model
     * @param config Configuration for the fine-tuning process
     * @param progress_callback Optional callback to track training progress
     * @return Result containing the fine-tuned model ID on success
     */
    TrainingResult<std::string> fine_tune_model(
        const CustomModelTrainingConfig& config,
        TrainingProgressCallback progress_callback = nullptr);

    /**
     * @brief Load a training dataset from file
     * @param dataset_path Path to the dataset file
     * @return TrainingDataset containing the loaded data
     */
    TrainingResult<TrainingDataset> load_dataset(const std::string& dataset_path);

    /**
     * @brief Validate training configuration
     * @param config The training configuration to validate
     * @return True if configuration is valid, false otherwise
     */
    bool validate_config(const CustomModelTrainingConfig& config) const;

    /**
     * @brief Get training metrics for a completed training job
     * @param training_job_id ID of the training job
     * @return Map of metric names to values
     */
    std::unordered_map<std::string, float> get_training_metrics(const std::string& training_job_id) const;

    /**
     * @brief Cancel a running training job
     * @param training_job_id ID of the training job to cancel
     * @return True if cancellation was successful, false otherwise
     */
    bool cancel_training_job(const std::string& training_job_id);

private:
    // Map of running training jobs
    std::unordered_map<std::string, bool> running_jobs_;
    
    // Internal method to train model based on input type
    TrainingResult<std::string> train_model_internal(
        const CustomModelTrainingConfig& config,
        const TrainingDataset& dataset,
        TrainingProgressCallback progress_callback);
    
    // Internal method to validate training data
    bool validate_dataset(const TrainingDataset& dataset) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_CUSTOM_MODEL_TRAINING_SERVICE_H