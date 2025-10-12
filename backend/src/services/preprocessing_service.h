#ifndef JADEVECTORDB_PREPROCESSING_SERVICE_H
#define JADEVECTORDB_PREPROCESSING_SERVICE_H

#include <string>
#include <vector>
#include <memory>

namespace jadevectordb {

/**
 * @brief Service to handle preprocessing for text and images
 * 
 * This service implements preprocessing pipelines for various types of input
 * including text tokenization, normalization and image resizing, normalization
 */
class PreprocessingService {
public:
    struct TextPreprocessingOptions {
        bool lowercase;                    // Convert text to lowercase
        bool remove_punctuation;          // Remove punctuation marks
        bool remove_extra_whitespace;     // Remove extra whitespace characters
        bool strip;                       // Strip leading/trailing whitespace
        std::string tokenizer_model;      // Model to use for tokenization (if needed)
        int max_length;                   // Maximum length of input
        std::string truncation_strategy;  // "start", "end", "middle" 
        std::string padding_strategy;     // "longest", "max_length", "do_not_pad"
    };
    
    struct ImagePreprocessingOptions {
        int target_width;                 // Target width for resizing
        int target_height;                // Target height for resizing
        bool normalize;                   // Whether to normalize pixel values
        std::vector<float> mean;          // Mean values for normalization (e.g., ImageNet: [0.485, 0.456, 0.406])
        std::vector<float> std;           // Std values for normalization (e.g., ImageNet: [0.229, 0.224, 0.225])
        std::string color_format;         // Expected color format ("RGB", "BGR", "GRAY")
        bool flip_channels;               // Whether to flip color channels (e.g., RGB to BGR)
        bool convert_to_float;            // Whether to convert to float32
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    
public:
    explicit PreprocessingService();
    ~PreprocessingService() = default;
    
    // Preprocess text with specified options
    Result<std::string> preprocess_text(const std::string& input, 
                                      const TextPreprocessingOptions& options) const;
    
    // Preprocess image with specified options
    Result<std::vector<float>> preprocess_image(const std::vector<uint8_t>& image_data,
                                              int width, int height, int channels,
                                              const ImagePreprocessingOptions& options) const;
    
    // Batch text preprocessing
    Result<std::vector<std::string>> preprocess_batch_text(
        const std::vector<std::string>& inputs,
        const TextPreprocessingOptions& options) const;
    
    // Batch image preprocessing
    Result<std::vector<std::vector<float>>> preprocess_batch_images(
        const std::vector<std::vector<uint8_t>>& images_data,
        const std::vector<std::tuple<int, int, int>>& dimensions, // width, height, channels
        const ImagePreprocessingOptions& options) const;
    
    // Tokenize text using specified tokenizer
    Result<std::vector<int>> tokenize_text(const std::string& text,
                                         const std::string& tokenizer_model) const;
    
    // Pad or truncate sequences to specified length
    Result<std::vector<int>> pad_sequence(const std::vector<int>& sequence,
                                        int target_length,
                                        const std::string& strategy = "longest") const;

private:
    // Helper methods for text preprocessing
    std::string apply_lowercase(const std::string& input) const;
    std::string remove_punctuation(const std::string& input) const;
    std::string remove_extra_whitespace(const std::string& input) const;
    std::string apply_strip(const std::string& input) const;
    
    // Helper methods for image preprocessing
    Result<std::vector<uint8_t>> resize_image(const std::vector<uint8_t>& image_data,
                                            int original_width, int original_height, 
                                            int original_channels,
                                            int target_width, int target_height) const;
    Result<std::vector<float>> normalize_image(const std::vector<uint8_t>& image_data,
                                             int width, int height, int channels,
                                             const std::vector<float>& mean,
                                             const std::vector<float>& std) const;
    std::vector<float> convert_to_float(const std::vector<uint8_t>& image_data) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_PREPROCESSING_SERVICE_H