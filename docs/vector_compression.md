# Vector Compression in JadeVectorDB

## Overview

JadeVectorDB provides advanced compression techniques for vector storage to reduce memory and disk usage while maintaining search quality. The compression system offers multiple algorithms with configurable lossy/lossless options.

## Compression Algorithms

### 1. SVD (Singular Value Decomposition)
- **Type**: Dimensionality reduction
- **Use Case**: Reduces vector dimensions while preserving most information
- **Quality**: Lossy (by nature of dimensionality reduction)
- **Performance**: Good for high-dimensional vectors

### 2. PCA (Principal Component Analysis) 
- **Type**: Dimensionality reduction
- **Use Case**: Similar to SVD, finds principal components to reduce dimensions
- **Quality**: Lossy (by nature of dimensionality reduction)
- **Performance**: Efficient for correlated features

### 3. Quantization
- **Type**: Value quantization
- **Use Case**: Reduces precision of floating-point values (e.g., 32-bit to 8-bit)
- **Quality**: Lossy but configurable precision
- **Performance**: High compression ratio with acceptable accuracy loss

## Configuration

Compression can be configured using the `CompressionConfig` struct:

```cpp
jadevectordb::compression::CompressionConfig config;
config.type = jadevectordb::compression::CompressionType::QUANTIZATION;
config.quality = jadevectordb::compression::CompressionQuality::MEDIUM;
config.compression_ratio = 0.5;  // Target 50% size reduction
config.target_dimensions = 128;  // For dimensionality reduction algorithms
```

## Usage in Vector Storage

The vector storage service provides compression methods:

```cpp
// Enable compression with specific configuration
auto result = vector_storage->enable_compression(config);
if (result.has_value()) {
    std::cout << "Compression enabled successfully" << std::endl;
}

// Check if compression is enabled
bool enabled = vector_storage->is_compression_enabled();

// Get current compression configuration
auto config_result = vector_storage->get_compression_config();
if (config_result.has_value()) {
    const auto& current_config = config_result.value();
    // Use current configuration
}

// Disable compression when needed
vector_storage->disable_compression();
```

## Integration with Storage

Compression is transparently applied during vector storage operations:

- `store_vector()` - Compresses vectors before storage
- `retrieve_vector()` - Decompresses vectors after retrieval
- `batch_store_vectors()` - Compresses vectors in batch operations
- `batch_retrieve_vectors()` - Decompresses vectors in batch operations

## Performance Considerations

### Trade-offs
- **Compression Ratio vs. Quality**: Higher compression ratios result in greater information loss
- **Compression Speed vs. Ratio**: More sophisticated algorithms may be slower but achieve better ratios
- **Memory Usage**: Compressed vectors use less memory but require processing for compression/decompression

### Recommended Settings
- **High-dimensional vectors** (>1000 dimensions): Use SVD or PCA for dimensionality reduction
- **Medium-dimensional vectors** (100-1000 dimensions): Use quantization for good balance
- **Low-dimensional vectors** (<100 dimensions): Compression may not be beneficial

## Lossy vs Lossless Options

Currently, the system implements lossy compression techniques which are appropriate for vector similarity search where small precision losses don't significantly impact search quality. True lossless compression could be added in future versions using algorithms like LZ77, Huffman coding, or specialized vector formats.

## Implementation Details

The compression system uses:

1. **Factory Pattern**: The `CompressionManager` manages different compression algorithms
2. **Strategy Pattern**: Different compression strategies can be selected at runtime
3. **RAII**: Automatic resource management for compression operations
4. **Batch Processing**: Efficient compression of multiple vectors at once
5. **Metadata Preservation**: Vector metadata is preserved separately from compressed values

The system automatically handles the complexity of compression/decompression, allowing users to work with vectors as usual while benefiting from reduced storage requirements.