#ifndef JADEVECTORDB_SERIALIZATION_H
#define JADEVECTORDB_SERIALIZATION_H

#include <string>
#include <vector>
#include <memory>
#include <flatbuffers/flatbuffers.h>
#include "models/vector.h"
#include "models/database.h"
#include "models/index.h"
#include "models/embedding_model.h"

namespace jadevectordb {

// Forward declarations for FlatBuffer generated code
namespace fb = flatbuffers;

// Serialization utilities using FlatBuffers
namespace serialization {

    // Serialize Vector to FlatBuffer
    std::vector<uint8_t> serialize_vector(const Vector& vec);
    
    // Deserialize Vector from FlatBuffer
    Vector deserialize_vector(const uint8_t* data, size_t size);
    
    // Serialize Database to FlatBuffer
    std::vector<uint8_t> serialize_database(const Database& db);
    
    // Deserialize Database from FlatBuffer
    Database deserialize_database(const uint8_t* data, size_t size);
    
    // Serialize Index to FlatBuffer
    std::vector<uint8_t> serialize_index(const Index& idx);
    
    // Deserialize Index from FlatBuffer
    Index deserialize_index(const uint8_t* data, size_t size);
    
    // Serialize EmbeddingModel to FlatBuffer
    std::vector<uint8_t> serialize_embedding_model(const EmbeddingModel& model);
    
    // Deserialize EmbeddingModel from FlatBuffer
    EmbeddingModel deserialize_embedding_model(const uint8_t* data, size_t size);
    
    // Batch serialization utilities
    std::vector<uint8_t> serialize_vector_batch(const std::vector<Vector>& vectors);
    std::vector<Vector> deserialize_vector_batch(const uint8_t* data, size_t size);
    
    // Utility functions for working with FlatBuffers
    std::vector<uint8_t> serialize_generic_vector(const Vector& vec);
    std::vector<uint8_t> serialize_generic_database(const Database& db);
    std::vector<uint8_t> serialize_generic_index(const Index& idx);

    Vector deserialize_generic_vector(const uint8_t* data, size_t size);
    Database deserialize_generic_database(const uint8_t* data, size_t size);
    Index deserialize_generic_index(const uint8_t* data, size_t size);
    
    // Helper functions for creating FlatBuffer builders
    std::unique_ptr<fb::FlatBufferBuilder> create_builder();
    
    // Helper functions for verifying FlatBuffer data
    bool verify_vector_buffer(const uint8_t* data, size_t size);
    bool verify_database_buffer(const uint8_t* data, size_t size);
    bool verify_index_buffer(const uint8_t* data, size_t size);
    bool verify_embedding_model_buffer(const uint8_t* data, size_t size);

} // namespace serialization

} // namespace jadevectordb

#endif // JADEVECTORDB_SERIALIZATION_H