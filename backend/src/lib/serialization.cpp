#include "serialization.h"
#include "generated/vector_generated.h"
#include "generated/database_generated.h"
#include "generated/index_generated.h"
#include <stdexcept>
#include <cstring>

namespace jadevectordb {

namespace serialization {

    // Serialize Vector to FlatBuffer
    std::vector<uint8_t> serialize_vector(const Vector& vec) {
        flatbuffers::FlatBufferBuilder builder(1024);

        // Create metadata
        auto metadata_fb = JadeVectorDB::Schema::CreateVectorMetadata(
            builder,
            builder.CreateString(vec.metadata.source),
            0,  // created_at (timestamp as ulong)
            0,  // updated_at (timestamp as ulong)
            builder.CreateString(vec.metadata.owner),
            builder.CreateString(vec.metadata.category),
            vec.metadata.score,
            builder.CreateString(vec.metadata.status),
            builder.CreateVectorOfStrings(vec.metadata.tags)
        );

        // Create vector
        auto vector_fb = JadeVectorDB::Schema::CreateVector(
            builder,
            builder.CreateString(vec.id),
            builder.CreateVector(vec.values),
            static_cast<uint32_t>(vec.values.size()),
            metadata_fb,
            1,  // version
            false  // deleted
        );

        builder.Finish(vector_fb);

        return std::vector<uint8_t>(builder.GetBufferPointer(),
                                    builder.GetBufferPointer() + builder.GetSize());
    }
    
    // Deserialize Vector from FlatBuffer
    Vector deserialize_vector(const uint8_t* data, size_t size) {
        // Verify buffer
        flatbuffers::Verifier verifier(data, size);
        if (!JadeVectorDB::Schema::VerifyVectorBuffer(verifier)) {
            throw std::runtime_error("Invalid FlatBuffer data for vector");
        }

        // Get root
        auto vector_fb = JadeVectorDB::Schema::GetVector(data);

        // Create Vector object
        Vector vec;
        vec.id = vector_fb->id()->str();

        // Copy values
        auto values_fb = vector_fb->values();
        if (values_fb) {
            vec.values.assign(values_fb->begin(), values_fb->end());
        }

        // Copy metadata
        auto metadata_fb = vector_fb->metadata();
        if (metadata_fb) {
            vec.metadata.source = metadata_fb->source() ? metadata_fb->source()->str() : "";
            vec.metadata.created_at = std::to_string(metadata_fb->created_at());
            vec.metadata.updated_at = std::to_string(metadata_fb->updated_at());
            vec.metadata.owner = metadata_fb->owner() ? metadata_fb->owner()->str() : "";
            vec.metadata.category = metadata_fb->category() ? metadata_fb->category()->str() : "";
            vec.metadata.score = metadata_fb->score();
            vec.metadata.status = metadata_fb->status() ? metadata_fb->status()->str() : "";

            auto tags_fb = metadata_fb->tags();
            if (tags_fb) {
                for (auto tag : *tags_fb) {
                    vec.metadata.tags.push_back(tag->str());
                }
            }
        }

        return vec;
    }
    
    // Serialize Database to FlatBuffer
    std::vector<uint8_t> serialize_database(const Database& db) {
        flatbuffers::FlatBufferBuilder builder(1024);

        auto database_fb = JadeVectorDB::Schema::CreateDatabase(
            builder,
            builder.CreateString(db.databaseId),
            builder.CreateString(db.name),
            builder.CreateString(db.description),
            static_cast<uint32_t>(db.vectorDimension),
            builder.CreateString(db.indexType),
            builder.CreateString("{}"),  // index_parameters_json
            builder.CreateString("{}"),  // sharding_config_json
            builder.CreateString("{}"),  // replication_config_json
            builder.CreateString("{}"),  // embedding_models_json
            builder.CreateString("{}"),  // metadata_schema_json
            builder.CreateString("{}"),  // retention_policy_json
            builder.CreateString("{}"),  // access_control_json
            0,  // created_timestamp
            0   // updated_timestamp
        );

        builder.Finish(database_fb);

        return std::vector<uint8_t>(builder.GetBufferPointer(),
                                    builder.GetBufferPointer() + builder.GetSize());
    }

    // Deserialize Database from FlatBuffer
    Database deserialize_database(const uint8_t* data, size_t size) {
        // Verify buffer
        flatbuffers::Verifier verifier(data, size);
        if (!JadeVectorDB::Schema::VerifyDatabaseBuffer(verifier)) {
            throw std::runtime_error("Invalid FlatBuffer data for database");
        }

        auto database_fb = JadeVectorDB::Schema::GetDatabase(data);

        Database db;
        db.databaseId = database_fb->database_id() ? database_fb->database_id()->str() : "";
        db.name = database_fb->name() ? database_fb->name()->str() : "";
        db.description = database_fb->description() ? database_fb->description()->str() : "";
        db.vectorDimension = database_fb->vector_dimension();
        db.indexType = database_fb->index_type() ? database_fb->index_type()->str() : "";

        return db;
    }
    
    // Serialize Index to FlatBuffer
    std::vector<uint8_t> serialize_index(const Index& idx) {
        flatbuffers::FlatBufferBuilder builder(1024);

        auto index_fb = JadeVectorDB::Schema::CreateIndex(
            builder,
            builder.CreateString(idx.indexId),
            builder.CreateString(idx.databaseId),
            builder.CreateString(idx.type),
            builder.CreateString("{}"),  // parameters_json
            builder.CreateString("ready"),  // status
            0,  // created_timestamp
            0,  // updated_timestamp
            0,  // vector_count
            0   // size_bytes
        );

        builder.Finish(index_fb);

        return std::vector<uint8_t>(builder.GetBufferPointer(),
                                    builder.GetBufferPointer() + builder.GetSize());
    }

    // Deserialize Index from FlatBuffer
    Index deserialize_index(const uint8_t* data, size_t size) {
        // Verify buffer
        flatbuffers::Verifier verifier(data, size);
        if (!JadeVectorDB::Schema::VerifyIndexBuffer(verifier)) {
            throw std::runtime_error("Invalid FlatBuffer data for index");
        }

        auto index_fb = JadeVectorDB::Schema::GetIndex(data);

        Index idx;
        idx.indexId = index_fb->index_id() ? index_fb->index_id()->str() : "";
        idx.databaseId = index_fb->database_id() ? index_fb->database_id()->str() : "";
        idx.type = index_fb->type() ? index_fb->type()->str() : "";

        return idx;
    }

    // Serialize EmbeddingModel to FlatBuffer
    std::vector<uint8_t> serialize_embedding_model(const EmbeddingModel& model) {
        // For now, use generic JSON serialization since we don't have a FlatBuffers schema
        // This would be implemented similarly to Vector/Database/Index
        return std::vector<uint8_t>();
    }

    // Deserialize EmbeddingModel from FlatBuffer
    EmbeddingModel deserialize_embedding_model(const uint8_t* data, size_t size) {
        // For now, return empty model
        return EmbeddingModel();
    }
    
    // Batch serialization utilities
    std::vector<uint8_t> serialize_vector_batch(const std::vector<Vector>& vectors) {
        // Simple approach: serialize each vector separately with a size prefix
        std::vector<uint8_t> result;

        // Write count of vectors (4 bytes)
        uint32_t count = static_cast<uint32_t>(vectors.size());
        result.insert(result.end(), reinterpret_cast<uint8_t*>(&count),
                      reinterpret_cast<uint8_t*>(&count) + sizeof(count));

        // Serialize each vector
        for (const auto& vec : vectors) {
            auto vec_data = serialize_vector(vec);

            // Write size of this vector data (4 bytes)
            uint32_t vec_size = static_cast<uint32_t>(vec_data.size());
            result.insert(result.end(), reinterpret_cast<uint8_t*>(&vec_size),
                         reinterpret_cast<uint8_t*>(&vec_size) + sizeof(vec_size));

            // Write vector data
            result.insert(result.end(), vec_data.begin(), vec_data.end());
        }

        return result;
    }

    std::vector<Vector> deserialize_vector_batch(const uint8_t* data, size_t size) {
        std::vector<Vector> result;

        if (size < sizeof(uint32_t)) {
            throw std::runtime_error("Invalid batch data: too small");
        }

        size_t offset = 0;

        // Read count
        uint32_t count;
        std::memcpy(&count, data + offset, sizeof(count));
        offset += sizeof(count);

        result.reserve(count);

        // Deserialize each vector
        for (uint32_t i = 0; i < count; ++i) {
            if (offset + sizeof(uint32_t) > size) {
                throw std::runtime_error("Invalid batch data: truncated");
            }

            // Read size of this vector
            uint32_t vec_size;
            std::memcpy(&vec_size, data + offset, sizeof(vec_size));
            offset += sizeof(vec_size);

            if (offset + vec_size > size) {
                throw std::runtime_error("Invalid batch data: vector size exceeds buffer");
            }

            // Deserialize vector
            auto vec = deserialize_vector(data + offset, vec_size);
            result.push_back(vec);

            offset += vec_size;
        }

        return result;
    }
    
    // Generic serialization functions
    std::vector<uint8_t> serialize_generic_vector(const Vector& vec) {
        return serialize_vector(vec);
    }

    std::vector<uint8_t> serialize_generic_database(const Database& db) {
        return serialize_database(db);
    }

    std::vector<uint8_t> serialize_generic_index(const Index& idx) {
        return serialize_index(idx);
    }

    // Generic deserialization functions
    Vector deserialize_generic_vector(const uint8_t* data, size_t size) {
        return deserialize_vector(data, size);
    }

    Database deserialize_generic_database(const uint8_t* data, size_t size) {
        return deserialize_database(data, size);
    }

    Index deserialize_generic_index(const uint8_t* data, size_t size) {
        return deserialize_index(data, size);
    }

    // Helper functions for creating FlatBuffer builders
    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_builder() {
        return std::make_unique<flatbuffers::FlatBufferBuilder>(1024);
    }

    // Helper functions for verifying FlatBuffer data
    bool verify_vector_buffer(const uint8_t* data, size_t size) {
        flatbuffers::Verifier verifier(data, size);
        return JadeVectorDB::Schema::VerifyVectorBuffer(verifier);
    }

    bool verify_database_buffer(const uint8_t* data, size_t size) {
        flatbuffers::Verifier verifier(data, size);
        return JadeVectorDB::Schema::VerifyDatabaseBuffer(verifier);
    }

    bool verify_index_buffer(const uint8_t* data, size_t size) {
        flatbuffers::Verifier verifier(data, size);
        return JadeVectorDB::Schema::VerifyIndexBuffer(verifier);
    }

    bool verify_embedding_model_buffer(const uint8_t* data, size_t size) {
        // As a fallback for now, return true since we don't have a specific schema for embedding models
        // In a real implementation, we'd check the actual buffer type
        flatbuffers::Verifier verifier(data, size);
        // Try verification against common schemas (since we don't have specific embedding model schema yet)
        return JadeVectorDB::Schema::VerifyVectorBuffer(verifier) ||
               JadeVectorDB::Schema::VerifyDatabaseBuffer(verifier) ||
               JadeVectorDB::Schema::VerifyIndexBuffer(verifier);
    }

} // namespace serialization

} // namespace jadevectordb