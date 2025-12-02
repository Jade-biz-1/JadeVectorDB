#include "arrow_utils.h"
#include "models/index.h"
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/compute/api.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/c/bridge.h>
#include <memory>
#include <fstream>

using arrow::default_memory_pool;

namespace jadevectordb {

namespace arrow_utils {

    // ArrowSchemaFactory implementation
    std::shared_ptr<arrow::Schema> ArrowSchemaFactory::create_vector_schema() {
        // Define fields for the vector schema
        std::vector<std::shared_ptr<arrow::Field>> fields = {
            arrow::field("id", arrow::utf8()),
            arrow::field("values", arrow::list(arrow::float32())),
            arrow::field("metadata_source", arrow::utf8()),
            arrow::field("created_at", arrow::timestamp(arrow::TimeUnit::MILLI)),
            arrow::field("updated_at", arrow::timestamp(arrow::TimeUnit::MILLI)),
            arrow::field("tags", arrow::list(arrow::utf8())),
            arrow::field("owner", arrow::utf8()),
            arrow::field("permissions", arrow::list(arrow::utf8())),
            arrow::field("category", arrow::utf8()),
            arrow::field("score", arrow::float32()),
            arrow::field("status", arrow::utf8()),
            arrow::field("custom", arrow::utf8()), // JSON string for custom metadata
            arrow::field("shard", arrow::utf8()),
            arrow::field("replicas", arrow::list(arrow::utf8())),
            arrow::field("version", arrow::int32()),
            arrow::field("deleted", arrow::boolean())
        };
        
        return arrow::schema(fields);
    }
    
    std::shared_ptr<arrow::Schema> ArrowSchemaFactory::create_database_schema() {
        // Define fields for the database schema
        std::vector<std::shared_ptr<arrow::Field>> fields = {
            arrow::field("database_id", arrow::utf8()),
            arrow::field("name", arrow::utf8()),
            arrow::field("description", arrow::utf8()),
            arrow::field("vector_dimension", arrow::int32()),
            arrow::field("index_type", arrow::utf8()),
            arrow::field("index_parameters", arrow::utf8()), // JSON string
            arrow::field("sharding_strategy", arrow::utf8()),
            arrow::field("num_shards", arrow::int32()),
            arrow::field("replication_factor", arrow::int32()),
            arrow::field("sync_replication", arrow::boolean()),
            arrow::field("embedding_models", arrow::utf8()), // JSON array string
            arrow::field("metadata_schema", arrow::utf8()), // JSON string
            arrow::field("retention_policy", arrow::utf8()), // JSON string
            arrow::field("access_control", arrow::utf8()), // JSON string
            arrow::field("created_at", arrow::timestamp(arrow::TimeUnit::MILLI)),
            arrow::field("updated_at", arrow::timestamp(arrow::TimeUnit::MILLI))
        };
        
        return arrow::schema(fields);
    }
    
    std::shared_ptr<arrow::Schema> ArrowSchemaFactory::create_index_schema() {
        // Define fields for the index schema
        std::vector<std::shared_ptr<arrow::Field>> fields = {
            arrow::field("index_id", arrow::utf8()),
            arrow::field("database_id", arrow::utf8()),
            arrow::field("type", arrow::utf8()),
            arrow::field("parameters", arrow::utf8()), // JSON string
            arrow::field("status", arrow::utf8()),
            arrow::field("created_at", arrow::timestamp(arrow::TimeUnit::MILLI)),
            arrow::field("updated_at", arrow::timestamp(arrow::TimeUnit::MILLI)),
            arrow::field("vector_count", arrow::int64()),
            arrow::field("size_bytes", arrow::int64()),
            arrow::field("build_time_ms", arrow::int32())
        };
        
        return arrow::schema(fields);
    }
    
    std::shared_ptr<arrow::Schema> ArrowSchemaFactory::create_embedding_model_schema() {
        // Define fields for the embedding model schema
        std::vector<std::shared_ptr<arrow::Field>> fields = {
            arrow::field("model_id", arrow::utf8()),
            arrow::field("name", arrow::utf8()),
            arrow::field("version", arrow::utf8()),
            arrow::field("provider", arrow::utf8()),
            arrow::field("input_type", arrow::utf8()),
            arrow::field("output_dimension", arrow::int32()),
            arrow::field("parameters", arrow::utf8()), // JSON string
            arrow::field("status", arrow::utf8())
        };
        
        return arrow::schema(fields);
    }
    
    std::shared_ptr<arrow::Schema> ArrowSchemaFactory::create_vector_batch_schema() {
        // For batch operations, we use the same schema as individual vectors
        return create_vector_schema();
    }
    
    // ArrowConverter implementation
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> ArrowConverter::vector_to_record_batch(
        const Vector& vector) {
        auto schema = ArrowSchemaFactory::create_vector_schema();
        
        // Create builders for each field
        arrow::StringBuilder id_builder;
        arrow::ListBuilder values_builder(default_memory_pool(), 
                                          std::make_shared<arrow::FloatBuilder>());
        arrow::StringBuilder metadata_source_builder;
        arrow::TimestampBuilder created_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI), 
                                                   default_memory_pool());
        arrow::TimestampBuilder updated_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI), 
                                                   default_memory_pool());
        arrow::ListBuilder tags_builder(default_memory_pool(), 
                                        std::make_shared<arrow::StringBuilder>());
        arrow::StringBuilder owner_builder;
        arrow::ListBuilder permissions_builder(default_memory_pool(), 
                                              std::make_shared<arrow::StringBuilder>());
        arrow::StringBuilder category_builder;
        arrow::FloatBuilder score_builder;
        arrow::StringBuilder status_builder;
        arrow::StringBuilder custom_builder;
        arrow::StringBuilder shard_builder;
        arrow::ListBuilder replicas_builder(default_memory_pool(), 
                                            std::make_shared<arrow::StringBuilder>());
        arrow::Int32Builder version_builder;
        arrow::BooleanBuilder deleted_builder;
        
        // Append values to builders
        ARROW_RETURN_NOT_OK(id_builder.Append(vector.id));
        
        // Append vector values
        auto float_builder = std::dynamic_pointer_cast<arrow::FloatBuilder>(
            values_builder.value_builder());
        ARROW_RETURN_NOT_OK(values_builder.Append());
        ARROW_RETURN_NOT_OK(float_builder->AppendValues(vector.values));
        
        ARROW_RETURN_NOT_OK(metadata_source_builder.Append(vector.metadata.source));
        ARROW_RETURN_NOT_OK(created_at_builder.Append(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::from_time_t(0).time_since_epoch()).count()));
        ARROW_RETURN_NOT_OK(updated_at_builder.Append(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::from_time_t(0).time_since_epoch()).count()));
        
        // Append tags
        ARROW_RETURN_NOT_OK(tags_builder.Append());
        auto tag_builder = std::dynamic_pointer_cast<arrow::StringBuilder>(
            tags_builder.value_builder());
        for (const auto& tag : vector.metadata.tags) {
            ARROW_RETURN_NOT_OK(tag_builder->Append(tag));
        }
        
        ARROW_RETURN_NOT_OK(owner_builder.Append(vector.metadata.owner));
        
        // Append permissions
        ARROW_RETURN_NOT_OK(permissions_builder.Append());
        auto perm_builder = std::dynamic_pointer_cast<arrow::StringBuilder>(
            permissions_builder.value_builder());
        for (const auto& perm : vector.metadata.permissions) {
            ARROW_RETURN_NOT_OK(perm_builder->Append(perm));
        }
        
        ARROW_RETURN_NOT_OK(category_builder.Append(vector.metadata.category));
        ARROW_RETURN_NOT_OK(score_builder.Append(vector.metadata.score));
        ARROW_RETURN_NOT_OK(status_builder.Append(vector.metadata.status));
        ARROW_RETURN_NOT_OK(custom_builder.Append("{}")); // Placeholder for custom metadata
        ARROW_RETURN_NOT_OK(shard_builder.Append(vector.shard));
        
        // Append replicas
        ARROW_RETURN_NOT_OK(replicas_builder.Append());
        auto replica_builder = std::dynamic_pointer_cast<arrow::StringBuilder>(
            replicas_builder.value_builder());
        for (const auto& replica : vector.replicas) {
            ARROW_RETURN_NOT_OK(replica_builder->Append(replica));
        }
        
        ARROW_RETURN_NOT_OK(version_builder.Append(vector.version));
        ARROW_RETURN_NOT_OK(deleted_builder.Append(vector.deleted));
        
        // Finish building arrays
        std::shared_ptr<arrow::Array> id_array;
        std::shared_ptr<arrow::Array> values_array;
        std::shared_ptr<arrow::Array> metadata_source_array;
        std::shared_ptr<arrow::Array> created_at_array;
        std::shared_ptr<arrow::Array> updated_at_array;
        std::shared_ptr<arrow::Array> tags_array;
        std::shared_ptr<arrow::Array> owner_array;
        std::shared_ptr<arrow::Array> permissions_array;
        std::shared_ptr<arrow::Array> category_array;
        std::shared_ptr<arrow::Array> score_array;
        std::shared_ptr<arrow::Array> status_array;
        std::shared_ptr<arrow::Array> custom_array;
        std::shared_ptr<arrow::Array> shard_array;
        std::shared_ptr<arrow::Array> replicas_array;
        std::shared_ptr<arrow::Array> version_array;
        std::shared_ptr<arrow::Array> deleted_array;
        
        ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
        ARROW_RETURN_NOT_OK(values_builder.Finish(&values_array));
        ARROW_RETURN_NOT_OK(metadata_source_builder.Finish(&metadata_source_array));
        ARROW_RETURN_NOT_OK(created_at_builder.Finish(&created_at_array));
        ARROW_RETURN_NOT_OK(updated_at_builder.Finish(&updated_at_array));
        ARROW_RETURN_NOT_OK(tags_builder.Finish(&tags_array));
        ARROW_RETURN_NOT_OK(owner_builder.Finish(&owner_array));
        ARROW_RETURN_NOT_OK(permissions_builder.Finish(&permissions_array));
        ARROW_RETURN_NOT_OK(category_builder.Finish(&category_array));
        ARROW_RETURN_NOT_OK(score_builder.Finish(&score_array));
        ARROW_RETURN_NOT_OK(status_builder.Finish(&status_array));
        ARROW_RETURN_NOT_OK(custom_builder.Finish(&custom_array));
        ARROW_RETURN_NOT_OK(shard_builder.Finish(&shard_array));
        ARROW_RETURN_NOT_OK(replicas_builder.Finish(&replicas_array));
        ARROW_RETURN_NOT_OK(version_builder.Finish(&version_array));
        ARROW_RETURN_NOT_OK(deleted_builder.Finish(&deleted_array));
        
        // Create RecordBatch
        std::vector<std::shared_ptr<arrow::Array>> arrays = {
            id_array, values_array, metadata_source_array, created_at_array, updated_at_array,
            tags_array, owner_array, permissions_array, category_array, score_array,
            status_array, custom_array, shard_array, replicas_array, version_array, deleted_array
        };
        
        return arrow::RecordBatch::Make(schema, 1, arrays);
    }
    
    arrow::Result<Vector> ArrowConverter::record_batch_to_vector(
        const std::shared_ptr<arrow::RecordBatch>& batch, int row_index) {
        Vector vector;
        
        // Extract values from the RecordBatch
        // This is a simplified implementation - in practice, we would need to handle
        // null values, type checking, and proper error handling
        
        // Extract ID
        auto id_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(0));
        if (id_array && row_index < id_array->length()) {
            vector.id = id_array->GetString(row_index);
        }
        
        // Extract values (vector data)
        auto values_list_array = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(1));
        if (values_list_array && row_index < values_list_array->length()) {
            auto float_array = std::dynamic_pointer_cast<arrow::FloatArray>(
                values_list_array->values());
            if (float_array) {
                int32_t start = values_list_array->value_offset(row_index);
                int32_t end = values_list_array->value_offset(row_index + 1);
                vector.values.resize(end - start);
                for (int32_t i = start; i < end; ++i) {
                    vector.values[i - start] = float_array->Value(i);
                }
            }
        }
        
        // Extract other fields as needed
        // This is a simplified implementation - in practice, we would extract all fields
        
        return vector;
    }
    
    // Other converter methods would be implemented similarly
    // For brevity, I'm providing placeholder implementations
    
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> ArrowConverter::vector_batch_to_record_batch(
        const std::vector<Vector>& vectors) {
        if (vectors.empty()) {
            return arrow::Status::Invalid("Cannot create RecordBatch from empty vector batch");
        }

        auto schema = ArrowSchemaFactory::create_vector_schema();

        // Create builders for each field
        arrow::StringBuilder id_builder;
        arrow::ListBuilder values_builder(arrow::default_memory_pool(),
                                          std::make_shared<arrow::FloatBuilder>());
        arrow::StringBuilder metadata_source_builder;
        arrow::TimestampBuilder created_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                   arrow::default_memory_pool());
        arrow::TimestampBuilder updated_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                   arrow::default_memory_pool());
        arrow::ListBuilder tags_builder(arrow::default_memory_pool(),
                                        std::make_shared<arrow::StringBuilder>());
        arrow::StringBuilder owner_builder;
        arrow::ListBuilder permissions_builder(arrow::default_memory_pool(),
                                              std::make_shared<arrow::StringBuilder>());
        arrow::StringBuilder category_builder;
        arrow::FloatBuilder score_builder;
        arrow::StringBuilder status_builder;
        arrow::StringBuilder custom_builder;
        arrow::StringBuilder shard_builder;
        arrow::ListBuilder replicas_builder(arrow::default_memory_pool(),
                                            std::make_shared<arrow::StringBuilder>());
        arrow::Int32Builder version_builder;
        arrow::BooleanBuilder deleted_builder;

        // Iterate through vectors and append to builders
        for (const auto& vector : vectors) {
            ARROW_RETURN_NOT_OK(id_builder.Append(vector.id));

            // Append vector values
            auto float_builder = std::dynamic_pointer_cast<arrow::FloatBuilder>(
                values_builder.value_builder());
            ARROW_RETURN_NOT_OK(values_builder.Append());
            ARROW_RETURN_NOT_OK(float_builder->AppendValues(vector.values));

            ARROW_RETURN_NOT_OK(metadata_source_builder.Append(vector.metadata.source));
            ARROW_RETURN_NOT_OK(created_at_builder.Append(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::from_time_t(0).time_since_epoch()).count()));
            ARROW_RETURN_NOT_OK(updated_at_builder.Append(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::from_time_t(0).time_since_epoch()).count()));

            // Append tags
            ARROW_RETURN_NOT_OK(tags_builder.Append());
            auto tag_builder = std::dynamic_pointer_cast<arrow::StringBuilder>(
                tags_builder.value_builder());
            for (const auto& tag : vector.metadata.tags) {
                ARROW_RETURN_NOT_OK(tag_builder->Append(tag));
            }

            ARROW_RETURN_NOT_OK(owner_builder.Append(vector.metadata.owner));

            // Append permissions
            ARROW_RETURN_NOT_OK(permissions_builder.Append());
            auto perm_builder = std::dynamic_pointer_cast<arrow::StringBuilder>(
                permissions_builder.value_builder());
            for (const auto& perm : vector.metadata.permissions) {
                ARROW_RETURN_NOT_OK(perm_builder->Append(perm));
            }

            ARROW_RETURN_NOT_OK(category_builder.Append(vector.metadata.category));
            ARROW_RETURN_NOT_OK(score_builder.Append(vector.metadata.score));
            ARROW_RETURN_NOT_OK(status_builder.Append(vector.metadata.status));
            ARROW_RETURN_NOT_OK(custom_builder.Append("{}"));
            ARROW_RETURN_NOT_OK(shard_builder.Append(vector.shard));

            // Append replicas
            ARROW_RETURN_NOT_OK(replicas_builder.Append());
            auto replica_builder = std::dynamic_pointer_cast<arrow::StringBuilder>(
                replicas_builder.value_builder());
            for (const auto& replica : vector.replicas) {
                ARROW_RETURN_NOT_OK(replica_builder->Append(replica));
            }

            ARROW_RETURN_NOT_OK(version_builder.Append(vector.version));
            ARROW_RETURN_NOT_OK(deleted_builder.Append(vector.deleted));
        }

        // Finish building arrays
        std::shared_ptr<arrow::Array> id_array, values_array, metadata_source_array;
        std::shared_ptr<arrow::Array> created_at_array, updated_at_array, tags_array;
        std::shared_ptr<arrow::Array> owner_array, permissions_array, category_array;
        std::shared_ptr<arrow::Array> score_array, status_array, custom_array;
        std::shared_ptr<arrow::Array> shard_array, replicas_array, version_array, deleted_array;

        ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
        ARROW_RETURN_NOT_OK(values_builder.Finish(&values_array));
        ARROW_RETURN_NOT_OK(metadata_source_builder.Finish(&metadata_source_array));
        ARROW_RETURN_NOT_OK(created_at_builder.Finish(&created_at_array));
        ARROW_RETURN_NOT_OK(updated_at_builder.Finish(&updated_at_array));
        ARROW_RETURN_NOT_OK(tags_builder.Finish(&tags_array));
        ARROW_RETURN_NOT_OK(owner_builder.Finish(&owner_array));
        ARROW_RETURN_NOT_OK(permissions_builder.Finish(&permissions_array));
        ARROW_RETURN_NOT_OK(category_builder.Finish(&category_array));
        ARROW_RETURN_NOT_OK(score_builder.Finish(&score_array));
        ARROW_RETURN_NOT_OK(status_builder.Finish(&status_array));
        ARROW_RETURN_NOT_OK(custom_builder.Finish(&custom_array));
        ARROW_RETURN_NOT_OK(shard_builder.Finish(&shard_array));
        ARROW_RETURN_NOT_OK(replicas_builder.Finish(&replicas_array));
        ARROW_RETURN_NOT_OK(version_builder.Finish(&version_array));
        ARROW_RETURN_NOT_OK(deleted_builder.Finish(&deleted_array));

        // Create RecordBatch
        std::vector<std::shared_ptr<arrow::Array>> arrays = {
            id_array, values_array, metadata_source_array, created_at_array, updated_at_array,
            tags_array, owner_array, permissions_array, category_array, score_array,
            status_array, custom_array, shard_array, replicas_array, version_array, deleted_array
        };

        return arrow::RecordBatch::Make(schema, vectors.size(), arrays);
    }
    
    arrow::Result<std::vector<Vector>> ArrowConverter::record_batch_to_vector_batch(
        const std::shared_ptr<arrow::RecordBatch>& batch) {
        std::vector<Vector> vectors;
        vectors.reserve(batch->num_rows());

        // Convert each row in the batch to a Vector
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            ARROW_ASSIGN_OR_RAISE(Vector vector, record_batch_to_vector(batch, row));
            vectors.push_back(std::move(vector));
        }

        return vectors;
    }
    
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> ArrowConverter::database_to_record_batch(
        const Database& database) {
        auto schema = ArrowSchemaFactory::create_database_schema();

        // Create builders
        arrow::StringBuilder database_id_builder, name_builder, description_builder;
        arrow::Int32Builder vector_dimension_builder, num_shards_builder, replication_factor_builder;
        arrow::StringBuilder index_type_builder, index_parameters_builder;
        arrow::StringBuilder sharding_strategy_builder;
        arrow::BooleanBuilder sync_replication_builder;
        arrow::StringBuilder embedding_models_builder, metadata_schema_builder;
        arrow::StringBuilder retention_policy_builder, access_control_builder;
        arrow::TimestampBuilder created_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                   arrow::default_memory_pool());
        arrow::TimestampBuilder updated_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                   arrow::default_memory_pool());

        // Append database data
        ARROW_RETURN_NOT_OK(database_id_builder.Append(database.databaseId));
        ARROW_RETURN_NOT_OK(name_builder.Append(database.name));
        ARROW_RETURN_NOT_OK(description_builder.Append(database.description));
        ARROW_RETURN_NOT_OK(vector_dimension_builder.Append(database.vectorDimension));
        ARROW_RETURN_NOT_OK(index_type_builder.Append(database.indexType));

        // Convert index parameters map to JSON string
        std::string index_params_json = "{";
        bool first = true;
        for (const auto& [key, value] : database.indexParameters) {
            if (!first) index_params_json += ",";
            index_params_json += "\"" + key + "\":\"" + value + "\"";
            first = false;
        }
        index_params_json += "}";
        ARROW_RETURN_NOT_OK(index_parameters_builder.Append(index_params_json));

        ARROW_RETURN_NOT_OK(sharding_strategy_builder.Append(database.sharding.strategy));
        ARROW_RETURN_NOT_OK(num_shards_builder.Append(database.sharding.numShards));
        ARROW_RETURN_NOT_OK(replication_factor_builder.Append(database.replication.factor));
        ARROW_RETURN_NOT_OK(sync_replication_builder.Append(database.replication.sync));

        // Convert embedding models to JSON string
        ARROW_RETURN_NOT_OK(embedding_models_builder.Append("[]")); // Simplified

        // Convert metadata schema to JSON string
        std::string metadata_schema_json = "{";
        first = true;
        for (const auto& [key, value] : database.metadataSchema) {
            if (!first) metadata_schema_json += ",";
            metadata_schema_json += "\"" + key + "\":\"" + value + "\"";
            first = false;
        }
        metadata_schema_json += "}";
        ARROW_RETURN_NOT_OK(metadata_schema_builder.Append(metadata_schema_json));

        ARROW_RETURN_NOT_OK(retention_policy_builder.Append("{}"));
        ARROW_RETURN_NOT_OK(access_control_builder.Append("{}"));
        ARROW_RETURN_NOT_OK(created_at_builder.Append(0));
        ARROW_RETURN_NOT_OK(updated_at_builder.Append(0));

        // Finish arrays
        std::shared_ptr<arrow::Array> database_id_array, name_array, description_array;
        std::shared_ptr<arrow::Array> vector_dimension_array, index_type_array, index_parameters_array;
        std::shared_ptr<arrow::Array> sharding_strategy_array, num_shards_array;
        std::shared_ptr<arrow::Array> replication_factor_array, sync_replication_array;
        std::shared_ptr<arrow::Array> embedding_models_array, metadata_schema_array;
        std::shared_ptr<arrow::Array> retention_policy_array, access_control_array;
        std::shared_ptr<arrow::Array> created_at_array, updated_at_array;

        ARROW_RETURN_NOT_OK(database_id_builder.Finish(&database_id_array));
        ARROW_RETURN_NOT_OK(name_builder.Finish(&name_array));
        ARROW_RETURN_NOT_OK(description_builder.Finish(&description_array));
        ARROW_RETURN_NOT_OK(vector_dimension_builder.Finish(&vector_dimension_array));
        ARROW_RETURN_NOT_OK(index_type_builder.Finish(&index_type_array));
        ARROW_RETURN_NOT_OK(index_parameters_builder.Finish(&index_parameters_array));
        ARROW_RETURN_NOT_OK(sharding_strategy_builder.Finish(&sharding_strategy_array));
        ARROW_RETURN_NOT_OK(num_shards_builder.Finish(&num_shards_array));
        ARROW_RETURN_NOT_OK(replication_factor_builder.Finish(&replication_factor_array));
        ARROW_RETURN_NOT_OK(sync_replication_builder.Finish(&sync_replication_array));
        ARROW_RETURN_NOT_OK(embedding_models_builder.Finish(&embedding_models_array));
        ARROW_RETURN_NOT_OK(metadata_schema_builder.Finish(&metadata_schema_array));
        ARROW_RETURN_NOT_OK(retention_policy_builder.Finish(&retention_policy_array));
        ARROW_RETURN_NOT_OK(access_control_builder.Finish(&access_control_array));
        ARROW_RETURN_NOT_OK(created_at_builder.Finish(&created_at_array));
        ARROW_RETURN_NOT_OK(updated_at_builder.Finish(&updated_at_array));

        std::vector<std::shared_ptr<arrow::Array>> arrays = {
            database_id_array, name_array, description_array, vector_dimension_array,
            index_type_array, index_parameters_array, sharding_strategy_array, num_shards_array,
            replication_factor_array, sync_replication_array, embedding_models_array,
            metadata_schema_array, retention_policy_array, access_control_array,
            created_at_array, updated_at_array
        };

        return arrow::RecordBatch::Make(schema, 1, arrays);
    }
    
    arrow::Result<Database> ArrowConverter::record_batch_to_database(
        const std::shared_ptr<arrow::RecordBatch>& batch, int row_index) {
        if (row_index >= batch->num_rows()) {
            return arrow::Status::Invalid("Row index out of bounds");
        }

        Database database;

        // Extract database_id
        auto database_id_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(0));
        if (database_id_array && row_index < database_id_array->length()) {
            database.databaseId = database_id_array->GetString(row_index);
        }

        // Extract name
        auto name_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(1));
        if (name_array && row_index < name_array->length()) {
            database.name = name_array->GetString(row_index);
        }

        // Extract description
        auto description_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(2));
        if (description_array && row_index < description_array->length()) {
            database.description = description_array->GetString(row_index);
        }

        // Extract vector_dimension
        auto vector_dimension_array = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(3));
        if (vector_dimension_array && row_index < vector_dimension_array->length()) {
            database.vectorDimension = vector_dimension_array->Value(row_index);
        }

        // Extract index_type
        auto index_type_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(4));
        if (index_type_array && row_index < index_type_array->length()) {
            database.indexType = index_type_array->GetString(row_index);
        }

        // Extract sharding_strategy
        auto sharding_strategy_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(6));
        if (sharding_strategy_array && row_index < sharding_strategy_array->length()) {
            database.sharding.strategy = sharding_strategy_array->GetString(row_index);
        }

        // Extract num_shards
        auto num_shards_array = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(7));
        if (num_shards_array && row_index < num_shards_array->length()) {
            database.sharding.numShards = num_shards_array->Value(row_index);
        }

        // Extract replication_factor
        auto replication_factor_array = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(8));
        if (replication_factor_array && row_index < replication_factor_array->length()) {
            database.replication.factor = replication_factor_array->Value(row_index);
        }

        // Extract sync_replication
        auto sync_replication_array = std::dynamic_pointer_cast<arrow::BooleanArray>(batch->column(9));
        if (sync_replication_array && row_index < sync_replication_array->length()) {
            database.replication.sync = sync_replication_array->Value(row_index);
        }

        return database;
    }
    
    arrow::Result<std::shared_ptr<arrow::RecordBatch>> ArrowConverter::index_to_record_batch(
        const Index& index) {
        auto schema = ArrowSchemaFactory::create_index_schema();

        // Create builders
        arrow::StringBuilder index_id_builder, database_id_builder, type_builder;
        arrow::StringBuilder parameters_builder, status_builder;
        arrow::TimestampBuilder created_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                   default_memory_pool());
        arrow::TimestampBuilder updated_at_builder(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                   default_memory_pool());
        arrow::Int64Builder vector_count_builder, size_bytes_builder;
        arrow::Int32Builder build_time_ms_builder;

        // Append index data
        ARROW_RETURN_NOT_OK(index_id_builder.Append(index.indexId));
        ARROW_RETURN_NOT_OK(database_id_builder.Append(index.databaseId));
        ARROW_RETURN_NOT_OK(type_builder.Append(index.type));

        // Convert parameters map to JSON string
        std::string params_json = "{";
        bool first = true;
        for (const auto& [key, value] : index.parameters) {
            if (!first) params_json += ",";
            params_json += "\"" + key + "\":\"" + value + "\"";
            first = false;
        }
        params_json += "}";
        ARROW_RETURN_NOT_OK(parameters_builder.Append(params_json));

        ARROW_RETURN_NOT_OK(status_builder.Append(index.status));
        ARROW_RETURN_NOT_OK(created_at_builder.Append(0));
        ARROW_RETURN_NOT_OK(updated_at_builder.Append(0));

        // Append statistics
        if (index.stats) {
            ARROW_RETURN_NOT_OK(vector_count_builder.Append(index.stats->vectorCount));
            ARROW_RETURN_NOT_OK(size_bytes_builder.Append(index.stats->sizeBytes));
            ARROW_RETURN_NOT_OK(build_time_ms_builder.Append(index.stats->buildTimeMs));
        } else {
            ARROW_RETURN_NOT_OK(vector_count_builder.Append(0));
            ARROW_RETURN_NOT_OK(size_bytes_builder.Append(0));
            ARROW_RETURN_NOT_OK(build_time_ms_builder.Append(0));
        }

        // Finish arrays
        std::shared_ptr<arrow::Array> index_id_array, database_id_array, type_array;
        std::shared_ptr<arrow::Array> parameters_array, status_array;
        std::shared_ptr<arrow::Array> created_at_array, updated_at_array;
        std::shared_ptr<arrow::Array> vector_count_array, size_bytes_array, build_time_ms_array;

        ARROW_RETURN_NOT_OK(index_id_builder.Finish(&index_id_array));
        ARROW_RETURN_NOT_OK(database_id_builder.Finish(&database_id_array));
        ARROW_RETURN_NOT_OK(type_builder.Finish(&type_array));
        ARROW_RETURN_NOT_OK(parameters_builder.Finish(&parameters_array));
        ARROW_RETURN_NOT_OK(status_builder.Finish(&status_array));
        ARROW_RETURN_NOT_OK(created_at_builder.Finish(&created_at_array));
        ARROW_RETURN_NOT_OK(updated_at_builder.Finish(&updated_at_array));
        ARROW_RETURN_NOT_OK(vector_count_builder.Finish(&vector_count_array));
        ARROW_RETURN_NOT_OK(size_bytes_builder.Finish(&size_bytes_array));
        ARROW_RETURN_NOT_OK(build_time_ms_builder.Finish(&build_time_ms_array));

        std::vector<std::shared_ptr<arrow::Array>> arrays = {
            index_id_array, database_id_array, type_array, parameters_array,
            status_array, created_at_array, updated_at_array,
            vector_count_array, size_bytes_array, build_time_ms_array
        };

        return arrow::RecordBatch::Make(schema, 1, arrays);
    }
    
    arrow::Result<Index> ArrowConverter::record_batch_to_index(
        const std::shared_ptr<arrow::RecordBatch>& batch, int row_index) {
        if (row_index >= batch->num_rows()) {
            return arrow::Status::Invalid("Row index out of bounds");
        }

        Index index;

        // Extract index_id
        auto index_id_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(0));
        if (index_id_array && row_index < index_id_array->length()) {
            index.indexId = index_id_array->GetString(row_index);
        }

        // Extract database_id
        auto database_id_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(1));
        if (database_id_array && row_index < database_id_array->length()) {
            index.databaseId = database_id_array->GetString(row_index);
        }

        // Extract type
        auto type_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(2));
        if (type_array && row_index < type_array->length()) {
            index.type = type_array->GetString(row_index);
        }

        // Extract status
        auto status_array = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(4));
        if (status_array && row_index < status_array->length()) {
            index.status = status_array->GetString(row_index);
        }

        // Extract statistics if available
        auto vector_count_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(7));
        auto size_bytes_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(8));
        auto build_time_ms_array = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(9));

        if (vector_count_array && size_bytes_array && build_time_ms_array &&
            row_index < vector_count_array->length()) {
            index.stats = std::make_unique<Index::Stats>();
            index.stats->vectorCount = vector_count_array->Value(row_index);
            index.stats->sizeBytes = size_bytes_array->Value(row_index);
            index.stats->buildTimeMs = build_time_ms_array->Value(row_index);
        }

        return index;
    }

} // namespace arrow_utils

} // namespace jadevectordb